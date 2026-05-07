#!/usr/bin/env python3
"""Build SE(3) toy unified metric table aligned with KLIFS-style reporting.

All outputs are written under toy_finalized/se3/.  Legacy SPD/toy_experiment
files are read only.
"""
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
LEGACY_FINAL = RELEASE_ROOT / "legacy"
ROOT = RELEASE_ROOT
LEGACY_CMF = LEGACY_FINAL / "cmf_comparison"
LEGACY_OUT = LEGACY_CMF / "output" / "se3"

sys.path.insert(0, str(LEGACY_FINAL))
sys.path.insert(0, str(LEGACY_CMF))

import config as cfg  # noqa: E402
import run_se3_cmf as old_se3  # noqa: E402
from cmf_module import load_cmf_results, plot_cmf_latent  # noqa: E402


METHOD_ORDER_EXTENDED = [
    "Original",
    "Oracle",
    "CMF",
    "CMF+Proj",
    "SFID",
    "SFID+Proj",
    "SPD",
    "SPD+Proj",
    "Geodesic",
    "IsoRot",
]

METHOD_ORDER_COMPACT = [
    "Original",
    "Oracle",
    "CMF",
    "SFID",
    "SPD",
    "IsoRot",
]

DISPLAY_NAME = {"IsoRot": "IsoAct"}

REPORT_METRICS = [
    "Linear Probe-Acc ↓",
    "Linear Probe-Recall ↓",
    "Linear Probe-F1 ↓",
    "Linear Probe-AUROC ↓",
    "Task MAE ↓",
    "Task In-Band ↑",
    "SO(3) Orthogonality Error Mean ↓",
    "SO(3) Orthogonality Error Max ↓",
    "SO(3) Determinant Error Mean ↓",
    "SO(3) Determinant Error Max ↓",
]

# Keep additional raw diagnostics in JSON/all-method CSV for reproducibility,
# but omit them from final compact/extended report tables.
RAW_EXTRA_METRICS = [
    "MLP Probe-Acc ↓",
    "MLP Probe-F1 ↓",
    "Translation Validity ↑",
    "SE(3) Valid Rate ↑",
    "SE(3) Valid Rate @1e-6",
]

COLS = ["Method"] + REPORT_METRICS
ALL_COLS = ["Method"] + REPORT_METRICS + RAW_EXTRA_METRICS

LOWER_IS_BETTER = {
    "Linear Probe-Acc ↓",
    "Linear Probe-Recall ↓",
    "Linear Probe-F1 ↓",
    "Linear Probe-AUROC ↓",
    "MLP Probe-Acc ↓",
    "MLP Probe-F1 ↓",
    "Task MAE ↓",
    "SO(3) Orthogonality Error Mean ↓",
    "SO(3) Orthogonality Error Max ↓",
    "SO(3) Determinant Error Mean ↓",
    "SO(3) Determinant Error Max ↓",
}

ORTH_TOL = 1e-10
DET_TOL = 1e-10


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure():
    OUT.mkdir(parents=True, exist_ok=True)


def finite_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def fmt(x):
    v = finite_float(x)
    if v is None:
        return ""
    if abs(v) < 1e-4 and v != 0:
        return f"{v:.3e}"
    return f"{v:.4f}"


def metric_arrays(X):
    R = X[:, :9].reshape(-1, 3, 3)
    t = X[:, 9:12]
    I = np.eye(3)
    orth = np.linalg.norm(np.matmul(np.transpose(R, (0, 2, 1)), R) - I[None, :, :], axis=(1, 2))
    det_err = np.abs(np.linalg.det(R) - 1.0)
    t_finite = np.isfinite(t).all(axis=1)
    return orth, det_err, t_finite


def structural_metrics(X):
    orth, det_err, t_finite = metric_arrays(X)
    se3_valid = (orth <= ORTH_TOL) & (det_err <= DET_TOL) & t_finite
    se3_valid_1e6 = (orth <= 1e-6) & (det_err <= 1e-6) & t_finite
    return {
        "SO(3) Orthogonality Error Mean ↓": float(np.mean(orth)),
        "SO(3) Orthogonality Error Max ↓": float(np.max(orth)),
        "SO(3) Determinant Error Mean ↓": float(np.mean(det_err)),
        "SO(3) Determinant Error Max ↓": float(np.max(det_err)),
        "Translation Validity ↑": float(np.mean(t_finite)),
        "SE(3) Valid Rate ↑": float(np.mean(se3_valid)),
        "SE(3) Valid Rate @1e-6": float(np.mean(se3_valid_1e6)),
    }


def project_to_so3_float64(M):
    """Nearest SO(3) projection using SVD/polar decomposition in float64."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def project_all_se3_float64(X):
    """Project R to SO(3) in float64 and keep translation unchanged."""
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X, dtype=np.float64)
    for i, row in enumerate(X):
        R = project_to_so3_float64(row[:9].reshape(3, 3))
        out[i, :9] = R.reshape(-1)
        out[i, 9:12] = row[9:12]
    return out


def probe_metrics(X, groups):
    Xtr, Xte, gtr, gte = train_test_split(
        X,
        groups,
        test_size=cfg.PROBE_TEST_SIZE,
        stratify=groups,
        random_state=cfg.PROBE_RANDOM_STATE,
    )
    lin = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        C=0.1,
        random_state=cfg.PROBE_RANDOM_STATE,
    )
    lin.fit(Xtr, gtr)
    pred_lin = lin.predict(Xte)
    if hasattr(lin, "predict_proba"):
        score_lin = lin.predict_proba(Xte)[:, 1]
    else:
        score_lin = lin.decision_function(Xte)

    mlp = MLPClassifier(
        hidden_layer_sizes=cfg.MLP_HIDDEN,
        max_iter=cfg.MLP_MAX_ITER,
        random_state=cfg.PROBE_RANDOM_STATE,
    )
    mlp.fit(Xtr, gtr)
    pred_mlp = mlp.predict(Xte)
    return {
        "Linear Probe-Acc ↓": float(accuracy_score(gte, pred_lin)),
        "Linear Probe-Recall ↓": float(recall_score(gte, pred_lin, average="macro", zero_division=0)),
        "Linear Probe-F1 ↓": float(f1_score(gte, pred_lin, average="macro", zero_division=0)),
        "Linear Probe-AUROC ↓": float(roc_auc_score(gte, score_lin)),
        "MLP Probe-Acc ↓": float(accuracy_score(gte, pred_mlp)),
        "MLP Probe-F1 ↓": float(f1_score(gte, pred_mlp, average="macro", zero_division=0)),
    }


def task_metrics(X_orig, X_new):
    _, tnorm_orig = old_se3.extract_phi_tnorm(X_orig)
    _, tnorm_new = old_se3.extract_phi_tnorm(X_new)
    return {
        "Task MAE ↓": float(np.mean(np.abs(tnorm_new - tnorm_orig))),
        "Task In-Band ↑": float(np.mean((tnorm_new >= 1.5) & (tnorm_new <= 2.5))),
    }


def build_method_arrays():
    """Regenerate toy data/methods deterministically and load cached CMF output."""
    X, groups, phi_all, t_mag, task_labels = old_se3.generate_data()
    neutral, bias_dir, sfid_idx = old_se3.compute_neutral(X, groups)

    cache_bl = LEGACY_OUT / "cache_baseline_ae.pt"
    cache_cmf = LEGACY_OUT / "cache_cmf.pt"
    baseline_cached = load_cmf_results(str(cache_bl), input_dim=12, hidden_dim=64)
    cmf_cached = load_cmf_results(str(cache_cmf), input_dim=12, hidden_dim=64)
    if baseline_cached is None or cmf_cached is None:
        raise RuntimeError("Legacy CMF caches are required for deterministic no-retrain consolidation.")
    _bl_model, bl_native_metrics, bl_rec, bl_z = baseline_cached
    _cmf_model, cmf_native_metrics, cmf_rec, cmf_z = cmf_cached

    raw = old_se3.run_all_methods(X, neutral, bias_dir, sfid_idx, cfg.PRIMARY_ALPHA, bl_rec, cmf_rec)
    arrays = {
        "Original": raw["Original"],
        "Oracle": raw["Oracle"],
        "CMF": np.asarray(cmf_rec),
        # KLIFS-style CMF+Proj: SVD/polar nearest-SO(3) projection in
        # float64, with translation unchanged.  Avoid preserving float32
        # reconstruction dtype because the main SE(3) validity threshold is
        # intentionally strict (1e-10).
        "CMF+Proj": project_all_se3_float64(np.asarray(cmf_rec)),
        "SFID": raw["SFID"],
        "SFID+Proj": raw["SFID+Proj"],
        "SPD": raw["SPD"],
        "SPD+Proj": raw["SPD+Proj"],
        "Geodesic": raw["Geodesic"],
        "IsoRot": raw["IsoRot"],
    }
    meta = {
        "seed": cfg.SEED,
        "n_per_group": cfg.N_PER_GROUP,
        "primary_alpha": cfg.PRIMARY_ALPHA,
        "alphas": cfg.ALPHAS,
        "probe_test_size": cfg.PROBE_TEST_SIZE,
        "probe_random_state": cfg.PROBE_RANDOM_STATE,
        "cmf_cache": str(cache_cmf),
        "baseline_cache": str(cache_bl),
        "cmf_native_metrics": cmf_native_metrics,
        "baseline_native_metrics": bl_native_metrics,
    }
    aux = {
        "X_orig": X,
        "groups": groups,
        "t_mag": t_mag,
        "task_labels": task_labels,
        "neutral": neutral,
        "bias_dir": bias_dir,
        "sfid_idx": sfid_idx,
        "bl_z": bl_z,
        "cmf_z": cmf_z,
    }
    return arrays, aux, meta


def compute_rows(arrays, X_orig, groups):
    rows = []
    for method in METHOD_ORDER_EXTENDED:
        X = arrays[method]
        row = {"Method": method}
        row.update(probe_metrics(X, groups))
        row.update(task_metrics(X_orig, X))
        row.update(structural_metrics(X))
        rows.append(row)
    return rows


def display_method(method):
    return DISPLAY_NAME.get(method, method)


def select_rows(rows, order):
    by_method = {r["Method"]: r for r in rows}
    return [by_method[m] for m in order if m in by_method]


def write_csv_file(rows, path, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in columns}
            out["Method"] = display_method(r["Method"])
            w.writerow(out)
    return path


def write_csvs(rows):
    compact = select_rows(rows, METHOD_ORDER_COMPACT)
    extended = select_rows(rows, METHOD_ORDER_EXTENDED)
    paths = {
        "compact": write_csv_file(compact, OUT / "se3_unified_metrics_compact.csv", COLS),
        "extended": write_csv_file(extended, OUT / "se3_unified_metrics_extended.csv", COLS),
        "all_methods": write_csv_file(extended, OUT / "se3_unified_metrics_all_methods.csv", ALL_COLS),
        # Backward-compatible representative table: extended report columns.
        "main": write_csv_file(extended, OUT / "se3_unified_metrics.csv", COLS),
    }
    return paths


def write_json(rows, meta, old_diff):
    path = OUT / "se3_unified_metrics.json"
    payload = {
        "generated_utc": now(),
        "deterministic_config": meta,
        "method_order_extended": METHOD_ORDER_EXTENDED,
        "method_order_compact": METHOD_ORDER_COMPACT,
        "display_name_map": DISPLAY_NAME,
        "report_columns": COLS,
        "all_columns": ALL_COLS,
        "thresholds": {"orth_tol": ORTH_TOL, "det_tol": DET_TOL},
        "metrics": rows,
        "legacy_comparison": old_diff,
        "notes": [
            "Linear Probe-Recall is macro recall; Linear Probe-F1 is macro F1.",
            "Linear Probe-AUROC uses the positive-class predicted probability from the logistic probe.",
            "Bias probe metrics are lower-is-better recoverable bias signal.",
            "CMF+Proj is SVD/polar SO(3) projection of raw CMF R with t unchanged.",
            "No legacy toy outputs were overwritten.",
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def parse_legacy_table():
    path = LEGACY_OUT / "result_table.txt"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 6:
            continue
        m = parts[0]
        if m in {"Method", ""} or m.startswith("-"):
            continue
        vals = {}
        # O (0.0000), O (t_MAE=0.0000), O (t_band=100.0%), 0.500 O
        m0 = re.search(r"\(([-+0-9.eE]+)\)", parts[1])
        m1 = re.search(r"t_MAE=([-+0-9.eE]+)", parts[2])
        m2 = re.search(r"t_band=([-+0-9.eE]+)%", parts[3])
        m3 = re.search(r"([-+0-9.eE]+)", parts[4])
        m4 = re.search(r"([-+0-9.eE]+)", parts[5])
        if m0:
            vals["old_on_manifold_mean"] = float(m0.group(1))
        if m1:
            vals["old_task_mae"] = float(m1.group(1))
        if m2:
            vals["old_task_inband"] = float(m2.group(1)) / 100.0
        if m3:
            vals["old_linear_acc"] = float(m3.group(1))
        if m4:
            vals["old_mlp_acc"] = float(m4.group(1))
        out[m] = vals
    return out


def compare_legacy(rows):
    old = parse_legacy_table()
    diffs = []
    map_cols = {
        "Task MAE ↓": "old_task_mae",
        "Task In-Band ↑": "old_task_inband",
        "Linear Probe-Acc ↓": "old_linear_acc",
        "MLP Probe-Acc ↓": "old_mlp_acc",
        "SO(3) Orthogonality Error Mean ↓": "old_on_manifold_mean",
    }
    for r in rows:
        m = r["Method"]
        if m not in old:
            diffs.append({"method": m, "status": "not_in_legacy_table"})
            continue
        for new_col, old_col in map_cols.items():
            old_val = old[m].get(old_col)
            new_val = r.get(new_col)
            if old_val is None:
                continue
            diffs.append(
                {
                    "method": m,
                    "metric": new_col,
                    "new": new_val,
                    "legacy": old_val,
                    "abs_diff": abs(float(new_val) - float(old_val)),
                }
            )
    return diffs


REFERENCE_METHODS = {"Original", "Oracle"}
DEBIAS_METRICS = {
    "Linear Probe-Acc ↓",
    "Linear Probe-Recall ↓",
    "Linear Probe-F1 ↓",
    "Linear Probe-AUROC ↓",
    "MLP Probe-Acc ↓",
    "MLP Probe-F1 ↓",
}
SO3_ERROR_METRICS = {
    "SO(3) Orthogonality Error Mean ↓",
    "SO(3) Orthogonality Error Max ↓",
    "SO(3) Determinant Error Mean ↓",
    "SO(3) Determinant Error Max ↓",
}
ANNOTATION_TOL = 1e-12


def _reference_value(rows, method, metric):
    for r in rows:
        if r.get("Method") == method:
            return finite_float(r.get(metric))
    return None


def _oracle_deviation(rows, metric, value):
    oracle = _reference_value(rows, "Oracle", metric)
    if oracle is None:
        return value
    return abs(value - oracle)


def display_metric_label(metric):
    if metric in DEBIAS_METRICS:
        return metric.replace(" ↓", " (raw)")
    return metric


def display_metric_value(rows, row, metric):
    # Markdown tables display raw metric values.  Only bold/underline
    # annotation for debiasing metrics uses Oracle deviation.
    return finite_float(row.get(metric))


def _effective_annotation_value(rows, metric, method, value):
    """Return value used only for bold/underline annotation.

    Original and Oracle are reference rows and never receive annotations.
    Linear-probe debiasing metrics are compared by absolute deviation from
    Oracle/chance-level probe performance; closer to Oracle is better. SO(3)
    error metrics use Original as a numerical-precision cap: any method at or
    below Original error is treated as tied at the cap.
    """
    if method in REFERENCE_METHODS:
        return None
    if metric in DEBIAS_METRICS:
        return _oracle_deviation(rows, metric, value)
    if metric in SO3_ERROR_METRICS:
        cap = _reference_value(rows, "Original", metric)
        if cap is not None and value <= cap + ANNOTATION_TOL:
            return cap
    return value


def best_second_annotations(rows, metric):
    vals = []
    for r in rows:
        method = r["Method"]
        v = finite_float(r.get(metric))
        if v is None:
            continue
        eff = _effective_annotation_value(rows, metric, method, v)
        if eff is not None:
            vals.append((method, eff))
    if len(vals) < 1:
        return {}

    reverse = metric not in LOWER_IS_BETTER
    ordered_values = sorted({v for _, v in vals}, reverse=reverse)
    best_v = ordered_values[0]
    second_v = ordered_values[1] if len(ordered_values) > 1 else None

    ann = {}
    for method, eff in vals:
        if abs(eff - best_v) <= ANNOTATION_TOL:
            ann[method] = "best"
        elif second_v is not None and abs(eff - second_v) <= ANNOTATION_TOL:
            ann[method] = "second"
    return ann


def md_table(rows, metrics):
    headers = ["Method"] + [display_metric_label(m) for m in metrics]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    ann = {m: best_second_annotations(rows, m) for m in metrics}
    for r in rows:
        vals = [display_method(r["Method"])]
        for m in metrics:
            s = fmt(display_metric_value(rows, r, m))
            a = ann.get(m, {}).get(r["Method"])
            if a == "best":
                s = f"**{s}**"
            elif a == "second":
                s = f"<u>{s}</u>"
            vals.append(s)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_md_table_file(rows, path, title, meta):
    lines = [
        f"# {title}",
        "",
        f"Generated: {now()} UTC",
        "",
        "- Linear Probe-Recall is **macro recall**.",
        "- Linear Probe-F1 is **macro F1**.",
        "- Linear Probe-AUROC uses logistic-probe positive-class probability.",
        "- Linear probe columns display **raw probe values**. Bold/underline annotations for these columns are computed by absolute deviation from Oracle: `|metric - Oracle metric|`.",
        "- Therefore, for linear probe columns, highlighted values mean closer-to-Oracle/chance-level debiasing, not simply lower raw value.",
        "- Existing internal `IsoRot` outputs are reported as **IsoAct**.",
        "- Bold/underline annotations exclude reference rows `Original` and `Oracle`.",
        "- Debiasing annotations rank non-reference methods by Oracle deviation, not raw probe value.",
        "- SO(3) error annotations use `Original` as a numerical-precision cap: non-reference methods at or below Original error are tied as best.",
        "",
        md_table(rows, REPORT_METRICS),
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def write_md(rows, meta, old_diff, figure_status):
    path = OUT / "se3_unified_metrics.md"
    legacy_large = [d for d in old_diff if d.get("abs_diff", 0) > 5e-4]
    lines = [
        "# SE(3) Toy Unified Metrics",
        "",
        f"Generated: {now()} UTC",
        "",
        "## Protocol",
        "",
        f"- Seed: `{meta['seed']}`",
        f"- Primary alpha: `{meta['primary_alpha']}`",
        f"- Probe split: test_size=`{meta['probe_test_size']}`, random_state=`{meta['probe_random_state']}`",
        "- Linear Probe Recall: **macro recall**",
        "- Linear Probe F1: **macro F1**",
        "- Linear Probe AUROC: binary AUROC using positive-class predicted probability.",
        "- Markdown tables display raw linear probe values. Bold/underline annotations for linear probe columns are computed by absolute deviation from Oracle: `|metric - Oracle metric|`.",
        "- Therefore, for linear probe columns, highlighted values mean closer-to-Oracle/chance-level debiasing, not simply lower raw value.",
        "- Existing internal `IsoRot` outputs are reported as **IsoAct**.",
        "- Bold/underline annotations exclude reference rows `Original` and `Oracle`.",
        "- Debiasing annotations rank non-reference methods by Oracle deviation, not raw probe value.",
        "- SO(3) error annotations use `Original` as a numerical-precision cap: non-reference methods at or below Original error are tied as best.",
        "- Task MAE: `mean(abs(||t_new|| - ||t_orig||))`.",
        "- Task In-Band: fraction with `||t|| in [1.5, 2.5]`, matching the existing toy tolerance band.",
        f"- Structural raw diagnostics still include strict SE(3) valid rate with orthogonality error <= `{ORTH_TOL}`, determinant error <= `{DET_TOL}`, and finite translation, but final report tables now use SO(3) error columns only.",
        "- CMF+Proj: raw CMF reconstruction with R projected to nearest SO(3) by SVD/polar decomposition; t unchanged.",
        "",
        "## Compact metric table",
        "",
        md_table(select_rows(rows, METHOD_ORDER_COMPACT), REPORT_METRICS),
        "",
        "## Extended metric table",
        "",
        md_table(select_rows(rows, METHOD_ORDER_EXTENDED), REPORT_METRICS),
        "",
        "## Float-threshold sensitivity note",
        "",
        "The main SE(3) Valid Rate uses the strict 1e-10 threshold. For learned raw CMF, SO(3) errors are order 1, so the strict threshold is not merely a float32 issue. For projected/original geometric methods, errors are near numerical precision and pass.",
        "",
        "## Legacy table comparison",
        "",
        f"- Legacy comparison source: `{LEGACY_OUT / 'result_table.txt'}`",
        f"- Large differences versus legacy rounded table (>5e-4): `{len(legacy_large)}`.",
    ]
    if legacy_large:
        lines.append("- Differences are expected where the legacy table rounded values to 3–4 decimals, and because this table adds macro F1 and CMF+Proj.")
        lines.append("")
        lines.append("| Method | Metric | New | Legacy | Abs diff |")
        lines.append("| --- | --- | --- | --- | --- |")
        for d in legacy_large[:30]:
            lines.append(f"| {d.get('method')} | {d.get('metric')} | {fmt(d.get('new'))} | {fmt(d.get('legacy'))} | {fmt(d.get('abs_diff'))} |")
    else:
        lines.append("- No material differences beyond legacy rounding for overlapping old metrics.")
    lines += [
        "",
        "## Figure regeneration status",
        "",
    ]
    for item in figure_status:
        lines.append(f"- `{item['path']}`: {item['status']}")
    lines += [
        "",
        "## Files created",
        "",
        "- `build_se3_unified_metrics.py`",
        "- `se3_unified_metrics.csv`",
        "- `se3_unified_metrics_compact.csv`",
        "- `se3_unified_metrics_extended.csv`",
        "- `se3_unified_metrics_all_methods.csv`",
        "- `se3_unified_metrics.json`",
        "- `se3_unified_metrics.md`",
        "- `se3_unified_metrics_compact.md`",
        "- `se3_unified_metrics_extended.md`",
        "- `se3_unified_metrics.png`",
        "- `se3_unified_arrays.npz`",
        "- regenerated figures: `bar_linear.png`, `bar_mlp.png`, `crosssection.png`, `latent_baseline.png`, `latent_cmf.png`, `scatter_a09.png`, `scatter_a10.png`, `scatter_a10_compact.png`, `scatter_a10_extended.png`, `setting.png`, `topdown_a09.png`, `topdown_a10.png`",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def color_for(method):
    base = method.replace("IsoAct", "IsoRot")
    return cfg.METHOD_COLORS.get(base.replace("+Proj", ""), cfg.METHOD_COLORS.get(base, "#999999"))


def savefig(fig, name, status):
    path = OUT / name
    fig.savefig(path, dpi=cfg.FIG_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    status.append({"path": str(path), "status": "regenerated"})


def plot_setting(X, groups, neutral, status):
    phis, tnorms = old_se3.extract_phi_tnorm(X)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for g in [0, 1]:
        mask = groups == g
        ax.scatter(np.cos(phis[mask]), np.sin(phis[mask]), tnorms[mask],
                   c=cfg.GROUP_COLORS[g], s=14, alpha=0.7, label=f"Group {g}")
    phi_n, tn_n = old_se3.extract_phi_tnorm(neutral[None, :])
    ax.scatter([np.cos(phi_n[0])], [np.sin(phi_n[0])], [tn_n[0]],
               c=cfg.NEUTRAL_COLOR, s=200, marker="*", edgecolors="k", linewidths=0.8)
    ax.set_xlabel("cos(φ)")
    ax.set_ylabel("sin(φ)")
    ax.set_zlabel("‖t‖")
    ax.legend()
    savefig(fig, "setting.png", status)


def plot_grid(results, groups, neutral, alpha, kind, filename, status, method_order=None):
    method_order = list(method_order or METHOD_ORDER_EXTENDED)
    n = len(method_order)
    ncols = 5 if n > 6 else 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)
    phi_n, _ = old_se3.extract_phi_tnorm(neutral[None, :])
    for idx, method in enumerate(method_order):
        ax = axes[idx]
        X = results[method]
        if kind == "scatter":
            phis, tnorms = old_se3.extract_phi_tnorm(X)
            ax.axhspan(1.5, 2.5, color="green", alpha=0.06, zorder=0)
            ax.axvline(np.degrees(phi_n[0]), color=cfg.NEUTRAL_COLOR, lw=2, ls="--", alpha=0.7)
            for g in [0, 1]:
                mask = groups == g
                ax.scatter(np.degrees(phis[mask]), tnorms[mask], c=cfg.GROUP_COLORS[g],
                           s=cfg.MARKER_SIZE, alpha=cfg.MARKER_ALPHA, zorder=3)
            ax.set_xlim(-200, 200)
            ax.set_ylim(0, 4)
            ax.set_xlabel("φ (deg)")
            ax.set_ylabel("‖t‖")
        elif kind == "topdown":
            tx, ty = X[:, 9], X[:, 10]
            circ = np.linspace(0, 2 * np.pi, 200)
            ax.plot(1.5 * np.cos(circ), 1.5 * np.sin(circ), "k--", lw=0.8, alpha=0.3)
            ax.plot(2.5 * np.cos(circ), 2.5 * np.sin(circ), "k--", lw=0.8, alpha=0.3)
            for g in [0, 1]:
                mask = groups == g
                ax.scatter(tx[mask], ty[mask], c=cfg.GROUP_COLORS[g],
                           s=cfg.MARKER_SIZE, alpha=cfg.MARKER_ALPHA, zorder=3)
            ax.set_aspect("equal")
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xlabel("t_x")
            ax.set_ylabel("t_y")
        elif kind == "crosssection":
            r00 = X[:, 0]
            tnorms = np.linalg.norm(X[:, 9:12], axis=1)
            ax.axhspan(1.5, 2.5, color="green", alpha=0.06, zorder=0)
            for g in [0, 1]:
                mask = groups == g
                ax.scatter(r00[mask], tnorms[mask], c=cfg.GROUP_COLORS[g],
                           s=cfg.MARKER_SIZE, alpha=cfg.MARKER_ALPHA, zorder=3)
            ax.set_xlabel("R[0,0]")
            ax.set_ylabel("‖t‖")
        ax.set_title(display_method(method))
    for ax in axes[len(method_order):]:
        ax.axis("off")
    fig.suptitle(f"SE(3) {kind} alpha={alpha}", fontweight="bold")
    fig.tight_layout()
    savefig(fig, filename, status)


def plot_bars(rows, metric, ylabel, filename, status):
    methods = [display_method(r["Method"]) for r in rows]
    vals = [r[metric] for r in rows]
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(methods)), vals, color=[color_for(m) for m in methods],
                  edgecolor="black", lw=0.5, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.05, max(vals) + 0.1))
    fig.tight_layout()
    savefig(fig, filename, status)


def plot_unified_heatmap(rows, status):
    metrics = REPORT_METRICS
    data = np.array([[r[m] for m in metrics] for r in rows], dtype=float)
    norm = np.zeros_like(data)
    for j, m in enumerate(metrics):
        col = data[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        if hi - lo < 1e-12:
            norm[:, j] = 0.5
        elif m in LOWER_IS_BETTER:
            norm[:, j] = 1 - (col - lo) / (hi - lo)
        else:
            norm[:, j] = (col - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([display_method(r["Method"]) for r in rows])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_title("SE(3) toy unified metrics (normalized: higher color = better)")
    plt.colorbar(im, ax=ax, label="normalized desirability")
    fig.tight_layout()
    savefig(fig, "se3_unified_metrics.png", status)


def regenerate_figures(rows, arrays, aux):
    status = []
    cfg.setup_style()
    plot_setting(aux["X_orig"], aux["groups"], aux["neutral"], status)
    # For alpha-specific figures regenerate post-hoc methods at alpha .9 and 1.0;
    # CMF/CMF+Proj are training-time cache outputs and do not depend on alpha.
    for alpha in cfg.ALPHAS:
        raw = old_se3.run_all_methods(
            aux["X_orig"], aux["neutral"], aux["bias_dir"], aux["sfid_idx"],
            alpha, arrays["CMF"], arrays["CMF"],
        )
        alpha_results = {
            "Original": raw["Original"],
            "Oracle": raw["Oracle"],
            "CMF": arrays["CMF"],
            "CMF+Proj": arrays["CMF+Proj"],
            "SFID": raw["SFID"],
            "SFID+Proj": raw["SFID+Proj"],
            "SPD": raw["SPD"],
            "SPD+Proj": raw["SPD+Proj"],
            "Geodesic": raw["Geodesic"],
            "IsoRot": raw["IsoRot"],
        }
        tag = f"a{str(alpha).replace('.', '')}"
        plot_grid(alpha_results, aux["groups"], aux["neutral"], alpha, "scatter", f"scatter_{tag}.png", status, METHOD_ORDER_EXTENDED)
        if abs(alpha - cfg.PRIMARY_ALPHA) < 1e-12:
            plot_grid(alpha_results, aux["groups"], aux["neutral"], alpha, "scatter", "scatter_a10_extended.png", status, METHOD_ORDER_EXTENDED)
            plot_grid(alpha_results, aux["groups"], aux["neutral"], alpha, "scatter", "scatter_a10_compact.png", status, METHOD_ORDER_COMPACT)
        plot_grid(alpha_results, aux["groups"], aux["neutral"], alpha, "topdown", f"topdown_{tag}.png", status, METHOD_ORDER_EXTENDED)
        if abs(alpha - cfg.PRIMARY_ALPHA) < 1e-12:
            plot_grid(alpha_results, aux["groups"], aux["neutral"], alpha, "crosssection", "crosssection.png", status, METHOD_ORDER_EXTENDED)
    plot_bars(rows, "Linear Probe-Acc ↓", "Bias Probe Accuracy (Linear)", "bar_linear.png", status)
    plot_bars(rows, "MLP Probe-Acc ↓", "Bias Probe Accuracy (MLP)", "bar_mlp.png", status)
    plot_cmf_latent(aux["bl_z"], aux["groups"], aux["t_mag"], "‖t‖", "Baseline AE",
                    str(OUT / "latent_baseline.png"))
    status.append({"path": str(OUT / "latent_baseline.png"), "status": "regenerated"})
    plot_cmf_latent(aux["cmf_z"], aux["groups"], aux["t_mag"], "‖t‖", "CMF",
                    str(OUT / "latent_cmf.png"))
    status.append({"path": str(OUT / "latent_cmf.png"), "status": "regenerated"})
    plot_unified_heatmap(rows, status)
    return status


def main():
    ensure()
    np.random.seed(cfg.SEED)
    arrays, aux, meta = build_method_arrays()
    rows = compute_rows(arrays, aux["X_orig"], aux["groups"])

    np.savez_compressed(
        OUT / "se3_unified_arrays.npz",
        **{f"X_{k.replace('+', 'plus')}": v for k, v in arrays.items()},
        groups=aux["groups"],
        X_orig=aux["X_orig"],
        split_seed=cfg.SEED,
        primary_alpha=cfg.PRIMARY_ALPHA,
    )

    old_diff = compare_legacy(rows)
    figure_status = regenerate_figures(rows, arrays, aux)

    csv_paths = write_csvs(rows)
    json_path = write_json(rows, meta, old_diff)
    md_path = write_md(rows, meta, old_diff, figure_status)
    compact_md = write_md_table_file(select_rows(rows, METHOD_ORDER_COMPACT), OUT / "se3_unified_metrics_compact.md", "SE(3) Toy Compact Metrics", meta)
    extended_md = write_md_table_file(select_rows(rows, METHOD_ORDER_EXTENDED), OUT / "se3_unified_metrics_extended.md", "SE(3) Toy Extended Metrics", meta)

    for csv_path in csv_paths.values():
        print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {compact_md}")
    print(f"Wrote {extended_md}")
    print(f"Wrote {OUT / 'se3_unified_metrics.png'}")


if __name__ == "__main__":
    main()
