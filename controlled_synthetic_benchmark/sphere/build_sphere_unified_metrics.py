#!/usr/bin/env python3
"""Build hypersphere S^2 toy unified metric tables aligned with SE(3) reporting.

All outputs are written under toy_finalized/sphere/. Legacy SPD/toy_experiment
files are read only.
"""
import csv
import json
import math
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
LEGACY_OUT = LEGACY_CMF / "output" / "sphere"

sys.path.insert(0, str(LEGACY_FINAL))
sys.path.insert(0, str(LEGACY_CMF))

import config as cfg  # noqa: E402
import run_sphere_cmf as old_sphere  # noqa: E402
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
    "Task Preserved @2° ↑",
    "Unit Norm Error Mean ↓",
    "Unit Norm Error Max ↓",
]

RAW_EXTRA_METRICS = [
    "MLP Probe-Acc ↓",
    "MLP Probe-F1 ↓",
    "Sphere Valid Rate ↑",
    "Sphere Valid Rate @1e-6",
]

COLS = ["Method"] + REPORT_METRICS
ALL_COLS = ["Method"] + REPORT_METRICS + RAW_EXTRA_METRICS

LOWER_IS_BETTER = {
    "Task MAE ↓",
    "Unit Norm Error Mean ↓",
    "Unit Norm Error Max ↓",
}

DEBIAS_METRICS = {
    "Linear Probe-Acc ↓",
    "Linear Probe-Recall ↓",
    "Linear Probe-F1 ↓",
    "Linear Probe-AUROC ↓",
    "MLP Probe-Acc ↓",
    "MLP Probe-F1 ↓",
}
UNIT_NORM_ERROR_METRICS = {
    "Unit Norm Error Mean ↓",
    "Unit Norm Error Max ↓",
}
REFERENCE_METHODS = {"Original", "Oracle"}
ANNOTATION_TOL = 1e-12
NORM_TOL = 1e-10
TASK_PRESERVE_TOL_DEG = 2.0


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


def display_method(method):
    return DISPLAY_NAME.get(method, method)


def color_for(method):
    base = method.replace("IsoAct", "IsoRot")
    return cfg.METHOD_COLORS.get(base.replace("+Proj", ""), cfg.METHOD_COLORS.get(base, "#999999"))


def project_to_sphere(X):
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def unit_norm_errors(X):
    return np.abs(np.linalg.norm(np.asarray(X), axis=1) - 1.0)


def structural_metrics(X):
    err = unit_norm_errors(X)
    return {
        "Unit Norm Error Mean ↓": float(np.mean(err)),
        "Unit Norm Error Max ↓": float(np.max(err)),
        "Sphere Valid Rate ↑": float(np.mean(err <= NORM_TOL)),
        "Sphere Valid Rate @1e-6": float(np.mean(err <= 1e-6)),
    }


def extract_phi_theta(X):
    return old_sphere.extract_phi_theta(np.asarray(X))


def probe_metrics(X, groups):
    Xtr, Xte, gtr, gte = train_test_split(
        X,
        groups,
        test_size=cfg.PROBE_TEST_SIZE,
        stratify=groups,
        random_state=cfg.PROBE_RANDOM_STATE,
    )
    lin = LogisticRegression(max_iter=1000, random_state=cfg.PROBE_RANDOM_STATE)
    lin.fit(Xtr, gtr)
    pred_lin = lin.predict(Xte)
    score_lin = lin.predict_proba(Xte)[:, 1]

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
    _, theta_orig = extract_phi_theta(X_orig)
    _, theta_new = extract_phi_theta(X_new)
    theta_abs_err = np.abs(theta_new - theta_orig)
    return {
        "Task MAE ↓": float(np.mean(theta_abs_err)),
        "Task Preserved @2° ↑": float(np.mean(theta_abs_err <= TASK_PRESERVE_TOL_DEG)),
    }


def build_method_arrays():
    X, groups, phi_all, theta_all_rad, task_labels = old_sphere.generate_data()
    theta_orig_deg = np.degrees(theta_all_rad)
    neutral, clf = old_sphere.compute_neutral(X, groups)
    bias_dir = old_sphere.compute_bias_dir(clf)
    sfid_dims = old_sphere.compute_sfid_dims(X, groups)

    cache_bl = LEGACY_OUT / "cache_baseline_ae.pt"
    cache_cmf = LEGACY_OUT / "cache_cmf.pt"
    baseline_cached = load_cmf_results(str(cache_bl), input_dim=3)
    cmf_cached = load_cmf_results(str(cache_cmf), input_dim=3)
    if baseline_cached is None or cmf_cached is None:
        raise RuntimeError("Legacy sphere CMF caches are required for deterministic no-retrain consolidation.")
    _bl_model, bl_native_metrics, bl_rec, bl_z = baseline_cached
    _cmf_model, cmf_native_metrics, cmf_rec, cmf_z = cmf_cached

    raw = old_sphere.run_all_methods(
        X, groups, neutral, bias_dir, sfid_dims, cfg.PRIMARY_ALPHA, bl_rec, cmf_rec
    )
    arrays = {
        "Original": raw["Original"],
        "Oracle": raw["Oracle"],
        "CMF": np.asarray(cmf_rec),
        "CMF+Proj": project_to_sphere(np.asarray(cmf_rec)),
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
        "theta_band_deg": old_sphere.THETA_BAND,
        "cmf_cache": str(cache_cmf),
        "baseline_cache": str(cache_bl),
        "cmf_native_metrics": cmf_native_metrics,
        "baseline_native_metrics": bl_native_metrics,
    }
    aux = {
        "X_orig": X,
        "groups": groups,
        "phi_all": phi_all,
        "theta_all_rad": theta_all_rad,
        "theta_orig_deg": theta_orig_deg,
        "task_labels": task_labels,
        "neutral": neutral,
        "bias_dir": bias_dir,
        "sfid_dims": sfid_dims,
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


def select_rows(rows, order):
    by_method = {r["Method"]: r for r in rows}
    return [by_method[m] for m in order if m in by_method]


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
    return finite_float(row.get(metric))


def _effective_annotation_value(rows, metric, method, value):
    if method in REFERENCE_METHODS:
        return None
    if metric in DEBIAS_METRICS:
        return _oracle_deviation(rows, metric, value)
    if metric in UNIT_NORM_ERROR_METRICS:
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
    reverse = metric not in LOWER_IS_BETTER and metric not in DEBIAS_METRICS
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
    return {
        "compact": write_csv_file(compact, OUT / "sphere_unified_metrics_compact.csv", COLS),
        "extended": write_csv_file(extended, OUT / "sphere_unified_metrics_extended.csv", COLS),
        "all_methods": write_csv_file(extended, OUT / "sphere_unified_metrics_all_methods.csv", ALL_COLS),
        "main": write_csv_file(extended, OUT / "sphere_unified_metrics.csv", COLS),
    }


def write_json(rows, meta):
    path = OUT / "sphere_unified_metrics.json"
    payload = {
        "generated_utc": now(),
        "deterministic_config": meta,
        "method_order_extended": METHOD_ORDER_EXTENDED,
        "method_order_compact": METHOD_ORDER_COMPACT,
        "display_name_map": DISPLAY_NAME,
        "report_columns": COLS,
        "all_columns": ALL_COLS,
        "thresholds": {"unit_norm_tol": NORM_TOL},
        "metrics": rows,
        "notes": [
            "Linear Probe-Recall is macro recall; Linear Probe-F1 is macro F1.",
            "Markdown tables display raw linear probe values; bold/underline for linear probes uses absolute deviation from Oracle.",
            "Unit norm error is abs(||x|| - 1).",
            "CMF+Proj is row-wise unit normalization of raw CMF reconstruction.",
            "No legacy toy outputs were overwritten.",
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def write_md_table_file(rows, path, title):
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
        "- Unit norm error annotations use `Original` as a numerical-precision cap: non-reference methods at or below Original error are tied as best.",
        "",
        md_table(rows, REPORT_METRICS),
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def write_md(rows, meta, figure_status):
    path = OUT / "sphere_unified_metrics.md"
    lines = [
        "# Hypersphere S^2 Toy Unified Metrics",
        "",
        f"Generated: {now()} UTC",
        "",
        "",
        "## Protocol",
        "",
        f"- Seed: `{meta['seed']}`",
        f"- Primary alpha: `{meta['primary_alpha']}`",
        f"- Probe split: test_size=`{meta['probe_test_size']}`, random_state=`{meta['probe_random_state']}`",
        f"- Theta in-band: `{meta['theta_band_deg']}` degrees",
        "- Linear Probe Recall: **macro recall**",
        "- Linear Probe F1: **macro F1**",
        "- Linear Probe AUROC: binary AUROC using positive-class predicted probability.",
        "- Markdown tables display raw linear probe values. Bold/underline annotations for linear probe columns are computed by absolute deviation from Oracle: `|metric - Oracle metric|`.",
        "- Existing internal `IsoRot` outputs are reported as **IsoAct**.",
        "- `Original` and `Oracle` are reference rows and are excluded from bold/underline annotation.",
        "- Task MAE: mean absolute error of polar angle θ in degrees.",
        f"- Task Preserved @2°: fraction with absolute θ error <= `{TASK_PRESERVE_TOL_DEG}` degrees relative to Original.",
        f"- Sphere valid rate raw diagnostic threshold: abs(||x|| - 1) <= `{NORM_TOL}`.",
        "- CMF+Proj: raw CMF reconstruction projected to the unit sphere by row-wise normalization.",
        "",
        "## Compact metric table",
        "",
        md_table(select_rows(rows, METHOD_ORDER_COMPACT), REPORT_METRICS),
        "",
        "## Extended metric table",
        "",
        md_table(select_rows(rows, METHOD_ORDER_EXTENDED), REPORT_METRICS),
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
        "- `build_sphere_unified_metrics.py`",
        "- `SPHERE_TOY_UNIFIED_METRIC_PLAN.md`",
        "- `sphere_unified_metrics.csv`",
        "- `sphere_unified_metrics_compact.csv`",
        "- `sphere_unified_metrics_extended.csv`",
        "- `sphere_unified_metrics_all_methods.csv`",
        "- `sphere_unified_metrics.json`",
        "- `sphere_unified_metrics.md`",
        "- `sphere_unified_metrics_compact.md`",
        "- `sphere_unified_metrics_extended.md`",
        "- `sphere_unified_arrays.npz`",
        "- figures: `setting.png`, `crosssection.png`, `topdown_a09.png`, `topdown_a10.png`, `scatter_a09.png`, `scatter_a10.png`, `scatter_a10_compact.png`, `scatter_a10_extended.png`, `bar_linear.png`, `bar_mlp.png`, `latent_baseline.png`, `latent_cmf.png`",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def savefig(fig, name, status):
    path = OUT / name
    fig.savefig(path, dpi=cfg.FIG_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    status.append({"path": str(path), "status": "regenerated"})


def plot_setting(X, groups, neutral, status):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    u_mesh = np.linspace(0, np.pi / 2, 25)
    v_mesh = np.linspace(-np.pi, np.pi, 50)
    Um, Vm = np.meshgrid(u_mesh, v_mesh)
    ax.plot_surface(np.sin(Um) * np.cos(Vm), np.sin(Um) * np.sin(Vm), np.cos(Um),
                    alpha=0.08, color="lightgray", edgecolor="gray", linewidth=0.08)
    for g in [0, 1]:
        mask = groups == g
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                   c=cfg.GROUP_COLORS[g], s=cfg.MARKER_SIZE, alpha=cfg.MARKER_ALPHA)
    ax.scatter([neutral[0]], [neutral[1]], [neutral[2]], c=cfg.NEUTRAL_COLOR,
               s=200, marker="*", edgecolors="k", linewidths=0.8, zorder=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=25, azim=30)
    savefig(fig, "setting.png", status)


def _make_grid(n, figsize=None):
    ncols = 5 if n > 6 else 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 5 * nrows))
    return fig, np.array(axes).reshape(-1)


def plot_grid(results, groups, neutral, alpha, kind, filename, status, method_order=None):
    method_order = list(method_order or METHOD_ORDER_EXTENDED)
    fig, axes = _make_grid(len(method_order), figsize=(25, 10) if len(method_order) > 6 else None)
    phi_n_deg = np.degrees(np.arctan2(neutral[1], neutral[0]))
    for idx, method in enumerate(method_order):
        ax = axes[idx]
        X = results[method]
        if kind == "scatter":
            phi, theta_deg = extract_phi_theta(X)
            phi_deg = np.degrees(phi)
            ax.axhspan(old_sphere.THETA_BAND[0], old_sphere.THETA_BAND[1], color="green", alpha=0.08, zorder=0)
            ax.axvline(phi_n_deg, color=cfg.NEUTRAL_COLOR, lw=1.5, ls="--", alpha=0.7)
            for g in [0, 1]:
                mask = groups == g
                ax.scatter(phi_deg[mask], theta_deg[mask], c=cfg.GROUP_COLORS[g],
                           s=cfg.MARKER_SIZE, alpha=cfg.MARKER_ALPHA, zorder=3)
            ax.set_xlim(-90, 90); ax.set_ylim(0, 60)
            ax.set_xlabel("φ (deg)"); ax.set_ylabel("θ (deg)")
        elif kind == "topdown":
            circ_t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(circ_t), np.sin(circ_t), "k--", lw=0.8, alpha=0.3)
            for g in [0, 1]:
                mask = groups == g
                ax.scatter(X[mask, 0], X[mask, 1], c=cfg.GROUP_COLORS[g],
                           s=cfg.MARKER_SIZE, alpha=cfg.MARKER_ALPHA, zorder=3)
            ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
            ax.set_xlabel("x"); ax.set_ylabel("y")
        elif kind == "crosssection":
            r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
            phi = np.arctan2(X[:, 1], X[:, 0])
            sr = np.where(phi >= 0, r, -r)
            curve_r = np.linspace(-1, 1, 300)
            curve_z = np.sqrt(np.maximum(1 - curve_r ** 2, 0))
            ax.plot(curve_r, curve_z, "k-", linewidth=1.5, alpha=0.4, zorder=1)
            for g in [0, 1]:
                mask = groups == g
                ax.scatter(sr[mask], X[mask, 2], c=cfg.GROUP_COLORS[g],
                           s=cfg.MARKER_SIZE, alpha=cfg.MARKER_ALPHA, zorder=3)
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.5, 1.5)
            ax.set_xlabel("signed r_xy"); ax.set_ylabel("z")
        ax.set_title(display_method(method))
    for ax in axes[len(method_order):]:
        ax.axis("off")
    fig.suptitle(f"S^2 {kind} alpha={alpha}", fontweight="bold")
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
        elif m in DEBIAS_METRICS:
            # desirability by closeness to Oracle for display heatmap
            oracle = _reference_value(rows, "Oracle", m)
            dev = np.abs(col - oracle)
            dlo, dhi = np.nanmin(dev), np.nanmax(dev)
            norm[:, j] = 0.5 if dhi - dlo < 1e-12 else 1 - (dev - dlo) / (dhi - dlo)
        elif m in LOWER_IS_BETTER:
            norm[:, j] = 1 - (col - lo) / (hi - lo)
        else:
            norm[:, j] = (col - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([display_method(r["Method"]) for r in rows])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([display_metric_label(m) for m in metrics], rotation=45, ha="right")
    ax.set_title("S^2 toy unified metrics (normalized: higher color = better)")
    plt.colorbar(im, ax=ax, label="normalized desirability")
    fig.tight_layout()
    savefig(fig, "sphere_unified_metrics.png", status)


def regenerate_figures(rows, arrays, aux):
    status = []
    cfg.setup_style()
    plot_setting(aux["X_orig"], aux["groups"], aux["neutral"], status)
    for alpha in cfg.ALPHAS:
        raw = old_sphere.run_all_methods(
            aux["X_orig"], aux["groups"], aux["neutral"], aux["bias_dir"], aux["sfid_dims"], alpha,
            arrays["CMF"], arrays["CMF"],
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
    plot_cmf_latent(aux["bl_z"], aux["groups"], aux["theta_orig_deg"], "θ (polar, deg)", "Baseline AE",
                    str(OUT / "latent_baseline.png"))
    status.append({"path": str(OUT / "latent_baseline.png"), "status": "regenerated"})
    plot_cmf_latent(aux["cmf_z"], aux["groups"], aux["theta_orig_deg"], "θ (polar, deg)", "CMF",
                    str(OUT / "latent_cmf.png"))
    status.append({"path": str(OUT / "latent_cmf.png"), "status": "regenerated"})
    plot_unified_heatmap(rows, status)
    return status


def write_plan_md():
    path = OUT / "SPHERE_TOY_UNIFIED_METRIC_PLAN.md"
    text = f"""# Hypersphere S^2 Toy Unified Metric Plan

Generated/updated: {now()} UTC

## Goal

Create finalized hypersphere toy outputs under:

```text
{OUT}
```

Toy data generation and method definitions follow the legacy source:

```text
{LEGACY_FINAL / 'run_sphere.py'}
```

CMF reconstruction is reused from the existing deterministic cache:

```text
{LEGACY_OUT / 'cache_cmf.pt'}
```

## Reporting conventions inherited from SE(3)

- Compact table: `Original, Oracle, CMF, SFID, SPD, IsoAct`.
- Extended table: `Original, Oracle, CMF, CMF+Proj, SFID, SFID+Proj, SPD, SPD+Proj, Geodesic, IsoAct`.
- Internal `IsoRot` is displayed as `IsoAct`.
- Linear probe cells display raw values.
- Linear probe bold/underline is computed by Oracle deviation: `|raw_metric(method) - raw_metric(Oracle)|`.
- `Original` and `Oracle` are reference rows and are excluded from bold/underline annotation.

## Metrics

- `Linear Probe-Acc (raw)`
- `Linear Probe-Recall (raw)`; macro recall
- `Linear Probe-F1 (raw)`; macro F1
- `Linear Probe-AUROC (raw)`
- `Task MAE ↓`; polar angle θ MAE in degrees
- `Task Preserved @2° ↑`; fraction with `|θ_method - θ_original| <= {TASK_PRESERVE_TOL_DEG}` degrees
- `Unit Norm Error Mean ↓`; mean `abs(||x|| - 1)`
- `Unit Norm Error Max ↓`; max `abs(||x|| - 1)`

Raw JSON/CSV additionally preserve MLP probe metrics and sphere valid-rate diagnostics.

## Hypersphere manifold constraint

The manifold constraint is unit norm membership. For each point:

```text
unit_norm_error_i = abs(||x_i|| - 1)
```

Final table reports mean and max unit norm error. For bold/underline annotation, `Original` is used as a numerical-precision cap: non-reference methods at or below Original unit-norm error are tied as best.

## Figures

Regenerate the sphere figures:

- `setting.png`
- `crosssection.png`
- `topdown_a09.png`, `topdown_a10.png`
- `scatter_a09.png`, `scatter_a10.png`
- `scatter_a10_compact.png`, `scatter_a10_extended.png`
- `bar_linear.png`, `bar_mlp.png`
- `latent_baseline.png`, `latent_cmf.png`
- `sphere_unified_metrics.png`
"""
    path.write_text(text)
    return path


def main():
    ensure()
    np.random.seed(cfg.SEED)
    arrays, aux, meta = build_method_arrays()
    rows = compute_rows(arrays, aux["X_orig"], aux["groups"])

    np.savez_compressed(
        OUT / "sphere_unified_arrays.npz",
        **{f"X_{k.replace('+', 'plus')}": v for k, v in arrays.items()},
        groups=aux["groups"],
        X_orig=aux["X_orig"],
        theta_orig_deg=aux["theta_orig_deg"],
        split_seed=cfg.SEED,
        primary_alpha=cfg.PRIMARY_ALPHA,
    )

    figure_status = regenerate_figures(rows, arrays, aux)
    csv_paths = write_csvs(rows)
    json_path = write_json(rows, meta)
    md_path = write_md(rows, meta, figure_status)
    compact_md = write_md_table_file(select_rows(rows, METHOD_ORDER_COMPACT), OUT / "sphere_unified_metrics_compact.md", "Hypersphere S^2 Toy Compact Metrics")
    extended_md = write_md_table_file(select_rows(rows, METHOD_ORDER_EXTENDED), OUT / "sphere_unified_metrics_extended.md", "Hypersphere S^2 Toy Extended Metrics")
    plan_path = write_plan_md()

    for csv_path in csv_paths.values():
        print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {compact_md}")
    print(f"Wrote {extended_md}")
    print(f"Wrote {plan_path}")
    print(f"Wrote {OUT / 'sphere_unified_metrics.png'}")


if __name__ == "__main__":
    main()
