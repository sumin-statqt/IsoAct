#!/usr/bin/env python3
"""
Sphere (S^2) experiment with CMF comparison.

Replicates run_sphere.py with Baseline AE + CMF added.
Bias = azimuthal angle phi, Task = polar angle theta.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from cmf_module import (
    train_cmf_model, get_reconstructed, compute_cmf_metrics,
    save_cmf_results, load_cmf_results, plot_cmf_latent
)

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

setup_style()
OUT = os.path.join(os.path.dirname(__file__), "output", "sphere")
rng = np.random.default_rng(SEED)

# CMF scripts now use config's METHOD_ORDER (9 methods) + fill_info_panel for 10th panel
SFID_N_DIMS = 2
THETA_BAND = (5.0, 45.0)  # degrees


# =============================================================================
# Helpers
# =============================================================================

def spherical_to_cartesian(theta_rad, phi_rad):
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    return np.column_stack([x, y, z])

def project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return X / norms

def manifold_residual(X):
    return np.mean(np.abs(np.linalg.norm(X, axis=1) - 1.0))

def extract_phi_theta(X):
    norms = np.linalg.norm(X, axis=1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    X_n = X / norms[:, None]
    phi = np.arctan2(X_n[:, 1], X_n[:, 0])
    theta = np.arccos(np.clip(X_n[:, 2], -1, 1))
    return phi, np.degrees(theta)

def theta_mae(X, theta_orig_deg):
    _, theta_deg = extract_phi_theta(X)
    return np.mean(np.abs(theta_deg - theta_orig_deg))

def theta_inband(X):
    _, theta_deg = extract_phi_theta(X)
    return np.mean((theta_deg >= THETA_BAND[0]) & (theta_deg <= THETA_BAND[1]))

def bias_probe_linear(X, groups):
    Xtr, Xte, ytr, yte = train_test_split(
        X, groups, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE, stratify=groups)
    clf = LogisticRegression(max_iter=1000, random_state=PROBE_RANDOM_STATE)
    clf.fit(Xtr, ytr)
    return clf.score(Xte, yte)

def bias_probe_mlp(X, groups):
    Xtr, Xte, ytr, yte = train_test_split(
        X, groups, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE, stratify=groups)
    clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
                        random_state=PROBE_RANDOM_STATE)
    clf.fit(Xtr, ytr)
    return clf.score(Xte, yte)


# =============================================================================
# Data Generation
# =============================================================================

def generate_data():
    print("=" * 70)
    print("Sphere (S^2): Generating data")
    print("=" * 70)
    phi_g0 = np.radians(30) + rng.standard_normal(N_PER_GROUP) * np.radians(8)
    phi_g1 = np.radians(-30) + rng.standard_normal(N_PER_GROUP) * np.radians(8)
    theta_g0 = np.radians(25) + rng.standard_normal(N_PER_GROUP) * np.radians(8)
    theta_g1 = np.radians(25) + rng.standard_normal(N_PER_GROUP) * np.radians(8)
    theta_g0 = np.clip(theta_g0, np.radians(2), np.radians(60))
    theta_g1 = np.clip(theta_g1, np.radians(2), np.radians(60))

    phi_all = np.concatenate([phi_g0, phi_g1])
    theta_all_rad = np.concatenate([theta_g0, theta_g1])
    groups = np.array([0] * N_PER_GROUP + [1] * N_PER_GROUP)

    X = spherical_to_cartesian(theta_all_rad, phi_all)
    task_labels = (theta_all_rad >= np.median(theta_all_rad)).astype(int)

    print(f"  N = {len(X)}")
    print(f"  Group 0 phi: {np.degrees(phi_g0.mean()):.1f}° ± {np.degrees(phi_g0.std()):.1f}°")
    print(f"  Group 1 phi: {np.degrees(phi_g1.mean()):.1f}° ± {np.degrees(phi_g1.std()):.1f}°")
    return X, groups, phi_all, theta_all_rad, task_labels


# =============================================================================
# Neutral & Bias direction
# =============================================================================

def compute_neutral(X, groups):
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X, groups)
    probs = clf.predict_proba(X).max(axis=1)
    low_conf = probs < LOWCONF_THR
    n_low = low_conf.sum()
    print(f"  Low-confidence: {n_low}/{len(X)}")
    if n_low >= 10:
        neutral_raw = X[low_conf].mean(axis=0)
    else:
        neutral_raw = X.mean(axis=0)
    neutral = neutral_raw / np.linalg.norm(neutral_raw)
    print(f"  Neutral: {neutral}")
    return neutral, clf

def compute_bias_dir(clf):
    w = clf.coef_[0]
    return w / np.linalg.norm(w)

def compute_sfid_dims(X, groups):
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X, groups)
    return np.argsort(rf.feature_importances_)[::-1][:SFID_N_DIMS]


# =============================================================================
# Post-hoc debiasing methods (from run_sphere.py)
# =============================================================================

def method_spd(X, bias_dir, neutral, alpha):
    u = bias_dir
    s_n = np.dot(neutral, u)
    projections = X @ u
    shifts = alpha * (s_n - projections)
    return X + shifts[:, None] * u[None, :]

def method_sfid(X, sfid_dims, neutral, alpha):
    X_out = X.copy()
    for j in sfid_dims:
        X_out[:, j] = X[:, j] + alpha * (neutral[j] - X[:, j])
    return X_out

def method_geodesic(X, bias_dir, neutral, alpha):
    u = bias_dir
    t_proj = float(u @ neutral)
    X_out = X.copy()
    for i in range(len(X)):
        x = X[i]
        u_tang = u - np.dot(u, x) * x
        norm_ut = np.linalg.norm(u_tang)
        if norm_ut < 1e-10:
            continue
        v = u_tang / norm_ut
        cur = np.dot(x, u)
        shift = alpha * (t_proj - cur)
        t = shift / norm_ut
        x_new = np.cos(t) * x + np.sin(t) * v
        X_out[i] = x_new / np.linalg.norm(x_new)
    return X_out

def method_isorot(X, neutral, alpha):
    phi_n = np.arctan2(neutral[1], neutral[0])
    X_out = X.copy()
    for i in range(len(X)):
        phi_i = np.arctan2(X[i, 1], X[i, 0])
        diff = np.arctan2(np.sin(phi_n - phi_i), np.cos(phi_n - phi_i))
        delta = alpha * diff
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        X_out[i, 0] = X[i, 0] * cos_d - X[i, 1] * sin_d
        X_out[i, 1] = X[i, 0] * sin_d + X[i, 1] * cos_d
    return X_out

def method_oracle(X, neutral, alpha):
    phi_n = np.arctan2(neutral[1], neutral[0])
    X_out = X.copy()
    for i in range(len(X)):
        phi_i = np.arctan2(X[i, 1], X[i, 0])
        theta_i = np.arccos(np.clip(X[i, 2], -1, 1))
        diff = np.arctan2(np.sin(phi_n - phi_i), np.cos(phi_n - phi_i))
        phi_new = phi_i + alpha * diff
        X_out[i, 0] = np.sin(theta_i) * np.cos(phi_new)
        X_out[i, 1] = np.sin(theta_i) * np.sin(phi_new)
        X_out[i, 2] = np.cos(theta_i)
    return X_out


# =============================================================================
# CMF Counterfactual for Sphere
# =============================================================================

def sphere_counterfactual(X_np, groups_np):
    """
    Counterfactual: keep theta (task), mirror phi across phi=0.
    Group 0: phi ~ N(+30°), Group 1: phi ~ N(-30°).
    Counterfactual: negate phi, keep theta.
    """
    X_cf = X_np.copy()
    for i in range(len(X_np)):
        phi_i = np.arctan2(X_np[i, 1], X_np[i, 0])
        theta_i = np.arccos(np.clip(X_np[i, 2], -1, 1))
        phi_cf = -phi_i  # mirror across phi=0
        X_cf[i, 0] = np.sin(theta_i) * np.cos(phi_cf)
        X_cf[i, 1] = np.sin(theta_i) * np.sin(phi_cf)
        X_cf[i, 2] = np.cos(theta_i)
    return X_cf


# =============================================================================
# Run all methods
# =============================================================================

def run_all_methods(X, groups, neutral, bias_dir, sfid_dims, alpha,
                    bl_rec, cmf_rec):
    results = {}
    results['Original'] = X.copy()
    results['Oracle'] = method_oracle(X, neutral, alpha)
    results['CMF'] = cmf_rec
    results['SFID'] = method_sfid(X, sfid_dims, neutral, alpha)
    results['SFID+Proj'] = project_to_sphere(results['SFID'])
    results['SPD'] = method_spd(X, bias_dir, neutral, alpha)
    results['SPD+Proj'] = project_to_sphere(results['SPD'])
    results['Geodesic'] = method_geodesic(X, bias_dir, neutral, alpha)
    results['IsoRot'] = method_isorot(X, neutral, alpha)
    # Baseline_AE stored separately for CMF-native metrics, not in main display
    results['_Baseline_AE'] = bl_rec
    return results


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(X_orig, results, groups, theta_orig_deg):
    metrics = {}
    for name in METHOD_ORDER:
        Xm = results[name]
        m = {}
        m['on_manifold'] = manifold_residual(Xm)
        m['theta_mae'] = theta_mae(Xm, theta_orig_deg)
        m['theta_inband'] = theta_inband(Xm)
        m['bias_linear'] = bias_probe_linear(Xm, groups)
        m['bias_mlp'] = bias_probe_mlp(Xm, groups)
        metrics[name] = m
    return metrics

def annotate_val(val, metric_name):
    if metric_name == 'on_manifold':
        if val < 1e-3: return 'O'
        elif val < 0.05: return 'triangle'
        else: return 'X'
    elif metric_name == 'theta_mae':
        if val < 0.5: return 'O'
        elif val < 3.0: return 'triangle'
        else: return 'X'
    elif metric_name == 'theta_inband':
        if val > 0.95: return 'O'
        elif val > 0.85: return 'triangle'
        else: return 'X'
    elif metric_name in ('bias_linear', 'bias_mlp'):
        if val < 0.55: return 'O'
        elif val < 0.65: return 'triangle'
        else: return 'X'
    return ''

def print_result_table(metrics, filepath):
    header = (f"{'Method':12s} | {'On-Mfld':14s} | {'θ MAE':18s} | "
              f"{'θ In-Band':18s} | {'Bias(Lin)':10s} | {'Bias(MLP)':10s}")
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name in METHOD_ORDER:
        m = metrics[name]
        a1 = annotate_val(m['on_manifold'], 'on_manifold')
        a2 = annotate_val(m['theta_mae'], 'theta_mae')
        a3 = annotate_val(m['theta_inband'], 'theta_inband')
        a4 = annotate_val(m['bias_linear'], 'bias_linear')
        a5 = annotate_val(m['bias_mlp'], 'bias_mlp')
        line = (f"{name:12s} | {a1} ({m['on_manifold']:.4f})"
                f"     | {a2} (t_MAE={m['theta_mae']:.4f})"
                f"   | {a3} (t_band={100*m['theta_inband']:.1f}%)"
                f" | {m['bias_linear']:.3f} {a4}"
                f"   | {m['bias_mlp']:.3f} {a5}")
        lines.append(line)
    lines.append(sep)
    table_str = "\n".join(lines)
    print("\n" + table_str)
    with open(filepath, 'w') as f:
        f.write(table_str + "\n")
    print(f"  Saved: {filepath}")


# =============================================================================
# Figures
# =============================================================================

def fig_setting(X, groups, neutral):
    print("\nPlotting setting figure ...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    u_mesh = np.linspace(0, np.pi / 2, 25)
    v_mesh = np.linspace(-np.pi, np.pi, 50)
    Um, Vm = np.meshgrid(u_mesh, v_mesh)
    ax.plot_surface(np.sin(Um)*np.cos(Vm), np.sin(Um)*np.sin(Vm), np.cos(Um),
                    alpha=0.08, color='lightgray', edgecolor='gray', linewidth=0.08)
    for g in [0, 1]:
        mask = groups == g
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                   c=GROUP_COLORS[g], s=MARKER_SIZE, alpha=MARKER_ALPHA, label=f'Group {g}')
    ax.scatter([neutral[0]], [neutral[1]], [neutral[2]],
               c=NEUTRAL_COLOR, s=200, marker='*', edgecolors='k', linewidths=0.8, zorder=10)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.view_init(elev=25, azim=30)
    save_fig(fig, os.path.join(OUT, "setting.png"))

def fig_crosssection(results, groups, neutral):
    print("\nPlotting cross-section ...")
    fig, axes = make_grid(figsize=(25, 10))
    # Cross-section: signed r_xy vs z, semicircle
    def signed_r_xy(X):
        r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        phi = np.arctan2(X[:, 1], X[:, 0])
        return np.where(phi >= 0, r, -r)

    curve_r = np.linspace(-1, 1, 300)
    curve_z = np.sqrt(np.maximum(1 - curve_r**2, 0))

    for idx, name in enumerate(METHOD_ORDER):
        ax = axes[idx]
        Xm = results[name]
        rs = signed_r_xy(Xm)
        ax.plot(curve_r, curve_z, 'k-', linewidth=1.5, alpha=0.4, zorder=1)
        for g in [0, 1]:
            mask = groups == g
            ax.scatter(rs[mask], Xm[mask, 2], c=GROUP_COLORS[g],
                       s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.set_title(name); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.5, 1.5)
    fill_info_panel(axes[len(METHOD_ORDER)], "Cross-section",
                    "Signed r_xy vs z.\n"
                    "Semicircle = unit sphere constraint.\n"
                    "Off-curve = off-manifold.")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "crosssection.png"))

def fig_topdown(results, groups, alpha, filename):
    print(f"\nPlotting top-down (alpha={alpha}) ...")
    fig, axes = make_grid(figsize=(25, 10))
    for idx, name in enumerate(METHOD_ORDER):
        ax = axes[idx]
        Xm = results[name]
        for g in [0, 1]:
            mask = groups == g
            ax.scatter(Xm[mask, 0], Xm[mask, 1], c=GROUP_COLORS[g],
                       s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        circ_t = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(circ_t), np.sin(circ_t), 'k--', lw=0.8, alpha=0.3)
        ax.set_title(name); ax.set_aspect('equal')
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    fill_info_panel(axes[len(METHOD_ORDER)], "Top-down",
                    f"x-y plane view (alpha={alpha}).\n"
                    "Dashed circle = equator.")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, filename))

def fig_scatter(results, groups, neutral, alpha, theta_orig_deg, filename):
    print(f"\nPlotting scatter (alpha={alpha}) ...")
    fig, axes = make_grid(figsize=(25, 10))
    phi_n_deg = np.degrees(np.arctan2(neutral[1], neutral[0]))
    for idx, name in enumerate(METHOD_ORDER):
        ax = axes[idx]
        Xm = results[name]
        phi, theta_deg = extract_phi_theta(Xm)
        phi_deg = np.degrees(phi)
        ax.axhspan(THETA_BAND[0], THETA_BAND[1], color='green', alpha=0.08, zorder=0)
        ax.axvline(phi_n_deg, color=NEUTRAL_COLOR, lw=1.5, ls='--', alpha=0.7)
        for g in [0, 1]:
            mask = groups == g
            ax.scatter(phi_deg[mask], theta_deg[mask], c=GROUP_COLORS[g],
                       s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.set_title(name); ax.set_xlim(-90, 90); ax.set_ylim(0, 60)
        ax.set_xlabel('φ (deg)'); ax.set_ylabel('θ (deg)')
    fill_info_panel(axes[len(METHOD_ORDER)], "Scatter",
                    f"phi vs theta (alpha={alpha}).\n"
                    "Green band = task-attribute range.\n"
                    "Gold line = neutral azimuth.")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, filename))

def fig_bar(metrics, probe_key, ylabel, filename):
    print(f"\nPlotting bar chart ({probe_key}) ...")
    fig, ax = plt.subplots(figsize=(12, 5))
    methods = METHOD_ORDER_NO_ORIG
    vals = [metrics[m][probe_key] for m in methods]
    colors = [METHOD_COLORS.get(m, '#999999') for m in methods]
    bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=TICK_SIZE, fontweight='bold')
    ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel(ylabel); ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, filename))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, groups, phi_all, theta_all_rad, task_labels = generate_data()
    theta_orig_deg = np.degrees(theta_all_rad)
    neutral, clf = compute_neutral(X, groups)
    bias_dir = compute_bias_dir(clf)
    sfid_dims = compute_sfid_dims(X, groups)
    print(f"  Bias dir: {bias_dir}, SFID dims: {sfid_dims}")

    # ── Train CMF ──
    print("\n" + "=" * 70)
    print("  Training CMF models")
    print("=" * 70)

    cache_bl = os.path.join(OUT, "cache_baseline_ae.pt")
    cache_cmf = os.path.join(OUT, "cache_cmf.pt")

    cached = load_cmf_results(cache_bl, input_dim=3)
    if cached:
        print("  Loaded cached Baseline AE")
        bl_model, bl_cmf_m, bl_rec, bl_z = cached
    else:
        bl_model = train_cmf_model(X, groups, task_labels, sphere_counterfactual,
                                   model_type="Baseline", input_dim=3, latent_dim=2,
                                   epochs=300, lr=1e-3)
        bl_rec, bl_z = get_reconstructed(bl_model, X)
        X_cf = sphere_counterfactual(X, groups)
        bl_cmf_m = compute_cmf_metrics(bl_model, X, X_cf, subset_size=32)
        save_cmf_results(bl_model, bl_cmf_m, bl_rec, bl_z, cache_bl)

    cached = load_cmf_results(cache_cmf, input_dim=3)
    if cached:
        print("  Loaded cached CMF")
        cmf_model, cmf_cmf_m, cmf_rec, cmf_z = cached
    else:
        cmf_model = train_cmf_model(X, groups, task_labels, sphere_counterfactual,
                                    model_type="CMF", input_dim=3, latent_dim=2,
                                    epochs=300, lr=1e-3, lambda_metric=10.0, lambda_curv=5.0)
        cmf_rec, cmf_z = get_reconstructed(cmf_model, X)
        X_cf = sphere_counterfactual(X, groups)
        cmf_cmf_m = compute_cmf_metrics(cmf_model, X, X_cf, subset_size=32)
        save_cmf_results(cmf_model, cmf_cmf_m, cmf_rec, cmf_z, cache_cmf)

    print(f"\n  Baseline AE: {bl_cmf_m}")
    print(f"  CMF: {cmf_cmf_m}")

    plot_cmf_latent(bl_z, groups, theta_orig_deg, "θ (polar, deg)", "Baseline AE",
                    os.path.join(OUT, "latent_baseline.png"))
    plot_cmf_latent(cmf_z, groups, theta_orig_deg, "θ (polar, deg)", "CMF",
                    os.path.join(OUT, "latent_cmf.png"))

    fig_setting(X, groups, neutral)

    all_metrics, all_results = {}, {}
    for alpha in ALPHAS:
        print(f"\n{'='*70}\n  ALPHA = {alpha}\n{'='*70}")
        results = run_all_methods(X, groups, neutral, bias_dir, sfid_dims, alpha, bl_rec, cmf_rec)
        all_results[alpha] = results
        metrics = compute_metrics(X, results, groups, theta_orig_deg)
        all_metrics[alpha] = metrics
        tag = f"a{str(alpha).replace('.','')}"
        fig_scatter(results, groups, neutral, alpha, theta_orig_deg, f"scatter_{tag}.png")
        fig_topdown(results, groups, alpha, f"topdown_{tag}.png")

    print(f"\n{'='*70}\n  RESULT TABLE (alpha={PRIMARY_ALPHA})\n{'='*70}")
    print_result_table(all_metrics[PRIMARY_ALPHA], os.path.join(OUT, "result_table.txt"))

    print(f"\n  CMF-Native Metrics (Tier 2):")
    print(f"  {'Method':12s} | {'Metric Error':>14s} | {'Curv Error':>12s} | {'Recon MSE':>10s}")
    print(f"  {'-'*55}")
    for name, m in [('Baseline_AE', bl_cmf_m), ('CMF', cmf_cmf_m)]:
        print(f"  {name:12s} | {m['metric_error']:14.4f} | {m['curvature_error']:12.4f} | {m['reconstruction_mse']:10.4f}")

    fig_crosssection(all_results[PRIMARY_ALPHA], groups, neutral)
    fig_bar(all_metrics[PRIMARY_ALPHA], 'bias_linear', 'Bias Probe Accuracy (Linear)', 'bar_linear.png')
    fig_bar(all_metrics[PRIMARY_ALPHA], 'bias_mlp', 'Bias Probe Accuracy (MLP)', 'bar_mlp.png')

    print(f"\n{'='*70}\n  ALL DONE. Outputs: {OUT}\n{'='*70}")
