#!/usr/bin/env python3
"""
Hyperbolic (H^2) experiment with CMF comparison.

Replicates run_hyperbolic.py with Baseline AE + CMF added.
Manifold: H^2 in Lorentz model, -x0^2 + x1^2 + x2^2 = -1, x0 > 0.
Bias = angular position theta, Task = radial distance r.
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
OUT = os.path.join(os.path.dirname(__file__), "output", "hyperbolic")
rng = np.random.default_rng(SEED)

# CMF scripts now use config's METHOD_ORDER (9 methods) + fill_info_panel for 10th panel


# =============================================================================
# Lorentz Operations (from run_hyperbolic.py)
# =============================================================================

def lorentz_inner(x, y):
    return -x[..., 0] * y[..., 0] + np.sum(x[..., 1:] * y[..., 1:], axis=-1)

def lorentz_norm(v):
    sq = -v[..., 0]**2 + np.sum(v[..., 1:]**2, axis=-1)
    return np.sqrt(np.maximum(sq, 0.0))

def lorentz_project(X):
    X_out = X.copy()
    spatial_sq = np.sum(X_out[..., 1:]**2, axis=-1)
    X_out[..., 0] = np.sqrt(1.0 + spatial_sq)
    return X_out

def constraint_residual_hyp(X):
    lip = -X[..., 0]**2 + np.sum(X[..., 1:]**2, axis=-1)
    return np.abs(lip + 1.0)

def tangent_project_lorentz(x, v):
    lip = lorentz_inner(x, v)
    return v + lip[..., None] * x

def exp_map_lorentz(x, v):
    vnorm = lorentz_norm(v)
    vnorm = np.minimum(vnorm, 50.0)
    vnorm_safe = np.where(vnorm < 1e-12, 1.0, vnorm)
    return (np.cosh(vnorm)[..., None] * x
            + np.sinh(vnorm)[..., None] * v / vnorm_safe[..., None])

def frechet_mean_lorentz(X, max_iter=50, lr=0.5, tol=1e-8):
    mu = lorentz_project(X.mean(axis=0, keepdims=True))
    for _ in range(max_iter):
        mu_broad = np.broadcast_to(mu, X.shape)
        neg_dot = -lorentz_inner(mu_broad, X)
        neg_dot = np.maximum(neg_dot, 1.0 + 1e-15)
        d = np.arccosh(neg_dot)
        coeff = d / np.sqrt(neg_dot**2 - 1.0 + 1e-30)
        log_vecs = coeff[:, None] * (X - neg_dot[:, None] * mu)
        avg_tangent = log_vecs.mean(axis=0, keepdims=True)
        avg_tangent = tangent_project_lorentz(mu, avg_tangent)
        tn = lorentz_norm(avg_tangent)
        if tn < tol:
            break
        mu = exp_map_lorentz(mu, lr * avg_tangent)
        mu = lorentz_project(mu)
    return mu.squeeze(0)


# =============================================================================
# Data Generation
# =============================================================================

def generate_data():
    print("=" * 70)
    print("Hyperbolic (H^2): Generating data")
    print("=" * 70)
    g0_lo, g0_hi = -np.pi / 4, np.pi / 4
    g1_lo, g1_hi = np.pi / 2, np.pi

    r_g0 = rng.uniform(1.5, 2.5, N_PER_GROUP)
    r_g1 = rng.uniform(1.5, 2.5, N_PER_GROUP)
    theta_g0 = rng.uniform(g0_lo, g0_hi, N_PER_GROUP)
    theta_g1 = rng.uniform(g1_lo, g1_hi, N_PER_GROUP)

    r_all = np.concatenate([r_g0, r_g1])
    theta_all = np.concatenate([theta_g0, theta_g1])
    groups = np.array([0] * N_PER_GROUP + [1] * N_PER_GROUP)
    task_labels = (r_all > np.median(r_all)).astype(int)

    X = np.column_stack([np.cosh(r_all),
                         np.sinh(r_all) * np.cos(theta_all),
                         np.sinh(r_all) * np.sin(theta_all)])

    print(f"  N = {len(X)}")
    print(f"  Manifold check: {constraint_residual_hyp(X).max():.2e}")
    return X, groups, r_all, theta_all, task_labels


def extract_theta_r(X):
    r = np.arccosh(np.clip(X[:, 0], 1.0, None))
    theta = np.arctan2(X[:, 2], X[:, 1])
    return theta, r


# =============================================================================
# Neutral & Bias direction
# =============================================================================

def compute_neutral(X, groups):
    clf = LogisticRegression(solver='lbfgs', max_iter=500, random_state=SEED)
    clf.fit(X, groups)
    probs = clf.predict_proba(X).max(axis=1)
    low_conf = probs < LOWCONF_THR
    n_low = low_conf.sum()
    print(f"  Low-confidence: {n_low}/{len(X)}")
    if n_low >= 10:
        neutral = frechet_mean_lorentz(X[low_conf])
    else:
        neutral = frechet_mean_lorentz(X)
    bias_dir = clf.coef_[0] / np.linalg.norm(clf.coef_[0])

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X, groups)
    sfid_idx = np.argsort(rf.feature_importances_)[-2:]

    theta_neutral = float(np.arctan2(neutral[2], neutral[1]))
    print(f"  Neutral: ({neutral[0]:.3f}, {neutral[1]:.3f}, {neutral[2]:.3f})")
    print(f"  Neutral θ: {np.degrees(theta_neutral):.1f}°")
    return neutral, bias_dir, sfid_idx


# =============================================================================
# Post-hoc Debiasing Methods (from run_hyperbolic.py)
# =============================================================================

def method_spd(X, u, neutral, alpha):
    s_n = float(u @ neutral)
    return X + alpha * (s_n - X @ u)[:, None] * u[None, :]

def method_sfid(X, idx, neutral, alpha):
    X_out = X.copy()
    X_out[:, idx] = X[:, idx] + alpha * (neutral[idx][None, :] - X[:, idx])
    return X_out

def method_geo_i(X, u, neutral, alpha):
    t_proj = float(neutral @ u)
    u_broad = np.broadcast_to(u[None, :], X.shape)
    u_tang = tangent_project_lorentz(X, u_broad)
    unorm = lorentz_norm(u_tang)
    valid = unorm > 1e-10
    u_hat = np.zeros_like(u_tang)
    u_hat[valid] = u_tang[valid] / unorm[valid, None]
    a = X @ u
    b = np.sum(u_hat * u[None, :], axis=1)
    target = (1.0 - alpha) * a + alpha * t_proj
    A_ = (a + b) / 2.0
    B_ = (a - b) / 2.0
    disc = target**2 - 4.0 * A_ * B_
    eta = np.zeros(len(X))
    eta[valid] = (target[valid] - a[valid]) / np.maximum(unorm[valid], 1e-12)
    solvable = valid & (disc >= 0) & (np.abs(A_) > 1e-12)
    if solvable.any():
        sqrt_d = np.sqrt(np.maximum(disc[solvable], 0.0))
        A_s, t_s = A_[solvable], target[solvable]
        w1 = (t_s + sqrt_d) / (2.0 * A_s)
        w2 = (t_s - sqrt_d) / (2.0 * A_s)
        eta1 = np.where(w1 > 1e-15, np.log(np.maximum(w1, 1e-15)), 1e10)
        eta2 = np.where(w2 > 1e-15, np.log(np.maximum(w2, 1e-15)), 1e10)
        use1 = np.abs(eta1) <= np.abs(eta2)
        eta_solved = np.where(use1, eta1, eta2)
        has_sol = (w1 > 1e-15) | (w2 > 1e-15)
        idx_s = np.where(solvable)[0]
        eta[idx_s[has_sol]] = eta_solved[has_sol]
    eta = np.clip(eta, -5.0, 5.0)
    return exp_map_lorentz(X, eta[:, None] * u_hat)

def method_isorot(X, neutral, alpha):
    theta_n = np.arctan2(neutral[2], neutral[1])
    theta_i = np.arctan2(X[:, 2], X[:, 1])
    diff = np.arctan2(np.sin(theta_n - theta_i), np.cos(theta_n - theta_i))
    delta = alpha * diff
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    X_out = X.copy()
    X_out[:, 1] = X[:, 1] * cos_d - X[:, 2] * sin_d
    X_out[:, 2] = X[:, 1] * sin_d + X[:, 2] * cos_d
    return X_out

def method_oracle(X, neutral, alpha):
    theta_n = np.arctan2(neutral[2], neutral[1])
    r = np.arccosh(np.clip(X[:, 0], 1.0, None))
    theta_i = np.arctan2(X[:, 2], X[:, 1])
    diff = np.arctan2(np.sin(theta_n - theta_i), np.cos(theta_n - theta_i))
    theta_new = theta_i + alpha * diff
    return np.column_stack([np.cosh(r), np.sinh(r)*np.cos(theta_new),
                            np.sinh(r)*np.sin(theta_new)])


# =============================================================================
# CMF Counterfactual for Hyperbolic
# =============================================================================

def hyp_counterfactual(X_np, groups_np):
    """
    Keep r (task), map theta to same relative position in opposite group.
    Group 0: θ ∈ [-π/4, π/4], Group 1: θ ∈ [π/2, π].
    """
    X_cf = X_np.copy()
    theta, r = extract_theta_r(X_np)
    g0_lo, g0_hi = -np.pi / 4, np.pi / 4
    g1_lo, g1_hi = np.pi / 2, np.pi

    for i in range(len(X_np)):
        if groups_np[i] == 0:
            frac = np.clip((theta[i] - g0_lo) / (g0_hi - g0_lo), 0, 1)
            theta_cf = g1_lo + frac * (g1_hi - g1_lo)
        else:
            frac = np.clip((theta[i] - g1_lo) / (g1_hi - g1_lo), 0, 1)
            theta_cf = g0_lo + frac * (g0_hi - g0_lo)
        X_cf[i] = [np.cosh(r[i]),
                    np.sinh(r[i]) * np.cos(theta_cf),
                    np.sinh(r[i]) * np.sin(theta_cf)]
    return X_cf


# =============================================================================
# Run all methods
# =============================================================================

def run_all_methods(X, neutral, bias_dir, sfid_idx, alpha, bl_rec, cmf_rec):
    X_sfid = method_sfid(X, sfid_idx, neutral, alpha)
    X_spd = method_spd(X, bias_dir, neutral, alpha)
    results = {
        'Original':    X.copy(),
        'Oracle':      method_oracle(X, neutral, alpha),
        'CMF':         cmf_rec,
        'SFID':        X_sfid,
        'SFID+Proj':   lorentz_project(X_sfid),
        'SPD':         X_spd,
        'SPD+Proj':    lorentz_project(X_spd),
        'Geodesic':    method_geo_i(X, bias_dir, neutral, alpha),
        'IsoRot':      method_isorot(X, neutral, alpha),
    }
    # Baseline_AE stored separately for CMF-native metrics, not in main display
    results['_Baseline_AE'] = bl_rec
    return results


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(X_orig, results, groups):
    _, r_orig = extract_theta_r(X_orig)
    metrics = {}
    for name in METHOD_ORDER:
        Xm = results[name]
        _, r = extract_theta_r(Xm)
        res = float(constraint_residual_hyp(Xm).mean())
        r_mae = float(np.mean(np.abs(r - r_orig)))
        r_inband = float(np.mean((r >= 1.5) & (r <= 2.5)))
        Xtr, Xte, gtr, gte = train_test_split(
            Xm, groups, test_size=PROBE_TEST_SIZE, stratify=groups,
            random_state=PROBE_RANDOM_STATE)
        clf_lin = LogisticRegression(solver='lbfgs', max_iter=500,
                                     random_state=PROBE_RANDOM_STATE)
        clf_lin.fit(Xtr, gtr)
        bias_lin = float(accuracy_score(gte, clf_lin.predict(Xte)))
        clf_mlp = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
                                random_state=PROBE_RANDOM_STATE)
        clf_mlp.fit(Xtr, gtr)
        bias_mlp = float(accuracy_score(gte, clf_mlp.predict(Xte)))
        metrics[name] = {
            'on_manifold': res, 'r_mae': r_mae, 'r_inband': r_inband,
            'bias_linear': bias_lin, 'bias_mlp': bias_mlp,
        }
    return metrics

def annotate_val(val, metric_name):
    if metric_name == 'on_manifold':
        if val < 1e-3: return 'O'
        elif val < 0.05: return 'triangle'
        else: return 'X'
    elif metric_name == 'r_mae':
        if val < 0.01: return 'O'
        elif val < 0.1: return 'triangle'
        else: return 'X'
    elif metric_name == 'r_inband':
        if val > 0.99: return 'O'
        elif val > 0.90: return 'triangle'
        else: return 'X'
    elif metric_name in ('bias_linear', 'bias_mlp'):
        if val < 0.55: return 'O'
        elif val < 0.65: return 'triangle'
        else: return 'X'
    return ''

def print_result_table(metrics, filepath):
    header = (f"{'Method':12s} | {'On-Mfld':14s} | {'r MAE':18s} | "
              f"{'r In-Band':18s} | {'Bias(Lin)':10s} | {'Bias(MLP)':10s}")
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name in METHOD_ORDER:
        m = metrics[name]
        a1 = annotate_val(m['on_manifold'], 'on_manifold')
        a2 = annotate_val(m['r_mae'], 'r_mae')
        a3 = annotate_val(m['r_inband'], 'r_inband')
        a4 = annotate_val(m['bias_linear'], 'bias_linear')
        a5 = annotate_val(m['bias_mlp'], 'bias_mlp')
        line = (f"{name:12s} | {a1} ({m['on_manifold']:.4f})"
                f"     | {a2} (t_MAE={m['r_mae']:.4f})"
                f"   | {a3} (t_band={100*m['r_inband']:.1f}%)"
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
    print("\nPlotting setting ...")
    _r_mesh = np.linspace(0, 3.2, 35)
    _th_mesh = np.linspace(0, 2*np.pi, 50)
    R_m, T_m = np.meshgrid(_r_mesh, _th_mesh)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.sinh(R_m)*np.cos(T_m), np.sinh(R_m)*np.sin(T_m), np.cosh(R_m),
                    alpha=0.08, color='lightgray', edgecolor='gray', linewidth=0.08)
    for g in [0, 1]:
        mask = groups == g
        ax.scatter(X[mask, 1], X[mask, 2], X[mask, 0],
                   c=GROUP_COLORS[g], s=14, alpha=0.7, label=f'Group {g}')
    ax.scatter(neutral[1], neutral[2], neutral[0],
               c=NEUTRAL_COLOR, s=200, marker='*', edgecolors='k', linewidths=0.8, zorder=5)
    ax.view_init(elev=18, azim=25)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_zlabel('x₀')
    save_fig(fig, os.path.join(OUT, "setting.png"))

def fig_crosssection(results, groups, neutral):
    print("\nPlotting cross-section ...")
    fig, axes = make_grid(figsize=(25, 10))
    def signed_r_spatial(X):
        r = np.sqrt(X[:, 1]**2 + X[:, 2]**2)
        theta = np.arctan2(X[:, 2], X[:, 1])
        return np.where(theta >= 0, r, -r)

    all_rs, all_x0 = [], []
    for name in METHOD_ORDER:
        X = results[name]
        all_rs.append(signed_r_spatial(X)); all_x0.append(X[:, 0])
    rs_min, rs_max = min(v.min() for v in all_rs), max(v.max() for v in all_rs)
    x0_min, x0_max = min(v.min() for v in all_x0), max(v.max() for v in all_x0)
    xlim = (rs_min - 0.5, rs_max + 0.5)
    ylim = (max(0.8, x0_min - 0.3), x0_max + 0.3)
    curve_r = np.linspace(xlim[0], xlim[1], 300)
    curve_x0 = np.sqrt(1 + curve_r**2)

    neutral_r = np.sqrt(neutral[1]**2 + neutral[2]**2)
    neutral_theta = np.arctan2(neutral[2], neutral[1])
    neutral_rs = neutral_r if neutral_theta >= 0 else -neutral_r

    for idx, name in enumerate(METHOD_ORDER):
        ax = axes[idx]
        X = results[name]
        rs, x0 = signed_r_spatial(X), X[:, 0]
        ax.plot(curve_r, curve_x0, 'k-', lw=1.5, alpha=0.4, zorder=1)
        ax.scatter([neutral_rs], [neutral[0]], c=NEUTRAL_COLOR, s=120, marker='*',
                   edgecolors='k', linewidths=0.6, zorder=10)
        for g in [0, 1]:
            mask = groups == g
            ax.scatter(rs[mask], x0[mask], c=GROUP_COLORS[g],
                       s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.set_title(name); ax.set_xlim(xlim); ax.set_ylim(ylim)
    fill_info_panel(axes[len(METHOD_ORDER)], "Cross-section",
                    "Signed r_spatial vs x0.\n"
                    "Curve = hyperboloid constraint.\n"
                    "Off-curve = off-manifold.")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "crosssection.png"))

def fig_topdown(results, groups, neutral, alpha, filename):
    print(f"\nPlotting top-down (alpha={alpha}) ...")
    fig, axes = make_grid(figsize=(25, 10))
    r_inner, r_outer = np.sinh(1.5), np.sinh(2.5)
    th_ring = np.linspace(0, 2*np.pi, 200)
    for idx, name in enumerate(METHOD_ORDER):
        ax = axes[idx]
        X = results[name]
        ax.plot(r_inner*np.cos(th_ring), r_inner*np.sin(th_ring), 'k--', lw=0.8, alpha=0.3)
        ax.plot(r_outer*np.cos(th_ring), r_outer*np.sin(th_ring), 'k--', lw=0.8, alpha=0.3)
        for g in [0, 1]:
            mask = groups == g
            ax.scatter(X[mask, 1], X[mask, 2], c=GROUP_COLORS[g],
                       s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.set_title(name); ax.set_aspect('equal')
        lim = max(r_outer*1.3, 8.0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fill_info_panel(axes[len(METHOD_ORDER)], "Top-down",
                    f"x1-x2 spatial view (alpha={alpha}).\n"
                    "Dashed rings = r in [1.5, 2.5].")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, filename))

def fig_scatter(results, groups, neutral, alpha, filename):
    print(f"\nPlotting scatter (alpha={alpha}) ...")
    fig, axes = make_grid(figsize=(25, 10))
    theta_neutral = float(np.arctan2(neutral[2], neutral[1]))
    for idx, name in enumerate(METHOD_ORDER):
        ax = axes[idx]
        X = results[name]
        theta, r = extract_theta_r(X)
        ax.axhspan(1.5, 2.5, color='green', alpha=0.06, zorder=0)
        ax.axhline(1.5, color='green', lw=0.8, ls=':', alpha=0.4)
        ax.axhline(2.5, color='green', lw=0.8, ls=':', alpha=0.4)
        ax.axvline(np.degrees(theta_neutral), color=NEUTRAL_COLOR, lw=2, ls='--', alpha=0.7)
        for g in [0, 1]:
            mask = groups == g
            ax.scatter(np.degrees(theta[mask]), r[mask], c=GROUP_COLORS[g],
                       s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.set_title(name); ax.set_xlim(-200, 200); ax.set_ylim(0, 4.5)
    fill_info_panel(axes[len(METHOD_ORDER)], "Scatter",
                    f"theta vs r (alpha={alpha}).\n"
                    "Green band = task-attribute range.\n"
                    "Gold line = neutral angle.")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, filename))

def fig_bar(metrics, probe_key, ylabel, filename):
    print(f"\nPlotting bar ({probe_key}) ...")
    fig, ax = plt.subplots(figsize=(12, 5))
    methods = METHOD_ORDER_NO_ORIG
    vals = [metrics[m][probe_key] for m in methods]
    colors = [METHOD_COLORS.get(m, '#999999') for m in methods]
    bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor='black', lw=0.5, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
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
    X, groups, r_all, theta_all, task_labels = generate_data()
    neutral, bias_dir, sfid_idx = compute_neutral(X, groups)

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
        bl_model = train_cmf_model(X, groups, task_labels, hyp_counterfactual,
                                   model_type="Baseline", input_dim=3, latent_dim=2,
                                   epochs=300, lr=1e-3)
        bl_rec, bl_z = get_reconstructed(bl_model, X)
        X_cf = hyp_counterfactual(X, groups)
        bl_cmf_m = compute_cmf_metrics(bl_model, X, X_cf, subset_size=32)
        save_cmf_results(bl_model, bl_cmf_m, bl_rec, bl_z, cache_bl)

    cached = load_cmf_results(cache_cmf, input_dim=3)
    if cached:
        print("  Loaded cached CMF")
        cmf_model, cmf_cmf_m, cmf_rec, cmf_z = cached
    else:
        cmf_model = train_cmf_model(X, groups, task_labels, hyp_counterfactual,
                                    model_type="CMF", input_dim=3, latent_dim=2,
                                    epochs=300, lr=1e-3, lambda_metric=10.0, lambda_curv=5.0)
        cmf_rec, cmf_z = get_reconstructed(cmf_model, X)
        X_cf = hyp_counterfactual(X, groups)
        cmf_cmf_m = compute_cmf_metrics(cmf_model, X, X_cf, subset_size=32)
        save_cmf_results(cmf_model, cmf_cmf_m, cmf_rec, cmf_z, cache_cmf)

    print(f"\n  Baseline AE: {bl_cmf_m}")
    print(f"  CMF: {cmf_cmf_m}")
    plot_cmf_latent(bl_z, groups, r_all, "r (radial)", "Baseline AE",
                    os.path.join(OUT, "latent_baseline.png"))
    plot_cmf_latent(cmf_z, groups, r_all, "r (radial)", "CMF",
                    os.path.join(OUT, "latent_cmf.png"))

    fig_setting(X, groups, neutral)

    all_metrics, all_results = {}, {}
    for alpha in ALPHAS:
        print(f"\n{'='*70}\n  ALPHA = {alpha}\n{'='*70}")
        results = run_all_methods(X, neutral, bias_dir, sfid_idx, alpha, bl_rec, cmf_rec)
        all_results[alpha] = results
        metrics = compute_metrics(X, results, groups)
        all_metrics[alpha] = metrics
        tag = f"a{str(alpha).replace('.','')}"
        fig_scatter(results, groups, neutral, alpha, f"scatter_{tag}.png")
        fig_topdown(results, groups, neutral, alpha, f"topdown_{tag}.png")

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
