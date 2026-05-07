#!/usr/bin/env python3
"""
SE(3) experiment with CMF comparison.

Replicates run_se3.py with Baseline AE + CMF added.
SE(3) = SO(3) x R^3, represented as 12-dim vector (flatten(R) + t).
Bias = z-rotation angle phi, Task = ||t|| (translation magnitude).
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
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

setup_style()
OUT = os.path.join(os.path.dirname(__file__), "output", "se3")
rng = np.random.default_rng(SEED)

# CMF scripts now use config's METHOD_ORDER (9 methods) + fill_info_panel for 10th panel
SFID_N_DIMS = 6


# =============================================================================
# SE(3) Operations (from run_se3.py)
# =============================================================================

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def make_rotation(phi, eps_x=0.0, eps_y=0.0):
    return Rx(eps_x) @ Ry(eps_y) @ Rz(phi)

def se3_to_vec(R, t):
    return np.concatenate([R.flatten(), t])

def vec_to_se3(v):
    return v[:9].reshape(3, 3), v[9:12].copy()

def project_to_SO3(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def se3_project(v):
    R_raw, t = vec_to_se3(v)
    R = project_to_SO3(R_raw)
    return se3_to_vec(R, t)

def project_all_se3(X):
    return np.array([se3_project(X[i]) for i in range(len(X))])

def so3_residual(X):
    res = np.zeros(len(X))
    for i in range(len(X)):
        R = X[i, :9].reshape(3, 3)
        res[i] = np.linalg.norm(R.T @ R - np.eye(3), 'fro')
    return res

def extract_z_angle(R):
    r = Rotation.from_matrix(R)
    angles = r.as_euler('XYZ')
    return angles[2], angles[0], angles[1]

def extract_phi_tnorm(X):
    phis, t_norms = np.zeros(len(X)), np.zeros(len(X))
    for i in range(len(X)):
        R_raw, t = vec_to_se3(X[i])
        R = project_to_SO3(R_raw)
        phi, _, _ = extract_z_angle(R)
        phis[i] = phi
        t_norms[i] = np.linalg.norm(t)
    return phis, t_norms

def so3_log(R):
    r = Rotation.from_matrix(R)
    rotvec = r.as_rotvec()
    theta = np.linalg.norm(rotvec)
    if theta < 1e-10:
        return np.zeros((3, 3))
    ax = rotvec / theta
    return np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]]) * theta

def so3_exp(Omega):
    theta = np.sqrt(max(0.0, -0.5 * np.trace(Omega @ Omega)))
    if theta < 1e-10:
        return np.eye(3) + Omega
    return (np.eye(3) + (np.sin(theta)/theta)*Omega
            + ((1-np.cos(theta))/theta**2)*(Omega@Omega))


# =============================================================================
# Data Generation
# =============================================================================

def generate_data():
    print("=" * 70)
    print("SE(3): Generating data")
    print("=" * 70)
    phi_g0 = rng.uniform(-np.pi/4, np.pi/4, N_PER_GROUP)
    phi_g1 = rng.uniform(np.pi/2, np.pi, N_PER_GROUP)
    phi_all = np.concatenate([phi_g0, phi_g1])
    groups = np.array([0]*N_PER_GROUP + [1]*N_PER_GROUP)
    N = len(groups)

    eps_x = rng.normal(0, 0.1, N)
    eps_y = rng.normal(0, 0.1, N)
    t_mag = rng.uniform(1.5, 2.5, N)
    t_dir = rng.standard_normal((N, 3))
    t_dir = t_dir / np.linalg.norm(t_dir, axis=1, keepdims=True)
    t_all = t_dir * t_mag[:, None]
    task_labels = (t_mag > np.median(t_mag)).astype(int)

    X_list = []
    for i in range(N):
        R = make_rotation(phi_all[i], eps_x[i], eps_y[i])
        X_list.append(se3_to_vec(R, t_all[i]))
    X = np.array(X_list)

    print(f"  N = {N}, dim = {X.shape[1]}")
    print(f"  SO(3) residual: {so3_residual(X).max():.2e}")
    return X, groups, phi_all, t_mag, task_labels


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
        neutral = X[low_conf].mean(axis=0)
    else:
        neutral = X.mean(axis=0)
    neutral = se3_project(neutral)
    bias_dir = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X, groups)
    sfid_idx = np.argsort(rf.feature_importances_)[-SFID_N_DIMS:]
    return neutral, bias_dir, sfid_idx


# =============================================================================
# Post-hoc Debiasing Methods (from run_se3.py)
# =============================================================================

def method_spd(X, u, neutral, alpha):
    s_n = float(u @ neutral)
    return X + alpha * (s_n - X @ u)[:, None] * u[None, :]

def method_sfid(X, idx, neutral, alpha):
    X_out = X.copy()
    X_out[:, idx] = X[:, idx] + alpha * (neutral[idx][None, :] - X[:, idx])
    return X_out

def method_geodesic(X, u, neutral, alpha):
    R_n = project_to_SO3(neutral[:9].reshape(3, 3))
    s_n = float(u @ neutral)
    X_out = X.copy()
    for i in range(len(X)):
        R_i = X[i, :9].reshape(3, 3)
        t_i = X[i, 9:12]
        Delta = R_i.T @ R_n
        Omega = so3_log(Delta)
        R_new = R_i @ so3_exp(alpha * Omega)
        R_new = project_to_SO3(R_new)
        cur = float(X[i] @ u)
        shift = alpha * (s_n - cur)
        t_new = t_i + shift * u[9:12]
        X_out[i] = se3_to_vec(R_new, t_new)
    return X_out

def method_isorot(X, neutral, alpha):
    R_n = project_to_SO3(neutral[:9].reshape(3, 3))
    phi_n, _, _ = extract_z_angle(R_n)
    X_out = X.copy()
    for i in range(len(X)):
        R_i = X[i, :9].reshape(3, 3)
        phi_i, _, _ = extract_z_angle(R_i)
        diff = np.arctan2(np.sin(phi_n - phi_i), np.cos(phi_n - phi_i))
        delta = alpha * diff
        R_new = R_i @ Rz(delta)
        X_out[i, :9] = R_new.flatten()
    return X_out

def method_oracle(X, neutral, alpha):
    R_n = project_to_SO3(neutral[:9].reshape(3, 3))
    phi_n, _, _ = extract_z_angle(R_n)
    X_out = X.copy()
    for i in range(len(X)):
        R_i = X[i, :9].reshape(3, 3)
        phi_i, eps_x_i, eps_y_i = extract_z_angle(R_i)
        diff = np.arctan2(np.sin(phi_n - phi_i), np.cos(phi_n - phi_i))
        phi_new = phi_i + alpha * diff
        R_new = make_rotation(phi_new, eps_x_i, eps_y_i)
        X_out[i, :9] = R_new.flatten()
    return X_out


# =============================================================================
# CMF Counterfactual for SE(3)
# =============================================================================

def se3_counterfactual(X_np, groups_np):
    """
    Keep t (task), map z-rotation angle phi to same relative position
    in opposite group. Keep eps_x, eps_y.
    Group 0: phi ∈ [-pi/4, pi/4], Group 1: phi ∈ [pi/2, pi].
    """
    X_cf = X_np.copy()
    g0_lo, g0_hi = -np.pi / 4, np.pi / 4
    g1_lo, g1_hi = np.pi / 2, np.pi

    for i in range(len(X_np)):
        R_i = X_np[i, :9].reshape(3, 3)
        t_i = X_np[i, 9:12]
        phi_i, eps_x_i, eps_y_i = extract_z_angle(project_to_SO3(R_i))

        if groups_np[i] == 0:
            frac = np.clip((phi_i - g0_lo) / (g0_hi - g0_lo), 0, 1)
            phi_cf = g1_lo + frac * (g1_hi - g1_lo)
        else:
            frac = np.clip((phi_i - g1_lo) / (g1_hi - g1_lo), 0, 1)
            phi_cf = g0_lo + frac * (g0_hi - g0_lo)

        R_cf = make_rotation(phi_cf, eps_x_i, eps_y_i)
        X_cf[i] = se3_to_vec(R_cf, t_i)
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
        'SFID+Proj':   project_all_se3(X_sfid),
        'SPD':         X_spd,
        'SPD+Proj':    project_all_se3(X_spd),
        'Geodesic':    method_geodesic(X, bias_dir, neutral, alpha),
        'IsoRot':      method_isorot(X, neutral, alpha),
    }
    # Baseline_AE stored separately for CMF-native metrics, not in main display
    results['_Baseline_AE'] = bl_rec
    return results


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(X_orig, results, groups):
    _, tnorm_orig = extract_phi_tnorm(X_orig)
    metrics = {}
    for name in METHOD_ORDER:
        Xm = results[name]
        _, tnorm = extract_phi_tnorm(Xm)
        res = float(so3_residual(Xm).mean())
        t_mae = float(np.mean(np.abs(tnorm - tnorm_orig)))
        t_inband = float(np.mean((tnorm >= 1.5) & (tnorm <= 2.5)))
        Xtr, Xte, gtr, gte = train_test_split(
            Xm, groups, test_size=PROBE_TEST_SIZE, stratify=groups,
            random_state=PROBE_RANDOM_STATE)
        clf_lin = LogisticRegression(solver='lbfgs', max_iter=500, C=0.1,
                                     random_state=PROBE_RANDOM_STATE)
        clf_lin.fit(Xtr, gtr)
        bias_lin = float(accuracy_score(gte, clf_lin.predict(Xte)))
        clf_mlp = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
                                random_state=PROBE_RANDOM_STATE)
        clf_mlp.fit(Xtr, gtr)
        bias_mlp = float(accuracy_score(gte, clf_mlp.predict(Xte)))
        metrics[name] = {
            'on_manifold': res, 't_mae': t_mae, 't_inband': t_inband,
            'bias_linear': bias_lin, 'bias_mlp': bias_mlp,
        }
    return metrics

def annotate_val(val, metric_name):
    if metric_name == 'on_manifold':
        if val < 1e-3: return 'O'
        elif val < 0.05: return 'triangle'
        else: return 'X'
    elif metric_name == 't_mae':
        if val < 0.01: return 'O'
        elif val < 0.1: return 'triangle'
        else: return 'X'
    elif metric_name == 't_inband':
        if val > 0.99: return 'O'
        elif val > 0.90: return 'triangle'
        else: return 'X'
    elif metric_name in ('bias_linear', 'bias_mlp'):
        if val < 0.55: return 'O'
        elif val < 0.65: return 'triangle'
        else: return 'X'
    return ''

def print_result_table(metrics, filepath):
    header = (f"{'Method':12s} | {'On-Mfld':14s} | {'‖t‖ MAE':18s} | "
              f"{'‖t‖ In-Band':18s} | {'Bias(Lin)':10s} | {'Bias(MLP)':10s}")
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name in METHOD_ORDER:
        m = metrics[name]
        a1 = annotate_val(m['on_manifold'], 'on_manifold')
        a2 = annotate_val(m['t_mae'], 't_mae')
        a3 = annotate_val(m['t_inband'], 't_inband')
        a4 = annotate_val(m['bias_linear'], 'bias_linear')
        a5 = annotate_val(m['bias_mlp'], 'bias_mlp')
        line = (f"{name:12s} | {a1} ({m['on_manifold']:.4f})"
                f"     | {a2} (t_MAE={m['t_mae']:.4f})"
                f"   | {a3} (t_band={100*m['t_inband']:.1f}%)"
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
    phis, tnorms = extract_phi_tnorm(X)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    for g in [0, 1]:
        mask = groups == g
        ax.scatter(np.cos(phis[mask]), np.sin(phis[mask]), tnorms[mask],
                   c=GROUP_COLORS[g], s=14, alpha=0.7, label=f'Group {g}')
    phi_n, tn_n = extract_phi_tnorm(neutral[None, :])
    ax.scatter([np.cos(phi_n[0])], [np.sin(phi_n[0])], [tn_n[0]],
               c=NEUTRAL_COLOR, s=200, marker='*', edgecolors='k', linewidths=0.8, zorder=5)
    ax.set_xlabel('cos(φ)'); ax.set_ylabel('sin(φ)'); ax.set_zlabel('‖t‖')
    save_fig(fig, os.path.join(OUT, "setting.png"))

def fig_crosssection(results, groups, neutral):
    print("\nPlotting cross-section ...")
    fig, axes = make_grid(figsize=(25, 10))
    for idx, name in enumerate(METHOD_ORDER):
        ax = axes[idx]
        Xm = results[name]
        # Show R[0,0] vs R[0,1] vs ||t|| in 2D: R[0,0] vs ||t||
        r00 = Xm[:, 0]  # R[0,0]
        tnorms = np.array([np.linalg.norm(Xm[i, 9:12]) for i in range(len(Xm))])
        for g in [0, 1]:
            mask = groups == g
            ax.scatter(r00[mask], tnorms[mask], c=GROUP_COLORS[g],
                       s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.set_title(name)
        ax.set_xlabel('R[0,0]'); ax.set_ylabel('‖t‖')
        ax.axhspan(1.5, 2.5, color='green', alpha=0.06, zorder=0)
    fill_info_panel(axes[len(METHOD_ORDER)], "Cross-section",
                    "R[0,0] vs ||t||.\n"
                    "Green band = task-attribute range.\n"
                    "R[0,0] deviates when SO(3) is broken.")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "crosssection.png"))

def fig_topdown(results, groups, alpha, filename):
    print(f"\nPlotting top-down (alpha={alpha}) ...")
    fig, axes = make_grid(figsize=(25, 10))
    for idx, name in enumerate(METHOD_ORDER):
        ax = axes[idx]
        Xm = results[name]
        t_x = Xm[:, 9]
        t_y = Xm[:, 10]
        circ_t = np.linspace(0, 2*np.pi, 200)
        ax.plot(1.5*np.cos(circ_t), 1.5*np.sin(circ_t), 'k--', lw=0.8, alpha=0.3)
        ax.plot(2.5*np.cos(circ_t), 2.5*np.sin(circ_t), 'k--', lw=0.8, alpha=0.3)
        for g in [0, 1]:
            mask = groups == g
            ax.scatter(t_x[mask], t_y[mask], c=GROUP_COLORS[g],
                       s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.set_title(name); ax.set_aspect('equal')
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    fill_info_panel(axes[len(METHOD_ORDER)], "Top-down",
                    f"t_x vs t_y (alpha={alpha}).\n"
                    "Dashed circles = ||t|| in [1.5, 2.5].")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, filename))

def fig_scatter(results, groups, neutral, alpha, filename):
    print(f"\nPlotting scatter (alpha={alpha}) ...")
    fig, axes = make_grid(figsize=(25, 10))
    phi_n, _ = extract_phi_tnorm(neutral[None, :])
    for idx, name in enumerate(METHOD_ORDER):
        ax = axes[idx]
        Xm = results[name]
        phis, tnorms = extract_phi_tnorm(Xm)
        ax.axhspan(1.5, 2.5, color='green', alpha=0.06, zorder=0)
        ax.axvline(np.degrees(phi_n[0]), color=NEUTRAL_COLOR, lw=2, ls='--', alpha=0.7)
        for g in [0, 1]:
            mask = groups == g
            ax.scatter(np.degrees(phis[mask]), tnorms[mask], c=GROUP_COLORS[g],
                       s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.set_title(name); ax.set_xlim(-200, 200); ax.set_ylim(0, 4)
    fill_info_panel(axes[len(METHOD_ORDER)], "Scatter",
                    f"phi vs ||t|| (alpha={alpha}).\n"
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
    X, groups, phi_all, t_mag, task_labels = generate_data()
    neutral, bias_dir, sfid_idx = compute_neutral(X, groups)

    print("\n" + "=" * 70)
    print("  Training CMF models")
    print("=" * 70)

    cache_bl = os.path.join(OUT, "cache_baseline_ae.pt")
    cache_cmf = os.path.join(OUT, "cache_cmf.pt")

    cached = load_cmf_results(cache_bl, input_dim=12, hidden_dim=64)
    if cached:
        print("  Loaded cached Baseline AE")
        bl_model, bl_cmf_m, bl_rec, bl_z = cached
    else:
        bl_model = train_cmf_model(X, groups, task_labels, se3_counterfactual,
                                   model_type="Baseline", input_dim=12, latent_dim=2,
                                   hidden_dim=64, epochs=300, lr=1e-3)
        bl_rec, bl_z = get_reconstructed(bl_model, X)
        X_cf = se3_counterfactual(X, groups)
        bl_cmf_m = compute_cmf_metrics(bl_model, X, X_cf, subset_size=32)
        save_cmf_results(bl_model, bl_cmf_m, bl_rec, bl_z, cache_bl)

    cached = load_cmf_results(cache_cmf, input_dim=12, hidden_dim=64)
    if cached:
        print("  Loaded cached CMF")
        cmf_model, cmf_cmf_m, cmf_rec, cmf_z = cached
    else:
        cmf_model = train_cmf_model(X, groups, task_labels, se3_counterfactual,
                                    model_type="CMF", input_dim=12, latent_dim=2,
                                    hidden_dim=64, epochs=300, lr=1e-3,
                                    lambda_metric=10.0, lambda_curv=5.0)
        cmf_rec, cmf_z = get_reconstructed(cmf_model, X)
        X_cf = se3_counterfactual(X, groups)
        cmf_cmf_m = compute_cmf_metrics(cmf_model, X, X_cf, subset_size=32)
        save_cmf_results(cmf_model, cmf_cmf_m, cmf_rec, cmf_z, cache_cmf)

    print(f"\n  Baseline AE: {bl_cmf_m}")
    print(f"  CMF: {cmf_cmf_m}")
    plot_cmf_latent(bl_z, groups, t_mag, "‖t‖", "Baseline AE",
                    os.path.join(OUT, "latent_baseline.png"))
    plot_cmf_latent(cmf_z, groups, t_mag, "‖t‖", "CMF",
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
