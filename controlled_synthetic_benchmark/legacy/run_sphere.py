#!/usr/bin/env python3
"""
Finalized Hypersphere S^2 Toy Experiment
=========================================
Manifold: Unit sphere S^2 in R^3
Bias attribute: azimuthal angle phi
Task attribute: polar angle theta (from z-axis)

Two groups separated in azimuth (phi ~ +/- 30 deg),
sharing similar polar angles (theta ~ N(25 deg, 8 deg)).

Methods: Original, SFID, SFID+Proj, SPD, SPD+Proj, Geodesic, IsoRot, Oracle
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'sphere')
os.makedirs(OUT_DIR, exist_ok=True)

setup_style()
rng = np.random.RandomState(SEED)

# Base scripts: 8 methods (no CMF), use config's METHOD_ORDER_BASE

# ═══════════════════════════════════════════════════════════════════════════════
#  1. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
THETA_BAND = (5.0, 45.0)   # degrees, for in-band metric
SFID_N_DIMS = 2

def spherical_to_cartesian(theta_rad, phi_rad):
    """theta = polar (from z-axis), phi = azimuthal."""
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    return np.column_stack([x, y, z])


# Group 0: phi ~ N(+30 deg, 8 deg)
# Group 1: phi ~ N(-30 deg, 8 deg)
phi_g0 = np.radians(30) + rng.randn(N_PER_GROUP) * np.radians(8)
phi_g1 = np.radians(-30) + rng.randn(N_PER_GROUP) * np.radians(8)

# Both groups: theta ~ N(25 deg, 8 deg), clipped to positive cone
theta_g0 = np.radians(25) + rng.randn(N_PER_GROUP) * np.radians(8)
theta_g1 = np.radians(25) + rng.randn(N_PER_GROUP) * np.radians(8)
theta_g0 = np.clip(theta_g0, np.radians(2), np.radians(60))
theta_g1 = np.clip(theta_g1, np.radians(2), np.radians(60))

X_g0 = spherical_to_cartesian(theta_g0, phi_g0)
X_g1 = spherical_to_cartesian(theta_g1, phi_g1)

X_orig = np.vstack([X_g0, X_g1])
groups = np.array([0] * N_PER_GROUP + [1] * N_PER_GROUP)
N = len(groups)

theta_all_rad = np.concatenate([theta_g0, theta_g1])
phi_all_rad = np.concatenate([phi_g0, phi_g1])

# Task labels: binary via median split on theta
theta_median = np.median(theta_all_rad)
task_labels = (theta_all_rad >= theta_median).astype(int)

print(f"Data: {N} points on S^2")
print(f"  Group 0 phi: {np.degrees(phi_g0).mean():.1f} +/- {np.degrees(phi_g0).std():.1f} deg")
print(f"  Group 1 phi: {np.degrees(phi_g1).mean():.1f} +/- {np.degrees(phi_g1).std():.1f} deg")
print(f"  Theta range: [{np.degrees(theta_all_rad).min():.1f}, {np.degrees(theta_all_rad).max():.1f}] deg")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. NEUTRAL VECTOR
# ═══════════════════════════════════════════════════════════════════════════════
lr_neutral = LogisticRegression(max_iter=1000, random_state=SEED)
lr_neutral.fit(X_orig, groups)
probs = lr_neutral.predict_proba(X_orig)
max_prob = probs.max(axis=1)

low_conf_mask = max_prob < LOWCONF_THR
n_low_conf = low_conf_mask.sum()
print(f"  Low-confidence samples (max_prob < {LOWCONF_THR}): {n_low_conf}")

if n_low_conf >= 10:
    neutral_raw = X_orig[low_conf_mask].mean(axis=0)
else:
    print("  Warning: fewer than 10 low-conf samples, using overall mean")
    neutral_raw = X_orig.mean(axis=0)

neutral = neutral_raw / np.linalg.norm(neutral_raw)
print(f"  Neutral vector: [{neutral[0]:.4f}, {neutral[1]:.4f}, {neutral[2]:.4f}]")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. BIAS DIRECTION & SFID DIMS
# ═══════════════════════════════════════════════════════════════════════════════
lr_bias = LogisticRegression(max_iter=1000, random_state=SEED)
lr_bias.fit(X_orig, groups)
bias_dir = lr_bias.coef_[0].copy()
bias_dir /= np.linalg.norm(bias_dir)
print(f"  Bias direction: [{bias_dir[0]:.4f}, {bias_dir[1]:.4f}, {bias_dir[2]:.4f}]")

# SFID: RandomForest feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(X_orig, groups)
importances = rf.feature_importances_
sfid_dims = np.argsort(importances)[::-1][:SFID_N_DIMS]
print(f"  RF importances: {importances} -> SFID dims: {sfid_dims}")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def project_to_sphere(X):
    """Normalize each row to unit norm."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return X / norms


def extract_phi_theta(X):
    """Extract azimuthal phi and polar theta (in degrees) from Cartesian."""
    norms = np.linalg.norm(X, axis=1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    X_n = X / norms[:, None]
    phi = np.arctan2(X_n[:, 1], X_n[:, 0])
    theta = np.arccos(np.clip(X_n[:, 2], -1, 1))
    return phi, np.degrees(theta)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. DEBIASING METHODS
# ═══════════════════════════════════════════════════════════════════════════════
def spd_debias(X, bias_dir, neutral, alpha):
    """SPD: x' = x + alpha * (neutral.u - x.u) * u  in ambient R^3."""
    u = bias_dir
    s_n = np.dot(neutral, u)
    projections = X @ u  # (N,)
    shifts = alpha * (s_n - projections)  # (N,)
    return X + shifts[:, None] * u[None, :]


def sfid_debias(X, sfid_dims, neutral, alpha):
    """SFID: replace top-k dims with neutral values."""
    X_out = X.copy()
    for j in sfid_dims:
        X_out[:, j] = X[:, j] + alpha * (neutral[j] - X[:, j])
    return X_out


def geodesic_debias_sphere(X, bias_dir, neutral, alpha):
    """Per-point geodesic debiasing on S^2 via great circle movement."""
    u = bias_dir
    t_proj = float(u @ neutral)
    X_out = X.copy()
    for i in range(len(X)):
        x = X[i]
        # Project bias direction onto tangent space at x
        u_tang = u - np.dot(u, x) * x
        norm_ut = np.linalg.norm(u_tang)
        if norm_ut < 1e-10:
            continue
        v = u_tang / norm_ut
        cur = np.dot(x, u)
        shift = alpha * (t_proj - cur)
        t = shift / norm_ut  # scale by tangent norm
        x_new = np.cos(t) * x + np.sin(t) * v
        X_out[i] = x_new / np.linalg.norm(x_new)  # safety renorm
    return X_out


def isorot_sphere(X, neutral, alpha):
    """Isometric rotation in (x,y) plane. Preserves z (theta) and norm."""
    phi_neutral = np.arctan2(neutral[1], neutral[0])
    X_out = X.copy()
    for i in range(len(X)):
        phi_i = np.arctan2(X[i, 1], X[i, 0])
        diff = np.arctan2(np.sin(phi_neutral - phi_i), np.cos(phi_neutral - phi_i))
        delta = alpha * diff
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        X_out[i, 0] = X[i, 0] * cos_d - X[i, 1] * sin_d
        X_out[i, 1] = X[i, 0] * sin_d + X[i, 1] * cos_d
        # z unchanged -> theta preserved
    return X_out


def oracle_sphere(X, neutral, alpha):
    """Oracle: directly interpolate phi toward neutral, keep theta fixed."""
    phi_n = np.arctan2(neutral[1], neutral[0])
    X_out = X.copy()
    for i in range(len(X)):
        r_xy = np.sqrt(X[i, 0]**2 + X[i, 1]**2)
        phi_i = np.arctan2(X[i, 1], X[i, 0])
        theta_i = np.arccos(np.clip(X[i, 2], -1, 1))
        diff = np.arctan2(np.sin(phi_n - phi_i), np.cos(phi_n - phi_i))
        phi_new = phi_i + alpha * diff
        X_out[i, 0] = np.sin(theta_i) * np.cos(phi_new)
        X_out[i, 1] = np.sin(theta_i) * np.sin(phi_new)
        X_out[i, 2] = np.cos(theta_i)  # z preserved
    return X_out


# ═══════════════════════════════════════════════════════════════════════════════
#  6. APPLY ALL METHODS AT ALL ALPHAS
# ═══════════════════════════════════════════════════════════════════════════════
def apply_methods(alpha):
    """Apply all 8 methods at a given alpha and return dict of results."""
    X_spd = spd_debias(X_orig, bias_dir, neutral, alpha)
    X_spd_proj = project_to_sphere(X_spd)
    X_sfid = sfid_debias(X_orig, sfid_dims, neutral, alpha)
    X_sfid_proj = project_to_sphere(X_sfid)
    X_geo = geodesic_debias_sphere(X_orig, bias_dir, neutral, alpha)
    X_iso = isorot_sphere(X_orig, neutral, alpha)
    X_oracle = oracle_sphere(X_orig, neutral, alpha)

    return {
        'Original':   X_orig,
        'Oracle':     X_oracle,
        'SFID':       X_sfid,
        'SFID+Proj':  X_sfid_proj,
        'SPD':        X_spd,
        'SPD+Proj':   X_spd_proj,
        'Geodesic':   X_geo,
        'IsoRot':     X_iso,
    }


all_results = {}  # alpha -> {method_name -> X_debiased}
for alpha in ALPHAS:
    all_results[alpha] = apply_methods(alpha)
    print(f"\nApplied all methods at alpha={alpha}")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def metric_on_manifold(X):
    """Mean |norm - 1|. Lower is better."""
    return np.mean(np.abs(np.linalg.norm(X, axis=1) - 1.0))


def metric_theta_mae(X, theta_orig_deg):
    """Mean absolute error in theta (degrees) after debiasing."""
    _, theta_deg = extract_phi_theta(X)
    return np.mean(np.abs(theta_deg - theta_orig_deg))


def metric_theta_inband(X, band_lo=THETA_BAND[0], band_hi=THETA_BAND[1]):
    """Fraction of points with theta in [band_lo, band_hi] degrees."""
    _, theta_deg = extract_phi_theta(X)
    return np.mean((theta_deg >= band_lo) & (theta_deg <= band_hi))


def metric_bias_linear(X, groups):
    """Logistic regression accuracy for predicting group from X."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, groups, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE,
        stratify=groups)
    clf = LogisticRegression(max_iter=1000, random_state=PROBE_RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


def metric_bias_mlp(X, groups):
    """MLP accuracy for predicting group from X."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, groups, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE,
        stratify=groups)
    clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
                        random_state=PROBE_RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


def compute_all_metrics(methods_dict, alpha_label=""):
    """Compute all 5 metrics for all methods. Returns dict of dicts."""
    theta_orig_deg = np.degrees(theta_all_rad)
    results = {}
    for name in METHOD_ORDER_BASE:
        X = methods_dict[name]
        m_on = metric_on_manifold(X)
        m_mae = metric_theta_mae(X, theta_orig_deg)
        m_band = metric_theta_inband(X)
        m_lin = metric_bias_linear(X, groups)
        m_mlp = metric_bias_mlp(X, groups)
        results[name] = {
            'on_manifold': m_on,
            'theta_mae': m_mae,
            'theta_inband': m_band,
            'bias_linear': m_lin,
            'bias_mlp': m_mlp,
        }
        print(f"  {alpha_label}{name:12s} | OnManif: {m_on:.5f} | "
              f"thetaMAE: {m_mae:.3f} deg | InBand: {m_band:.3f} | "
              f"BiasLin: {m_lin:.3f} | BiasMLP: {m_mlp:.3f}")
    return results


print("\n" + "=" * 90)
print("METRICS")
print("=" * 90)

all_metrics = {}
for alpha in ALPHAS:
    print(f"\n--- alpha = {alpha} ---")
    all_metrics[alpha] = compute_all_metrics(all_results[alpha],
                                              alpha_label=f"[a={alpha}] ")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Shared data for plots ────────────────────────────────────────────────────
phi_neutral_deg = np.degrees(np.arctan2(neutral[1], neutral[0]))
theta_orig_deg = np.degrees(theta_all_rad)


# --------------------------------------------------------------------------
#  8.1  SETTING FIGURE (setting.png)
# --------------------------------------------------------------------------
print("\n--- Generating setting.png ---")
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Sphere wireframe (upper hemisphere)
u_sph = np.linspace(0, 2 * np.pi, 50)
v_sph = np.linspace(0, np.pi / 2, 25)
sx = np.outer(np.cos(u_sph), np.sin(v_sph))
sy = np.outer(np.sin(u_sph), np.sin(v_sph))
sz = np.outer(np.ones_like(u_sph), np.cos(v_sph))
ax.plot_wireframe(sx, sy, sz, color='gray', alpha=0.08, linewidth=0.3)

# Data points colored by group
for g, color in GROUP_COLORS.items():
    mask = groups == g
    ax.scatter(X_orig[mask, 0], X_orig[mask, 1], X_orig[mask, 2],
               c=color, s=MARKER_SIZE, alpha=MARKER_ALPHA, edgecolors='none')

# Neutral point
ax.scatter(*neutral, c=NEUTRAL_COLOR, s=120, marker='*',
           edgecolors='k', linewidths=0.5, zorder=10)

ax.set_xlim([-0.7, 0.7])
ax.set_ylim([-0.7, 0.7])
ax.set_zlim([0.3, 1.05])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=30, azim=45)
fig.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'setting.png'))


# --------------------------------------------------------------------------
#  8.2  RESULT TABLE (result_table.txt)
# --------------------------------------------------------------------------
print("\n--- Generating result_table.txt ---")
alpha_tbl = PRIMARY_ALPHA
metrics_tbl = all_metrics[alpha_tbl]


def annotate_val(val, metric_name):
    """Return O/X/triangle annotation for a metric value."""
    if metric_name == 'on_manifold':
        if val < 1e-3:
            return 'O'
        elif val < 0.05:
            return 'triangle'
        else:
            return 'X'
    elif metric_name == 'theta_mae':
        if val < 0.5:
            return 'O'
        elif val < 2.0:
            return 'triangle'
        else:
            return 'X'
    elif metric_name == 'theta_inband':
        if val > 0.99:
            return 'O'
        elif val > 0.90:
            return 'triangle'
        else:
            return 'X'
    elif metric_name in ('bias_linear', 'bias_mlp'):
        if val < 0.55:
            return 'O'
        elif val < 0.65:
            return 'triangle'
        else:
            return 'X'
    return ''


header = f"{'Method':12s} | {'On-Mfld':14s} | {'theta MAE':20s} | {'theta In-Band':20s} | {'Bias(Lin)':10s} | {'Bias(MLP)':10s}"
sep = "-" * len(header)

lines = [sep, header, sep]
for name in METHOD_ORDER_BASE:
    m = metrics_tbl[name]
    a1 = annotate_val(m['on_manifold'], 'on_manifold')
    a2 = annotate_val(m['theta_mae'], 'theta_mae')
    a3 = annotate_val(m['theta_inband'], 'theta_inband')
    a4 = annotate_val(m['bias_linear'], 'bias_linear')
    a5 = annotate_val(m['bias_mlp'], 'bias_mlp')
    line = (f"{name:12s} | {a1} ({m['on_manifold']:.4f})"
            f"     | {a2} (MAE={m['theta_mae']:.3f} deg)"
            f"   | {a3} (band={100*m['theta_inband']:.1f}%)"
            f" | {m['bias_linear']:.3f} {a4}"
            f"   | {m['bias_mlp']:.3f} {a5}")
    lines.append(line)
lines.append(sep)

table_str = "\n".join(lines)
print(table_str)

with open(os.path.join(OUT_DIR, 'result_table.txt'), 'w') as f:
    f.write(table_str + "\n")
print(f"  Saved: {os.path.join(OUT_DIR, 'result_table.txt')}")


# --------------------------------------------------------------------------
#  8.3  CROSS-SECTION (crosssection.png) — alpha=1.0
# --------------------------------------------------------------------------
print("\n--- Generating crosssection.png ---")
alpha_cs = PRIMARY_ALPHA
methods_cs = all_results[alpha_cs]

# Neutral marker in cross-section coords
neutral_rxy = np.sqrt(neutral[0]**2 + neutral[1]**2)
neutral_signed_rxy = np.sign(neutral[1]) * neutral_rxy
neutral_z = neutral[2]

# Compute global axis range
all_srxy, all_zv = [], []
for name in METHOD_ORDER_BASE:
    X_m = methods_cs[name]
    rxy = np.linalg.norm(X_m[:, :2], axis=1)
    srxy = np.sign(X_m[:, 1]) * rxy
    all_srxy.append(srxy)
    all_zv.append(X_m[:, 2])
srxy_min = min(s.min() for s in all_srxy)
srxy_max = max(s.max() for s in all_srxy)
z_min = min(z.min() for z in all_zv)
z_max = max(z.max() for z in all_zv)
xpad = max((srxy_max - srxy_min) * 0.15, 0.15)
ypad = max((z_max - z_min) * 0.15, 0.05)
xlim = (min(srxy_min - xpad, -0.8), max(srxy_max + xpad, 0.8))
ylim = (min(z_min - ypad, 0.5), max(z_max + ypad, 1.05))

# Manifold curve: upper unit semicircle (rxy^2 + z^2 = 1)
curve_rxy = np.linspace(-1, 1, 300)
curve_z = np.sqrt(np.maximum(1 - curve_rxy**2, 0))

fig, axes = make_grid(figsize=(25, 10))
for idx, name in enumerate(METHOD_ORDER_BASE):
    ax = axes[idx]
    X_m = methods_cs[name]

    rxy = np.linalg.norm(X_m[:, :2], axis=1)
    signed_rxy = np.sign(X_m[:, 1]) * rxy
    z_vals = X_m[:, 2]

    # Manifold curve
    ax.plot(curve_rxy, curve_z, 'k-', linewidth=1.5, alpha=0.4, zorder=1)

    # Neutral star
    ax.scatter([neutral_signed_rxy], [neutral_z],
               c=NEUTRAL_COLOR, s=120, marker='*', edgecolors='k',
               linewidths=0.6, zorder=10)

    # Data points
    for g, color in GROUP_COLORS.items():
        mask = groups == g
        ax.scatter(signed_rxy[mask], z_vals[mask],
                   c=color, s=MARKER_SIZE, alpha=MARKER_ALPHA,
                   edgecolors='none', zorder=3)

    ax.set_title(name)
    ax.set_xlabel(r'$r_{xy}$ (signed by $y$)')
    ax.set_ylabel('z')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

fill_info_panel(axes[len(METHOD_ORDER_BASE)], "Cross-section",
                "Signed r_xy vs z.\n"
                "Semicircle = unit sphere constraint.\n"
                "Off-curve = off-manifold.")
for extra_idx in range(len(METHOD_ORDER_BASE) + 1, len(axes)):
    axes[extra_idx].axis('off')

fig.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'crosssection.png'))


# --------------------------------------------------------------------------
#  8.4-8.5  TOP-DOWN VIEWS (topdown_a09.png, topdown_a10.png)
# --------------------------------------------------------------------------
for alpha_td in ALPHAS:
    tag = f"a{str(alpha_td).replace('.', '')}"
    fname = f"topdown_{tag}.png"
    print(f"\n--- Generating {fname} ---")
    methods_td = all_results[alpha_td]

    fig, axes = make_grid(figsize=(25, 10))
    for idx, name in enumerate(METHOD_ORDER_BASE):
        ax = axes[idx]
        X_m = methods_td[name]

        # Reference circles for task band: theta in [5°, 45°] → r = sin(theta)
        circle_t = np.linspace(0, 2 * np.pi, 200)
        r_inner = np.sin(np.radians(THETA_BAND[0]))
        r_outer = np.sin(np.radians(THETA_BAND[1]))
        ax.plot(r_inner * np.cos(circle_t), r_inner * np.sin(circle_t),
                'k--', linewidth=0.8, alpha=0.3)
        ax.plot(r_outer * np.cos(circle_t), r_outer * np.sin(circle_t),
                'k--', linewidth=0.8, alpha=0.3)

        # Debiased points
        for g, color in GROUP_COLORS.items():
            mask = groups == g
            ax.scatter(X_m[mask, 0], X_m[mask, 1],
                       c=color, s=MARKER_SIZE, alpha=MARKER_ALPHA,
                       edgecolors='none')

        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_xlim([-0.7, 0.7])
        ax.set_ylim([-0.7, 0.7])

    fill_info_panel(axes[len(METHOD_ORDER_BASE)], "Top-down",
                    f"x-y plane view (alpha={alpha_td}).\n"
                    "Dashed circles = task-attribute band.")
    for extra_idx in range(len(METHOD_ORDER_BASE) + 1, len(axes)):
        axes[extra_idx].axis('off')

    fig.tight_layout()
    save_fig(fig, os.path.join(OUT_DIR, fname))


# --------------------------------------------------------------------------
#  8.6-8.7  SCATTER PLOTS: phi vs theta (scatter_a09.png, scatter_a10.png)
# --------------------------------------------------------------------------
for alpha_sc in ALPHAS:
    tag = f"a{str(alpha_sc).replace('.', '')}"
    fname = f"scatter_{tag}.png"
    print(f"\n--- Generating {fname} ---")
    methods_sc = all_results[alpha_sc]

    fig, axes = make_grid(figsize=(25, 10))
    for idx, name in enumerate(METHOD_ORDER_BASE):
        ax = axes[idx]
        X_m = methods_sc[name]

        phi_m, theta_m = extract_phi_theta(X_m)
        phi_m_deg = np.degrees(phi_m)

        # Green band for theta in-band
        ax.axhspan(THETA_BAND[0], THETA_BAND[1], color='#4CAF50', alpha=0.08)

        # Gold vertical line at phi_neutral
        ax.axvline(phi_neutral_deg, color=NEUTRAL_COLOR, linewidth=1.5,
                   linestyle='--', alpha=0.7)

        # Debiased points
        for g, color in GROUP_COLORS.items():
            mask = groups == g
            ax.scatter(phi_m_deg[mask], theta_m[mask],
                       c=color, s=MARKER_SIZE, alpha=MARKER_ALPHA,
                       edgecolors='none')

        ax.set_title(name)
        ax.set_xlabel(r'$\varphi$ (deg)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.set_xlim([-80, 80])
        ax.set_ylim([0, 60])

    fill_info_panel(axes[len(METHOD_ORDER_BASE)], "Scatter",
                    f"phi vs theta (alpha={alpha_sc}).\n"
                    "Green band = task-attribute range.\n"
                    "Gold line = neutral azimuth.")
    for extra_idx in range(len(METHOD_ORDER_BASE) + 1, len(axes)):
        axes[extra_idx].axis('off')

    fig.tight_layout()
    save_fig(fig, os.path.join(OUT_DIR, fname))


# --------------------------------------------------------------------------
#  8.8-8.9  BAR CHARTS (bar_linear.png, bar_mlp.png)
# --------------------------------------------------------------------------
alpha_bar = PRIMARY_ALPHA
metrics_bar = all_metrics[alpha_bar]

for probe_key, probe_label, fname in [
    ('bias_linear', 'Bias Probe Accuracy (Linear)', 'bar_linear.png'),
    ('bias_mlp', 'Bias Probe Accuracy (MLP)', 'bar_mlp.png'),
]:
    print(f"\n--- Generating {fname} ---")
    fig, ax = plt.subplots(figsize=(9, 5))

    methods_bar = METHOD_ORDER_BASE_NO_ORIG
    values = [metrics_bar[m][probe_key] for m in methods_bar]
    colors = [METHOD_COLORS[m] for m in methods_bar]

    bars = ax.bar(range(len(methods_bar)), values, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.7)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=TICK_SIZE,
                fontweight='bold')

    # Horizontal line at chance (0.5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xticks(range(len(methods_bar)))
    ax.set_xticklabels(methods_bar, rotation=25, ha='right')
    ax.set_ylabel(probe_label)
    ax.set_ylim([0.0, 1.05])

    fig.tight_layout()
    save_fig(fig, os.path.join(OUT_DIR, fname))


# ═══════════════════════════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print(f"All outputs saved to: {OUT_DIR}")
print("=" * 90)
