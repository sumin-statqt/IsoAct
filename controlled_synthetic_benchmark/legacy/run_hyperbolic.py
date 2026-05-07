#!/usr/bin/env python3
"""
Finalized Hyperbolic H² toy experiment.

Manifold: H² in Lorentz model R³, -x₀² + x₁² + x₂² = -1, x₀ > 0.
Bias = angular position θ, Task = radial distance r.
Group 0: θ ∈ [-π/4, π/4] (right), Group 1: θ ∈ [π/2, π] (left-upper).

Methods: Original, SFID, SFID+Proj, SPD, SPD+Proj, Geodesic, IsoRot, Oracle
Figures: setting, result_table, crosssection, topdown (×2α), scatter (×2α),
         bar_linear, bar_mlp
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "hyperbolic")
os.makedirs(OUT_DIR, exist_ok=True)
setup_style()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Lorentz Operations                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

def lorentz_inner(x, y):
    """Minkowski inner product: ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + x₂y₂."""
    return -x[..., 0] * y[..., 0] + np.sum(x[..., 1:] * y[..., 1:], axis=-1)


def lorentz_norm(v):
    """Lorentz norm: √|⟨v,v⟩_L| (for spacelike tangent vectors)."""
    sq = -v[..., 0]**2 + np.sum(v[..., 1:]**2, axis=-1)
    return np.sqrt(np.maximum(sq, 0.0))


def lorentz_project(X):
    """Spatial-preserving projection to H²: keep (x₁,x₂), recompute x₀."""
    X_out = X.copy()
    spatial_sq = np.sum(X_out[..., 1:]**2, axis=-1)
    X_out[..., 0] = np.sqrt(1.0 + spatial_sq)
    return X_out


def constraint_residual_hyp(X):
    """Per-point: |⟨x,x⟩_L + 1|."""
    lip = -X[..., 0]**2 + np.sum(X[..., 1:]**2, axis=-1)
    return np.abs(lip + 1.0)


def tangent_project_lorentz(x, v):
    """Project v onto T_x(H²): v_tang = v + ⟨v,x⟩_L · x."""
    lip = lorentz_inner(x, v)
    return v + lip[..., None] * x


def exp_map_lorentz(x, v):
    """Exponential map on H²: Exp_x(v) = cosh(‖v‖)x + sinh(‖v‖)v/‖v‖."""
    vnorm = lorentz_norm(v)
    vnorm = np.minimum(vnorm, 50.0)
    vnorm_safe = np.where(vnorm < 1e-12, 1.0, vnorm)
    return (np.cosh(vnorm)[..., None] * x
            + np.sinh(vnorm)[..., None] * v / vnorm_safe[..., None])


def log_map_lorentz(x, y):
    """Log map: tangent vector at x pointing toward y."""
    neg_dot = -lorentz_inner(x, y)
    neg_dot = np.maximum(neg_dot, 1.0 + 1e-15)
    d = np.arccosh(neg_dot)
    direction = y - neg_dot[..., None] * x
    sinh_d = np.sinh(d)
    safe_sinh = np.where(sinh_d < 1e-12, 1.0, sinh_d)
    return (d / safe_sinh)[..., None] * direction


def geodesic_distance_lorentz(x, y):
    """arccosh(-⟨x,y⟩_L)."""
    neg_dot = -lorentz_inner(x, y)
    neg_dot = np.maximum(neg_dot, 1.0)
    return np.arccosh(neg_dot)


def frechet_mean_lorentz(X, max_iter=50, lr=0.5, tol=1e-8):
    """Iterative Frechet mean on H² via tangent-space averaging."""
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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Data Generation                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

rng = np.random.default_rng(SEED)

# Base scripts: 8 methods (no CMF), use config's METHOD_ORDER_BASE

g0_lo, g0_hi = -np.pi / 4, np.pi / 4
g1_lo, g1_hi = np.pi / 2, np.pi

r_g0 = rng.uniform(1.5, 2.5, N_PER_GROUP)
r_g1 = rng.uniform(1.5, 2.5, N_PER_GROUP)
theta_g0 = rng.uniform(g0_lo, g0_hi, N_PER_GROUP)
theta_g1 = rng.uniform(g1_lo, g1_hi, N_PER_GROUP)

r_all = np.concatenate([r_g0, r_g1])
theta_all = np.concatenate([theta_g0, theta_g1])
groups = np.array([0] * N_PER_GROUP + [1] * N_PER_GROUP)
N = len(groups)

# Task labels: binary via median split on r
task_labels = (r_all > np.median(r_all)).astype(int)


def hyp_map(r, theta):
    """Parametric map (r,θ) → (cosh r, sinh r cos θ, sinh r sin θ)."""
    return np.column_stack([
        np.cosh(r),
        np.sinh(r) * np.cos(theta),
        np.sinh(r) * np.sin(theta),
    ])


X_orig = hyp_map(r_all, theta_all)
print(f"Manifold check (max residual): {constraint_residual_hyp(X_orig).max():.2e}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Feature Extraction                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

def extract_theta_r(X):
    """Extract intrinsic coordinates (θ, r) from ambient Lorentz coords."""
    r = np.arccosh(np.clip(X[:, 0], 1.0, None))
    theta = np.arctan2(X[:, 2], X[:, 1])
    return theta, r


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Neutral Vector & Directions                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

clf_lr = LogisticRegression(solver='lbfgs', max_iter=500, random_state=SEED)
clf_lr.fit(X_orig, groups)
probs = clf_lr.predict_proba(X_orig).max(axis=1)
low_conf = probs < LOWCONF_THR
n_low = low_conf.sum()
print(f"Low-confidence samples: {n_low}/{N}")

if n_low >= 10:
    neutral_vec = frechet_mean_lorentz(X_orig[low_conf])
else:
    neutral_vec = frechet_mean_lorentz(X_orig)

bias_dir = clf_lr.coef_[0] / np.linalg.norm(clf_lr.coef_[0])

# SFID dims (RF feature importances, top 2)
SFID_N_DIMS = 2
clf_rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
clf_rf.fit(X_orig, groups)
sfid_idx = np.argsort(clf_rf.feature_importances_)[-SFID_N_DIMS:]

theta_neutral = float(np.arctan2(neutral_vec[2], neutral_vec[1]))
print(f"Neutral: ({neutral_vec[0]:.3f}, {neutral_vec[1]:.3f}, {neutral_vec[2]:.3f})")
print(f"Bias dir: {bias_dir}")
print(f"SFID dims: {sfid_idx}")
print(f"Neutral θ: {np.degrees(theta_neutral):.1f}°")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Debiasing Methods                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

def method_spd(X, u, neutral, alpha):
    """SPD: x' = x + α(s_n - x·u)u in R³ ambient (Euclidean dot)."""
    s_n = float(u @ neutral)
    return X + alpha * (s_n - X @ u)[:, None] * u[None, :]


def method_sfid(X, idx, neutral, alpha):
    """SFID: coordinate replacement on important dims."""
    X_out = X.copy()
    X_out[:, idx] = X[:, idx] + alpha * (neutral[idx][None, :] - X[:, idx])
    return X_out


def method_geo_i(X, u, neutral, alpha):
    """
    Geodesic (Geo-I): intrinsic bias direction field → tangent project → exp_map.

    For each point x:
      1. Project bias direction u onto T_x(H²)
      2. Normalize to unit tangent vector
      3. Compute shift needed: cur_proj = x·u, target = neutral·u
      4. Solve for step size η via quadratic (exact solution for ambient dot product)
      5. Move: x' = Exp_x(η · v)

    Ported from phase4_approx_compare.py: geo_i_rf().
    """
    t_proj = float(neutral @ u)

    # Project u onto tangent space at each x
    u_broad = np.broadcast_to(u[None, :], X.shape)
    u_tang = tangent_project_lorentz(X, u_broad)
    unorm = lorentz_norm(u_tang)

    valid = unorm > 1e-10
    u_hat = np.zeros_like(u_tang)
    u_hat[valid] = u_tang[valid] / unorm[valid, None]

    # Current and target Euclidean projections onto bias direction
    a = X @ u  # current projection
    b = np.sum(u_hat * u[None, :], axis=1)  # d/dη of projection at η=0

    target = (1.0 - alpha) * a + alpha * t_proj

    # Quadratic solve: after exp_map, projection = cosh(η)·a + sinh(η)·b
    # Using A, B decomposition for the quadratic
    A = (a + b) / 2.0
    B = (a - b) / 2.0
    disc = target**2 - 4.0 * A * B

    eta = np.zeros(len(X))
    # Default: linear approximation for non-solvable points
    eta[valid] = (target[valid] - a[valid]) / np.maximum(unorm[valid], 1e-12)

    solvable = valid & (disc >= 0) & (np.abs(A) > 1e-12)
    if solvable.any():
        sqrt_d = np.sqrt(np.maximum(disc[solvable], 0.0))
        A_s = A[solvable]
        t_s = target[solvable]
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
    """
    Isometric Rotation: rotate (x₁, x₂) spatial plane toward neutral angle.

    This is a Lorentz isometry: preserves x₀ = cosh(r) hence r,
    preserves x₁²+x₂² hence manifold constraint, preserves all pairwise
    geodesic distances.
    """
    theta_n = np.arctan2(neutral[2], neutral[1])
    theta_i = np.arctan2(X[:, 2], X[:, 1])
    diff = np.arctan2(np.sin(theta_n - theta_i), np.cos(theta_n - theta_i))
    delta = alpha * diff

    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    X_out = X.copy()
    X_out[:, 1] = X[:, 1] * cos_d - X[:, 2] * sin_d
    X_out[:, 2] = X[:, 1] * sin_d + X[:, 2] * cos_d
    # x₀ unchanged → r preserved, manifold constraint preserved
    return X_out


def method_oracle(X, neutral, alpha):
    """Oracle: directly interpolate θ toward neutral, keep r fixed."""
    theta_n = np.arctan2(neutral[2], neutral[1])
    r = np.arccosh(np.clip(X[:, 0], 1.0, None))
    theta_i = np.arctan2(X[:, 2], X[:, 1])
    diff = np.arctan2(np.sin(theta_n - theta_i), np.cos(theta_n - theta_i))
    theta_new = theta_i + alpha * diff
    return hyp_map(r, theta_new)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Apply Methods at Each Alpha                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

def apply_all_methods(alpha):
    """Apply all 8 methods at given alpha, return dict name→X."""
    X_sfid = method_sfid(X_orig, sfid_idx, neutral_vec, alpha)
    X_sfid_proj = lorentz_project(X_sfid)
    X_spd = method_spd(X_orig, bias_dir, neutral_vec, alpha)
    X_spd_proj = lorentz_project(X_spd)
    X_geo = method_geo_i(X_orig, bias_dir, neutral_vec, alpha)
    X_isorot = method_isorot(X_orig, neutral_vec, alpha)
    X_oracle = method_oracle(X_orig, neutral_vec, alpha)
    return {
        'Original':  X_orig,
        'Oracle':    X_oracle,
        'SFID':      X_sfid,
        'SFID+Proj': X_sfid_proj,
        'SPD':       X_spd,
        'SPD+Proj':  X_spd_proj,
        'Geodesic':  X_geo,
        'IsoRot':    X_isorot,
    }


# Pre-compute for all alphas
results = {}
for alpha in ALPHAS:
    print(f"\nApplying methods at α={alpha}...")
    results[alpha] = apply_all_methods(alpha)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Metrics Computation                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def compute_metrics(X, name):
    """Compute all 5 metrics for one method's output."""
    theta, r = extract_theta_r(X)
    theta_orig, r_orig = extract_theta_r(X_orig)

    # 1. On-manifold residual
    res = float(constraint_residual_hyp(X).mean())
    on_manifold = res < 1e-3

    # 2. r MAE
    r_mae = float(np.mean(np.abs(r - r_orig)))

    # 3. r in-band fraction
    r_inband = float(np.mean((r >= 1.5) & (r <= 2.5)))

    # 4. Bias Linear probe
    X_tr, X_te, g_tr, g_te = train_test_split(
        X, groups, test_size=PROBE_TEST_SIZE, stratify=groups,
        random_state=PROBE_RANDOM_STATE)
    clf_lin = LogisticRegression(solver='lbfgs', max_iter=500,
                                  random_state=PROBE_RANDOM_STATE)
    clf_lin.fit(X_tr, g_tr)
    bias_lin = float(accuracy_score(g_te, clf_lin.predict(X_te)))

    # 5. Bias MLP probe
    clf_mlp = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
                             random_state=PROBE_RANDOM_STATE)
    clf_mlp.fit(X_tr, g_tr)
    bias_mlp = float(accuracy_score(g_te, clf_mlp.predict(X_te)))

    return {
        'name':       name,
        'res':        res,
        'on_manifold': on_manifold,
        'r_mae':      r_mae,
        'r_inband':   r_inband,
        'bias_lin':   bias_lin,
        'bias_mlp':   bias_mlp,
    }


# Compute metrics at primary alpha
print(f"\nComputing metrics at α={PRIMARY_ALPHA}...")
metrics = {}
for name in METHOD_ORDER_BASE:
    X = results[PRIMARY_ALPHA][name]
    m = compute_metrics(X, name)
    metrics[name] = m
    tag = "ON " if m['on_manifold'] else "OFF"
    print(f"  {name:12s} | Res={m['res']:.4f} [{tag}] | rMAE={m['r_mae']:.4f} | "
          f"rBand={m['r_inband']:.3f} | BiasLin={m['bias_lin']:.3f} | "
          f"BiasMLP={m['bias_mlp']:.3f}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. Setting Figure                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\nGenerating setting figure...")

# Hyperboloid surface mesh
_r_mesh = np.linspace(0, 3.2, 35)
_th_mesh = np.linspace(0, 2 * np.pi, 50)
R_m, T_m = np.meshgrid(_r_mesh, _th_mesh)
surf_X = np.sinh(R_m) * np.cos(T_m)
surf_Y = np.sinh(R_m) * np.sin(T_m)
surf_Z = np.cosh(R_m)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(surf_X, surf_Y, surf_Z,
                alpha=0.08, color='lightgray', edgecolor='gray', linewidth=0.08)

C0, C1 = GROUP_COLORS[0], GROUP_COLORS[1]

ax.scatter(X_orig[groups == 0, 1], X_orig[groups == 0, 2], X_orig[groups == 0, 0],
           c=C0, s=14, alpha=0.7, label='Group 0')
ax.scatter(X_orig[groups == 1, 1], X_orig[groups == 1, 2], X_orig[groups == 1, 0],
           c=C1, s=14, alpha=0.7, label='Group 1')
ax.scatter(neutral_vec[1], neutral_vec[2], neutral_vec[0],
           c=NEUTRAL_COLOR, s=200, marker='*', edgecolors='k', linewidths=0.8,
           zorder=5)

ax.view_init(elev=18, azim=25)
ax.set_xlabel('x₁', fontsize=AXIS_LABEL_SIZE, labelpad=2)
ax.set_ylabel('x₂', fontsize=AXIS_LABEL_SIZE, labelpad=2)
ax.set_zlabel('x₀', fontsize=AXIS_LABEL_SIZE, labelpad=2)
ax.tick_params(labelsize=TICK_SIZE)

save_fig(fig, os.path.join(OUT_DIR, "setting.png"))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. Result Table                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("Writing result table...")


def annotate_val(val, metric_name):
    """Return O/X/triangle annotation for a metric value."""
    if metric_name == 'on_manifold':
        if val < 1e-3:
            return 'O'
        elif val < 0.05:
            return 'triangle'
        else:
            return 'X'
    elif metric_name == 'r_mae':
        if val < 0.01:
            return 'O'
        elif val < 0.1:
            return 'triangle'
        else:
            return 'X'
    elif metric_name == 'r_inband':
        if val > 0.99:
            return 'O'
        elif val > 0.90:
            return 'triangle'
        else:
            return 'X'
    elif metric_name in ('bias_lin', 'bias_mlp'):
        if val < 0.55:
            return 'O'
        elif val < 0.65:
            return 'triangle'
        else:
            return 'X'
    return ''


table_path = os.path.join(OUT_DIR, "result_table.txt")
header = f"{'Method':12s} | {'On-Mfld':14s} | {'r MAE':18s} | {'r In-Band':18s} | {'Bias(Lin)':10s} | {'Bias(MLP)':10s}"
sep = "-" * len(header)

lines = [sep, header, sep]
for name in METHOD_ORDER_BASE:
    m = metrics[name]
    a1 = annotate_val(m['res'], 'on_manifold')
    a2 = annotate_val(m['r_mae'], 'r_mae')
    a3 = annotate_val(m['r_inband'], 'r_inband')
    a4 = annotate_val(m['bias_lin'], 'bias_lin')
    a5 = annotate_val(m['bias_mlp'], 'bias_mlp')
    line = (f"{name:12s} | {a1} ({m['res']:.4f})"
            f"     | {a2} (r_MAE={m['r_mae']:.4f})"
            f"   | {a3} (r_band={100*m['r_inband']:.1f}%)"
            f" | {m['bias_lin']:.3f} {a4}"
            f"   | {m['bias_mlp']:.3f} {a5}")
    lines.append(line)
lines.append(sep)

with open(table_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f"  Saved: {table_path}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. Cross-section  (α = PRIMARY_ALPHA)                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("Generating cross-section figure...")

# Use r_spatial = sqrt(x₁² + x₂²) signed by θ = atan2(x₂, x₁).
# Constraint: -x₀² + x₁² + x₂² = -1  =>  x₀ = sqrt(1 + r_spatial²).
# This way ALL on-manifold points lie exactly on the reference curve,
# regardless of their x₂ value.

def signed_r_spatial(X):
    """Signed spatial radius: sign(θ) · √(x₁² + x₂²)."""
    r = np.sqrt(X[:, 1]**2 + X[:, 2]**2)
    theta = np.arctan2(X[:, 2], X[:, 1])
    return np.where(theta >= 0, r, -r)

# Compute global axis range
all_rs, all_x0 = [], []
for name in METHOD_ORDER_BASE:
    X = results[PRIMARY_ALPHA][name]
    all_rs.append(signed_r_spatial(X))
    all_x0.append(X[:, 0])
rs_min = min(v.min() for v in all_rs)
rs_max = max(v.max() for v in all_rs)
x0_min = min(v.min() for v in all_x0)
x0_max = max(v.max() for v in all_x0)
rs_pad = max((rs_max - rs_min) * 0.1, 0.5)
x0_pad = max((x0_max - x0_min) * 0.1, 0.3)
xlim_cs = (rs_min - rs_pad, rs_max + rs_pad)
ylim_cs = (max(0.8, x0_min - x0_pad), x0_max + x0_pad)

# Manifold curve: x₀ = √(1 + r²)
curve_r = np.linspace(xlim_cs[0], xlim_cs[1], 300)
curve_x0 = np.sqrt(1 + curve_r**2)

# Neutral signed r
neutral_r = np.sqrt(neutral_vec[1]**2 + neutral_vec[2]**2)
neutral_theta = np.arctan2(neutral_vec[2], neutral_vec[1])
neutral_rs = neutral_r if neutral_theta >= 0 else -neutral_r

fig, axes = make_grid(figsize=(25, 10))

for idx, name in enumerate(METHOD_ORDER_BASE):
    ax = axes[idx]
    X = results[PRIMARY_ALPHA][name]

    rs_vals = signed_r_spatial(X)
    x0_vals = X[:, 0]

    # Manifold curve
    ax.plot(curve_r, curve_x0, 'k-', linewidth=1.5, alpha=0.4, zorder=1)

    # Neutral star
    ax.scatter([neutral_rs], [neutral_vec[0]],
               c=NEUTRAL_COLOR, s=120, marker='*', edgecolors='k',
               linewidths=0.6, zorder=10)

    # Data points
    ax.scatter(rs_vals[groups == 0], x0_vals[groups == 0],
               c=C0, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
    ax.scatter(rs_vals[groups == 1], x0_vals[groups == 1],
               c=C1, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)

    ax.set_title(name)
    ax.set_xlabel(r'$r_{spatial}$ (signed by $\theta$)')
    ax.set_ylabel(r'$x_0$')
    ax.set_xlim(xlim_cs)
    ax.set_ylim(ylim_cs)

fill_info_panel(axes[len(METHOD_ORDER_BASE)], "Cross-section",
                "Signed r_spatial vs x0.\n"
                "Curve = hyperboloid constraint.\n"
                "Off-curve = off-manifold.")
for extra_idx in range(len(METHOD_ORDER_BASE) + 1, len(axes)):
    axes[extra_idx].axis('off')

plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "crosssection.png"))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Helper: top-down and scatter figure generation                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

r_inner, r_outer = np.sinh(1.5), np.sinh(2.5)
th_ring = np.linspace(0, 2 * np.pi, 200)
theta_orig_vals, r_orig_vals = extract_theta_r(X_orig)


def make_topdown_fig(alpha, path):
    """Generate 2×5 top-down view (x₁ vs x₂)."""
    fig, axes = make_grid(figsize=(25, 10))
    method_data = results[alpha]

    for idx, name in enumerate(METHOD_ORDER_BASE):
        ax = axes[idx]
        X = method_data[name]

        # Reference circles
        ax.plot(r_inner * np.cos(th_ring), r_inner * np.sin(th_ring),
                'k--', lw=0.8, alpha=0.3)
        ax.plot(r_outer * np.cos(th_ring), r_outer * np.sin(th_ring),
                'k--', lw=0.8, alpha=0.3)

        # Debiased
        ax.scatter(X[groups == 0, 1], X[groups == 0, 2],
                   c=C0, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.scatter(X[groups == 1, 1], X[groups == 1, 2],
                   c=C1, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)

        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(name)
        ax.set_aspect('equal')
        lim = max(r_outer * 1.3, 8.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.15)
        ax.axhline(0, color='k', lw=0.3, alpha=0.3)
        ax.axvline(0, color='k', lw=0.3, alpha=0.3)

    fill_info_panel(axes[len(METHOD_ORDER_BASE)], "Top-down",
                    f"x1-x2 spatial view (alpha={alpha}).\n"
                    "Dashed rings = r in [1.5, 2.5].")
    for extra_idx in range(len(METHOD_ORDER_BASE) + 1, len(axes)):
        axes[extra_idx].axis('off')

    plt.tight_layout()
    save_fig(fig, path)


def make_scatter_fig(alpha, path):
    """Generate 2×5 scatter view (θ vs r)."""
    fig, axes = make_grid(figsize=(25, 10))
    method_data = results[alpha]

    for idx, name in enumerate(METHOD_ORDER_BASE):
        ax = axes[idx]
        X = method_data[name]
        theta, r = extract_theta_r(X)

        # Green band for r ∈ [1.5, 2.5]
        ax.axhspan(1.5, 2.5, color='green', alpha=0.06, zorder=0)
        ax.axhline(1.5, color='green', lw=0.8, ls=':', alpha=0.4)
        ax.axhline(2.5, color='green', lw=0.8, ls=':', alpha=0.4)

        # Gold vertical line at θ_neutral
        ax.axvline(np.degrees(theta_neutral), color=NEUTRAL_COLOR, lw=2,
                    ls='--', alpha=0.7)

        # Debiased
        ax.scatter(np.degrees(theta[groups == 0]), r[groups == 0],
                   c=C0, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.scatter(np.degrees(theta[groups == 1]), r[groups == 1],
                   c=C1, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)

        ax.set_xlabel('θ (degrees)')
        ax.set_ylabel('r (task)')
        ax.set_title(name)
        ax.set_xlim(-200, 200)
        ax.set_ylim(0, 4.5)
        ax.grid(True, alpha=0.15)

    fill_info_panel(axes[len(METHOD_ORDER_BASE)], "Scatter",
                    f"theta vs r (alpha={alpha}).\n"
                    "Green band = task-attribute range.\n"
                    "Gold line = neutral angle.")
    for extra_idx in range(len(METHOD_ORDER_BASE) + 1, len(axes)):
        axes[extra_idx].axis('off')

    plt.tight_layout()
    save_fig(fig, path)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4-5. Top-Down Figures (α=0.9 and α=1.0)                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("Generating top-down figures...")
make_topdown_fig(0.9, os.path.join(OUT_DIR, "topdown_a09.png"))
make_topdown_fig(1.0, os.path.join(OUT_DIR, "topdown_a10.png"))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6-7. Scatter Figures (α=0.9 and α=1.0)                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("Generating scatter figures...")
make_scatter_fig(0.9, os.path.join(OUT_DIR, "scatter_a09.png"))
make_scatter_fig(1.0, os.path.join(OUT_DIR, "scatter_a10.png"))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  8-9. Bar Charts (Linear & MLP Bias Probes)                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("Generating bar charts...")

bar_names = METHOD_ORDER_BASE_NO_ORIG
bar_colors = [METHOD_COLORS.get(n, '#9E9E9E') for n in bar_names]
x_pos = np.arange(len(bar_names))


def make_bar_fig(metric_key, ylabel, path, hline=0.5, higher_better=False):
    """Generate single bar chart for one metric."""
    vals = [metrics[n][metric_key] for n in bar_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x_pos, vals, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_names, rotation=35, ha='right', fontsize=TICK_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.grid(axis='y', alpha=0.2)

    if hline is not None:
        ax.axhline(hline, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    for bar, val in zip(bars, vals):
        yoff = max(max(vals) * 0.01, 0.003)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yoff,
                f'{val:.3f}', ha='center', va='bottom', fontsize=TICK_SIZE)

    ax.set_ylim(0.0, 1.05)
    plt.tight_layout()
    save_fig(fig, path)


make_bar_fig('bias_lin', 'Bias Probe Accuracy (Linear)',
             os.path.join(OUT_DIR, "bar_linear.png"),
             hline=0.5)

make_bar_fig('bias_mlp', 'Bias Probe Accuracy (MLP)',
             os.path.join(OUT_DIR, "bar_mlp.png"),
             hline=0.5)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Summary                                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

print(f"\nAll outputs saved to: {OUT_DIR}/")
print("Files generated:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
print("\nDone.")
