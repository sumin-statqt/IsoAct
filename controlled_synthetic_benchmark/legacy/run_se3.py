#!/usr/bin/env python3
"""
SE(3) Toy Experiment — Finalized version with Geodesic method.

SE(3) = SO(3) x R^3 (rotation + translation), the rigid body transformation group.

Setup:
  - Bias  = z-rotation angle phi
  - Task  = translation magnitude ||t||
  - Group 0: phi in [-pi/4, pi/4]   (near identity rotation)
  - Group 1: phi in [pi/2, pi]      (large z-rotation)
  - ||t|| in [1.5, 2.5], random direction
  - Small x,y rotation noise for realism

Methods (8):
  Original, SFID, SFID+Proj, SPD, SPD+Proj, Geodesic, IsoRot, Oracle

Ambient representation: flatten(R) + t = 12-dim vector
Projection to SE(3): nearest SO(3) via SVD polar decomposition + keep t
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.transform import Rotation

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "se3")
os.makedirs(OUT_DIR, exist_ok=True)

setup_style()

SFID_N_DIMS = 6  # half of 12-dim ambient

rng = np.random.default_rng(SEED)

# Base scripts: 8 methods (no CMF), use config's METHOD_ORDER_BASE


# ══════════════════════════════════════════════════════════════════════════════
# SO(3) / SE(3) helpers
# ══════════════════════════════════════════════════════════════════════════════

def Rz(angle):
    """Z-axis rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Rx(angle):
    """X-axis rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(angle):
    """Y-axis rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_rotation(phi, eps_x=0.0, eps_y=0.0):
    """R = Rx(eps_x) @ Ry(eps_y) @ Rz(phi)"""
    return Rx(eps_x) @ Ry(eps_y) @ Rz(phi)


def extract_z_angle(R):
    """Extract z-rotation angle from R via scipy Euler decomposition.
    Returns (phi, eps_x, eps_y)."""
    r = Rotation.from_matrix(R)
    angles = r.as_euler('XYZ')  # [eps_x, eps_y, phi]
    return angles[2], angles[0], angles[1]  # phi, eps_x, eps_y


def project_to_SO3(M):
    """Project 3x3 matrix to nearest SO(3) via SVD polar decomposition."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def se3_to_vec(R, t):
    """Flatten SE(3) element to 12-dim vector: flatten(R) + t."""
    return np.concatenate([R.flatten(), t])


def vec_to_se3(v):
    """Recover (R_raw, t) from 12-dim vector. R_raw may not be in SO(3)."""
    R_raw = v[:9].reshape(3, 3)
    t = v[9:12].copy()
    return R_raw, t


def se3_project(v):
    """Project 12-dim vector back to SE(3): nearest SO(3) + keep t."""
    R_raw, t = vec_to_se3(v)
    R = project_to_SO3(R_raw)
    return se3_to_vec(R, t)


# ══════════════════════════════════════════════════════════════════════════════
# Data generation
# ══════════════════════════════════════════════════════════════════════════════

# z-rotation angles (bias attribute)
phi_g0 = rng.uniform(-np.pi / 4, np.pi / 4, N_PER_GROUP)
phi_g1 = rng.uniform(np.pi / 2, np.pi, N_PER_GROUP)
phi_all = np.concatenate([phi_g0, phi_g1])

# Small x,y rotation noise
eps_x_all = rng.normal(0, 0.1, 2 * N_PER_GROUP)
eps_y_all = rng.normal(0, 0.1, 2 * N_PER_GROUP)

# Translation: random direction, magnitude in [1.5, 2.5] (task attribute)
t_mag = rng.uniform(1.5, 2.5, 2 * N_PER_GROUP)
t_dir = rng.standard_normal((2 * N_PER_GROUP, 3))
t_dir = t_dir / np.linalg.norm(t_dir, axis=1, keepdims=True)
t_all = t_dir * t_mag[:, None]

groups = np.array([0] * N_PER_GROUP + [1] * N_PER_GROUP)
N = len(groups)

# Task labels: median split on ||t||
task_labels = (t_mag > np.median(t_mag)).astype(int)

# Build SE(3) elements and flatten to 12-dim
rotations = []
X_orig_list = []
for i in range(N):
    R = make_rotation(phi_all[i], eps_x_all[i], eps_y_all[i])
    rotations.append(R)
    X_orig_list.append(se3_to_vec(R, t_all[i]))

X_orig = np.array(X_orig_list)
print(f"Data shape: {X_orig.shape} (N={N}, D=12)")

# Verify SO(3) membership
so3_residuals_init = []
for i in range(N):
    R_raw = X_orig[i, :9].reshape(3, 3)
    so3_residuals_init.append(np.linalg.norm(R_raw.T @ R_raw - np.eye(3), 'fro'))
print(f"SO(3) residual: max={max(so3_residuals_init):.2e}, "
      f"mean={np.mean(so3_residuals_init):.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# Neutral vector
# ══════════════════════════════════════════════════════════════════════════════

clf_lr = LogisticRegression(solver='lbfgs', max_iter=500, random_state=SEED)
clf_lr.fit(X_orig, groups)
probs = clf_lr.predict_proba(X_orig).max(axis=1)
low_conf = probs < LOWCONF_THR
n_low = low_conf.sum()
print(f"Low-confidence samples: {n_low}/{N}")

if n_low >= 10:
    neutral_vec = X_orig[low_conf].mean(axis=0)
else:
    neutral_vec = X_orig.mean(axis=0)

# Project neutral to SE(3)
neutral_vec = se3_project(neutral_vec)
R_neutral, t_neutral = vec_to_se3(neutral_vec)
R_neutral = project_to_SO3(R_neutral)
phi_neutral, _, _ = extract_z_angle(R_neutral)
print(f"Neutral phi: {np.degrees(phi_neutral):.1f} deg, "
      f"||t_neutral||: {np.linalg.norm(t_neutral):.3f}")

bias_dir = clf_lr.coef_[0] / np.linalg.norm(clf_lr.coef_[0])

# SFID dims (RF feature importance)
clf_rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
clf_rf.fit(X_orig, groups)
sfid_idx = np.argsort(clf_rf.feature_importances_)[-SFID_N_DIMS:]
print(f"SFID dims: {sfid_idx} "
      f"(importances: {clf_rf.feature_importances_[sfid_idx]})")


# ══════════════════════════════════════════════════════════════════════════════
# Debiasing methods
# ══════════════════════════════════════════════════════════════════════════════

def spd(X, u, neutral, alpha):
    """SPD: linear subspace projection in ambient 12-dim space."""
    s_n = float(u @ neutral)
    return X + alpha * (s_n - X @ u)[:, None] * u[None, :]


def sfid(X, idx, neutral, alpha):
    """SFID: coordinate replacement in ambient 12-dim space."""
    X_out = X.copy()
    X_out[:, idx] = X[:, idx] + alpha * (neutral[idx][None, :] - X[:, idx])
    return X_out


def project_all_se3(X):
    """Project each row back to SE(3)."""
    X_out = np.zeros_like(X)
    for i in range(len(X)):
        X_out[i] = se3_project(X[i])
    return X_out


def so3_log(R):
    """Compute so(3) logarithm: returns skew-symmetric Omega s.t. exp(Omega)=R."""
    r = Rotation.from_matrix(R)
    rotvec = r.as_rotvec()  # axis * angle
    theta = np.linalg.norm(rotvec)
    if theta < 1e-10:
        return np.zeros((3, 3))
    ax = rotvec / theta
    return np.array([
        [0, -ax[2], ax[1]],
        [ax[2], 0, -ax[0]],
        [-ax[1], ax[0], 0]
    ]) * theta


def so3_exp(Omega):
    """SO(3) exponential map via Rodrigues' formula."""
    theta = np.sqrt(max(0.0, -0.5 * np.trace(Omega @ Omega)))
    if theta < 1e-10:
        return np.eye(3) + Omega
    return (np.eye(3)
            + (np.sin(theta) / theta) * Omega
            + ((1.0 - np.cos(theta)) / theta**2) * (Omega @ Omega))


def geodesic_se3(X, u, neutral, alpha):
    """Geodesic debiasing on SE(3).

    Exploits SE(3) = SO(3) x R^3 product structure:
      - Rotation (SO(3), curved): geodesic interpolation toward R_neutral
        via SO(3) Log/Exp maps.
      - Translation (R^3, flat): SPD-style linear displacement along
        the bias direction's translation component.

    This is the intrinsic analog of SPD: the rotation part follows the
    manifold geodesic instead of moving through the ambient space, while
    the translation part is handled identically (R^3 is already flat).
    """
    R_n = neutral[:9].reshape(3, 3)
    R_n = project_to_SO3(R_n)
    s_n = float(u @ neutral)
    X_out = X.copy()

    for i in range(len(X)):
        R_i = X[i, :9].reshape(3, 3)
        t_i = X[i, 9:12]

        # --- Rotation: SO(3) geodesic toward R_neutral ---
        # Log_{R_i}(R_n) = log(R_i^T R_n) in so(3)
        Delta = R_i.T @ R_n
        Omega = so3_log(Delta)
        # Geodesic at parameter alpha: R_i @ exp(alpha * Omega)
        R_new = R_i @ so3_exp(alpha * Omega)
        R_new = project_to_SO3(R_new)  # numerical safety

        # --- Translation: SPD-style linear shift (R^3 is flat) ---
        cur = float(X[i] @ u)
        shift = alpha * (s_n - cur)
        d_trans = shift * u[9:12]
        t_new = t_i + d_trans

        X_out[i] = se3_to_vec(R_new, t_new)
    return X_out


def isometric_rotation_se3(X, neutral_v, alpha):
    """IsoRot: right-multiply R by corrective Rz(delta_phi), keep t.

    This is a right group action, hence an isometry. It:
      - Preserves SO(3) membership (product of rotations)
      - Preserves ||t|| exactly (translation untouched)
      - Preserves relative rotation distances
    """
    R_n = neutral_v[:9].reshape(3, 3)
    R_n = project_to_SO3(R_n)
    phi_n, _, _ = extract_z_angle(R_n)

    X_out = X.copy()
    for i in range(len(X)):
        R_i = X[i, :9].reshape(3, 3)
        phi_i, _, _ = extract_z_angle(R_i)

        # Signed angular difference (proper wrapping)
        diff = np.arctan2(np.sin(phi_n - phi_i), np.cos(phi_n - phi_i))
        delta = alpha * diff

        # Right-multiply by Rz(delta)
        R_new = R_i @ Rz(delta)
        X_out[i, :9] = R_new.flatten()
        # t unchanged

    return X_out


def oracle_se3(X, neutral_v, alpha):
    """Oracle: directly interpolate phi toward neutral, keep t and eps.

    phi_new = phi + alpha * (phi_neutral - phi)  [with angle wrapping]
    R_new = Rx(eps_x) @ Ry(eps_y) @ Rz(phi_new)
    t unchanged
    """
    R_n = neutral_v[:9].reshape(3, 3)
    R_n = project_to_SO3(R_n)
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


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction & metrics
# ══════════════════════════════════════════════════════════════════════════════

def extract_phi_tnorm(X):
    """Extract z-rotation angle and ||t|| from 12-dim vectors."""
    phis = np.zeros(len(X))
    eps_xs = np.zeros(len(X))
    eps_ys = np.zeros(len(X))
    t_norms = np.zeros(len(X))
    for i in range(len(X)):
        R_raw, t = vec_to_se3(X[i])
        R = project_to_SO3(R_raw)
        phi, ex, ey = extract_z_angle(R)
        phis[i] = phi
        eps_xs[i] = ex
        eps_ys[i] = ey
        t_norms[i] = np.linalg.norm(t)
    return phis, t_norms, eps_xs, eps_ys


def so3_residual(X):
    """Per-point SO(3) residual: ||R^T R - I||_F."""
    residuals = np.zeros(len(X))
    for i in range(len(X)):
        R_raw = X[i, :9].reshape(3, 3)
        residuals[i] = np.linalg.norm(R_raw.T @ R_raw - np.eye(3), 'fro')
    return residuals


# ══════════════════════════════════════════════════════════════════════════════
# Apply all methods
# ══════════════════════════════════════════════════════════════════════════════

def run_all_methods(X, alpha):
    """Run all 8 methods and return list of (name, X_debiased)
    in the canonical METHOD_ORDER_BASE order."""
    X_sfid_v    = sfid(X, sfid_idx, neutral_vec, alpha)
    X_sfid_proj = project_all_se3(X_sfid_v)
    X_spd_v     = spd(X, bias_dir, neutral_vec, alpha)
    X_spd_proj  = project_all_se3(X_spd_v)
    X_geo       = geodesic_se3(X, bias_dir, neutral_vec, alpha)
    X_isorot    = isometric_rotation_se3(X, neutral_vec, alpha)
    X_oracle    = oracle_se3(X, neutral_vec, alpha)

    return [
        ("Original",  X),
        ("Oracle",    X_oracle),
        ("SFID",      X_sfid_v),
        ("SFID+Proj", X_sfid_proj),
        ("SPD",       X_spd_v),
        ("SPD+Proj",  X_spd_proj),
        ("Geodesic",  X_geo),
        ("IsoRot",    X_isorot),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Compute & print metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(methods_list, X_ref, grps, t_labels):
    """Compute metrics for all methods. Returns list of dicts."""
    phi_ref, tnorm_ref, _, _ = extract_phi_tnorm(X_ref)

    header = (f"  {'Method':15s} | {'SO3_Res':>8s} | {'t_MAE':>8s} | "
              f"{'t_band':>8s} | {'BiasLin':>8s} | {'BiasMLP':>8s}")
    print(f"\n{'=' * 80}")
    print(header)
    print(f"{'=' * 80}")

    results = []
    for name, X in methods_list:
        phi, tnorm, ex, ey = extract_phi_tnorm(X)
        res_mean = so3_residual(X).mean()
        on_manif = res_mean < 1e-3

        # ||t|| preservation
        tnorm_mae = float(np.mean(np.abs(tnorm - tnorm_ref)))
        tnorm_in_band = float(np.mean((tnorm >= 1.5) & (tnorm <= 2.5)))

        # Bias probe -- Linear
        X_tr, X_te, g_tr, g_te = train_test_split(
            X, grps, test_size=PROBE_TEST_SIZE,
            stratify=grps, random_state=PROBE_RANDOM_STATE)
        clf_b_lin = LogisticRegression(
            solver='lbfgs', max_iter=500, C=0.1,
            random_state=PROBE_RANDOM_STATE)
        clf_b_lin.fit(X_tr, g_tr)
        bias_lin = accuracy_score(g_te, clf_b_lin.predict(X_te))

        # Bias probe -- MLP
        clf_b_mlp = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
            random_state=PROBE_RANDOM_STATE)
        clf_b_mlp.fit(X_tr, g_tr)
        bias_mlp = accuracy_score(g_te, clf_b_mlp.predict(X_te))

        print(f"  {name:15s} | {res_mean:8.5f} | {tnorm_mae:8.4f} | "
              f"{tnorm_in_band:8.3f} | {bias_lin:8.4f} | {bias_mlp:8.4f}")

        results.append({
            'name': name,
            'so3_res': res_mean,
            'tnorm_mae': tnorm_mae,
            'tnorm_in_band': tnorm_in_band,
            'bias_lin': bias_lin,
            'bias_mlp': bias_mlp,
            'on_manif': on_manif,
        })

    print(f"{'=' * 80}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Result table (text)
# ══════════════════════════════════════════════════════════════════════════════

def annotate_val(val, metric_name):
    """Return O/X/triangle annotation for a metric value."""
    if metric_name == 'on_manifold':
        if val < 1e-3:
            return 'O'
        elif val < 0.05:
            return 'triangle'
        else:
            return 'X'
    elif metric_name == 'tnorm_mae':
        if val < 0.01:
            return 'O'
        elif val < 0.1:
            return 'triangle'
        else:
            return 'X'
    elif metric_name == 'tnorm_in_band':
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


def save_result_table(results, alpha, path):
    """Write a plain-text result table with O/X annotations."""
    header = f"{'Method':12s} | {'On-Mfld':14s} | {'t MAE':18s} | {'t In-Band':18s} | {'Bias(Lin)':10s} | {'Bias(MLP)':10s}"
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for r in results:
        a1 = annotate_val(r['so3_res'], 'on_manifold')
        a2 = annotate_val(r['tnorm_mae'], 'tnorm_mae')
        a3 = annotate_val(r['tnorm_in_band'], 'tnorm_in_band')
        a4 = annotate_val(r['bias_lin'], 'bias_lin')
        a5 = annotate_val(r['bias_mlp'], 'bias_mlp')
        line = (f"{r['name']:12s} | {a1} ({r['so3_res']:.4f})"
                f"     | {a2} (t_MAE={r['tnorm_mae']:.4f})"
                f"   | {a3} (t_band={100*r['tnorm_in_band']:.1f}%)"
                f" | {r['bias_lin']:.3f} {a4}"
                f"   | {r['bias_mlp']:.3f} {a5}")
        lines.append(line)
    lines.append(sep)

    txt = "\n".join(lines)
    with open(path, 'w') as f:
        f.write(txt + "\n")
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

C0, C1 = GROUP_COLORS[0], GROUP_COLORS[1]
phi_orig_vals, tnorm_orig_vals, _, _ = extract_phi_tnorm(X_orig)


# ---------- Setting figure --------------------------------------------------

def plot_setting():
    """Setting figure: 3D view of SE(3) data.
    x = cos(phi), y = sin(phi) shows rotation on a circle,
    z = ||t|| shows task attribute as height.
    This creates a cylinder-like structure where groups are angularly separated.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Cylinder wireframe: circles at z=1.5, 2.0, 2.5
    circle_t = np.linspace(0, 2 * np.pi, 120)
    for z_ref in [1.5, 2.0, 2.5]:
        ax.plot(np.cos(circle_t), np.sin(circle_t), z_ref,
                'k-', linewidth=0.5, alpha=0.15)

    # Vertical lines at key angles for wireframe effect
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ax.plot([np.cos(angle)] * 2, [np.sin(angle)] * 2, [1.5, 2.5],
                'k-', linewidth=0.3, alpha=0.1)

    # Data points: (cos φ, sin φ, ||t||)
    for g, color in GROUP_COLORS.items():
        mask = groups == g
        ax.scatter(np.cos(phi_orig_vals[mask]),
                   np.sin(phi_orig_vals[mask]),
                   tnorm_orig_vals[mask],
                   c=color, s=MARKER_SIZE, alpha=MARKER_ALPHA)

    # Neutral star
    ax.scatter([np.cos(phi_neutral)], [np.sin(phi_neutral)],
               [np.linalg.norm(t_neutral)],
               c=NEUTRAL_COLOR, s=200, marker='*', edgecolors='k',
               linewidths=0.8, zorder=10)

    ax.set_xlabel(r'$\cos(\phi)$', labelpad=5)
    ax.set_ylabel(r'$\sin(\phi)$', labelpad=5)
    ax.set_zlabel(r'$\|t\|$', labelpad=5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(1.0, 3.0)
    ax.view_init(elev=20, azim=35)
    ax.tick_params(labelsize=TICK_SIZE)

    path = os.path.join(OUT_DIR, "setting.png")
    save_fig(fig, path)


# ---------- Cross-section: SO(3) residual vs phi ----------------------------

def plot_crosssection(methods_list, alpha):
    """Cross-section: 3D cylinder view (cos φ, sin φ, ‖t‖) for each method.
    SE(3) manifold = SO(3) × R³, which projects to a cylinder
    cos²φ + sin²φ = 1 in the (cos φ, sin φ, ‖t‖) space.
    Points off-manifold (broken SO(3)) will appear inside/outside the cylinder."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(30, 12))

    # Cylinder wireframe data
    circle_t = np.linspace(0, 2 * np.pi, 120)

    for idx, (name, X) in enumerate(methods_list):
        ax = fig.add_subplot(GRID_ROWS, GRID_COLS, idx + 1, projection='3d')

        # Extract phi and ||t|| from each point
        phi_m, tnorm_m, _, _ = extract_phi_tnorm(X)
        cx = np.cos(phi_m)
        cy = np.sin(phi_m)

        # But for off-manifold methods, use raw R[0,0], R[0,1] to show deviation
        # R[0,0] ≈ cos(φ), R[0,1] ≈ -sin(φ) when on-manifold
        r00 = X[:, 0]
        r01 = X[:, 1]
        # For SO(3) matrices, first row norm should be 1
        # Use raw coordinates to show off-manifold behavior
        row_norm = np.sqrt(r00**2 + r01**2 + X[:, 2]**2)
        # If off-manifold (row_norm != 1), the point is inside/outside cylinder
        # Plot using raw r00, r01 (which deviate from circle when off-manifold)
        # and actual t norms
        t_norms = np.array([np.linalg.norm(X[i, 9:12]) for i in range(len(X))])

        # Cylinder wireframe
        for z_ref in [1.5, 2.0, 2.5]:
            ax.plot(np.cos(circle_t), np.sin(circle_t), z_ref,
                    'k-', linewidth=0.5, alpha=0.15)
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            ax.plot([np.cos(angle)] * 2, [np.sin(angle)] * 2, [1.5, 2.5],
                    'k-', linewidth=0.3, alpha=0.1)

        # Data points using raw rotation matrix entries
        for g, color in GROUP_COLORS.items():
            mask = groups == g
            ax.scatter(r00[mask], r01[mask], t_norms[mask],
                       c=color, s=MARKER_SIZE, alpha=MARKER_ALPHA)

        # Neutral star
        ax.scatter([neutral_vec[0]], [neutral_vec[1]],
                   [np.linalg.norm(neutral_vec[9:12])],
                   c=NEUTRAL_COLOR, s=150, marker='*', edgecolors='k',
                   linewidths=0.6, zorder=10)

        ax.set_title(name, fontsize=PANEL_TITLE_SIZE)
        ax.set_xlabel(r'$R_{00}$', labelpad=2, fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(r'$R_{01}$', labelpad=2, fontsize=AXIS_LABEL_SIZE)
        ax.set_zlabel(r'$\|t\|$', labelpad=2, fontsize=AXIS_LABEL_SIZE)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0.5, 3.5)
        ax.view_init(elev=20, azim=35)
        ax.tick_params(labelsize=TICK_SIZE)

    # Info panel and blank panel for remaining slots
    for extra_idx in range(len(methods_list) + 1, GRID_ROWS * GRID_COLS + 1):
        ax_extra = fig.add_subplot(GRID_ROWS, GRID_COLS, extra_idx)
        ax_extra.axis('off')
    # Info in first extra panel
    ax_info = fig.add_subplot(GRID_ROWS, GRID_COLS, len(methods_list) + 1)
    ax_info.axis('off')
    ax_info.text(0.5, 0.95, "Cross-section", transform=ax_info.transAxes,
                 fontsize=PANEL_TITLE_SIZE, fontweight='bold',
                 ha='center', va='top')
    ax_info.text(0.5, 0.5, "R[0,0] vs R[0,1] vs ||t|| (3D).\n"
                 "Cylinder = SO(3) constraint.\n"
                 "Off-cylinder = broken SO(3).",
                 transform=ax_info.transAxes, fontsize=AXIS_LABEL_SIZE,
                 ha='center', va='center', wrap=True, linespacing=1.5,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                           edgecolor='#BDBDBD', alpha=0.8))

    plt.tight_layout()
    save_fig(fig, os.path.join(OUT_DIR, "crosssection.png"))


# ---------- Top-down translation (t_x vs t_y) --------------------------------

def plot_topdown(methods_list, alpha, suffix=""):
    """Top-down translation space: t_x vs t_y. 2x5 grid."""
    fig, axes = make_grid(figsize=(25, 10))
    th_ring = np.linspace(0, 2 * np.pi, 200)

    for idx, (name, X) in enumerate(methods_list):
        ax = axes[idx]
        t_x = X[:, 9]
        t_y = X[:, 10]

        # Debiased
        ax.scatter(t_x[groups == 0], t_y[groups == 0],
                   c=C0, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.scatter(t_x[groups == 1], t_y[groups == 1],
                   c=C1, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)

        # Reference circles
        ax.plot(1.5 * np.cos(th_ring), 1.5 * np.sin(th_ring),
                'k--', lw=0.8, alpha=0.3)
        ax.plot(2.5 * np.cos(th_ring), 2.5 * np.sin(th_ring),
                'k--', lw=0.8, alpha=0.3)

        ax.set_xlabel('t_x')
        ax.set_ylabel('t_y')
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.15)
        ax.axhline(0, color='k', lw=0.3, alpha=0.3)
        ax.axvline(0, color='k', lw=0.3, alpha=0.3)

    fill_info_panel(axes[len(methods_list)], "Top-down",
                    f"t_x vs t_y (alpha={alpha}).\n"
                    "Dashed circles = ||t|| in [1.5, 2.5].")
    for extra_idx in range(len(methods_list) + 1, len(axes)):
        axes[extra_idx].axis('off')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"topdown{suffix}.png")
    save_fig(fig, path)


# ---------- Scatter: phi vs ||t|| -------------------------------------------

def plot_scatter(methods_list, alpha, suffix=""):
    """Scatter: phi vs ||t||. 2x5 grid."""
    fig, axes = make_grid(figsize=(25, 10))

    for idx, (name, X) in enumerate(methods_list):
        ax = axes[idx]
        phi, tnorm, _, _ = extract_phi_tnorm(X)

        # Debiased
        ax.scatter(np.degrees(phi[groups == 0]), tnorm[groups == 0],
                   c=C0, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)
        ax.scatter(np.degrees(phi[groups == 1]), tnorm[groups == 1],
                   c=C1, s=MARKER_SIZE, alpha=MARKER_ALPHA, zorder=3)

        # Neutral phi line
        ax.axvline(np.degrees(phi_neutral), color=NEUTRAL_COLOR,
                   lw=2, ls='--', alpha=0.7)
        # Green band for ||t|| in [1.5, 2.5]
        ax.axhspan(1.5, 2.5, color='green', alpha=0.06, zorder=0)
        ax.axhline(1.5, color='green', lw=0.8, ls=':', alpha=0.4)
        ax.axhline(2.5, color='green', lw=0.8, ls=':', alpha=0.4)

        ax.set_xlabel('phi (degrees)')
        ax.set_ylabel('||t||')
        ax.set_title(name)
        ax.set_xlim(-200, 200)
        ax.set_ylim(0, 4.5)
        ax.grid(True, alpha=0.15)

    fill_info_panel(axes[len(methods_list)], "Scatter",
                    f"phi vs ||t|| (alpha={alpha}).\n"
                    "Green band = task-attribute range.\n"
                    "Gold line = neutral angle.")
    for extra_idx in range(len(methods_list) + 1, len(axes)):
        axes[extra_idx].axis('off')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"scatter{suffix}.png")
    save_fig(fig, path)


# ---------- Bar charts -------------------------------------------------------

def plot_bar(results, metric_key, ylabel, title, lower_better, path,
             chance_line=None, orig_val=None, ylim=None):
    """Single bar chart for one metric, methods excluding Original."""
    res_filt = [r for r in results if r['name'] != 'Original']
    names = [r['name'] for r in res_filt]
    vals = [r[metric_key] for r in res_filt]
    colors = [METHOD_COLORS.get(n, '#9E9E9E') for n in names]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=TICK_SIZE)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.2)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        yoff = max(max(vals) * 0.01, 0.003) if max(vals) > 0 else 0.003
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yoff,
                f'{val:.3f}', ha='center', va='bottom', fontsize=TICK_SIZE)

    if chance_line is not None:
        ax.axhline(chance_line, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    if orig_val is not None:
        ax.axhline(orig_val, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    save_fig(fig, path)


def plot_bars_linear(results, alpha):
    """Bar chart for Bias Linear probe accuracy."""
    plot_bar(results, 'bias_lin',
             ylabel='Bias Probe Accuracy (Linear)',
             title='',
             lower_better=True,
             path=os.path.join(OUT_DIR, "bar_linear.png"),
             chance_line=0.5,
             ylim=(0.0, 1.05))


def plot_bars_mlp(results, alpha):
    """Bar chart for Bias MLP probe accuracy."""
    plot_bar(results, 'bias_mlp',
             ylabel='Bias Probe Accuracy (MLP)',
             title='',
             lower_better=True,
             path=os.path.join(OUT_DIR, "bar_mlp.png"),
             chance_line=0.5,
             ylim=(0.0, 1.05))


# ══════════════════════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SE(3) Toy Experiment — Finalized")
    print("=" * 60)

    # Setting figure (always)
    plot_setting()

    for alpha in ALPHAS:
        suffix = f"_a{int(alpha * 10):02d}"
        print(f"\n{'#' * 60}")
        print(f"# alpha = {alpha}")
        print(f"{'#' * 60}")

        methods_list = run_all_methods(X_orig, alpha)
        results = compute_metrics(methods_list, X_orig, groups, task_labels)

        # Scatter
        plot_scatter(methods_list, alpha, suffix)

        # Top-down translation
        plot_topdown(methods_list, alpha, suffix)

        # Cross-section and result table at primary alpha only
        if alpha == PRIMARY_ALPHA:
            plot_crosssection(methods_list, alpha)
            save_result_table(results, alpha,
                              os.path.join(OUT_DIR, "result_table.txt"))

        # Bar charts at primary alpha
        if alpha == PRIMARY_ALPHA:
            plot_bars_linear(results, alpha)
            plot_bars_mlp(results, alpha)

    print("\nDone! All SE(3) figures saved to:", OUT_DIR)
