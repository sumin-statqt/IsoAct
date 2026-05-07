"""
CMF (Causal Manifold Fairness) module for toy experiments.

Faithful reimplementation of https://github.com/vidhirathore/cmf
adapted for our manifold toy experiments (Arc, Sphere, Hyperbolic, SE(3)).

Key design: CMF is a TRAINING-TIME method. It trains an autoencoder
with geometric constraints (metric tensor + Hessian invariance).
This is NOT a post-hoc version of CMF.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.functional as F_auto
import numpy as np
import os
import json
from torch.utils.data import DataLoader, TensorDataset


# ======================================================================
# 1. Autoencoder Architecture (faithful to original CMF)
# ======================================================================

class CMFAutoencoder(nn.Module):
    """
    ELU-based autoencoder as in original CMF.
    ELU is chosen over ReLU because it is C2-continuous,
    required for stable Hessian computation.
    """
    def __init__(self, input_dim, latent_dim=2, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ELU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        y_pred = self.classifier(z)
        return x_rec, y_pred, z

    def decode(self, z):
        return self.decoder(z)


# ======================================================================
# 2. Geometric Loss Computation (faithful to original CMF savfig.py)
# ======================================================================

def compute_geometric_loss(model, z, z_cf, calc_hessian=True, subset_size=8):
    """
    Compute Metric Tensor (1st order) and Hessian (2nd order) differences.
    Faithful to original CMF: savfig.py compute_geometric_loss().

    Returns (metric_loss, curvature_loss) as torch tensors.
    """
    batch_size = z.shape[0]
    actual_subset = min(subset_size, batch_size)
    subset_idx = torch.randperm(batch_size)[:actual_subset]
    z_sub = z[subset_idx]
    z_cf_sub = z_cf[subset_idx]

    metric_loss = 0.0
    curvature_loss = 0.0

    dummy_out = model.decode(z_sub[0:1])
    output_dim = dummy_out.shape[1]

    for i in range(len(z_sub)):
        z_i = z_sub[i].unsqueeze(0)
        z_cf_i = z_cf_sub[i].unsqueeze(0)

        # --- 1. Jacobian & Metric Tensor ---
        J = F_auto.jacobian(model.decode, z_i, create_graph=True).squeeze()
        J_cf = F_auto.jacobian(model.decode, z_cf_i, create_graph=True).squeeze()

        G = torch.matmul(J.T, J)
        G_cf = torch.matmul(J_cf.T, J_cf)
        metric_loss += torch.norm(G - G_cf, p='fro')

        # --- 2. Hessian (Curvature) ---
        if calc_hessian:
            H_list, H_cf_list = [], []
            for k in range(output_dim):
                def scalar_decode_k(z_in, _k=k):
                    return model.decode(z_in).squeeze()[_k]

                h_k = F_auto.hessian(scalar_decode_k, z_i, create_graph=True).squeeze()
                h_cf_k = F_auto.hessian(scalar_decode_k, z_cf_i, create_graph=True).squeeze()
                H_list.append(h_k)
                H_cf_list.append(h_cf_k)

            H = torch.stack(H_list)
            H_cf = torch.stack(H_cf_list)
            curvature_loss += torch.norm(H - H_cf, p='fro')

    return metric_loss / len(z_sub), curvature_loss / len(z_sub)


# ======================================================================
# 3. Training Loop (faithful to original CMF savfig.py)
# ======================================================================

def train_cmf_model(X_np, groups_np, task_labels_np,
                    counterfactual_fn,
                    model_type="CMF",
                    input_dim=3, latent_dim=2, hidden_dim=64,
                    epochs=300, lr=1e-3, batch_size=64,
                    lambda_metric=10.0, lambda_curv=5.0,
                    calc_hessian=True,
                    seed=42, verbose=True):
    """
    Train a CMF or Baseline autoencoder.

    Args:
        X_np: (N, D) numpy array of data points
        groups_np: (N,) numpy array of group labels (0/1)
        task_labels_np: (N,) numpy array of binary task labels
        counterfactual_fn: function(X_np, groups_np) -> X_cf_np
            Returns counterfactual data (same task var, flipped group)
        model_type: "CMF" or "Baseline"
        input_dim: ambient dimension
        latent_dim: latent space dimension
        hidden_dim: hidden layer width
        epochs, lr, batch_size: training hyperparameters
        lambda_metric, lambda_curv: geometric loss weights
        calc_hessian: whether to compute Hessian loss (expensive)
        seed: random seed
        verbose: print progress
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CMFAutoencoder(input_dim, latent_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(task_labels_np, dtype=torch.float32).unsqueeze(1)

    # Generate counterfactual data
    X_cf_np = counterfactual_fn(X_np, groups_np)
    X_cf = torch.tensor(X_cf_np, dtype=torch.float32)

    dataset = TensorDataset(X, Y, X_cf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if verbose:
        print(f"  Training {model_type} (epochs={epochs}, lr={lr}, "
              f"lambda_m={lambda_metric}, lambda_c={lambda_curv})...")

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_Y, batch_X_cf in loader:
            optimizer.zero_grad()

            rec_X, pred_Y, z = model(batch_X)
            _, _, z_cf = model(batch_X_cf)

            # Standard losses (same for Baseline and CMF)
            loss_rec = nn.MSELoss()(rec_X, batch_X)
            loss_clf = nn.BCELoss()(pred_Y, batch_Y)
            loss_align = nn.MSELoss()(z, z_cf)
            loss = loss_rec + loss_clf + loss_align

            # CMF geometric losses
            if model_type == "CMF":
                l_metric, l_curv = compute_geometric_loss(
                    model, z, z_cf, calc_hessian=calc_hessian, subset_size=8)
                loss += lambda_metric * l_metric + lambda_curv * l_curv

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose and epoch % 50 == 0:
            print(f"    Epoch {epoch}: loss={total_loss:.4f}")

    return model


# ======================================================================
# 4. Evaluation: extract debiased output from trained model
# ======================================================================

def get_reconstructed(model, X_np):
    """Get reconstructed X from trained AE. Returns numpy array."""
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        x_rec, _, z = model(X_t)
    return x_rec.numpy(), z.numpy()


# ======================================================================
# 5. CMF-native metrics (Tier 2)
# ======================================================================

def compute_cmf_metrics(model, X_np, X_cf_np, subset_size=16):
    """
    Compute CMF-native metrics: Metric Error, Curvature Error, Reconstruction MSE.
    These require a trained autoencoder with a decoder.

    Returns dict with metric_error, curvature_error, reconstruction_mse.
    """
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32)
    X_cf_t = torch.tensor(X_cf_np, dtype=torch.float32)

    with torch.no_grad():
        x_rec, _, z = model(X_t)
        rec_mse = float(nn.MSELoss()(x_rec, X_t).item())

    # Geometric metrics need gradients
    z = model.encoder(X_t)
    z_cf = model.encoder(X_cf_t)

    # Use larger subset for eval
    actual_subset = min(subset_size, len(X_np))
    metric_err, curv_err = compute_geometric_loss(
        model, z[:actual_subset], z_cf[:actual_subset],
        calc_hessian=True, subset_size=actual_subset)

    return {
        'metric_error': float(metric_err.item()),
        'curvature_error': float(curv_err.item()),
        'reconstruction_mse': rec_mse,
    }


# ======================================================================
# 6. Caching utilities
# ======================================================================

def save_cmf_results(model, metrics, X_rec, z, path):
    """Save model weights, metrics, and outputs to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'X_rec': X_rec,
        'z': z,
    }, path)
    print(f"  Saved CMF results to {path}")


def load_cmf_results(path, input_dim, latent_dim=2, hidden_dim=64):
    """Load cached CMF results. Returns (model, metrics, X_rec, z) or None."""
    if not os.path.exists(path):
        return None
    data = torch.load(path, weights_only=False)
    model = CMFAutoencoder(input_dim, latent_dim, hidden_dim)
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    return model, data['metrics'], data['X_rec'], data['z']


# ======================================================================
# 7. Visualization: CMF-style latent space plots
# ======================================================================

def plot_cmf_latent(z, groups, task_var, task_name, title, path,
                    group_colors=None):
    """
    Generate CMF-author-style latent space visualization:
    Left: z colored by sensitive attribute (A)
    Right: z colored by task variable
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if group_colors is None:
        group_colors = {0: '#2196F3', 1: '#F44336'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by group
    for g in [0, 1]:
        mask = groups == g
        axes[0].scatter(z[mask, 0], z[mask, 1],
                        c=group_colors[g], s=12, alpha=0.7,
                        label=f'Group {g}')
    axes[0].set_title(f"{title}\nColored by Sensitive Attr (A)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: colored by task variable
    sc = axes[1].scatter(z[:, 0], z[:, 1], c=task_var, cmap='viridis',
                         s=12, alpha=0.7)
    axes[1].set_title(f"{title}\nColored by {task_name}")
    plt.colorbar(sc, ax=axes[1], label=task_name)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
