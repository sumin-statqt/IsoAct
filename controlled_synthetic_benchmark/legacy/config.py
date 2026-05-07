"""
Unified configuration for finalized toy experiments.
Shared styles, colors, method order, and hyperparameters.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Method order (uniform across all settings) ─────────────────────────────
# Full order (9 methods, for CMF comparison scripts)
METHOD_ORDER = ['Original', 'Oracle', 'CMF',
                'SFID', 'SFID+Proj', 'SPD', 'SPD+Proj',
                'Geodesic', 'IsoRot']

# Base order (8 methods, for scripts without CMF)
METHOD_ORDER_BASE = ['Original', 'Oracle',
                     'SFID', 'SFID+Proj', 'SPD', 'SPD+Proj',
                     'Geodesic', 'IsoRot']

# Methods excluding Original (for bar charts, result tables)
METHOD_ORDER_NO_ORIG = METHOD_ORDER[1:]
METHOD_ORDER_BASE_NO_ORIG = METHOD_ORDER_BASE[1:]

# ── Colors ──────────────────────────────────────────────────────────────────
GROUP_COLORS = {0: '#2196F3', 1: '#F44336'}  # blue, red
FAINT_ALPHA = 0.15  # for original-position ghost points
NEUTRAL_COLOR = '#FFD600'  # gold

# Method colors for bar charts
METHOD_COLORS = {
    'Original':  '#9E9E9E',   # gray
    'Oracle':    '#FFA726',   # orange
    'CMF':       '#FF7043',   # deep orange (training-based)
    'Baseline_AE': '#FF7043', # deep orange (training-based)
    'SFID':      '#ef5350',   # red (extrinsic)
    'SFID+Proj': '#ef5350',
    'SPD':       '#ef5350',
    'SPD+Proj':  '#ef5350',
    'Geodesic':  '#7E57C2',   # purple
    'IsoRot':    '#4CAF50',   # green (ours)
}

# ── Figure style ────────────────────────────────────────────────────────────
PANEL_TITLE_SIZE = 16
AXIS_LABEL_SIZE = 13
TICK_SIZE = 10
MARKER_SIZE = 12
MARKER_ALPHA = 0.6
FIG_DPI = 150

def setup_style():
    """Apply unified matplotlib style."""
    plt.rcParams.update({
        'font.size': TICK_SIZE,
        'axes.titlesize': PANEL_TITLE_SIZE,
        'axes.titleweight': 'bold',
        'axes.labelsize': AXIS_LABEL_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'figure.dpi': FIG_DPI,
        'savefig.dpi': FIG_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

# ── Data generation ─────────────────────────────────────────────────────────
N_PER_GROUP = 200
SEED = 42
LOWCONF_THR = 0.7

# ── Probe hyperparameters ───────────────────────────────────────────────────
PROBE_TEST_SIZE = 0.3
PROBE_RANDOM_STATE = 42
MLP_HIDDEN = (32, 16)
MLP_MAX_ITER = 500

# ── Alpha values ────────────────────────────────────────────────────────────
ALPHAS = [0.9, 1.0]
PRIMARY_ALPHA = 1.0  # for result table, cross-section, bar charts

# ── Grid layout for 9-method + 1-info panel figures ────────────────────────
GRID_ROWS, GRID_COLS = 2, 5

def make_grid(figsize=(25, 10)):
    """Create a 2x5 grid figure. 9 method panels + 1 info panel."""
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=figsize)
    return fig, axes.flatten()

def fill_info_panel(ax, title, description):
    """Fill the last (10th) panel with figure description instead of data."""
    ax.axis('off')
    ax.text(0.5, 0.95, title, transform=ax.transAxes,
            fontsize=PANEL_TITLE_SIZE, fontweight='bold',
            ha='center', va='top')
    ax.text(0.5, 0.5, description, transform=ax.transAxes,
            fontsize=AXIS_LABEL_SIZE, ha='center', va='center',
            wrap=True, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                      edgecolor='#BDBDBD', alpha=0.8))

def save_fig(fig, path):
    """Save figure and close."""
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {path}")
