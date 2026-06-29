"""
UBDS visualization helpers
==========================

Centralised plotting style for the solver.  Hard constraint for this project:
**no pure black is ever used** - all axes, text, ticks, spines, grid and line
artwork use a dark navy ('#1b2a4a') instead of black, and all colour maps are
chosen so that they contain no black band.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# ---------------------------------------------------------------------------
# Global "no-black" style
# ---------------------------------------------------------------------------
INK = "#1b2a4a"          # dark navy used everywhere instead of black
GRIDC = "#9fb3c8"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRIDC,
    "grid.alpha": 0.5,
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.prop_cycle": plt.cycler(color=[
        "#d1495b", "#2a6f97", "#3a9278", "#e09f3e", "#7b5ea7",
        "#118ab2", "#ef8354", "#06a77d"]),
})

# Salinity colour map: light cream -> blue -> green -> amber -> magenta.
# Deliberately contains NO black band; the blue->green transition is aligned
# with the 36 psu threshold of regulatory interest.
_SAL_COLORS = ["#eaf4fb", "#bde0ef", "#7ec8e3", "#3a9bd5", "#2a6f97",
               "#3a9278", "#7cbf6a", "#e6c84f", "#e08a3c", "#d1495b", "#9b2226"]
SAL_CMAP = LinearSegmentedColormap.from_list("ubds_sal", _SAL_COLORS)

# Velocity / bathymetry maps (no black)
VEL_CMAP = LinearSegmentedColormap.from_list(
    "ubds_vel", ["#f7fbff", "#c6dbef", "#6baed6", "#3182bd", "#08519c",
                 "#54278f", "#9b2226"])
BATHY_CMAP = LinearSegmentedColormap.from_list(
    "ubds_bathy", ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f2efe9"])

# Salinity-RISE (increment above ambient) map - deliberately different palette
# (teal -> violet -> orange) so rise-contour figures are visually distinct from
# the absolute-salinity envelope figures.
RISE_CMAP = LinearSegmentedColormap.from_list(
    "ubds_rise", ["#e6fff7", "#7fd4c1", "#3a9eb0", "#5566b0", "#7b5ea7",
                  "#b5567f", "#e08a3c", "#9b2226"])

TIDAL_VEL_COLORS = {"slack": "#d1495b", "neap": "#2a6f97", "spring": "#3a9278"}


def new_ax(figsize=(7, 5.5), title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return fig, ax


def save(fig, path):
    # respect figures that already manage their own layout (constrained_layout)
    try:
        engaged = fig.get_constrained_layout()
    except Exception:
        engaged = False
    if not engaged:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def nonlinear_salinity_levels(s_bg=34.2):
    """
    Non-linear contour levels (denser near background): fine 0.25-0.5 psu bands
    near background, coarser higher up.  The 36 psu threshold of regulatory
    interest sits exactly on a band edge.
    """
    fine = np.arange(s_bg, 36.01, 0.25)
    mid = np.arange(36.5, 42.01, 0.5)
    coarse = np.array([43, 45, 48, 52, 56, 60])
    return np.unique(np.concatenate([fine, mid, coarse]))
