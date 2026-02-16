from __future__ import annotations

import os

BASE_M = 250
GAMMAS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
SIGMA = 1.0
SEED = 0

ZERO_T_TRIALS = 50

BASE_M_VER3 = 400
ZERO_T_TRIALS_VER3 = 10

GAMMAS_VER4 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
BASE_M_VER4 = 400
ZERO_T_TRIALS_VER4 = 10

BETA_MAX = 1.0
K_LIST = [-0.5, -1.0, -1.5]

X_SAMPLES_FINITE_T = 8000
FINITE_T_TRIALS = 10

X_SAMPLES_FINITE_T_RIFIED = 12000
FINITE_T_TRIALS_RIFIED = 10

RS_GH_N = 64
RS_TOL = 1e-10
RS_MAX_ITER = 500

_VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_VIS_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIG_DIR = os.path.join(REPO_ROOT, "fig")

PLOT_STYLE = {
    "font.family": "sans-serif",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "axes.linewidth": 1.0,
    "axes.xmargin": 0.01,
    "axes.ymargin": 0.01,
    "legend.fancybox": False,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
}
GRID_LINESTYLE = "--"
SHOW_TITLE = False

FONT_SIZE_LABEL = 38
FONT_SIZE_TICK = 22
FONT_SIZE_LEGEND = 22

FIG_W = 6.4
FIG_H = 4.8

MARKER_SIZE = 12
EB_ALPHA = 0.6
FILL_STYLE = "none"
LINE_WIDTH = 2

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINE_STYLES = ["-", "--", "-.", ":"]


def apply_plot_style():
    import matplotlib.pyplot as plt
    for k, v in PLOT_STYLE.items():
        plt.rcParams[k] = v


def err_kw(color):
    return {
        "capsize": 3,
        "ecolor": color,
        "elinewidth": 1.5,
        "alpha": EB_ALPHA,
        "capthick": 1.5,
    }


def apply_axis_style(ax, font_size_label=None, font_size_tick=None):
    fs_label = font_size_label or FONT_SIZE_LABEL
    fs_tick = font_size_tick or FONT_SIZE_TICK
    ax.grid(ls=GRID_LINESTYLE)
    if not SHOW_TITLE:
        ax.set_title("")
    ax.set_xlabel(ax.get_xlabel(), fontsize=fs_label)
    ax.set_ylabel(ax.get_ylabel(), fontsize=fs_label)
    ax.tick_params(labelsize=fs_tick)
