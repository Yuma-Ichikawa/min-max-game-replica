#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vis.config import (
    DATA_DIR, FIG_DIR,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    MARKER_SIZE, EB_ALPHA, FILL_STYLE, LINE_WIDTH, GRID_LINESTYLE,
    COLORS, MARKERS, apply_plot_style, err_kw as _cfg_err_kw,
)

apply_plot_style()

FONT_LABEL = FONT_SIZE_LABEL
FONT_TICK = FONT_SIZE_TICK
FONT_LEGEND = FONT_SIZE_LEGEND
MS = MARKER_SIZE
LW_THEORY = LINE_WIDTH
LW_ERR = 1.5
FILL = FILL_STYLE

colors = list(COLORS)
markers = list(MARKERS)


def _apply_style(ax, xlabel, ylabel):
    ax.grid(ls=GRID_LINESTYLE)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)


def bmax_to_str(bmax):
    return f"{bmax}".replace(".", "p")


def load_k_dep_data(M, bmax):
    bmax_str = bmax_to_str(bmax)
    fname = f"finiteT_data_k_dep_M{M}_bmax{bmax_str}.npz"
    path = os.path.join(DATA_DIR, fname)
    if not os.path.isfile(path):
        print(f"  [SKIP] {fname} not found", flush=True)
        return None

    data = np.load(path)
    k_list = np.atleast_1d(data["k_list"])
    abs_k = np.abs(k_list)

    v_theory = []
    v_mc_mean = []
    v_mc_std = []
    energy_theory = []
    energy_mc_mean = []
    energy_mc_std = []

    gammas_theory = data["gammas_theory"] if "gammas_theory" in data else data["gammas"]

    for k in k_list:
        key = f"k_{k:.2f}".replace(".", "p").replace("-", "m")

        v_th_arr = data.get(f"{key}_v_theory", None)
        e_th_arr = data.get(f"{key}_energy_theory", None)

        if v_th_arr is not None and len(v_th_arr) > 0:
            idx_g1 = np.argmin(np.abs(gammas_theory - 1.0))
            v_theory.append(float(v_th_arr[idx_g1]))
        else:
            v_theory.append(np.nan)

        if e_th_arr is not None and len(e_th_arr) > 0:
            idx_g1 = np.argmin(np.abs(gammas_theory - 1.0))
            energy_theory.append(float(e_th_arr[idx_g1]))
        else:
            energy_theory.append(np.nan)

        v_mc = data.get(f"{key}_v_mc_mean", None)
        v_sd = data.get(f"{key}_v_mc_std", None)
        e_mc = data.get(f"{key}_energy_mc_mean", None)
        e_sd = data.get(f"{key}_energy_mc_std", None)

        v_mc_mean.append(float(v_mc[0]) if v_mc is not None else np.nan)
        v_mc_std.append(float(v_sd[0]) if v_sd is not None else np.nan)
        energy_mc_mean.append(float(e_mc[0]) if e_mc is not None else np.nan)
        energy_mc_std.append(float(e_sd[0]) if e_sd is not None else np.nan)

    result = {
        "abs_k": abs_k,
        "v_theory": np.array(v_theory),
        "v_mc_mean": np.array(v_mc_mean),
        "v_mc_std": np.array(v_mc_std),
        "energy_theory": np.array(energy_theory),
        "energy_mc_mean": np.array(energy_mc_mean),
        "energy_mc_std": np.array(energy_mc_std),
        "M": M,
        "beta_max": bmax,
    }
    print(f"  Loaded {fname}: {len(abs_k)} |k| points", flush=True)
    return result


def plot_per_bmax(M_list, bmax, all_data, se_factor=1.0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, M in enumerate(M_list):
        d = all_data.get((M, bmax))
        if d is None:
            continue
        c = colors[i % len(colors)]
        mk = markers[i % len(markers)]
        abs_k = d["abs_k"]

        order = np.argsort(abs_k)
        abs_k = abs_k[order]

        v_th = d["v_theory"][order]
        v_mc = d["v_mc_mean"][order]
        v_sd = d["v_mc_std"][order]
        valid = ~np.isnan(v_th)
        ax1.plot(abs_k[valid], v_th[valid], "-", color=c, lw=LW_THEORY,
                 label=rf"RS $M={M}$", zorder=3)
        valid_mc = ~np.isnan(v_mc)
        if valid_mc.any():
            ax1.errorbar(abs_k[valid_mc], v_mc[valid_mc],
                         yerr=v_sd[valid_mc] * se_factor,
                         fmt=mk, color=c, ms=MS, fillstyle=FILL,
                         markeredgewidth=1.5, zorder=4,
                         label=rf"AIS $M={M}$",
                         **_cfg_err_kw(c))

        e_th = d["energy_theory"][order]
        e_mc = d["energy_mc_mean"][order]
        e_sd = d["energy_mc_std"][order]
        valid = ~np.isnan(e_th)
        ax2.plot(abs_k[valid], e_th[valid], "-", color=c, lw=LW_THEORY,
                 label=rf"RS $M={M}$", zorder=3)
        valid_mc = ~np.isnan(e_mc)
        if valid_mc.any():
            ax2.errorbar(abs_k[valid_mc], e_mc[valid_mc],
                         yerr=e_sd[valid_mc] * se_factor,
                         fmt=mk, color=c, ms=MS, fillstyle=FILL,
                         markeredgewidth=1.5, zorder=4,
                         label=rf"AIS $M={M}$",
                         **_cfg_err_kw(c))

    bmax_str = bmax_to_str(bmax)
    for ax, ylabel, title_letter in [
        (ax1, r"$v = -g / \beta_{\min}$", "(a)"),
        (ax2, r"$\mathbb{E}[V]/L$", "(b)"),
    ]:
        _apply_style(ax, r"$|k| = \beta_{\min}/\beta_{\max}$", ylabel)
        ax.legend(fontsize=FONT_LEGEND, loc="best", ncol=2)
        ax.set_title(title_letter, fontsize=FONT_LABEL, loc="left", fontweight="bold")

    fig.suptitle(rf"$\gamma=1.0$,  $\beta_{{\max}}={bmax}$", fontsize=16, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"k_dep_v_energy_bmax{bmax_str}.pdf")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def plot_per_M(M, bmax_list, all_data, se_factor=1.0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, bmax in enumerate(bmax_list):
        d = all_data.get((M, bmax))
        if d is None:
            continue
        c = colors[i % len(colors)]
        mk = markers[i % len(markers)]
        abs_k = d["abs_k"]

        order = np.argsort(abs_k)
        abs_k = abs_k[order]

        v_th = d["v_theory"][order]
        v_mc = d["v_mc_mean"][order]
        v_sd = d["v_mc_std"][order]
        valid = ~np.isnan(v_th)
        ax1.plot(abs_k[valid], v_th[valid], "-", color=c, lw=LW_THEORY,
                 label=rf"RS $\beta_{{\max}}={bmax}$", zorder=3)
        valid_mc = ~np.isnan(v_mc)
        if valid_mc.any():
            ax1.errorbar(abs_k[valid_mc], v_mc[valid_mc],
                         yerr=v_sd[valid_mc] * se_factor,
                         fmt=mk, color=c, ms=MS, fillstyle=FILL,
                         markeredgewidth=1.5, zorder=4,
                         label=rf"AIS $\beta_{{\max}}={bmax}$",
                         **_cfg_err_kw(c))

        e_th = d["energy_theory"][order]
        e_mc = d["energy_mc_mean"][order]
        e_sd = d["energy_mc_std"][order]
        valid = ~np.isnan(e_th)
        ax2.plot(abs_k[valid], e_th[valid], "-", color=c, lw=LW_THEORY,
                 label=rf"RS $\beta_{{\max}}={bmax}$", zorder=3)
        valid_mc = ~np.isnan(e_mc)
        if valid_mc.any():
            ax2.errorbar(abs_k[valid_mc], e_mc[valid_mc],
                         yerr=e_sd[valid_mc] * se_factor,
                         fmt=mk, color=c, ms=MS, fillstyle=FILL,
                         markeredgewidth=1.5, zorder=4,
                         label=rf"AIS $\beta_{{\max}}={bmax}$",
                         **_cfg_err_kw(c))

    for ax, ylabel, title_letter in [
        (ax1, r"$v = -g / \beta_{\min}$", "(a)"),
        (ax2, r"$\mathbb{E}[V]/L$", "(b)"),
    ]:
        _apply_style(ax, r"$|k| = \beta_{\min}/\beta_{\max}$", ylabel)
        ax.legend(fontsize=FONT_LEGEND, loc="best", ncol=2)
        ax.set_title(title_letter, fontsize=FONT_LABEL, loc="left", fontweight="bold")

    fig.suptitle(rf"$\gamma=1.0$,  $M={M}$", fontsize=16, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"k_dep_v_energy_M{M}.pdf")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def plot_summary_grid(M_list, bmax_list, all_data, se_factor=1.0):
    n_bmax = len(bmax_list)
    n_M = len(M_list)

    fig, axes = plt.subplots(n_bmax, n_M, figsize=(5 * n_M, 4 * n_bmax),
                             squeeze=False)

    for row, bmax in enumerate(bmax_list):
        for col, M in enumerate(M_list):
            ax = axes[row, col]
            d = all_data.get((M, bmax))

            if d is not None:
                abs_k = d["abs_k"]
                order = np.argsort(abs_k)
                abs_k = abs_k[order]

                v_th = d["v_theory"][order]
                valid = ~np.isnan(v_th)
                ax.plot(abs_k[valid], v_th[valid], "-", color=colors[0],
                        lw=2.0, label="RS", zorder=3)

                v_mc = d["v_mc_mean"][order]
                v_sd = d["v_mc_std"][order]
                valid_mc = ~np.isnan(v_mc)
                if valid_mc.any():
                    ax.errorbar(abs_k[valid_mc], v_mc[valid_mc],
                                yerr=v_sd[valid_mc] * se_factor,
                                fmt="o", color=colors[1], ms=6, fillstyle=FILL,
                                markeredgewidth=1.2, zorder=4, label="AIS",
                                capsize=2, ecolor=colors[1], elinewidth=0.8,
                                capthick=0.8)

                ax.legend(fontsize=9, loc="best")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")

            ax.set_title(rf"$M={M}$, $\beta_{{\max}}={bmax}$", fontsize=FONT_TICK)
            ax.grid(ls=GRID_LINESTYLE)
            ax.tick_params(labelsize=FONT_TICK)

            if row == n_bmax - 1:
                ax.set_xlabel(r"$|k|$", fontsize=FONT_LABEL)
            if col == 0:
                ax.set_ylabel(r"$v$", fontsize=14)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "k_dep_v_summary_grid.pdf")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Plot v and E[V]/L vs |k| at gamma=1.0")
    ap.add_argument("--M_list", nargs="+", type=int, default=[40, 80, 120],
                     help="M values to plot (default: 40 80 120)")
    ap.add_argument("--bmax_list", nargs="+", type=float, default=[0.1, 0.5, 1.0],
                     help="beta_max values to plot (default: 0.1 0.5 1.0)")
    ap.add_argument("--n_trials", type=int, default=10,
                     help="Number of AIS trials (for SE scaling)")
    args = ap.parse_args()

    M_list = args.M_list
    bmax_list = args.bmax_list
    se_factor = 1.0 / math.sqrt(args.n_trials)

    print("=" * 60, flush=True)
    print(f"Loading |k|-dependence data: M={M_list}, bmax={bmax_list}", flush=True)
    print("=" * 60, flush=True)

    all_data = {}
    for M in M_list:
        for bmax in bmax_list:
            result = load_k_dep_data(M, bmax)
            if result is not None:
                all_data[(M, bmax)] = result

    if not all_data:
        print("\nNo data files found! Run sbatch_k_dep_*.sh first.", flush=True)
        sys.exit(1)

    n_loaded = len(all_data)
    n_total = len(M_list) * len(bmax_list)
    print(f"\nLoaded {n_loaded}/{n_total} data files", flush=True)

    print("\n--- Per-bmax figures ---", flush=True)
    for bmax in bmax_list:
        if any((M, bmax) in all_data for M in M_list):
            plot_per_bmax(M_list, bmax, all_data, se_factor)

    print("\n--- Per-M figures ---", flush=True)
    for M in M_list:
        if any((M, bmax) in all_data for bmax in bmax_list):
            plot_per_M(M, bmax_list, all_data, se_factor)

    print("\n--- Summary grid ---", flush=True)
    plot_summary_grid(M_list, bmax_list, all_data, se_factor)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
