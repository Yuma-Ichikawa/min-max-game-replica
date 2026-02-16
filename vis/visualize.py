from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from vis.config import (
    DATA_DIR,
    FIG_DIR,
    PLOT_STYLE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_LEGEND,
    FIG_W,
    FIG_H,
    MARKER_SIZE,
    EB_ALPHA,
    FILL_STYLE,
    LINE_WIDTH,
    COLORS,
    MARKERS,
    LINE_STYLES,
    apply_plot_style,
    apply_axis_style,
    err_kw,
)

line_styles = LINE_STYLES
markers = MARKERS
colors = COLORS
COLOR_RHO_X = colors[0]
COLOR_RHO_Y = colors[1]

MS = MARKER_SIZE
FILL = FILL_STYLE


def apply_style(ax, font_size_label=None, font_size_tick=None):
    apply_axis_style(ax, font_size_label=font_size_label, font_size_tick=font_size_tick)


def plot_zeroT_three_panels(out_path=None, ver2=False, ver3=False, ver4=False):
    if ver4:
        npz_basename = "zeroT_data_ver4"
    elif ver3:
        npz_basename = "zeroT_data_ver3"
    else:
        npz_basename = "zeroT_data"
    path = os.path.join(DATA_DIR, f"{npz_basename}.npz")
    if not os.path.isfile(path):
        print(f"Run data/run_zeroT.py {'--ver4' if ver4 else '--ver3' if ver3 else ''} first to create {path}")
        return
    apply_plot_style()
    data = np.load(path, allow_pickle=False)
    gam = data["gammas"]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 3, FIG_H))
    ax = axes[0]
    ax.plot(gam, data["v_theory"], "-", color=colors[0], linewidth=LINE_WIDTH)
    ax.errorbar(gam, data["v_mean"], yerr=data["v_std"],
                fmt=markers[0], color=colors[0], fillstyle=FILL, markersize=MS,
                **err_kw(colors[0]))
    ax.set_xlabel(r"$\gamma = N/M$")
    ax.set_ylabel(r"$v$")
    apply_style(ax)
    ax = axes[1]
    ax.plot(gam, data["rho_x_theory"], "-", color=COLOR_RHO_X, linewidth=LINE_WIDTH, label=r"$\rho_x$")
    ax.plot(gam, data["rho_y_theory"], "-", color=COLOR_RHO_Y, linewidth=LINE_WIDTH, label=r"$\rho_y$")
    ax.errorbar(gam, data["rho_x_mean"], yerr=data["rho_x_std"], fmt=markers[0], color=COLOR_RHO_X, fillstyle=FILL, markersize=MS, **err_kw(COLOR_RHO_X))
    ax.errorbar(gam, data["rho_y_mean"], yerr=data["rho_y_std"], fmt=markers[1], color=COLOR_RHO_Y, fillstyle=FILL, markersize=MS, **err_kw(COLOR_RHO_Y))
    ax.set_xlabel(r"$\gamma = N/M$")
    ax.set_ylabel(r"$\rho_x,\ \rho_y$")
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False)
    apply_style(ax)
    ax = axes[2]
    ax.plot(gam, data["qx_theory"], "-", color=COLOR_RHO_X, linewidth=LINE_WIDTH, label=r"$q_x$")
    ax.plot(gam, data["qy_theory"], "-", color=COLOR_RHO_Y, linewidth=LINE_WIDTH, label=r"$q_y$")
    ax.errorbar(gam, data["qx_mean"], yerr=data["qx_std"], fmt=markers[0], color=COLOR_RHO_X, fillstyle=FILL, markersize=MS, **err_kw(COLOR_RHO_X))
    ax.errorbar(gam, data["qy_mean"], yerr=data["qy_std"], fmt=markers[1], color=COLOR_RHO_Y, fillstyle=FILL, markersize=MS, **err_kw(COLOR_RHO_Y))
    ax.set_xlabel(r"$\gamma = N/M$")
    ax.set_ylabel(r"$q_x,\ q_y$")
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False)
    apply_style(ax)
    fig.tight_layout()
    out_path = out_path or os.path.join(FIG_DIR, "zeroT_three_panels.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_finiteT_v_vs_gamma(out_path=None, finiteT_npz_basename="finiteT_data", zeroT_npz_basename="zeroT_data"):
    path_z = os.path.join(DATA_DIR, f"{zeroT_npz_basename}.npz")
    path_f = os.path.join(DATA_DIR, f"{finiteT_npz_basename}.npz")
    if not os.path.isfile(path_z):
        print(f"Run data/run_zeroT.py first")
        return
    if not os.path.isfile(path_f):
        print(f"Run data/run_finiteT.py first")
        return
    apply_plot_style()
    data_z = np.load(path_z)
    data_f = np.load(path_f)
    gam_val = data_f["gammas"]
    gam_th = data_f["gammas_theory"] if "gammas_theory" in data_f else gam_val
    k_list = np.atleast_1d(data_f["k_list"]).tolist()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    idx = 0
    ax.plot(gam_th, data_f["zeroT_v_theory"], line_styles[0], color=colors[idx], linewidth=LINE_WIDTH, label=r"$\beta_{\min},\ \beta_{\max}\to\infty$")
    ax.errorbar(gam_val, data_z["v_mean"], yerr=data_z["v_std"],
                fmt=markers[idx], color=colors[idx], fillstyle=FILL, markersize=MS, **err_kw(colors[idx]))
    idx += 1
    for k in k_list:
        key = f"k_{k:.2f}".replace(".", "p").replace("-", "m")
        v_th = data_f[f"{key}_v_theory"]
        v_mc = data_f[f"{key}_v_mc_mean"]
        v_std = data_f[f"{key}_v_mc_std"]
        c = colors[idx % len(colors)]
        mk = markers[idx % len(markers)]
        ax.plot(gam_th, v_th, line_styles[0], color=c, linewidth=LINE_WIDTH, label=rf"$|k|={abs(k)}$")
        ax.errorbar(gam_val, v_mc, yerr=v_std, fmt=mk, color=c, fillstyle=FILL, markersize=MS, **err_kw(c))
        idx += 1
    ax.set_xlabel(r"$\gamma = N/M$")
    ax.set_ylabel(r"$v$")
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, ncol=2)
    apply_style(ax)
    fig.tight_layout()
    out_path = out_path or os.path.join(FIG_DIR, "finiteT_v_vs_gamma.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_finiteT_q_vs_gamma(out_path=None, finiteT_npz_basename="finiteT_data", zeroT_npz_basename="zeroT_data"):
    path_z = os.path.join(DATA_DIR, f"{zeroT_npz_basename}.npz")
    path_f = os.path.join(DATA_DIR, f"{finiteT_npz_basename}.npz")
    if not os.path.isfile(path_z) or not os.path.isfile(path_f):
        print(f"Missing data files: {path_z} or {path_f}")
        return
    apply_plot_style()

    data_z = np.load(path_z)
    data_f = np.load(path_f)
    gam_val = data_f["gammas"]
    gam_th = data_f["gammas_theory"] if "gammas_theory" in data_f else gam_val
    k_list = np.atleast_1d(data_f["k_list"]).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H))

    for panel_idx, (qty_label, qty_x, qty_y, ylabel) in enumerate([
        ("Qx", "qx", "Qx", r"$Q_x$"),
        ("Qy", "qy", "Qy", r"$Q_y$"),
    ]):
        ax = axes[panel_idx]
        idx = 0

        zeroT_key = f"{qty_x}_theory"
        if zeroT_key in data_z:
            gam_z = data_z["gammas"]
            ax.plot(gam_z, data_z[zeroT_key], line_styles[0], color=colors[idx],
                    linewidth=LINE_WIDTH, label=r"$\beta_{\min},\beta_{\max}\to\infty$")
            zeroT_mean_key = f"{qty_x}_mean"
            zeroT_std_key = f"{qty_x}_std"
            if zeroT_mean_key in data_z and zeroT_std_key in data_z:
                ax.errorbar(gam_z, data_z[zeroT_mean_key], yerr=data_z[zeroT_std_key],
                            fmt=markers[idx], color=colors[idx], fillstyle=FILL, markersize=MS,
                            **err_kw(colors[idx]))
        idx += 1

        for k_val in k_list:
            key = f"k_{k_val:.2f}".replace(".", "p").replace("-", "m")
            theory_key = f"{key}_{qty_y}_theory"
            mc_mean_key = f"{key}_{qty_y}_mc_mean"
            mc_std_key = f"{key}_{qty_y}_mc_std"

            c = colors[idx % len(colors)]
            mk = markers[idx % len(markers)]
            if theory_key in data_f:
                ax.plot(gam_th, data_f[theory_key], line_styles[0], color=c,
                        linewidth=LINE_WIDTH, label=rf"$|k|={abs(k_val)}$")
            if mc_mean_key in data_f and mc_std_key in data_f:
                ax.errorbar(gam_val, data_f[mc_mean_key], yerr=data_f[mc_std_key],
                            fmt=mk, color=c, fillstyle=FILL, markersize=MS, **err_kw(c))
            idx += 1

        ax.set_xlabel(r"$\gamma = N/M$")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, ncol=2)
        apply_style(ax)

    fig.tight_layout()
    out_path = out_path or os.path.join(FIG_DIR, "finiteT_q_vs_gamma.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_finiteT_energy_vs_gamma(out_path=None, finiteT_npz_basename="finiteT_data"):
    path_f = os.path.join(DATA_DIR, f"{finiteT_npz_basename}.npz")
    if not os.path.isfile(path_f):
        print(f"Data file not found: {path_f}")
        return
    apply_plot_style()
    data_f = np.load(path_f)
    gam_val = data_f["gammas"]
    gam_th = data_f["gammas_theory"] if "gammas_theory" in data_f else gam_val
    k_list = np.atleast_1d(data_f["k_list"]).tolist()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    idx = 0

    if "zeroT_v_theory" in data_f:
        ax.plot(gam_th, data_f["zeroT_v_theory"], line_styles[0], color=colors[idx],
                linewidth=LINE_WIDTH, label=r"$\beta_{\min},\ \beta_{\max}\to\infty$")
        idx += 1

    for ki in k_list:
        key = f"k_{ki:.2f}".replace(".", "p").replace("-", "m")
        energy_key = f"{key}_energy_theory"
        if energy_key not in data_f:
            continue
        c = colors[idx % len(colors)]
        mk = markers[idx % len(markers)]
        ax.plot(gam_th, data_f[energy_key], line_styles[0], color=c, linewidth=LINE_WIDTH,
                label=rf"$|k|={abs(ki):.1f}$")
        mc_mean_key = f"{key}_energy_mc_mean"
        mc_std_key = f"{key}_energy_mc_std"
        if mc_mean_key in data_f and mc_std_key in data_f:
            ax.errorbar(gam_val, data_f[mc_mean_key], yerr=data_f[mc_std_key],
                        fmt=mk, color=c, fillstyle=FILL, markersize=MS, **err_kw(c))
        idx += 1

    ax.set_xlabel(r"$\gamma = N/M$")
    ax.set_ylabel(r"$\mathbb{E}_{p,q}[V] / L$")
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, ncol=2)
    apply_style(ax)
    fig.tight_layout()
    out_path = out_path or os.path.join(FIG_DIR, "finiteT_energy_vs_gamma.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeroT", action="store_true", help="Plot zero-T 3 panels")
    ap.add_argument("--finiteT", action="store_true", help="Plot finite-T v vs γ")
    ap.add_argument("--all", action="store_true", help="Plot all")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--suffix", type=str, default="", help="Suffix for output filenames (e.g. _rified)")
    ap.add_argument("--ver2", action="store_true", help="Zero-T 3 panels ver2: larger y-label, no legend frame, larger legend")
    ap.add_argument("--ver3", action="store_true", help="Zero-T 3 panels ver3: larger M/trials (smaller error bars), even larger y-label")
    ap.add_argument("--ver4", action="store_true", help="Zero-T 3 panels ver4: hollow markers, transparent errorbars")
    args = ap.parse_args()
    out_dir = args.out_dir or FIG_DIR
    suffix = args.suffix if args.suffix else ""
    do_z = args.zeroT or args.all or args.ver4
    do_f = args.finiteT or args.all
    if not do_z and not do_f:
        do_z = do_f = True
    if do_z:
        if args.ver4:
            plot_zeroT_three_panels(
                out_path=os.path.join(out_dir, "zeroT_three_panels_ver4.pdf"),
                ver2=False,
                ver3=False,
                ver4=True,
            )
        elif args.ver3:
            plot_zeroT_three_panels(
                out_path=os.path.join(out_dir, "zeroT_three_panels_ver3.pdf"),
                ver2=False,
                ver3=True,
                ver4=False,
            )
        elif args.ver2:
            plot_zeroT_three_panels(
                out_path=os.path.join(out_dir, "zeroT_three_panels_ver2.pdf"),
                ver2=True,
                ver3=False,
                ver4=False,
            )
        else:
            plot_zeroT_three_panels(
                out_path=os.path.join(out_dir, f"zeroT_three_panels{suffix}.pdf"),
                ver2=False,
                ver3=False,
                ver4=False,
            )
    if do_f:
        finiteT_basename = "finiteT_data_rified" if suffix == "_rified" else "finiteT_data"
        plot_finiteT_v_vs_gamma(
            out_path=os.path.join(out_dir, f"finiteT_v_vs_gamma{suffix}.pdf"),
            finiteT_npz_basename=finiteT_basename,
        )


if __name__ == "__main__":
    main()
