from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vis.config import (
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    FIG_W, FIG_H, MARKER_SIZE, EB_ALPHA, FILL_STYLE, LINE_WIDTH,
    COLORS, MARKERS, LINE_STYLES,
    apply_plot_style, apply_axis_style, err_kw,
)

MS = MARKER_SIZE
FILL = FILL_STYLE

colors = list(COLORS)
markers = list(MARKERS)
line_styles = list(LINE_STYLES)
COLOR_X = colors[0]
COLOR_Y = colors[1]


def _err_kw(c):
    return err_kw(c)


def _apply_style(ax):
    apply_axis_style(ax)


def plot_finiteT_two_panels(
    finiteT_npz_path: str,
    zeroT_npz_path: str,
    out_path: str,
):
    apply_plot_style()

    data_f = np.load(finiteT_npz_path)
    data_z = np.load(zeroT_npz_path)

    gam_val = data_f["gammas"]
    gam_th = data_f["gammas_theory"]
    k_list = np.atleast_1d(data_f["k_list"]).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H))

    ax = axes[0]
    idx = 0

    c0 = colors[idx]
    ax.plot(gam_th, data_f["zeroT_v_theory"], line_styles[0], color=c0,
            linewidth=LINE_WIDTH, label=r"$\beta_{\min},\ \beta_{\max}\to\infty$")
    ax.errorbar(gam_val, data_z["v_mean"], yerr=data_z["v_std"],
                fmt=markers[0], color=c0, fillstyle=FILL, markersize=MS,
                **_err_kw(c0))
    idx += 1

    for k in k_list:
        key = f"k_{k:.2f}".replace(".", "p").replace("-", "m")
        v_th_key = f"{key}_v_theory"
        v_mc_mean_key = f"{key}_v_mc_mean"
        v_mc_std_key = f"{key}_v_mc_std"

        if v_th_key not in data_f:
            continue

        c = colors[idx % len(colors)]
        mk = markers[idx % len(markers)]

        ax.plot(gam_th, data_f[v_th_key], line_styles[0], color=c, linewidth=LINE_WIDTH,
                label=rf"$|k|={abs(k):.1f}$")

        if v_mc_mean_key in data_f:
            ax.errorbar(gam_val, data_f[v_mc_mean_key], yerr=data_f[v_mc_std_key],
                        fmt=mk, color=c, fillstyle=FILL, markersize=MS,
                        **_err_kw(c))
        idx += 1

    ax.set_xlabel(r"$\gamma = N/M$")
    ax.set_ylabel(r"$v$")
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, ncol=2)
    _apply_style(ax)

    ax = axes[1]

    zeroT_qx_fine = data_f.get("zeroT_qx_theory", data_z.get("qx_theory_fine", None))
    zeroT_qy_fine = data_f.get("zeroT_qy_theory", data_z.get("qy_theory_fine", None))

    if zeroT_qx_fine is not None:
        ax.plot(gam_th, zeroT_qx_fine, line_styles[0], color=COLOR_X,
                linewidth=LINE_WIDTH, label=r"$q_x\ (\beta\to\infty)$")
    if zeroT_qy_fine is not None:
        ax.plot(gam_th, zeroT_qy_fine, line_styles[0], color=COLOR_Y,
                linewidth=LINE_WIDTH, label=r"$q_y\ (\beta\to\infty)$")

    if "qx_mean" in data_z:
        ax.errorbar(gam_val, data_z["qx_mean"], yerr=data_z["qx_std"],
                    fmt=markers[0], color=COLOR_X, fillstyle=FILL, markersize=MS,
                    **_err_kw(COLOR_X))
    if "qy_mean" in data_z:
        ax.errorbar(gam_val, data_z["qy_mean"], yerr=data_z["qy_std"],
                    fmt=markers[1], color=COLOR_Y, fillstyle=FILL, markersize=MS,
                    **_err_kw(COLOR_Y))

    ls_idx = 1
    for k in k_list:
        key = f"k_{k:.2f}".replace(".", "p").replace("-", "m")
        qx_key = f"{key}_qx_theory"
        q0_key = f"{key}_q0_theory"

        if qx_key not in data_f:
            continue

        ls = line_styles[ls_idx % len(line_styles)]

        qx_vals = data_f[qx_key]
        ax.plot(gam_th, qx_vals, ls, color=COLOR_X, linewidth=1.8,
                label=rf"$q_x\ (|k|={abs(k):.1f})$")

        if q0_key in data_f:
            q0_vals = data_f[q0_key]
            ax.plot(gam_th, q0_vals, ls, color=COLOR_Y, linewidth=1.8,
                    label=rf"$q_0\ (|k|={abs(k):.1f})$")

        ls_idx += 1

    ax.set_xlabel(r"$\gamma = N/M$")
    ax.set_ylabel(r"$q_x,\ q_y$")
    ax.legend(fontsize=14, frameon=False, ncol=2)
    _apply_style(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_finiteT_energy(
    finiteT_npz_path: str,
    out_path: str,
):
    apply_plot_style()

    data_f = np.load(finiteT_npz_path)
    gam_th = data_f["gammas_theory"]
    k_list = np.atleast_1d(data_f["k_list"]).tolist()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    if "zeroT_v_theory" in data_f:
        ax.plot(gam_th, data_f["zeroT_v_theory"], line_styles[0], color=colors[0],
                linewidth=LINE_WIDTH, label=r"$\beta_{\min},\ \beta_{\max}\to\infty$")

    idx = 1
    for k in k_list:
        key = f"k_{k:.2f}".replace(".", "p").replace("-", "m")
        energy_key = f"{key}_energy_theory"

        if energy_key not in data_f:
            continue

        c = colors[idx % len(colors)]
        ax.plot(gam_th, data_f[energy_key], line_styles[0], color=c, linewidth=LINE_WIDTH,
                label=rf"$|k|={abs(k):.1f}$")
        idx += 1

    ax.set_xlabel(r"$\gamma = N/M$")
    ax.set_ylabel(r"$\mathbb{E}_{p,q}[V] / L$")
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, ncol=2)
    _apply_style(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot finite-T two panels from npz")
    ap.add_argument("--finiteT_npz", type=str, required=True,
                    help="Path to finiteT_data*.npz")
    ap.add_argument("--zeroT_npz", type=str, default=None,
                    help="Path to zeroT_data*.npz")
    ap.add_argument("--out", type=str, default="fig/finiteT_two_panels.pdf",
                    help="Output PDF path")
    ap.add_argument("--energy_out", type=str, default=None,
                    help="Output PDF path for energy plot")
    args = ap.parse_args()

    if args.zeroT_npz is not None:
        plot_finiteT_two_panels(
            finiteT_npz_path=args.finiteT_npz,
            zeroT_npz_path=args.zeroT_npz,
            out_path=args.out,
        )

    energy_out = args.energy_out or args.out.replace(".pdf", "_energy.pdf")
    plot_finiteT_energy(
        finiteT_npz_path=args.finiteT_npz,
        out_path=energy_out,
    )


if __name__ == "__main__":
    main()
