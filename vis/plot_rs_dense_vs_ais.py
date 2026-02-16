#!/usr/bin/env python
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

from src.special import DTYPE, DEVICE
from src.rs_finiteT import (
    solve_rs_finiteT, FiniteTRSParams, FiniteTRSResult,
    compute_energy,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(REPO_ROOT, "fig")
data_dir = os.path.join(REPO_ROOT, "data")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
print(f"Device: {DEVICE}", flush=True)

from vis.config import (
    PLOT_STYLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    MARKER_SIZE, EB_ALPHA, FILL_STYLE, LINE_WIDTH, GRID_LINESTYLE,
    COLORS, MARKERS, apply_plot_style, apply_axis_style, err_kw,
)

FONT_LABEL = FONT_SIZE_LABEL
FONT_TICK = FONT_SIZE_TICK
FONT_LEGEND = FONT_SIZE_LEGEND
MS = MARKER_SIZE
FILL = FILL_STYLE

colors = list(COLORS) + ["tab:pink", "tab:gray"]
markers = list(MARKERS) + ["*", "X"]

apply_plot_style()


def _err_kw(c):
    return err_kw(c)


def _apply_style(ax, xlabel, ylabel):
    ax.grid(ls=GRID_LINESTYLE)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)


data_path = os.path.join(data_dir, "rs_vs_ais_k_dep_data.npz")
saved = np.load(data_path) if os.path.exists(data_path) else None
if saved is not None:
    print(f"Loaded AIS data from {data_path}", flush=True)

ais_data = {}
gammas = [1.0, 1.5, 2.0]
NM_for_gamma = {1.0: (30, 30), 1.5: (30, 20), 2.0: (40, 20)}

for gm in gammas:
    prefix = f"g{gm:.1f}_".replace(".", "p")
    d = {}
    if saved is not None:
        for key in saved.files:
            if key.startswith(prefix):
                d[key[len(prefix):]] = saved[key]
    d["N"], d["M"] = NM_for_gamma[gm]
    d["L"] = math.sqrt(d["N"] * d["M"])
    ais_data[gm] = d

n_trials = 12
se_factor = 1.0 / math.sqrt(n_trials)


def _is_degenerate(res):
    if res is None:
        return True
    s = res.state
    if s.Qx < 0.01 or s.qx < 0.001:
        return True
    if abs(res.g) > 1e10:
        return True
    if float(np.max(np.abs(res.residuals))) > 0.1:
        return True
    return False


def solve_rs_curve_single_sweep(gamma, abs_k_array, sigma=1.0, beta_max=1.0,
                                gh_n=100, x_max=500.0, y_max=80.0,
                                max_iter=5000, tol=1e-13, damping=0.10,
                                direction="forward"):
    nk = len(abs_k_array)

    def _make_params(abs_k):
        return FiniteTRSParams(
            sigma=sigma, gamma=gamma, beta_max=beta_max,
            beta_min=abs_k * beta_max,
            gh_n=gh_n, x_max=x_max, y_max=y_max,
            damping=damping, max_iter=max_iter, tol=tol,
        )

    def _solve(abs_k, init_state):
        p = _make_params(abs_k)
        try:
            return solve_rs_finiteT(p, init=init_state, device=DEVICE)
        except Exception:
            return None

    if direction == "reverse":
        indices = range(nk - 1, -1, -1)
    else:
        indices = range(nk)

    results = [None] * nk
    prev = None
    n_retry = 0

    done = 0
    for i in indices:
        res = _solve(abs_k_array[i], prev)
        if _is_degenerate(res):
            res = _solve(abs_k_array[i], None)
            if not _is_degenerate(res):
                n_retry += 1
        results[i] = res
        if res is not None and not _is_degenerate(res):
            prev = res.state
        done += 1
        if done % 10 == 0 or done == nk:
            print(f"        [{done}/{nk}] |k|={abs_k_array[i]:.3f}", flush=True)

    n_valid = sum(1 for r in results if not _is_degenerate(r))
    print(f"      {n_valid}/{nk} valid ({n_retry} recovered by multi-init)",
          flush=True)

    return results


print("\n" + "=" * 60, flush=True)
print("Computing RS curves", flush=True)
print("=" * 60, flush=True)

sigma = 1.0
beta_max = 1.0
abs_k_dense = np.arange(0.10, 1.52, 0.04)
print(f"|k| grid: {len(abs_k_dense)} points [{abs_k_dense[0]:.2f}, {abs_k_dense[-1]:.2f}]",
      flush=True)

rs_data_finite = {}
rs_data_thermo = {}

for gm in gammas:
    N, M = NM_for_gamma[gm]
    print(f"\n--- γ={gm:.2f}  (N={N}, M={M}) ---", flush=True)

    print(f"  Finite-size RS (x_max={N}, y_max={M}, dense grid, forward sweep)...",
          flush=True)
    t0 = time.time()
    fs_results = solve_rs_curve_single_sweep(
        gm, abs_k_dense,
        sigma=sigma, beta_max=beta_max,
        gh_n=100, x_max=float(N), y_max=float(M),
        max_iter=5000, tol=1e-13, damping=0.10,
        direction="forward",
    )
    elapsed = time.time() - t0

    fs_v = np.array([r.v if r else np.nan for r in fs_results])
    fs_e = np.array([r.energy if r else np.nan for r in fs_results])
    fs_g = np.array([r.g if r else np.nan for r in fs_results])

    n_valid = sum(1 for r in fs_results if r is not None and not _is_degenerate(r))
    print(f"  Done in {elapsed:.1f}s ({n_valid}/{len(abs_k_dense)} valid)", flush=True)

    for idx in range(len(abs_k_dense)):
        r = fs_results[idx]
        if r and not _is_degenerate(r):
            s = r.state
            print(f"    |k|={abs_k_dense[idx]:.2f}: v={r.v:+.6f} E={r.energy:+.6f} "
                  f"g={r.g:+.8f} Qx={s.Qx:.4f} qx={s.qx:.4f} "
                  f"Qy={s.Qy:.4f} q1={s.q1:.4f} q0={s.q0:.4f} "
                  f"res={float(np.max(np.abs(r.residuals))):.1e}", flush=True)
        else:
            print(f"    |k|={abs_k_dense[idx]:.2f}: FAILED", flush=True)

    rs_data_finite[gm] = {
        "abs_k": abs_k_dense.copy(),
        "v": fs_v,
        "energy": fs_e,
        "g": fs_g,
    }

    print(f"  Thermodynamic limit RS (x_max=500, y_max=80, forward sweep)...",
          flush=True)
    t0 = time.time()
    th_results = solve_rs_curve_single_sweep(
        gm, abs_k_dense,
        sigma=sigma, beta_max=beta_max,
        gh_n=100, x_max=500.0, y_max=80.0,
        max_iter=5000, tol=1e-13, damping=0.10,
        direction="forward",
    )
    elapsed = time.time() - t0

    th_v = np.array([r.v if r else np.nan for r in th_results])
    th_e = np.array([r.energy if r else np.nan for r in th_results])
    th_g = np.array([r.g if r else np.nan for r in th_results])

    n_valid = sum(1 for r in th_results if r is not None and not _is_degenerate(r))
    print(f"  Done in {elapsed:.1f}s ({n_valid}/{len(abs_k_dense)} valid)", flush=True)

    rs_data_thermo[gm] = {
        "abs_k": abs_k_dense.copy(),
        "v": th_v,
        "energy": th_e,
        "g": th_g,
    }

    _save = {}
    for g_ in gammas:
        if g_ in rs_data_finite:
            pf = f"rs_finite_g{g_:.1f}_".replace(".", "p")
            for k_, v_ in rs_data_finite[g_].items():
                if isinstance(v_, np.ndarray):
                    _save[pf + k_] = v_
        if g_ in rs_data_thermo:
            pf = f"rs_thermo_g{g_:.1f}_".replace(".", "p")
            for k_, v_ in rs_data_thermo[g_].items():
                if isinstance(v_, np.ndarray):
                    _save[pf + k_] = v_
    np.savez(os.path.join(data_dir, "rs_dense_data.npz"), **_save)
    print(f"  [incremental save done]", flush=True)


print("\n" + "=" * 60, flush=True)
print("Generating figures", flush=True)
print("=" * 60, flush=True)

fig, ax = plt.subplots(figsize=(9, 6.5))
for i, gm in enumerate(gammas):
    fd = rs_data_finite[gm]
    ad = ais_data[gm]
    N_val, M_val = NM_for_gamma[gm]
    c = colors[i]
    mk = markers[i]

    valid = ~np.isnan(fd["v"])
    ax.plot(fd["abs_k"][valid], fd["v"][valid], "-", color=c, lw=2.5,
            label=rf"RS ($N\!={N_val}$) $\gamma={gm:.1f}$", zorder=3)

    td = rs_data_thermo[gm]
    valid_t = ~np.isnan(td["v"])
    ax.plot(td["abs_k"][valid_t], td["v"][valid_t], "--", color=c, lw=1.2,
            alpha=0.5, zorder=2)

    if "ais_mean_v" in ad:
        ax.errorbar(ad["abs_k"], ad["ais_mean_v"],
                    yerr=ad["ais_std_v"] * se_factor,
                    fmt=mk, color=c, ms=MS, fillstyle=FILL,
                    markeredgewidth=1.5, zorder=4,
                    label=rf"AIS $\gamma={gm:.1f}$",
                    **_err_kw(c))

_apply_style(ax, r"$|k| = \beta_{\min}/\beta_{\max}$",
             r"$v = -g / \beta_{\min}$")
ax.legend(fontsize=FONT_LEGEND, loc="best")
ax.set_title(r"RS (finite-size) vs AIS: $v$ vs $|k|$", fontsize=18)
ax.set_xlim(0, 1.6)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "rs_vs_ais_v_vs_k_all_gammas.pdf"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: rs_vs_ais_v_vs_k_all_gammas.pdf", flush=True)

fig, ax = plt.subplots(figsize=(9, 6.5))
for i, gm in enumerate(gammas):
    fd = rs_data_finite[gm]
    ad = ais_data[gm]
    N_val, M_val = NM_for_gamma[gm]
    c = colors[i]
    mk = markers[i]

    valid = ~np.isnan(fd["energy"])
    ax.plot(fd["abs_k"][valid], fd["energy"][valid], "-", color=c, lw=2.5,
            label=rf"RS ($N\!={N_val}$) $\gamma={gm:.1f}$", zorder=3)

    if "ais_mean_energy" in ad:
        ax.errorbar(ad["abs_k"], ad["ais_mean_energy"],
                    yerr=ad.get("ais_std_energy", ad["ais_std_v"]) * se_factor,
                    fmt=mk, color=c, ms=MS, fillstyle=FILL,
                    markeredgewidth=1.5, zorder=4,
                    label=rf"AIS $\gamma={gm:.1f}$",
                    **_err_kw(c))

_apply_style(ax, r"$|k| = \beta_{\min}/\beta_{\max}$",
             r"$\mathbb{E}[V]/L$")
ax.legend(fontsize=FONT_LEGEND, loc="best")
ax.set_title(r"RS (finite-size) vs AIS: $\mathbb{E}[V]/L$ vs $|k|$", fontsize=18)
ax.set_xlim(0, 1.6)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "rs_vs_ais_energy_vs_k_all_gammas.pdf"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: rs_vs_ais_energy_vs_k_all_gammas.pdf", flush=True)

for gm in gammas:
    fd = rs_data_finite[gm]
    td = rs_data_thermo[gm]
    ad = ais_data[gm]
    N_val, M_val = NM_for_gamma[gm]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        rf"RS vs AIS ($N={N_val}, M={M_val}$): $\gamma={gm:.1f}$",
        fontsize=18, y=0.98)

    ax = axes[0, 0]
    valid = ~np.isnan(fd["v"])
    ax.plot(fd["abs_k"][valid], fd["v"][valid], "-", color=colors[0], lw=2.5,
            label=f"RS ($N={N_val}$)", zorder=3)
    valid_t = ~np.isnan(td["v"])
    ax.plot(td["abs_k"][valid_t], td["v"][valid_t], "--", color=colors[0],
            lw=1.2, alpha=0.5, label="RS (thermo.)", zorder=2)
    if "ais_mean_v" in ad:
        ax.errorbar(ad["abs_k"], ad["ais_mean_v"],
                    yerr=ad["ais_std_v"] * se_factor,
                    fmt="o", color=colors[1], ms=MS, fillstyle=FILL,
                    markeredgewidth=1.5, label="AIS", zorder=4,
                    **_err_kw(colors[1]))
    ax.set_title(r"$v$", fontsize=16)
    _apply_style(ax, r"$|k|$", r"$v$")
    ax.set_xlim(0, 1.6)
    ax.legend(fontsize=FONT_LEGEND)

    ax = axes[0, 1]
    valid = ~np.isnan(fd["energy"])
    ax.plot(fd["abs_k"][valid], fd["energy"][valid], "-", color=colors[0], lw=2.5,
            label=f"RS ($N={N_val}$)", zorder=3)
    if "ais_mean_energy" in ad:
        ax.errorbar(ad["abs_k"], ad["ais_mean_energy"],
                    yerr=ad.get("ais_std_energy", ad["ais_std_v"]) * se_factor,
                    fmt="o", color=colors[1], ms=MS, fillstyle=FILL,
                    markeredgewidth=1.5, label="AIS", zorder=4,
                    **_err_kw(colors[1]))
    ax.set_title(r"$\mathbb{E}[V]/L$", fontsize=16)
    _apply_style(ax, r"$|k|$", r"$\mathbb{E}[V]/L$")
    ax.set_xlim(0, 1.6)
    ax.legend(fontsize=FONT_LEGEND)

    ax = axes[1, 0]
    if "ais_ess" in ad:
        ax.plot(ad["abs_k"], ad["ais_ess"], "-o", color=colors[4],
                ms=MS, fillstyle=FILL, markeredgewidth=1.5, lw=2.0)
        ax.axhline(0.1, ls=":", color="gray", alpha=0.5)
        ax.set_ylim(bottom=0)
    ax.set_title("ESS fraction (AIS)", fontsize=16)
    _apply_style(ax, r"$|k|$", "ESS fraction")

    ax = axes[1, 1]
    if "ais_mean_v" in ad:
        ais_k = ad["abs_k"]
        v_valid = ~np.isnan(fd["v"])
        rs_v_interp = np.interp(ais_k, fd["abs_k"][v_valid], fd["v"][v_valid])

        gap_v = np.abs(ad["ais_mean_v"] - rs_v_interp) / np.maximum(
            np.abs(rs_v_interp), 1e-10)
        ax.plot(ais_k, gap_v, "-o", color=colors[0], ms=MS-2,
                fillstyle=FILL, markeredgewidth=1.2, lw=1.5,
                label=r"$|v_{\rm AIS} - v_{\rm RS}|/|v_{\rm RS}|$")

        if "ais_mean_energy" in ad:
            e_valid = ~np.isnan(fd["energy"])
            rs_e_interp = np.interp(ais_k, fd["abs_k"][e_valid],
                                    fd["energy"][e_valid])
            valid_ei = (np.abs(rs_e_interp) > 1e-10)
            if valid_ei.any():
                gap_e = np.abs(ad["ais_mean_energy"] - rs_e_interp)
                gap_e = np.where(valid_ei, gap_e / np.abs(rs_e_interp), np.nan)
                ax.plot(ais_k[valid_ei], gap_e[valid_ei],
                        "-D", color=colors[3],
                        ms=MS-2, fillstyle=FILL, markeredgewidth=1.2, lw=1.5,
                        label=r"$|E_{\rm AIS} - E_{\rm RS}|/|E_{\rm RS}|$")

    ax.set_yscale("log")
    ax.set_title("Relative deviations (RS finite vs AIS)", fontsize=14)
    _apply_style(ax, r"$|k|$", "Relative error")
    ax.legend(fontsize=FONT_LEGEND - 1, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, f"rs_vs_ais_k_dep_gamma{gm:.1f}.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: rs_vs_ais_k_dep_gamma{gm:.1f}.pdf", flush=True)

save_dict = {}
for gm in gammas:
    if gm in rs_data_finite:
        pf = f"rs_finite_g{gm:.1f}_".replace(".", "p")
        for key, val in rs_data_finite[gm].items():
            if isinstance(val, np.ndarray):
                save_dict[pf + key] = val
    if gm in rs_data_thermo:
        pf = f"rs_thermo_g{gm:.1f}_".replace(".", "p")
        for key, val in rs_data_thermo[gm].items():
            if isinstance(val, np.ndarray):
                save_dict[pf + key] = val
np.savez(os.path.join(data_dir, "rs_dense_data.npz"), **save_dict)
print(f"Saved: rs_dense_data.npz", flush=True)

print("\nALL DONE", flush=True)
