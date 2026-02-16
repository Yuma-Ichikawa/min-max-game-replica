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
from src.finiteT_ais import estimate_Phi_ais_for_C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(REPO_ROOT, "fig")
data_dir = os.path.join(REPO_ROOT, "data")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
print(f"Device: {DEVICE}", flush=True)

sigma = 1.0
beta_max = 1.0

gammas = [0.8, 1.0, 1.2]
NM_for_gamma = {
    0.8: (32, 40),
    1.0: (30, 30),
    1.2: (36, 30),
}

K_MIN, K_MAX = 0.80, 1.20

abs_k_rs = np.arange(K_MIN, K_MAX + 0.001, 0.005)

abs_k_ais = np.linspace(K_MIN, K_MAX, 9)

n_trials = 12
n_chains = 1500
n_temps = 400
n_mcmc = 3

gh_n = 100
rs_max_iter = 5000
rs_tol = 1e-13
rs_damping = 0.10

print(f"gammas: {gammas}", flush=True)
print(f"|k| range: [{K_MIN}, {K_MAX}]", flush=True)
print(f"RS grid: {len(abs_k_rs)} points (step=0.005)", flush=True)
print(f"AIS grid: {len(abs_k_ais)} points", flush=True)
print(f"AIS: n_trials={n_trials}, n_chains={n_chains}, n_temps={n_temps}, n_mcmc={n_mcmc}",
      flush=True)
print(f"RS: gh_n={gh_n}, max_iter={rs_max_iter}, tol={rs_tol}, damping={rs_damping}",
      flush=True)
print(f"beta_max={beta_max}, sigma={sigma}", flush=True)
print(flush=True)

from vis.config import (
    PLOT_STYLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
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

colors_rs = [COLORS[0], COLORS[3], COLORS[2]]
colors_ais = colors_rs
markers_ais = [MARKERS[0], MARKERS[1], MARKERS[2]]
fills = [FILL_STYLE] * 3


def _is_degenerate(res):
    if res is None:
        return True
    s = res.state
    if s.Qx < 0.01 or s.qx < 0.001:
        return True
    if abs(res.g) > 1e10:
        return True
    if float(np.max(np.abs(res.residuals))) > 0.01:
        return True
    return False


def solve_rs_dense(gamma, abs_k_array, N, M):
    nk = len(abs_k_array)
    results = [None] * nk
    prev = None
    n_retry = 0

    for i in range(nk):
        p = FiniteTRSParams(
            sigma=sigma, gamma=gamma, beta_max=beta_max,
            beta_min=abs_k_array[i] * beta_max,
            gh_n=gh_n, x_max=float(N), y_max=float(M),
            damping=rs_damping, max_iter=rs_max_iter, tol=rs_tol,
        )
        try:
            res = solve_rs_finiteT(p, init=prev, device=DEVICE)
        except Exception:
            res = None

        if _is_degenerate(res):
            try:
                res = solve_rs_finiteT(p, init=None, device=DEVICE)
            except Exception:
                res = None
            if not _is_degenerate(res):
                n_retry += 1

        results[i] = res
        if res is not None and not _is_degenerate(res):
            prev = res.state

        if (i + 1) % 20 == 0 or i == nk - 1:
            print(f"      [{i+1}/{nk}] |k|={abs_k_array[i]:.3f}", flush=True)

    n_valid = sum(1 for r in results if not _is_degenerate(r))
    print(f"    {n_valid}/{nk} valid ({n_retry} recovered by multi-init)", flush=True)
    return results


print("=" * 60, flush=True)
print("Computing RS (finite-size) curves", flush=True)
print("=" * 60, flush=True)

rs_data = {}
for gm in gammas:
    N, M = NM_for_gamma[gm]
    print(f"\n--- γ={gm:.2f}  (N={N}, M={M}) ---", flush=True)
    print(f"  RS forward sweep (gh_n={gh_n}, {len(abs_k_rs)} points)...", flush=True)
    t0 = time.time()
    results = solve_rs_dense(gm, abs_k_rs, N, M)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)

    vs = np.array([r.v if r else np.nan for r in results])
    es = np.array([r.energy if r else np.nan for r in results])
    gs = np.array([r.g if r else np.nan for r in results])

    for idx in [0, len(results)//4, len(results)//2, 3*len(results)//4, -1]:
        r = results[idx]
        if r and not _is_degenerate(r):
            s = r.state
            print(f"    |k|={abs_k_rs[idx]:.3f}: v={r.v:+.6f} E={r.energy:+.6f} "
                  f"g={r.g:+.8f} res={float(np.max(np.abs(r.residuals))):.1e}",
                  flush=True)

    rs_data[gm] = {"abs_k": abs_k_rs.copy(), "v": vs, "energy": es, "g": gs}


print("\n" + "=" * 60, flush=True)
print("Computing AIS estimates", flush=True)
print("=" * 60, flush=True)

ais_data = {}
for gm in gammas:
    N, M = NM_for_gamma[gm]
    L = math.sqrt(N * M)
    print(f"\n--- γ={gm:.2f}  (N={N}, M={M}, L={L:.2f}) ---", flush=True)

    block_size = min(8, N - 1)

    ais_mean_v = []
    ais_std_v = []
    ais_mean_e = []
    ais_std_e = []
    ais_ess_list = []

    for ki, abs_k in enumerate(abs_k_ais):
        beta_min = abs_k * beta_max

        print(f"  |k|={abs_k:.3f} [{ki+1}/{len(abs_k_ais)}]...", end="", flush=True)
        t0 = time.time()

        rng = np.random.default_rng(42)
        trial_v = []
        trial_e = []
        trial_ess = []

        for trial in range(n_trials):
            C = rng.standard_normal((N, M))
            ais = estimate_Phi_ais_for_C(
                C, sigma=sigma, beta_max=beta_max, beta_min=beta_min,
                n_chains=n_chains, n_temps=n_temps, n_mcmc=n_mcmc,
                schedule="sigmoid", proposal="block_dirichlet",
                block_size=block_size, pairwise_step=0.3,
                rng=np.random.default_rng(trial * 1000 + 7),
                device=DEVICE,
            )
            trial_v.append(ais.Phi_over_L)
            trial_e.append(ais.energy)
            trial_ess.append(ais.ess_fraction)

        elapsed = time.time() - t0
        arr_v = np.array(trial_v)
        arr_e = np.array(trial_e)
        ais_mean_v.append(arr_v.mean())
        ais_std_v.append(arr_v.std(ddof=1))
        ais_mean_e.append(arr_e.mean())
        ais_std_e.append(arr_e.std(ddof=1))
        ais_ess_list.append(np.mean(trial_ess))

        print(f" v={arr_v.mean():+.6f}±{arr_v.std(ddof=1):.4f} "
              f"E={arr_e.mean():+.6f}±{arr_e.std(ddof=1):.4f} "
              f"ESS={np.mean(trial_ess):.3f} ({elapsed:.1f}s)", flush=True)

    ais_data[gm] = {
        "abs_k": abs_k_ais.copy(),
        "N": N, "M": M, "L": L,
        "ais_mean_v": np.array(ais_mean_v),
        "ais_std_v": np.array(ais_std_v),
        "ais_mean_energy": np.array(ais_mean_e),
        "ais_std_energy": np.array(ais_std_e),
        "ais_ess": np.array(ais_ess_list),
    }


save_dict = {}
for gm in gammas:
    pf = f"g{gm:.1f}_".replace(".", "p")
    for key, val in rs_data[gm].items():
        if isinstance(val, np.ndarray):
            save_dict[f"rs_{pf}{key}"] = val
    for key, val in ais_data[gm].items():
        if isinstance(val, np.ndarray):
            save_dict[f"ais_{pf}{key}"] = val
np.savez(os.path.join(data_dir, "paper_energy_data.npz"), **save_dict)
print(f"\nSaved: paper_energy_data.npz", flush=True)


print("\n" + "=" * 60, flush=True)
print("Generating publication figure", flush=True)
print("=" * 60, flush=True)

se_factor = 1.0 / math.sqrt(n_trials)

fig, ax = plt.subplots(figsize=(7, 5))

for i, gm in enumerate(gammas):
    rd = rs_data[gm]
    ad = ais_data[gm]
    N_val, M_val = NM_for_gamma[gm]
    c = colors_rs[i]
    mk = markers_ais[i]

    valid = ~np.isnan(rd["energy"])
    ax.plot(rd["abs_k"][valid], rd["energy"][valid], "-", color=c, lw=LW_THEORY,
            label=rf"RS $\gamma={gm}$", zorder=3)

    ax.errorbar(ad["abs_k"], ad["ais_mean_energy"],
                yerr=ad["ais_std_energy"] * se_factor,
                fmt=mk, color=c, ms=MS, fillstyle=fills[i],
                markeredgewidth=1.5, zorder=4,
                label=rf"AIS $\gamma={gm}$",
                **_cfg_err_kw(c))

ax.set_xlabel(r"$|k| = \beta_{\min}/\beta_{\max}$", fontsize=FONT_LABEL)
ax.set_ylabel(r"$\mathbb{E}[V]/L$", fontsize=FONT_LABEL)
ax.tick_params(labelsize=FONT_TICK)
ax.set_xlim(K_MIN - 0.02, K_MAX + 0.02)
ax.grid(ls=GRID_LINESTYLE)
ax.legend(fontsize=FONT_LEGEND, loc="best", ncol=2)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "paper_energy_vs_k.pdf"),
            dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: paper_energy_vs_k.pdf", flush=True)


fig, ax = plt.subplots(figsize=(7, 5))

for i, gm in enumerate(gammas):
    rd = rs_data[gm]
    ad = ais_data[gm]
    c = colors_rs[i]
    mk = markers_ais[i]

    valid = ~np.isnan(rd["v"])
    ax.plot(rd["abs_k"][valid], rd["v"][valid], "-", color=c, lw=LW_THEORY,
            label=rf"RS $\gamma={gm}$", zorder=3)

    ax.errorbar(ad["abs_k"], ad["ais_mean_v"],
                yerr=ad["ais_std_v"] * se_factor,
                fmt=mk, color=c, ms=MS, fillstyle=fills[i],
                markeredgewidth=1.5, zorder=4,
                label=rf"AIS $\gamma={gm}$",
                **_cfg_err_kw(c))

ax.set_xlabel(r"$|k| = \beta_{\min}/\beta_{\max}$", fontsize=FONT_LABEL)
ax.set_ylabel(r"$v = -g / \beta_{\min}$", fontsize=FONT_LABEL)
ax.tick_params(labelsize=FONT_TICK)
ax.set_xlim(K_MIN - 0.02, K_MAX + 0.02)
ax.grid(ls=GRID_LINESTYLE)
ax.legend(fontsize=FONT_LEGEND, loc="best", ncol=2)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "paper_v_vs_k.pdf"),
            dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: paper_v_vs_k.pdf", flush=True)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for i, gm in enumerate(gammas):
    rd = rs_data[gm]
    ad = ais_data[gm]
    c = colors_rs[i]
    mk = markers_ais[i]

    valid = ~np.isnan(rd["v"])
    ax1.plot(rd["abs_k"][valid], rd["v"][valid], "-", color=c, lw=LW_THEORY,
             label=rf"RS $\gamma={gm}$", zorder=3)
    ax1.errorbar(ad["abs_k"], ad["ais_mean_v"],
                 yerr=ad["ais_std_v"] * se_factor,
                 fmt=mk, color=c, ms=MS, fillstyle=fills[i],
                 markeredgewidth=1.5, zorder=4,
                 label=rf"AIS $\gamma={gm}$",
                 **_cfg_err_kw(c))

    valid = ~np.isnan(rd["energy"])
    ax2.plot(rd["abs_k"][valid], rd["energy"][valid], "-", color=c, lw=LW_THEORY,
             label=rf"RS $\gamma={gm}$", zorder=3)
    ax2.errorbar(ad["abs_k"], ad["ais_mean_energy"],
                 yerr=ad["ais_std_energy"] * se_factor,
                 fmt=mk, color=c, ms=MS, fillstyle=fills[i],
                 markeredgewidth=1.5, zorder=4,
                 label=rf"AIS $\gamma={gm}$",
                 **_cfg_err_kw(c))

for ax, ylabel, title_letter in [
    (ax1, r"$v = -g / \beta_{\min}$", "(a)"),
    (ax2, r"$\mathbb{E}[V]/L$", "(b)"),
]:
    ax.set_xlabel(r"$|k| = \beta_{\min}/\beta_{\max}$", fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_xlim(K_MIN - 0.02, K_MAX + 0.02)
    ax.grid(ls=GRID_LINESTYLE)
    ax.legend(fontsize=FONT_LEGEND - 1, loc="best", ncol=2)
    ax.set_title(title_letter, fontsize=FONT_LABEL, loc="left", fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "paper_v_and_energy_vs_k.pdf"),
            dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: paper_v_and_energy_vs_k.pdf", flush=True)

print("\nALL DONE", flush=True)
