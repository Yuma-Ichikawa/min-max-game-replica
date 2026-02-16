#!/usr/bin/env python
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

from src.special import DTYPE, DEVICE
from src.rs_finiteT import solve_rs_finiteT, FiniteTRSParams
from src.finiteT_ais import estimate_Phi_ais_for_C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


sigma = 1.0
beta_max = 1.0

gammas = [1.0, 1.5, 2.0]
NM_for_gamma = {
    1.0: (30, 30),
    1.5: (30, 20),
    2.0: (40, 20),
}

abs_k_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

n_trials = 12
n_chains_base = 1500

gh_n = 100
rs_max_iter = 5000
rs_tol = 1e-13

qty_keys = ["v", "Qx", "qx", "Qy", "q1", "energy"]

print(f"Device: {DEVICE}", flush=True)
print(f"Gammas: {gammas}", flush=True)
print(f"|k| values: {abs_k_values}", flush=True)
print(flush=True)


def adaptive_ais_params(abs_k, N, M):
    n_temps = int(500 + 300 * abs_k)
    n_mcmc = max(3, int(2 + abs_k))
    n_chains = n_chains_base
    block_size = min(8, N - 1)
    return n_temps, n_mcmc, n_chains, block_size


all_data = {}

for gamma in gammas:
    N, M = NM_for_gamma[gamma]
    L = math.sqrt(N * M)
    print(f"\n{'='*60}", flush=True)
    print(f"gamma={gamma:.2f}, N={N}, M={M}, L={L:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    rs_results = {q: [] for q in qty_keys}
    ais_means = {q: [] for q in qty_keys}
    ais_stds = {q: [] for q in qty_keys}
    ais_ess_list = []

    for abs_k in abs_k_values:
        beta_min = abs_k * beta_max
        n_temps, n_mcmc, n_chains, block_size = adaptive_ais_params(abs_k, N, M)

        print(f"\n  |k|={abs_k:.2f} (n_temps={n_temps}, n_mcmc={n_mcmc})",
              flush=True)

        p = FiniteTRSParams(
            sigma=sigma, gamma=gamma, beta_max=beta_max, beta_min=beta_min,
            gh_n=gh_n, x_max=float(N), y_max=float(M),
            damping=0.10, max_iter=rs_max_iter, tol=rs_tol,
        )
        torch.cuda.synchronize()
        t0 = time.time()
        rs = solve_rs_finiteT(p, device=DEVICE)
        torch.cuda.synchronize()
        t_rs = time.time() - t0

        rv = {
            "v": rs.v, "Qx": rs.state.Qx, "qx": rs.state.qx,
            "Qy": rs.state.Qy, "q1": rs.state.q1, "energy": rs.energy,
        }
        for q in qty_keys:
            rs_results[q].append(rv[q])

        print(f"    [RS] conv={rs.converged} iter={rs.n_iter} "
              f"v={rs.v:+.6f} E={rs.energy:+.6f} time={t_rs:.1f}s", flush=True)

        rng = np.random.default_rng(42)
        trial_data = {q: [] for q in qty_keys}
        trial_ess = []

        for trial in range(n_trials):
            C = rng.standard_normal((N, M))
            torch.cuda.synchronize()
            t0 = time.time()
            ais = estimate_Phi_ais_for_C(
                C, sigma=sigma, beta_max=beta_max, beta_min=beta_min,
                n_chains=n_chains, n_temps=n_temps, n_mcmc=n_mcmc,
                schedule="sigmoid", proposal="block_dirichlet",
                block_size=block_size, pairwise_step=0.3,
                rng=np.random.default_rng(trial * 1000 + 7),
                device=DEVICE,
            )
            torch.cuda.synchronize()
            t_ais = time.time() - t0

            trial_data["v"].append(ais.Phi_over_L)
            trial_data["Qx"].append(ais.Qx)
            trial_data["qx"].append(ais.qx)
            trial_data["Qy"].append(ais.Qy)
            trial_data["q1"].append(ais.q1)
            trial_data["energy"].append(ais.energy)
            trial_ess.append(ais.ess_fraction)

        for q in qty_keys:
            arr = np.array(trial_data[q])
            ais_means[q].append(arr.mean())
            ais_stds[q].append(arr.std(ddof=1))

        ais_ess_list.append(np.mean(trial_ess))

        m_v = np.mean(trial_data["v"])
        m_e = np.mean(trial_data["energy"])
        s_v = np.std(trial_data["v"], ddof=1)
        print(f"    [AIS] v={m_v:+.6f}±{s_v:.4f} E={m_e:+.6f} "
              f"ESS={np.mean(trial_ess):.3f}", flush=True)

    all_data[gamma] = {
        "abs_k": np.array(abs_k_values),
        "N": N, "M": M, "L": L,
        "ais_ess": np.array(ais_ess_list),
    }
    for q in qty_keys:
        all_data[gamma][f"rs_{q}"] = np.array(rs_results[q])
        all_data[gamma][f"ais_mean_{q}"] = np.array(ais_means[q])
        all_data[gamma][f"ais_std_{q}"] = np.array(ais_stds[q])


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(REPO_ROOT, "fig")
data_dir = os.path.join(REPO_ROOT, "data")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "rs_vs_ais_k_dep_data.npz")
save_dict = {}
for gamma, d in all_data.items():
    prefix = f"g{gamma:.1f}_".replace(".", "p")
    for key, val in d.items():
        if isinstance(val, np.ndarray):
            save_dict[prefix + key] = val
        else:
            save_dict[prefix + key] = np.array([val])
np.savez(data_path, **save_dict)
print(f"\nData saved: {data_path}", flush=True)


def make_6panel(gamma_val, data, out_path):
    abs_k = data["abs_k"]
    N, M, L_val = data["N"], data["M"], data["L"]
    se_factor = 1.0 / math.sqrt(n_trials)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        rf"RS theory vs AIS:  $\gamma={gamma_val:.1f}$,  $N={N}$, $M={M}$,  "
        rf"$L={L_val:.1f}$",
        fontsize=20, y=0.98,
    )

    ax = axes[0, 0]
    ax.plot(abs_k, data["rs_v"], "-", color=colors[0], lw=2.0,
            label="RS theory", zorder=3)
    ax.errorbar(abs_k, data["ais_mean_v"],
                yerr=data["ais_std_v"] * se_factor,
                fmt=markers[0], color=colors[1], ms=MS,
                fillstyle=FILL, markeredgewidth=1.5,
                label=f"AIS ($N$={N})", zorder=4, **_err_kw(colors[1]))
    ax.set_title(r"$v = -g/\beta_{\min}$", fontsize=16, pad=8)
    _apply_style(ax, r"$|k|$", r"$v$")
    ax.legend(fontsize=FONT_LEGEND, loc="best")

    ax = axes[0, 1]
    ax.plot(abs_k, data["rs_energy"], "-", color=colors[0], lw=2.0,
            label="RS theory", zorder=3)
    ax.errorbar(abs_k, data["ais_mean_energy"],
                yerr=data["ais_std_energy"] * se_factor,
                fmt=markers[0], color=colors[1], ms=MS,
                fillstyle=FILL, markeredgewidth=1.5,
                label=f"AIS ($N$={N})", zorder=4, **_err_kw(colors[1]))
    ax.set_title(r"$\mathbb{E}[V]/L$", fontsize=16, pad=8)
    _apply_style(ax, r"$|k|$", r"$\mathbb{E}[V]/L$")
    ax.legend(fontsize=FONT_LEGEND, loc="best")

    ax = axes[0, 2]
    ax.plot(abs_k, data["ais_ess"], "-o", color=colors[4],
            ms=MS, fillstyle=FILL, markeredgewidth=1.5, lw=2.0)
    ax.axhline(0.1, ls=":", color="gray", alpha=0.5, label="ESS=0.1")
    ax.set_title("ESS", fontsize=16, pad=8)
    _apply_style(ax, r"$|k|$", "ESS fraction")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=FONT_LEGEND, loc="best")

    ax = axes[1, 0]
    for q, lab, c, mk in [
        ("Qx", r"$Q_x$", colors[0], markers[0]),
        ("qx", r"$q_x$", colors[2], markers[2]),
    ]:
        ax.plot(abs_k, data[f"rs_{q}"], "-", color=c, lw=2.0,
                label=f"{lab} RS", zorder=3)
        ax.errorbar(abs_k, data[f"ais_mean_{q}"],
                    yerr=data[f"ais_std_{q}"] * se_factor,
                    fmt=mk, color=c, ms=MS, fillstyle=FILL,
                    markeredgewidth=1.5,
                    label=f"{lab} AIS", zorder=4, **_err_kw(c))
    ax.set_title(r"$x$-side order parameters", fontsize=16, pad=8)
    _apply_style(ax, r"$|k|$", r"$Q_x,\; q_x$")
    ax.legend(fontsize=FONT_LEGEND, loc="best")

    ax = axes[1, 1]
    for q, lab, c, mk in [
        ("Qy", r"$Q_y$", colors[1], markers[1]),
        ("q1", r"$q_1$", colors[3], markers[3]),
    ]:
        ax.plot(abs_k, data[f"rs_{q}"], "-", color=c, lw=2.0,
                label=f"{lab} RS", zorder=3)
        ax.errorbar(abs_k, data[f"ais_mean_{q}"],
                    yerr=data[f"ais_std_{q}"] * se_factor,
                    fmt=mk, color=c, ms=MS, fillstyle=FILL,
                    markeredgewidth=1.5,
                    label=f"{lab} AIS", zorder=4, **_err_kw(c))
    ax.set_title(r"$y$-side order parameters", fontsize=16, pad=8)
    _apply_style(ax, r"$|k|$", r"$Q_y,\; q_1$")
    ax.legend(fontsize=FONT_LEGEND, loc="best")

    ax = axes[1, 2]
    for i, (q, c, mk) in enumerate(zip(
        ["v", "Qx", "energy"],
        [colors[0], colors[1], colors[3]],
        markers[:3],
    )):
        rs_v = data[f"rs_{q}"]
        ais_m = data[f"ais_mean_{q}"]
        rel = np.abs(ais_m - rs_v) / np.maximum(np.abs(rs_v), 1e-10)
        ax.plot(abs_k, rel, f"-{mk}", color=c, ms=MS-2,
                fillstyle=FILL, markeredgewidth=1.2, lw=1.5,
                label=q)
    ax.set_yscale("log")
    ax.set_title(r"|AIS$-$RS|/|RS|", fontsize=16, pad=8)
    _apply_style(ax, r"$|k|$", "Relative error")
    ax.legend(fontsize=FONT_LEGEND, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


for gamma in gammas:
    out_path = os.path.join(out_dir, f"rs_vs_ais_k_dep_gamma{gamma:.1f}.pdf")
    make_6panel(gamma, all_data[gamma], out_path)

se_factor = 1.0 / math.sqrt(n_trials)

fig, ax = plt.subplots(figsize=(8, 6))
for i, gamma in enumerate(gammas):
    d = all_data[gamma]
    abs_k = d["abs_k"]
    N = d["N"]
    c = colors[i]
    mk = markers[i]
    ax.plot(abs_k, d["rs_v"], "-", color=c, lw=2.0,
            label=rf"RS $\gamma={gamma:.1f}$", zorder=3)
    ax.errorbar(abs_k, d["ais_mean_v"],
                yerr=d["ais_std_v"] * se_factor,
                fmt=mk, color=c, ms=MS, fillstyle=FILL,
                markeredgewidth=1.5, zorder=4,
                label=rf"AIS $\gamma={gamma:.1f}$ ($N$={N})",
                **_err_kw(c))
_apply_style(ax, r"$|k| = \beta_{\min}/\beta_{\max}$",
             r"$v = -g / \beta_{\min}$")
ax.legend(fontsize=FONT_LEGEND, loc="best")
ax.set_title(r"RS vs AIS: $v$ vs $|k|$", fontsize=18)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "rs_vs_ais_v_vs_k_all_gammas.pdf"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: rs_vs_ais_v_vs_k_all_gammas.pdf", flush=True)

fig, ax = plt.subplots(figsize=(8, 6))
for i, gamma in enumerate(gammas):
    d = all_data[gamma]
    abs_k = d["abs_k"]
    N = d["N"]
    c = colors[i]
    mk = markers[i]
    ax.plot(abs_k, d["rs_energy"], "-", color=c, lw=2.0,
            label=rf"RS $\gamma={gamma:.1f}$", zorder=3)
    ax.errorbar(abs_k, d["ais_mean_energy"],
                yerr=d["ais_std_energy"] * se_factor,
                fmt=mk, color=c, ms=MS, fillstyle=FILL,
                markeredgewidth=1.5, zorder=4,
                label=rf"AIS $\gamma={gamma:.1f}$ ($N$={N})",
                **_err_kw(c))
_apply_style(ax, r"$|k| = \beta_{\min}/\beta_{\max}$",
             r"$\mathbb{E}[V]/L$")
ax.legend(fontsize=FONT_LEGEND, loc="best")
ax.set_title(r"RS vs AIS: $\mathbb{E}[V]/L$ vs $|k|$", fontsize=18)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "rs_vs_ais_energy_vs_k_all_gammas.pdf"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: rs_vs_ais_energy_vs_k_all_gammas.pdf", flush=True)

print("\nALL DONE", flush=True)
