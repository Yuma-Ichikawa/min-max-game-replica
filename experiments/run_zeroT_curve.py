"""
Run a comparison between zero-temperature RS theory and finite-size LP numerics.

Example:
    python -m experiments.run_zeroT_curve --gammas 0.3 0.5 0.8 1.0 1.3 --trials 50 --base_M 80 --seed 0
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.rs_zeroT import solve_zeroT_rs
from src.zeroT_lp import solve_minmax_lp, support_fraction, second_moment_scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", nargs="+", type=float, required=True, help="List of gamma=N/M values.")
    ap.add_argument("--trials", type=int, default=50, help="Number of random matrices per gamma.")
    ap.add_argument("--base_M", type=int, default=80, help="Base M; N is round(gamma*M).")
    ap.add_argument("--sigma", type=float, default=1.0, help="Sigma scaling (theory multiplies by sqrt(sigma)).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_figs", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    rows = []
    for gamma in args.gammas:
        M = int(args.base_M)
        N = int(round(gamma * M))
        gamma_eff = N / M

        theory = solve_zeroT_rs(gamma_eff)

        f_samples = []
        rho_x_s = []
        rho_y_s = []
        qx_s = []
        qy_s = []

        for _ in tqdm(range(args.trials), desc=f"gamma={gamma_eff:.3f} (N={N},M={M})"):
            C = rng.standard_normal(size=(N, M))
            lp = solve_minmax_lp(C, return_strategies=True)
            t = lp.value
            f_L = (N * M) ** 0.25 * t  # matches manuscript
            f_samples.append(f_L)

            p = lp.p
            q = lp.q
            rho_x_s.append(support_fraction(p))
            rho_y_s.append(support_fraction(q))
            qx_s.append(second_moment_scaled(p, mass=float(N)))
            qy_s.append(second_moment_scaled(q, mass=float(M)))

        row = dict(
            gamma=gamma_eff,
            N=N,
            M=M,
            f_theory=np.sqrt(args.sigma) * theory.f,
            f_emp_mean=float(np.mean(f_samples)) * np.sqrt(args.sigma),
            f_emp_std=float(np.std(f_samples, ddof=1)) * np.sqrt(args.sigma),
            rho_x_theory=theory.rho_x,
            rho_x_emp=float(np.mean(rho_x_s)),
            rho_y_theory=theory.rho_y,
            rho_y_emp=float(np.mean(rho_y_s)),
            qx_theory=theory.qx,
            qx_emp=float(np.mean(qx_s)),
            qy_theory=theory.qy,
            qy_emp=float(np.mean(qy_s)),
        )
        rows.append(row)

    # pretty print
    print("\n=== Zero-T RS theory vs LP numerics ===")
    for r in rows:
        print(
            f"gamma={r['gamma']:.3f} (N={r['N']}, M={r['M']}): "
            f"f_theory={r['f_theory']:.4f} | f_emp={r['f_emp_mean']:.4f} ± {r['f_emp_std']:.4f} ; "
            f"rho_x (th/emp)={r['rho_x_theory']:.3f}/{r['rho_x_emp']:.3f} ; "
            f"rho_y (th/emp)={r['rho_y_theory']:.3f}/{r['rho_y_emp']:.3f}"
        )

    if args.save_figs:
        gam = np.array([r["gamma"] for r in rows])
        f_th = np.array([r["f_theory"] for r in rows])
        f_emp = np.array([r["f_emp_mean"] for r in rows])
        f_std = np.array([r["f_emp_std"] for r in rows])

        plt.figure()
        plt.errorbar(gam, f_emp, yerr=f_std, fmt="o", label="LP (finite-size)")
        plt.plot(gam, f_th, "-", label="RS theory")
        plt.xlabel(r"$\gamma=N/M$")
        plt.ylabel(r"$E_0/L \;\;(\sigma^{1/2} f(\gamma))$")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/zeroT_value_curve.pdf")

        plt.figure()
        plt.plot(gam, [r["rho_x_emp"] for r in rows], "o", label=r"$\rho_x$ emp")
        plt.plot(gam, [r["rho_x_theory"] for r in rows], "-", label=r"$\rho_x$ theory")
        plt.plot(gam, [r["rho_y_emp"] for r in rows], "s", label=r"$\rho_y$ emp")
        plt.plot(gam, [r["rho_y_theory"] for r in rows], "--", label=r"$\rho_y$ theory")
        plt.xlabel(r"$\gamma$")
        plt.ylabel("support fraction")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/zeroT_support.pdf")

        print("\nSaved figures to figures/")

if __name__ == "__main__":
    main()
