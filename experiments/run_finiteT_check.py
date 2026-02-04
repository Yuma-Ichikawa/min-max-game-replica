"""
Finite-temperature comparison: RS saddle vs finite-size Monte Carlo.

Example:
  python -m experiments.run_finiteT_check --gamma 0.5 --beta_max 1 --beta_min 1 --sigma 1 \
    --N 40 --M 80 --x_samples 2000 --trials 10 --seed 0
"""
from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from src.rs_finiteT import FiniteTRSParams, solve_rs_finiteT
from src.finiteT_mc import estimate_Phi_for_C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, required=True, help="gamma=N/M (used for RS theory).")
    ap.add_argument("--beta_max", type=float, required=True)
    ap.add_argument("--beta_min", type=float, required=True)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--N", type=int, required=True, help="Finite-size N (x dimension).")
    ap.add_argument("--M", type=int, required=True, help="Finite-size M (y dimension).")
    ap.add_argument("--x_samples", type=int, default=2000, help="Outer MC samples for x integral.")
    ap.add_argument("--trials", type=int, default=10, help="Number of random matrices.")
    ap.add_argument("--gh_n", type=int, default=40, help="Gauss-Hermite order for RS saddle.")
    ap.add_argument("--y_max", type=float, default=None, help="Upper bound for y-site integral in RS solver; default=M.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose_saddle", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    gamma_eff = args.gamma
    if args.y_max is None:
        y_max = float(args.M)
    else:
        y_max = float(args.y_max)

    # --- RS saddle ---
    rs_params = FiniteTRSParams(
        sigma=args.sigma,
        gamma=gamma_eff,
        beta_max=args.beta_max,
        beta_min=args.beta_min,
        gh_n=args.gh_n,
        y_max=y_max,
        damping=0.2,
        max_iter=300,
        tol=1e-8,
    )
    rs = solve_rs_finiteT(rs_params, verbose=args.verbose_saddle)
    print("\n=== RS saddle (finite-T) ===")
    print(f"converged={rs.converged} in {rs.n_iter} iter")
    print(f"v_RS = {rs.v:.6f}  (g={rs.g:.6f}, k={rs.k:.3f})")
    print(f"residuals = {rs.residuals}")

    # --- finite-size MC ---
    v_samples = []
    for t in tqdm(range(args.trials), desc=f"finite-size MC (N={args.N},M={args.M})"):
        C = rng.standard_normal(size=(args.N, args.M))
        mc = estimate_Phi_for_C(
            C,
            sigma=args.sigma,
            beta_max=args.beta_max,
            beta_min=args.beta_min,
            x_samples=args.x_samples,
            rng=rng,
        )
        v_samples.append(mc.Phi_over_L)

    v_samples = np.array(v_samples, dtype=float)
    print("\n=== Finite-size Monte Carlo ===")
    print(f"mean(Phi/L) = {v_samples.mean():.6f}  ± {v_samples.std(ddof=1):.6f}   (over {args.trials} trials)")
    print("\nDifference (MC - RS) = {:.6f}".format(v_samples.mean() - rs.v))


if __name__ == "__main__":
    main()
