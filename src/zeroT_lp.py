from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class LPResult:
    value: float
    p: Optional[np.ndarray]
    q: Optional[np.ndarray]


def solve_minmax_lp(C: np.ndarray, *, return_strategies: bool = True) -> LPResult:
    C = np.asarray(C, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be a 2D array")
    N, M = C.shape

    c = np.zeros(N + 1)
    c[-1] = 1.0

    A_ub = np.hstack([C.T, -np.ones((M, 1))])
    b_ub = np.zeros(M)

    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * N + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Primal LP failed: {res.message}")

    p = res.x[:N]
    value = float(res.x[-1])

    q = None
    if return_strategies:
        c2 = np.zeros(M + 1)
        c2[-1] = -1.0

        A_ub2 = np.hstack([-C, np.ones((N, 1))])
        b_ub2 = np.zeros(N)

        A_eq2 = np.zeros((1, M + 1))
        A_eq2[0, :M] = 1.0
        b_eq2 = np.array([1.0])

        bounds2 = [(0.0, None)] * M + [(None, None)]
        res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
        if not res2.success:
            raise RuntimeError(f"Dual LP failed: {res2.message}")
        q = res2.x[:M]

    return LPResult(value=value, p=p if return_strategies else None, q=q)


def support_fraction(v: np.ndarray, tol: float = 1e-10) -> float:
    v = np.asarray(v)
    return float(np.mean(v > tol))


def second_moment_scaled(p: np.ndarray, mass: float) -> float:
    p = np.asarray(p, dtype=float)
    x = mass * p
    return float(np.mean(x * x))
