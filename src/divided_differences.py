"""
Exact simplex integral of exp(linear form) via divided differences.

We need, for a vector a ∈ R^M, and a "mass" parameter m>0:

    Z = ∫_{y_j >= 0} δ(m - Σ_j y_j) exp( Σ_j a_j y_j ) dy.

For distinct a_j, one has the identity (see manuscript Appendix; also a classical result):

    Z = DD_{a_1,...,a_M} [ exp(m t) ],

i.e. the (M-1)-th divided difference of the function t ↦ exp(m t) at points a_j.

A stable O(M^2) recursion computes this divided difference:

    d_j^{(0)} = exp(m a_j)
    d_j^{(r)} = ( d_{j+1}^{(r-1)} - d_j^{(r-1)} ) / ( a_{j+r} - a_j )
    Z = d_1^{(M-1)}

We implement a numerically robust version:
- sort points a_j,
- shift by max(a) to reduce overflow,
- optionally use longdouble for wider exponent range,
- add a tiny jitter if points are nearly identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DivDiffResult:
    logZ: float
    Z: float
    used_jitter: bool


def simplex_exp_integral_divdiff(
    a: np.ndarray,
    mass: float,
    *,
    use_longdouble: bool = True,
    jitter_tol: float = 1e-12,
    jitter_scale: float = 1e-9,
) -> DivDiffResult:
    """
    Compute Z = ∫ δ(mass - sum y) exp(a·y) dy exactly via divided differences.

    Parameters
    ----------
    a : array_like, shape (M,)
        Coefficients in the exponent.
    mass : float
        The simplex mass (in the manuscript, mass=M).
    use_longdouble : bool
        Use np.longdouble for extra range/precision.
    jitter_tol : float
        If min spacing between sorted a is below this, add a small jitter.
    jitter_scale : float
        Relative jitter scale (multiplied by max(1,std(a))).

    Returns
    -------
    DivDiffResult
        logZ, Z, and whether jitter was applied.
    """
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("a must be a 1D array")
    M = a.size
    if M < 1:
        raise ValueError("a must have length >= 1")
    mass = float(mass)
    if mass <= 0:
        raise ValueError("mass must be positive")

    dtype = np.longdouble if use_longdouble else np.float64
    a = a.astype(dtype, copy=False)

    # sort a for recursion stability
    idx = np.argsort(a)
    a = a[idx]

    used_jitter = False
    if M >= 2:
        diffs = np.diff(a)
        min_diff = np.min(np.abs(diffs))
        if min_diff < jitter_tol:
            used_jitter = True
            scale = max(1.0, float(np.std(a)))
            eps = dtype(jitter_scale * scale)
            # deterministic, symmetric jitter
            jitter = eps * (np.arange(M, dtype=dtype) - (M - 1) / 2.0) / M
            a = a + jitter

    a_max = np.max(a)
    a_shift = a - a_max

    # d^{(0)}
    d = np.exp(dtype(mass) * a_shift)

    # recursion
    for r in range(1, M):
        denom = a[r:] - a[:-r]
        num = d[1:] - d[:-1]
        d = num / denom

    Z_scaled = d[0]
    # Z = exp(mass*a_max) * Z_scaled
    # Z_scaled should be >0; handle small negative from roundoff
    if float(Z_scaled) <= 0.0:
        # The true quantity is positive; a small negative indicates cancellation/roundoff.
        Z_scaled = np.abs(Z_scaled)

    logZ = float(dtype(mass) * a_max + np.log(Z_scaled))
    Z = float(np.exp(logZ))  # may overflow; user can rely on logZ
    return DivDiffResult(logZ=logZ, Z=Z, used_jitter=used_jitter)
