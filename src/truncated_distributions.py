"""
Moments and log-partition functions for truncated distributions used in the RS saddle.

We need:
1) x-sector: density proportional to exp(-A/2 x^2 + B x) on x ∈ [0, ∞)
   (a shifted/truncated Gaussian).

2) y-sector (at the finite-temperature RS saddle): hatQ_y=0 so the on-site
   integral is a truncated exponential on [0, y_max]:
        ∫_0^{y_max} exp(h y) dy
   plus its first two moments under the normalized density.

We implement numerically stable expressions, including asymptotic branches for
very large |u| = |h| y_max.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import log_ndtr

from .special import LOG_SQRT_2PI


@dataclass(frozen=True)
class TruncStats:
    logZ: np.ndarray
    mean: np.ndarray
    second: np.ndarray


def trunc_gauss_lower0(A: float, B: np.ndarray) -> TruncStats:
    """
    For density proportional to exp(-A/2 x^2 + B x) on x >= 0, with A>0.

    Interpreting:
        -A/2 x^2 + B x = -A/2 (x - μ)^2 + B^2/(2A),
        μ = B/A,  σ^2 = 1/A.

    Then x is Normal(μ, σ^2) truncated to [0, ∞), and:

        Z = exp(B^2/(2A)) * sqrt(2π/A) * Φ(B/√A).

    Returns logZ, mean, second moment (E[x^2]) of the normalized density.
    """
    if A <= 0:
        raise ValueError("A must be positive for trunc_gauss_lower0 with upper=∞.")
    B = np.asarray(B, dtype=float)
    sqrtA = np.sqrt(A)

    mu = B / A
    sigma = 1.0 / sqrtA

    # α = (0 - μ)/σ = -B/√A
    alpha = -B / sqrtA

    # Z0 = P[X >= 0] = 1 - Φ(alpha) = Φ(-alpha) = Φ(B/√A)
    logZ0 = log_ndtr(-alpha)

    logZ = (B * B) / (2.0 * A) + 0.5 * np.log(2.0 * np.pi) - 0.5 * np.log(A) + logZ0

    # λ = φ(alpha) / Z0
    logphi = -0.5 * alpha * alpha - LOG_SQRT_2PI
    lam = np.exp(logphi - logZ0)

    mean = mu + sigma * lam
    var = (sigma * sigma) * (1.0 + alpha * lam - lam * lam)
    second = var + mean * mean

    return TruncStats(logZ=logZ, mean=mean, second=second)


def trunc_exp_0_ymax(h: np.ndarray, y_max: float) -> TruncStats:
    r"""
    For density proportional to exp(h y) on y ∈ [0, y_max].

    Partition function:
        Z = (exp(u) - 1)/h,  u = h y_max.
    A numerically stable form is:
        Z = y_max * (exp(u) - 1)/u.

    Moments (exact):
        E[y]   = y_max * [ (u-1) exp(u) + 1 ] / [ u (exp(u) - 1) ]
        E[y^2] = y_max^2 * [ (u^2 - 2u + 2) exp(u) - 2 ] / [ u^2 (exp(u) - 1) ]

    We use:
    - a Taylor branch for |u| small,
    - asymptotic branches for u very large positive/negative,
    - the exact formulas in the moderate regime using expm1(u).
    """
    h = np.asarray(h, dtype=float)
    y_max = float(y_max)
    u = h * y_max

    logZ = np.empty_like(u)
    mean = np.empty_like(u)
    second = np.empty_like(u)

    small = np.abs(u) < 1e-6
    large_pos = u > 50.0
    large_neg = u < -50.0
    mid = ~(small | large_pos | large_neg)

    # small-u Taylor expansions:
    #   Z ≈ y_max (1 + u/2 + u^2/6 + u^3/24 + u^4/120)
    #   mean/y_max ≈ 1/2 + u/12 - u^3/720
    #   second/y_max^2 ≈ 1/3 + u/12 + u^2/60 - u^4/2520
    if np.any(small):
        us = u[small]
        Zs = y_max * (1.0 + us/2.0 + us*us/6.0 + us**3/24.0 + us**4/120.0)
        logZ[small] = np.log(Zs)
        mean[small] = y_max * (0.5 + us/12.0 - us**3/720.0)
        second[small] = (y_max**2) * (1.0/3.0 + us/12.0 + us*us/60.0 - us**4/2520.0)

    # large u>0: exp(u) >> 1
    #   Z ≈ y_max exp(u)/u, mean ≈ y_max (1 - 1/u), second ≈ y_max^2 (1 - 2/u + 2/u^2)
    if np.any(large_pos):
        up = u[large_pos]
        logZ[large_pos] = np.log(y_max) + up - np.log(up)
        mean[large_pos] = y_max * (1.0 - 1.0/up)
        second[large_pos] = (y_max**2) * (1.0 - 2.0/up + 2.0/(up*up))

    # large u<0: exp(u) ~ 0
    #   Z ≈ y_max/(-u), mean ≈ y_max/(-u), second ≈ 2 y_max^2 / u^2
    if np.any(large_neg):
        un = u[large_neg]
        logZ[large_neg] = np.log(y_max) - np.log(-un)
        mean[large_neg] = y_max/(-un)
        second[large_neg] = 2.0 * (y_max**2) / (un*un)

    # moderate u: exact with expm1
    if np.any(mid):
        um = u[mid]
        den = np.expm1(um)          # exp(u) - 1
        eu = den + 1.0              # exp(u)
        logZ[mid] = np.log(y_max) + np.log(np.abs(den)) - np.log(np.abs(um))
        mean[mid] = y_max * (((um - 1.0)*eu + 1.0) / (um * den))
        second[mid] = (y_max**2) * ((((um*um - 2.0*um + 2.0)*eu - 2.0) / (um*um * den)))

    return TruncStats(logZ=logZ, mean=mean, second=second)
