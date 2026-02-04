"""
Finite-temperature finite-size numerics.

We estimate, for a given random matrix C (N×M) with i.i.d. N(0,1):

    Φ = -(1/β_min) log ∫_{x∈X} dx [ Z_y(x) ]^k,
    k = -β_min/β_max,

where:
    Z_y(x) = ∫_{y∈Y} dy exp( β_max V(x,y) ),
    V(x,y) = sqrt(σ/L) Σ_{i,j} C_{ij} x_i y_j,
    L = sqrt(N M),
    X: {x>=0, sum x = N},
    Y: {y>=0, sum y = M}.

Strategy:
- sample x uniformly on X (Dirichlet trick),
- compute Z_y(x) exactly using divided differences,
- Monte Carlo estimate the outer integral.

Important:
- The inner integral is exact (up to floating-point), but the outer integral is Monte Carlo.
- Results fluctuate strongly for small N,M and/or small x_samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.special import logsumexp

from .divided_differences import simplex_exp_integral_divdiff
from .simplex import sample_uniform_simplex, log_simplex_volume


@dataclass(frozen=True)
class FiniteTMCResult:
    Phi: float
    Phi_over_L: float
    logZ_outer: float
    mc_log_mean_weight: float


def estimate_Phi_for_C(
    C: np.ndarray,
    *,
    sigma: float,
    beta_max: float,
    beta_min: float,
    x_samples: int,
    rng: np.random.Generator,
    use_longdouble: bool = True,
) -> FiniteTMCResult:
    """
    Monte Carlo estimate of Φ for a single matrix C.

    Returns Φ, Φ/L and some diagnostics.
    """
    C = np.asarray(C, dtype=float)
    N, M = C.shape
    L = float(np.sqrt(N * M))
    k = -beta_min / beta_max

    # sample x uniformly on simplex sum=N
    X = sample_uniform_simplex(dim=N, mass=float(N), n_samples=x_samples, rng=rng)

    # compute log weights: log w = k log Z_y(x)
    logw = np.empty(x_samples, dtype=float)

    # precompute scale for a_j
    scale = beta_max * np.sqrt(sigma / L)

    for t in range(x_samples):
        x = X[t]
        # a_j = scale * Σ_i C_{ij} x_i  = scale * (x^T C)_j
        a = scale * (x @ C)  # shape (M,)
        # Z_y(x) = DD_{a_1,...,a_M} exp(M t) (mass=M)
        res = simplex_exp_integral_divdiff(a, mass=float(M), use_longdouble=use_longdouble)
        logw[t] = k * res.logZ

    # outer integral: ∫_X dx w(x) = Vol(X) * E_uniform[w]
    logVolX = log_simplex_volume(dim=N, mass=float(N))
    mc_log_mean_w = float(logsumexp(logw) - np.log(x_samples))
    logZ_outer = float(logVolX + mc_log_mean_w)

    Phi = -(1.0 / beta_min) * logZ_outer
    Phi_over_L = Phi / L
    return FiniteTMCResult(Phi=Phi, Phi_over_L=Phi_over_L, logZ_outer=logZ_outer, mc_log_mean_weight=mc_log_mean_w)
