"""
Simplex utilities: sampling and volumes.

We use the (N-1)-simplex of "mass" N:
    X = {x_i >= 0, Σ_i x_i = N}.

Lebesgue measure on this set has volume:
    Vol(X) = N^{N-1} / (N-1)!.

Uniform sampling w.r.t. the normalized Lebesgue measure is equivalent to:
    p ~ Dirichlet(1,...,1),  x = N p.

We use the exponential trick:
    g_i ~ Exp(1), p_i = g_i / Σ g_i.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import gammaln


def log_simplex_volume(dim: int, mass: float) -> float:
    """
    log volume of {x>=0, sum x = mass} in R^dim, with dim >= 1.

    Here dim is the number of coordinates (N), so geometric dimension is dim-1.

    Vol = mass^(dim-1) / (dim-1)! = mass^(dim-1) / Γ(dim).

    Returns log Vol.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    mass = float(mass)
    if mass <= 0:
        raise ValueError("mass must be > 0")
    return (dim - 1) * np.log(mass) - gammaln(dim)


def sample_uniform_simplex(
    dim: int,
    mass: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample uniformly from the simplex {x>=0, sum x = mass} (Lebesgue-uniform).

    Returns array of shape (n_samples, dim).
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    mass = float(mass)
    if mass <= 0:
        raise ValueError("mass must be > 0")

    g = rng.exponential(scale=1.0, size=(n_samples, dim))
    p = g / g.sum(axis=1, keepdims=True)
    return mass * p
