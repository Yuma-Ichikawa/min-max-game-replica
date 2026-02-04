"""
Gauss–Hermite quadrature utilities.

We frequently need integrals over the standard normal measure

    ∫ Dz f(z),    Dz = (2π)^(-1/2) exp(-z^2/2) dz.

Gauss–Hermite quadrature (GH) provides nodes/weights for

    ∫_{-∞}^{∞} exp(-x^2) g(x) dx ≈ Σ w_i g(x_i).

Using z = √2 x, we have:

    ∫ Dz f(z) = (1/√π) ∫ exp(-x^2) f(√2 x) dx
              ≈ Σ (w_i/√π) f(√2 x_i).

This module returns nodes z_i and weights w_i such that:

    ∫ Dz f(z) ≈ Σ w_i f(z_i),
    Σ w_i = 1 (up to numerical error).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss


@dataclass(frozen=True)
class GHQuadrature:
    """Nodes/weights for standard-normal expectation."""
    z: np.ndarray
    w: np.ndarray

    def check(self) -> None:
        if self.z.ndim != 1 or self.w.ndim != 1:
            raise ValueError("z and w must be 1D arrays")
        if self.z.shape != self.w.shape:
            raise ValueError("z and w must have the same shape")


def gh_standard_normal(n: int, dtype=np.float64) -> GHQuadrature:
    """
    Return Gauss–Hermite nodes/weights (z, w) for ∫ Dz f(z).

    Parameters
    ----------
    n : int
        Quadrature order (typically 20–80).
    dtype :
        Floating dtype (float64 recommended; longdouble for extra safety).

    Returns
    -------
    GHQuadrature
        z: nodes (shape (n,))
        w: weights (shape (n,)) so that sum(w)=1 and ∫ Dz f ≈ sum(w*f(z))
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    x, w = hermgauss(n)  # for ∫ e^{-x^2} g(x) dx
    z = (np.sqrt(2.0) * x).astype(dtype, copy=False)
    w = (w / np.sqrt(np.pi)).astype(dtype, copy=False)  # for standard normal
    return GHQuadrature(z=z, w=w)
