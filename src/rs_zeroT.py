"""
Zero-temperature RS theory (α-equations) for the Gaussian matrix game.

We solve for α_x, α_y (real scalars) from:

    Φ(α_y) = γ Φ(α_x),
    γ^{1/2} α_x √q(α_y) + α_y √q(α_x) = 0,

where:

    A(α) = ∫ Dz (z+α)_+ = α Φ(α) + φ(α),
    B(α) = ∫ Dz (z+α)_+^2 = (α^2+1) Φ(α) + α φ(α),
    q(α) = B(α)/A(α)^2.

The predicted value scaling is:

    f(γ) = 1/2 ( γ^{-1/4} α_x √q(α_y) - γ^{1/4} α_y √q(α_x) ),

so that for σ=1:
    E0/L ≈ f(γ),
and for general σ:
    E0/L ≈ √σ f(γ).

This matches the manuscript.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import root
from scipy.special import ndtr

from .special import phi


def Phi(x: np.ndarray) -> np.ndarray:
    return ndtr(x)


def A(alpha: float) -> float:
    alpha = float(alpha)
    return alpha * float(Phi(alpha)) + float(phi(alpha))


def B(alpha: float) -> float:
    alpha = float(alpha)
    return (alpha * alpha + 1.0) * float(Phi(alpha)) + alpha * float(phi(alpha))


def q_of(alpha: float) -> float:
    a = A(alpha)
    b = B(alpha)
    return b / (a * a)


@dataclass(frozen=True)
class ZeroTRSResult:
    gamma: float
    alpha_x: float
    alpha_y: float
    f: float
    rho_x: float
    rho_y: float
    qx: float
    qy: float


def solve_zeroT_rs(gamma: float, *, x0: Tuple[float, float] = (0.2, -0.2)) -> ZeroTRSResult:
    """
    Solve the RS zero-temperature system for a given gamma = N/M > 0.

    Parameters
    ----------
    gamma : float
        Aspect ratio.
    x0 : (float, float)
        Initial guess for (alpha_x, alpha_y).
        For gamma<1, alpha_x>0, alpha_y<0 typically.

    Returns
    -------
    ZeroTRSResult
    """
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    sqrt_gamma = np.sqrt(gamma)

    def F(u):
        ax, ay = u
        eq1 = Phi(ay) - gamma * Phi(ax)
        eq2 = sqrt_gamma * ax * np.sqrt(q_of(ay)) + ay * np.sqrt(q_of(ax))
        return np.array([eq1, eq2], dtype=float)

    sol = root(F, np.array(x0, dtype=float), method="hybr")
    if not sol.success:
        # Try a couple of alternative initializations
        for guess in [(0.5, -0.5), (1.0, -1.0), (0.1, -0.5), (0.5, -0.1)]:
            sol = root(F, np.array(guess, dtype=float), method="hybr")
            if sol.success:
                break
    if not sol.success:
        raise RuntimeError(f"RS zero-T solver failed: {sol.message}")

    ax, ay = map(float, sol.x)
    qx = q_of(ax)
    qy = q_of(ay)
    f = 0.5 * (gamma**(-0.25) * ax * np.sqrt(qy) - gamma**(0.25) * ay * np.sqrt(qx))
    rho_x = float(Phi(ax))
    rho_y = float(Phi(ay))
    return ZeroTRSResult(gamma=gamma, alpha_x=ax, alpha_y=ay, f=f, rho_x=rho_x, rho_y=rho_y, qx=qx, qy=qy)
