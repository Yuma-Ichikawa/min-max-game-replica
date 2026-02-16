from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import root
from scipy.special import ndtr


_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _phi_scalar(x: float) -> float:
    return float(np.exp(-0.5 * x * x - _LOG_SQRT_2PI))


def Phi(x) -> np.ndarray:
    return ndtr(x)


def A(alpha: float) -> float:
    alpha = float(alpha)
    return alpha * float(Phi(alpha)) + _phi_scalar(alpha)


def B(alpha: float) -> float:
    alpha = float(alpha)
    return (alpha * alpha + 1.0) * float(Phi(alpha)) + alpha * _phi_scalar(alpha)


def q_of(alpha: float) -> float:
    a = A(alpha)
    b = B(alpha)
    if a < 1e-50:
        return 1e30
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
