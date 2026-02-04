"""
Finite-temperature RS saddle solver.

We implement the RS/1RSB saddle-point equations presented in the manuscript
for the two-temperature free energy density.

Unknowns:
    Qx, qx, Qy, q1, q0, mx, my.

Conjugates (from stationarity of the variational functional):
    χx  = (σ β_max^2 / √γ) k^2 q0
    χ0  = (√γ σ β_max^2) qx
    χ1  = (√γ σ β_max^2) (Qx - qx)
    hatQy = 0
    hatQx = χx - (σ β_max^2/√γ) [k Qy + k(k-1) q1]

Site measures:
- x-sector: proportional to exp(-hatQx/2 x^2 + (mx + √χx z) x) on x ≥ 0
- y-sector: proportional to exp( (my + √χ0 z + √χ1 η) y ) on y ∈ [0, y_max]
  and the η-measure is reweighted by [Z_y(z,η)]^k.

We solve the coupled moment constraints by:
- 1D bracketing solve for mx and my to enforce E[x]=1 and E[y]=1,
- damped fixed-point iteration for overlaps (Qx,qx,Qy,q1,q0).

The returned free energy density is:
    v = -(1/β_min) g_*,
where g_* is the variational functional evaluated at the saddle.

Notes
-----
1) The y-upper-bound y_max should be taken large (or set to the finite-size M)
   because hatQy=0 at the saddle. In practice, using y_max=M for comparisons
   to finite-size experiments is natural.

2) This solver assumes hatQx > 0 (so the x-site integral is convergent on [0,∞)).
   In regimes where a bounded x-domain is essential, the theory needs careful
   treatment; we raise a clear error in that case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .quadrature import gh_standard_normal, GHQuadrature
from .truncated_distributions import trunc_gauss_lower0, trunc_exp_0_ymax


@dataclass(frozen=True)
class FiniteTRSParams:
    sigma: float = 1.0
    gamma: float = 0.5
    beta_max: float = 1.0
    beta_min: float = 1.0
    gh_n: int = 40
    y_max: float = 80.0  # set to M for finite-size comparisons
    damping: float = 0.2
    max_iter: int = 200
    tol: float = 1e-8


@dataclass(frozen=True)
class FiniteTRSState:
    Qx: float
    qx: float
    Qy: float
    q1: float
    q0: float
    mx: float
    my: float


@dataclass(frozen=True)
class FiniteTRSResult:
    params: FiniteTRSParams
    state: FiniteTRSState
    k: float
    hatQx: float
    hatchix: float
    hatchi0: float
    hatchi1: float
    g: float
    v: float
    residuals: np.ndarray
    n_iter: int
    converged: bool


def _conjugates(state: FiniteTRSState, p: FiniteTRSParams) -> Tuple[float, float, float, float, float]:
    """
    Return (k, hatQx, χx, χ0, χ1).
    """
    sigma = p.sigma
    gamma = p.gamma
    beta_max = p.beta_max
    beta_min = p.beta_min
    k = -beta_min / beta_max
    sqrt_gamma = np.sqrt(gamma)

    Qx, qx, Qy, q1, q0 = state.Qx, state.qx, state.Qy, state.q1, state.q0

    chi_x = (sigma * beta_max**2 / sqrt_gamma) * (k**2) * q0
    chi0 = (sqrt_gamma * sigma * beta_max**2) * qx
    chi1 = (sqrt_gamma * sigma * beta_max**2) * (Qx - qx)

    hatQx = chi_x - (sigma * beta_max**2 / sqrt_gamma) * (k * Qy + k * (k - 1.0) * q1)
    return k, hatQx, chi_x, chi0, chi1


def _x_moments(mx: float, state: FiniteTRSState, p: FiniteTRSParams, gh: GHQuadrature) -> Tuple[float, float, float, float]:
    """
    Compute:
        Ex = ∫ Dz E[x|z],
        Ex2 = ∫ Dz E[x^2|z],
        Exm2 = ∫ Dz (E[x|z])^2,
        logZx_int = ∫ Dz log Z_x(z).
    """
    k, hatQx, chi_x, _, _ = _conjugates(state, p)
    if hatQx <= 0:
        raise RuntimeError(f"hatQx must be positive for x-integral on [0,∞); got {hatQx:g}.")

    z = gh.z
    w = gh.w

    B = mx + np.sqrt(chi_x) * z
    stats = trunc_gauss_lower0(hatQx, B)

    Ex = float(np.sum(w * stats.mean))
    Ex2 = float(np.sum(w * stats.second))
    Exm2 = float(np.sum(w * (stats.mean * stats.mean)))
    logZx_int = float(np.sum(w * stats.logZ))
    return Ex, Ex2, Exm2, logZx_int


def _y_moments(my: float, state: FiniteTRSState, p: FiniteTRSParams, gh_z: GHQuadrature, gh_eta: GHQuadrature) -> Tuple[float, float, float, float, float]:
    """
    Compute:
        Ey   = ∫ Dz <<y>>,
        Ey2  = ∫ Dz <<y^2>>,
        Eym2 = ∫ Dz <<(E[y|z,η])^2>>,
        Ey0  = ∫ Dz ( <<E[y|z,η]>>_η )^2,
        logPsi_int = ∫ Dz log ∫ Dη [Z_y(z,η)]^k.
    """
    k, _, _, chi0, chi1 = _conjugates(state, p)

    z = gh_z.z
    wz = gh_z.w
    eta = gh_eta.z
    weta = gh_eta.w

    sqrt_chi0 = np.sqrt(max(chi0, 0.0))
    sqrt_chi1 = np.sqrt(max(chi1, 0.0))

    Ey = Ey2 = Eym2 = Ey0 = 0.0
    logPsi_int = 0.0

    for zi, wi in zip(z, wz):
        h = my + sqrt_chi0 * zi + sqrt_chi1 * eta  # shape (n_eta,)

        stats = trunc_exp_0_ymax(h, y_max=p.y_max)  # vectorized over η
        logw = np.log(weta) + k * stats.logZ

        logPsi = float(logsumexp(logw))
        # normalized weights over eta (the 1/√π cancels)
        wnorm = np.exp(logw - logPsi)

        y_mean = float(np.sum(wnorm * stats.mean))
        y2 = float(np.sum(wnorm * stats.second))
        ym2 = float(np.sum(wnorm * (stats.mean * stats.mean)))

        Ey += wi * y_mean
        Ey2 += wi * y2
        Eym2 += wi * ym2
        Ey0 += wi * (y_mean * y_mean)
        logPsi_int += wi * logPsi

    return Ey, Ey2, Eym2, Ey0, logPsi_int


def _bracket_root(fun, a0: float, b0: float, expand: float = 1.5, max_expand: int = 50) -> Tuple[float, float]:
    """
    Expand [a0,b0] until it brackets a root (fun(a)*fun(b) < 0).
    """
    a, b = float(a0), float(b0)
    fa, fb = fun(a), fun(b)
    for _ in range(max_expand):
        if np.isnan(fa) or np.isnan(fb):
            raise RuntimeError("Function returned NaN during bracketing.")
        if fa * fb < 0:
            return a, b
        a *= expand
        b *= expand
        fa, fb = fun(a), fun(b)
    raise RuntimeError("Failed to bracket root.")


def solve_rs_finiteT(
    params: FiniteTRSParams,
    *,
    init: Optional[FiniteTRSState] = None,
    verbose: bool = False,
) -> FiniteTRSResult:
    """
    Solve the finite-temperature RS saddle.

    Returns a FiniteTRSResult with the final state and free energy density v.

    The default initializer is chosen to avoid hatQx≈0 at k≈-1.
    """
    p = params

    if init is None:
        init = FiniteTRSState(
            Qx=2.0,
            qx=1.5,
            Qy=2.0,
            q1=1.0,
            q0=0.5,
            mx=0.0,
            my=-1.0,
        )

    gh_z = gh_standard_normal(p.gh_n)
    gh_eta = gh_standard_normal(p.gh_n)

    state = init
    converged = False
    last_delta = np.inf

    for it in range(1, p.max_iter + 1):
        # --- Solve mx from Ex(mx)=1 ---
        def fx(mx):
            Ex, _, _, _ = _x_moments(mx, state, p, gh_z)
            return Ex - 1.0

        a, b = _bracket_root(fx, -10.0, 10.0, expand=1.5, max_expand=60)
        mx = float(brentq(fx, a, b, maxiter=200))

        Ex, Ex2, Exm2, logZx_int = _x_moments(mx, state, p, gh_z)

        # --- Solve my from Ey(my)=1 ---
        def fy(my):
            Ey, _, _, _, _ = _y_moments(my, state, p, gh_z, gh_eta)
            return Ey - 1.0

        a, b = _bracket_root(fy, -50.0, 10.0, expand=1.4, max_expand=80)
        my = float(brentq(fy, a, b, maxiter=300))

        Ey, Ey2, Eym2, Ey0, logPsi_int = _y_moments(my, state, p, gh_z, gh_eta)

        # predicted overlaps from moments
        Qx_new = Ex2
        qx_new = Exm2
        Qy_new = Ey2
        q1_new = Eym2
        q0_new = Ey0

        # damped update
        d = p.damping
        state_new = FiniteTRSState(
            Qx=(1 - d) * state.Qx + d * Qx_new,
            qx=(1 - d) * state.qx + d * qx_new,
            Qy=(1 - d) * state.Qy + d * Qy_new,
            q1=(1 - d) * state.q1 + d * q1_new,
            q0=(1 - d) * state.q0 + d * q0_new,
            mx=mx,
            my=my,
        )

        # convergence metric
        vec_old = np.array([state.Qx, state.qx, state.Qy, state.q1, state.q0], dtype=float)
        vec_new = np.array([state_new.Qx, state_new.qx, state_new.Qy, state_new.q1, state_new.q0], dtype=float)
        delta = float(np.max(np.abs(vec_new - vec_old) / (1.0 + np.abs(vec_old))))

        if verbose and (it == 1 or it % 10 == 0):
            k, hatQx, _, _, _ = _conjugates(state_new, p)
            print(
                f"[iter {it:4d}] delta={delta:.3e} "
                f"hatQx={hatQx:.3e} mx={mx:.3f} my={my:.3f} "
                f"Qx={state_new.Qx:.3f} qx={state_new.qx:.3f} "
                f"Qy={state_new.Qy:.3f} q1={state_new.q1:.3f} q0={state_new.q0:.3f}"
            )

        state = state_new
        last_delta = delta
        if delta < p.tol:
            converged = True
            break

    # compute g and residuals at final state
    k, hatQx, chi_x, chi0, chi1 = _conjugates(state, p)

    # moments at final multipliers (for residuals)
    Ex, Ex2, Exm2, logZx_int = _x_moments(state.mx, state, p, gh_z)
    Ey, Ey2, Eym2, Ey0, logPsi_int = _y_moments(state.my, state, p, gh_z, gh_eta)

    residuals = np.array([
        Ex - 1.0,
        state.Qx - Ex2,
        state.qx - Exm2,
        Ey - 1.0,
        state.Qy - Ey2,
        state.q1 - Eym2,
        state.q0 - Ey0,
    ], dtype=float)

    # Variational functional g (hatQy=0)
    sigma = p.sigma
    gamma = p.gamma
    beta_max = p.beta_max
    beta_min = p.beta_min
    sqrt_gamma = np.sqrt(gamma)

    Qx, qx, Qy, q1, q0 = state.Qx, state.qx, state.Qy, state.q1, state.q0
    mx, my = state.mx, state.my

    term1 = (sigma * beta_max**2 / 2.0) * (k * Qx * Qy + k * (k - 1.0) * Qx * q1 - (k**2) * qx * q0)
    term2 = (sqrt_gamma / 2.0) * (hatQx * Qx - chi_x * (Qx - qx))
    term3 = (1.0 / (2.0 * sqrt_gamma)) * ( -k * (chi1 + chi0) * (Qy + (k - 1.0) * q1) + (k**2) * chi0 * q0 )
    term4 = -sqrt_gamma * mx - (k / sqrt_gamma) * my
    term5 = sqrt_gamma * logZx_int + (1.0 / sqrt_gamma) * logPsi_int

    g = float(term1 + term2 + term3 + term4 + term5)
    v = float(-(1.0 / beta_min) * g)

    return FiniteTRSResult(
        params=p,
        state=state,
        k=k,
        hatQx=float(hatQx),
        hatchix=float(chi_x),
        hatchi0=float(chi0),
        hatchi1=float(chi1),
        g=g,
        v=v,
        residuals=residuals,
        n_iter=it,
        converged=converged,
    )
