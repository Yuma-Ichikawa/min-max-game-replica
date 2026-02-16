from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import numpy as np
import torch
from scipy.special import roots_hermite as hermgauss

from .special import DEVICE, DTYPE


@dataclass(frozen=True)
class FiniteTRSParams:
    sigma: float = 1.0
    gamma: float = 0.5
    beta_max: float = 1.0
    beta_min: float = 1.0
    gh_n: int = 80
    x_max: float = np.inf
    y_max: float = 80.0
    damping: float = 0.2
    max_iter: int = 2000
    tol: float = 1e-12


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
    energy: float
    residuals: np.ndarray
    n_iter: int
    converged: bool


def compute_energy(state: FiniteTRSState, p: FiniteTRSParams) -> float:
    k = -p.beta_min / p.beta_max
    Qx, qx = state.Qx, state.qx
    Qy, q1, q0 = state.Qy, state.q1, state.q0
    return float(p.sigma * p.beta_max * (
        Qx * Qy + (k - 1.0) * Qx * q1 - k * qx * q0
    ))


_GH_CACHE: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
_GH_LOG_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}
_GL_CACHE: dict[tuple[int, float, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_gh(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (n, device)
    if key not in _GH_CACHE:
        x, w = hermgauss(n)
        z = torch.tensor(np.sqrt(2.0) * x, dtype=DTYPE, device=device)
        w = torch.tensor(w / np.sqrt(np.pi), dtype=DTYPE, device=device)
        _GH_CACHE[key] = (z, w)
    return _GH_CACHE[key]


def _get_gh_log_w(n: int, device: torch.device) -> torch.Tensor:
    key = (n, device)
    if key not in _GH_LOG_CACHE:
        _, w = _get_gh(n, device)
        _GH_LOG_CACHE[key] = torch.log(w)
    return _GH_LOG_CACHE[key]


def _get_gl(n: int, U: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (n, U, device)
    if key not in _GL_CACHE:
        from numpy.polynomial.legendre import leggauss
        xg_np, wg_np = leggauss(n)
        x = torch.tensor(0.5 * (xg_np + 1.0) * U, dtype=DTYPE, device=device)
        w = torch.tensor(0.5 * U * wg_np, dtype=DTYPE, device=device)
        _GL_CACHE[key] = (x, w)
    return _GL_CACHE[key]


@torch.no_grad()
def _trunc_exp_gpu(h: torch.Tensor, y_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = h * y_max
    abs_u = u.abs()
    log_ym = math.log(y_max)
    ym2 = y_max * y_max

    u_pos = u.clamp(min=0.5)
    u_neg = (-u).clamp(min=0.5)
    u_mid = u.clamp(min=-50.0, max=50.0)

    logZ_s = log_ym + torch.log1p(u * 0.5 + u * u / 6.0 + u**3 / 24.0 + u**4 / 120.0)
    mean_s = y_max * (0.5 + u / 12.0 - u**3 / 720.0)
    sec_s = ym2 * (1.0/3.0 + u / 12.0 + u * u / 360.0 - u**3 / 720.0 - u**4 / 15120.0)

    logZ_p = log_ym + u - torch.log(u_pos)
    inv_up = 1.0 / u_pos
    mean_p = y_max * (1.0 - inv_up)
    sec_p = ym2 * (1.0 - 2.0 * inv_up + 2.0 * inv_up * inv_up)

    logZ_n = log_ym - torch.log(u_neg)
    inv_un = 1.0 / u_neg
    mean_n = y_max * inv_un
    sec_n = 2.0 * ym2 * inv_un * inv_un

    expm1_m = torch.expm1(u_mid)
    eu_m = expm1_m + 1.0
    expm1_safe = torch.where(expm1_m.abs() < 1e-15, torch.ones_like(expm1_m), expm1_m)
    abs_u_mid = u_mid.abs().clamp(min=1e-15)
    logZ_m = log_ym + torch.log(expm1_m.abs().clamp(min=1e-300)) - torch.log(abs_u_mid)
    mean_m = y_max * ((u_mid - 1.0) * eu_m + 1.0) / (u_mid * expm1_safe)
    sec_m = ym2 * ((u_mid * u_mid - 2 * u_mid + 2) * eu_m - 2.0) / (u_mid * u_mid * expm1_safe)

    small = abs_u < 1e-4
    lp = u > 50.0
    ln = u < -50.0

    logZ = torch.where(small, logZ_s, torch.where(lp, logZ_p, torch.where(ln, logZ_n, logZ_m)))
    mean = torch.where(small, mean_s, torch.where(lp, mean_p, torch.where(ln, mean_n, mean_m)))
    sec = torch.where(small, sec_s, torch.where(lp, sec_p, torch.where(ln, sec_n, sec_m)))

    return logZ, mean, sec


def _trunc_quad_gpu(A: float, B: torch.Tensor, U: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if A > 0.0:
        sqrtA = math.sqrt(A)
        sigma = 1.0 / sqrtA
        mu = B / A
        alpha = -B / sqrtA
        beta = U * sqrtA - B / sqrtA

        if hasattr(torch.special, 'log_ndtr'):
            logPhi_b = torch.special.log_ndtr(beta)
            logPhi_a = torch.special.log_ndtr(alpha)
        else:
            from .special import logPhi
            logPhi_b = logPhi(beta)
            logPhi_a = logPhi(alpha)

        from .special import logdiffexp
        logZ0 = logdiffexp(logPhi_b, logPhi_a)
        Z0 = torch.exp(logZ0)
        safe = Z0 > 1e-300

        logZ = B*B/(2*A) + 0.5*math.log(2*math.pi) - 0.5*math.log(A) + logZ0

        LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
        logphi_a = -0.5*alpha*alpha - LOG_SQRT_2PI
        logphi_b = -0.5*beta*beta - LOG_SQRT_2PI

        mask_phi = logphi_a >= logphi_b
        sign = torch.where(mask_phi, torch.ones_like(B), -torch.ones_like(B))
        log_larger = torch.where(mask_phi, logphi_a, logphi_b)
        log_smaller = torch.where(mask_phi, logphi_b, logphi_a)
        logabs = log_larger + torch.log1p(-torch.exp(torch.clamp(log_smaller - log_larger, -745.0, 0.0)))
        Z0_safe = Z0.clamp(min=1e-300)
        lam = sign * torch.exp(logabs - torch.log(Z0_safe))

        mean_val = mu + sigma * lam
        phi_a = torch.exp(logphi_a)
        phi_b = torch.exp(logphi_b)
        term = (alpha * phi_a - beta * phi_b) / Z0_safe
        var = sigma**2 * (1.0 + term - lam*lam)
        sec_val = var + mean_val**2

        LARGE_NEG = -700.0
        logZ = torch.where(safe, logZ, torch.tensor(LARGE_NEG, dtype=DTYPE, device=B.device))
        mean_val = torch.where(safe, mean_val, torch.zeros_like(B))
        sec_val = torch.where(safe, sec_val, torch.zeros_like(B))
        return logZ, mean_val, sec_val

    elif A == 0.0:
        return _trunc_exp_gpu(B, U)

    else:
        x, w = _get_gl(80, U, B.device)
        f = (-A*0.5)*(x.unsqueeze(0)**2) + B.unsqueeze(1)*x.unsqueeze(0)
        fmax = f.max(dim=1, keepdim=True).values
        es = torch.exp(f - fmax)
        I0 = (w.unsqueeze(0)*es).sum(1)
        I1 = (w.unsqueeze(0)*x.unsqueeze(0)*es).sum(1)
        I2 = (w.unsqueeze(0)*(x.unsqueeze(0)**2)*es).sum(1)
        return fmax[:,0]+torch.log(I0), I1/I0, I2/I0


@torch.no_grad()
def _vectorized_bisection(
    f_batch,
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 8,
    K: int = 128,
    device: torch.device = None,
) -> float:
    if device is None:
        device = DEVICE

    lo, hi = float(a), float(b)

    for _ in range(max_iter):
        if abs(hi - lo) < tol:
            break
        xs = torch.linspace(lo, hi, K, dtype=DTYPE, device=device)
        fs = f_batch(xs)

        signs = fs[:-1] * fs[1:]
        sign_neg = signs <= 0
        has_change = sign_neg.any()

        if has_change:
            idx = sign_neg.to(torch.int32).argmax().item()
            lo = xs[idx].item()
            hi = xs[idx + 1].item()
        else:
            mid = (lo + hi) / 2
            f0 = fs[0].item()
            f_mid = f_batch(torch.tensor([mid], dtype=DTYPE, device=device))[0].item()
            if f_mid * f0 <= 0:
                hi = mid
            else:
                lo = mid

    return (lo + hi) / 2


def _conjugates(state: FiniteTRSState, p: FiniteTRSParams) -> Tuple[float, float, float, float, float]:
    sigma = p.sigma
    gamma = p.gamma
    k = -p.beta_min / p.beta_max
    sg = math.sqrt(gamma)
    Qx, qx, Qy, q1, q0 = state.Qx, state.qx, state.Qy, state.q1, state.q0
    hatchix = (sigma * p.beta_max**2 / sg) * k**2 * q0
    hatchi0 = (sg * sigma * p.beta_max**2) * qx
    hatchi1 = (sg * sigma * p.beta_max**2) * (Qx - qx)
    hatQx = hatchix - (sigma * p.beta_max**2 / sg) * (k * Qy + k * (k-1) * q1)
    return k, hatQx, hatchix, hatchi0, hatchi1


def _x_moments_gpu(
    mx_vals: torch.Tensor,
    state: FiniteTRSState,
    p: FiniteTRSParams,
    gh_z: torch.Tensor,
    gh_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, hatQx, hatchix, _, _ = _conjugates(state, p)
    sqrt_hx = math.sqrt(max(hatchix, 0.0))

    B = mx_vals.unsqueeze(1) + sqrt_hx * gh_z.unsqueeze(0)
    K, n = B.shape

    B_flat = B.reshape(-1)
    logZ_f, mean_f, sec_f = _trunc_quad_gpu(hatQx, B_flat, p.x_max)
    logZ = logZ_f.reshape(K, n)
    mean = mean_f.reshape(K, n)
    sec = sec_f.reshape(K, n)

    w = gh_w.unsqueeze(0)
    Ex = (w * mean).sum(1)
    Ex2 = (w * sec).sum(1)
    Exm2 = (w * mean**2).sum(1)
    logZx = (w * logZ).sum(1)

    return Ex, Ex2, Exm2, logZx


def _y_moments_core(
    h_flat: torch.Tensor,
    y_max: float,
    k_val: float,
    K: int,
    n_z: int,
    n_eta: int,
    log_weta: torch.Tensor,
    wz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u = h_flat * y_max
    abs_u = u.abs()
    log_ym = math.log(y_max)
    ym2 = y_max * y_max

    u_pos = u.clamp(min=0.5)
    u_neg = (-u).clamp(min=0.5)
    u_mid = u.clamp(min=-50.0, max=50.0)

    logZ_s = log_ym + torch.log1p(u * 0.5 + u * u / 6.0 + u**3 / 24.0 + u**4 / 120.0)
    mean_s = y_max * (0.5 + u / 12.0 - u**3 / 720.0)
    sec_s = ym2 * (1.0/3.0 + u / 12.0 + u * u / 360.0 - u**3 / 720.0 - u**4 / 15120.0)

    inv_up = 1.0 / u_pos
    logZ_p = log_ym + u - torch.log(u_pos)
    mean_p = y_max * (1.0 - inv_up)
    sec_p = ym2 * (1.0 - 2.0 * inv_up + 2.0 * inv_up * inv_up)

    inv_un = 1.0 / u_neg
    logZ_n = log_ym - torch.log(u_neg)
    mean_n = y_max * inv_un
    sec_n = 2.0 * ym2 * inv_un * inv_un

    expm1_m = torch.expm1(u_mid)
    eu_m = expm1_m + 1.0
    expm1_safe = torch.where(expm1_m.abs() < 1e-15, torch.ones_like(expm1_m), expm1_m)
    abs_u_mid = u_mid.abs().clamp(min=1e-15)
    logZ_m = log_ym + torch.log(expm1_m.abs().clamp(min=1e-300)) - torch.log(abs_u_mid)
    mean_m = y_max * ((u_mid - 1.0) * eu_m + 1.0) / (u_mid * expm1_safe)
    sec_m = ym2 * ((u_mid * u_mid - 2 * u_mid + 2) * eu_m - 2.0) / (u_mid * u_mid * expm1_safe)

    small = abs_u < 1e-4
    lp = u > 50.0
    ln = u < -50.0

    s_logZ = torch.where(small, logZ_s, torch.where(lp, logZ_p, torch.where(ln, logZ_n, logZ_m)))
    s_mean = torch.where(small, mean_s, torch.where(lp, mean_p, torch.where(ln, mean_n, mean_m)))
    s_sec = torch.where(small, sec_s, torch.where(lp, sec_p, torch.where(ln, sec_n, sec_m)))

    s_logZ = s_logZ.reshape(K, n_z, n_eta)
    s_mean = s_mean.reshape(K, n_z, n_eta)
    s_sec = s_sec.reshape(K, n_z, n_eta)

    logw = log_weta.reshape(1, 1, n_eta) + k_val * s_logZ
    logPsi = torch.logsumexp(logw, dim=2)
    wnorm = torch.exp(logw - logPsi.unsqueeze(2))

    y_mean = (wnorm * s_mean).sum(2)
    y2 = (wnorm * s_sec).sum(2)
    ym2_val = (wnorm * s_mean * s_mean).sum(2)

    wz_2d = wz.reshape(1, n_z)
    Ey = (wz_2d * y_mean).sum(1)
    Ey2 = (wz_2d * y2).sum(1)
    Eym2 = (wz_2d * ym2_val).sum(1)
    Ey0 = (wz_2d * y_mean * y_mean).sum(1)
    logPsi_int = (wz_2d * logPsi).sum(1)

    return Ey, Ey2, Eym2, Ey0, logPsi_int


@torch.no_grad()
def _y_moments_gpu(
    my_vals: torch.Tensor,
    state: FiniteTRSState,
    p: FiniteTRSParams,
    gh_z: torch.Tensor,
    gh_wz: torch.Tensor,
    gh_eta: torch.Tensor,
    gh_weta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k_val, _, _, hatchi0, hatchi1 = _conjugates(state, p)
    sqrt_c0 = math.sqrt(max(hatchi0, 0.0))
    sqrt_c1 = math.sqrt(max(hatchi1, 0.0))

    K = my_vals.shape[0]
    n_z = gh_z.shape[0]
    n_eta = gh_eta.shape[0]

    h = (my_vals.reshape(K, 1, 1)
         + sqrt_c0 * gh_z.reshape(1, n_z, 1)
         + sqrt_c1 * gh_eta.reshape(1, 1, n_eta))
    h_flat = h.reshape(-1)

    log_weta = _get_gh_log_w(n_eta, gh_weta.device)

    return _y_moments_core(h_flat, p.y_max, k_val, K, n_z, n_eta, log_weta, gh_wz)


@torch.no_grad()
def _find_mx_gpu(
    state: FiniteTRSState,
    p: FiniteTRSParams,
    gh_z: torch.Tensor,
    gh_w: torch.Tensor,
    tol: float = 1e-12,
    prev_mx: Optional[float] = None,
) -> tuple[float, float, float, float, float]:
    device = gh_z.device

    def f_batch(mx_tensor):
        Ex, _, _, _ = _x_moments_gpu(mx_tensor, state, p, gh_z, gh_w)
        return Ex - 1.0

    if prev_mx is not None and math.isfinite(prev_mx):
        lo, hi = prev_mx - 3.0, prev_mx + 3.0
    else:
        lo, hi = -10.0, 10.0
    bracket_pts = torch.empty(2, dtype=DTYPE, device=device)
    for _ in range(60):
        bracket_pts[0] = lo
        bracket_pts[1] = hi
        fs = f_batch(bracket_pts)
        fvals = fs.tolist()
        nan_lo = math.isnan(fvals[0])
        nan_hi = math.isnan(fvals[1])
        if nan_lo or nan_hi:
            mid = (lo + hi) * 0.5
            if nan_lo and not nan_hi:
                lo = mid
            elif nan_hi and not nan_lo:
                hi = mid
            else:
                lo = mid - 0.8 * (mid - lo)
                hi = mid + 0.8 * (hi - mid)
            continue
        if fvals[0] * fvals[1] < 0:
            break
        width = max(hi - lo, 1.0)
        lo -= 0.5 * width
        hi += 0.5 * width

    mx = _vectorized_bisection(f_batch, lo, hi, tol=tol, K=128, device=device)

    mx_t = torch.tensor([mx], dtype=DTYPE, device=device)
    Ex, Ex2, Exm2, logZx = _x_moments_gpu(mx_t, state, p, gh_z, gh_w)
    vals = torch.stack([Ex[0], Ex2[0], Exm2[0], logZx[0]]).tolist()
    return mx, vals[0], vals[1], vals[2], vals[3]


@torch.no_grad()
def _find_my_gpu(
    state: FiniteTRSState,
    p: FiniteTRSParams,
    gh_z: torch.Tensor,
    gh_wz: torch.Tensor,
    gh_eta: torch.Tensor,
    gh_weta: torch.Tensor,
    tol: float = 1e-12,
    prev_my: Optional[float] = None,
) -> tuple[float, float, float, float, float, float]:
    device = gh_z.device

    def f_batch(my_tensor):
        Ey, _, _, _, _ = _y_moments_gpu(my_tensor, state, p, gh_z, gh_wz, gh_eta, gh_weta)
        return Ey - 1.0

    if prev_my is not None and math.isfinite(prev_my):
        lo, hi = prev_my - 5.0, prev_my + 5.0
    else:
        lo, hi = -200.0, 50.0
    bracket_pts = torch.empty(2, dtype=DTYPE, device=device)
    for _ in range(50):
        bracket_pts[0] = lo
        bracket_pts[1] = hi
        fs = f_batch(bracket_pts)
        fvals = fs.tolist()
        nan_lo = math.isnan(fvals[0])
        nan_hi = math.isnan(fvals[1])
        if nan_lo or nan_hi:
            mid = (lo + hi) * 0.5
            if nan_lo and not nan_hi:
                lo = mid
            elif nan_hi and not nan_lo:
                hi = mid
            else:
                lo = mid - 0.8 * (mid - lo)
                hi = mid + 0.8 * (hi - mid)
            continue
        if fvals[0] * fvals[1] < 0:
            break
        width = max(hi - lo, 1.0)
        lo -= 0.6 * width
        hi += 0.6 * width

    my = _vectorized_bisection(f_batch, lo, hi, tol=tol, K=128, device=device)

    my_t = torch.tensor([my], dtype=DTYPE, device=device)
    Ey, Ey2, Eym2, Ey0, logPsi = _y_moments_gpu(my_t, state, p, gh_z, gh_wz, gh_eta, gh_weta)
    vals = torch.stack([Ey[0], Ey2[0], Eym2[0], Ey0[0], logPsi[0]]).tolist()
    return my, vals[0], vals[1], vals[2], vals[3], vals[4]


def _build_result(
    state: FiniteTRSState,
    p: FiniteTRSParams,
    device: torch.device,
    gh_z: torch.Tensor,
    gh_w: torch.Tensor,
    n_iter: int,
    converged: bool,
    residuals_override: Optional[np.ndarray] = None,
) -> FiniteTRSResult:
    k, hatQx, hatchix, hatchi0, hatchi1 = _conjugates(state, p)

    mx_t = torch.tensor([state.mx], dtype=DTYPE, device=device)
    Ex_t, Ex2_t, Exm2_t, logZx_t = _x_moments_gpu(mx_t, state, p, gh_z, gh_w)
    xvals = torch.stack([Ex_t[0], Ex2_t[0], Exm2_t[0], logZx_t[0]]).tolist()
    Ex, Ex2, Exm2, logZx_int = xvals

    my_t = torch.tensor([state.my], dtype=DTYPE, device=device)
    Ey_t, Ey2_t, Eym2_t, Ey0_t, logPsi_t = _y_moments_gpu(
        my_t, state, p, gh_z, gh_w, gh_z, gh_w)
    yvals = torch.stack([Ey_t[0], Ey2_t[0], Eym2_t[0], Ey0_t[0], logPsi_t[0]]).tolist()
    Ey, Ey2, Eym2, Ey0, logPsi_int = yvals

    if residuals_override is not None:
        residuals = residuals_override
    else:
        residuals = np.array([
            Ex - 1.0, state.Qx - Ex2, state.qx - Exm2,
            Ey - 1.0, state.Qy - Ey2, state.q1 - Eym2, state.q0 - Ey0,
        ], dtype=float)

    sigma = p.sigma
    sqrt_gamma = math.sqrt(p.gamma)
    Qx, qx, Qy, q1, q0 = state.Qx, state.qx, state.Qy, state.q1, state.q0
    mx, my = state.mx, state.my

    term1 = (sigma * p.beta_max**2 / 2) * (k*Qx*Qy + k*(k-1)*Qx*q1 - k**2*qx*q0)
    term2 = (sqrt_gamma / 2) * (hatQx*Qx - hatchix*(Qx-qx))
    term3 = (1/(2*sqrt_gamma)) * (-k*(hatchi1+hatchi0)*(Qy+(k-1)*q1) + k**2*hatchi0*q0)
    term4 = -sqrt_gamma*mx - (k/sqrt_gamma)*my
    term5 = sqrt_gamma*logZx_int + (1/sqrt_gamma)*logPsi_int

    g = float(term1 + term2 + term3 + term4 + term5)
    v = float(-(1/p.beta_min)*g)
    ev = compute_energy(state, p)

    return FiniteTRSResult(
        params=p, state=state, k=k,
        hatQx=float(hatQx), hatchix=float(hatchix),
        hatchi0=float(hatchi0), hatchi1=float(hatchi1),
        g=g, v=v, energy=ev, residuals=residuals, n_iter=n_iter, converged=converged,
    )


def _solve_rs_finiteT_single(
    p: FiniteTRSParams,
    init: FiniteTRSState,
    device: torch.device,
    verbose: bool,
    early_stop_iter: int = 0,
    skip_warmup: bool = False,
) -> FiniteTRSResult:
    gh_z, gh_w = _get_gh(p.gh_n, device)

    state = init
    converged = False

    if skip_warmup:
        d_start = p.damping
        warmup_iters = 0
    else:
        d_start = max(p.damping, 0.3)
        warmup_iters = min(300, p.max_iter // 4)
    d_end = p.damping

    best_residual = float('inf')
    best_delta = float('inf')
    stagnation_count = 0
    prev_delta = float('inf')
    oscillation_count = 0
    oscillation_cooldown = 0

    for it in range(1, p.max_iter + 1):
        tol_iter = max(1e-10, p.tol * 10) if it <= 5 else max(1e-12, p.tol)

        if it <= warmup_iters:
            d = d_start - (d_start - d_end) * (it - 1) / max(warmup_iters - 1, 1)
        else:
            d = d_end

        if oscillation_cooldown > 0:
            d = max(d * 0.5, 0.02)
            oscillation_cooldown -= 1

        mx, Ex, Ex2, Exm2, logZx_int = _find_mx_gpu(
            state, p, gh_z, gh_w, tol=tol_iter,
            prev_mx=state.mx)

        my, Ey, Ey2, Eym2, Ey0, logPsi_int = _find_my_gpu(
            state, p, gh_z, gh_w, gh_z, gh_w, tol=tol_iter,
            prev_my=state.my)

        state_new = FiniteTRSState(
            Qx=(1-d)*state.Qx + d*Ex2,
            qx=(1-d)*state.qx + d*Exm2,
            Qy=(1-d)*state.Qy + d*Ey2,
            q1=(1-d)*state.q1 + d*Eym2,
            q0=(1-d)*state.q0 + d*Ey0,
            mx=mx, my=my,
        )

        delta = max(
            abs(state_new.Qx - state.Qx) / (1.0 + abs(state.Qx)),
            abs(state_new.qx - state.qx) / (1.0 + abs(state.qx)),
            abs(state_new.Qy - state.Qy) / (1.0 + abs(state.Qy)),
            abs(state_new.q1 - state.q1) / (1.0 + abs(state.q1)),
            abs(state_new.q0 - state.q0) / (1.0 + abs(state.q0)),
        )
        max_residual = delta / max(d, 1e-15)

        if verbose and (it == 1 or it % 50 == 0):
            k_val, hatQx_val, _, _, _ = _conjugates(state_new, p)
            print(f"[iter {it:4d}] delta={delta:.3e} res={max_residual:.3e} "
                  f"d={d:.3f} hatQx={hatQx_val:.3e}")

        if delta > prev_delta * 1.05 and it > warmup_iters:
            oscillation_count += 1
            if oscillation_count >= 5:
                oscillation_cooldown = 30
                oscillation_count = 0
        else:
            oscillation_count = max(0, oscillation_count - 1)
        prev_delta = delta

        if max_residual < best_residual * 0.99:
            best_residual = max_residual
            stagnation_count = 0
        else:
            stagnation_count += 1

        if delta < best_delta * 0.99:
            best_delta = delta

        if early_stop_iter > 0 and it >= early_stop_iter and best_residual > 1e-4:
            break

        if stagnation_count >= 500 and it >= 800 and best_residual < 5e-6:
            converged = True
            break
        if stagnation_count >= 800 and it >= 1000:
            if best_residual < 1e-4:
                converged = True
            break

        state = state_new
        if max_residual < max(p.tol, 1e-10):
            converged = True
            break

    if best_residual > 5e-6:
        if skip_warmup:
            d_polish = min(0.3, max(d_end * 2, 0.10))
        else:
            d_polish = min(0.5, max(d_end * 3, 0.20))
        p_res = best_residual
        for pit in range(1, 51):
            mx, Ex, Ex2, Exm2, logZx_int = _find_mx_gpu(
                state, p, gh_z, gh_w, tol=max(1e-13, p.tol),
                prev_mx=state.mx)
            my, Ey, Ey2, Eym2, Ey0, logPsi_int = _find_my_gpu(
                state, p, gh_z, gh_w, gh_z, gh_w, tol=max(1e-13, p.tol),
                prev_my=state.my)

            state_new = FiniteTRSState(
                Qx=(1-d_polish)*state.Qx + d_polish*Ex2,
                qx=(1-d_polish)*state.qx + d_polish*Exm2,
                Qy=(1-d_polish)*state.Qy + d_polish*Ey2,
                q1=(1-d_polish)*state.q1 + d_polish*Eym2,
                q0=(1-d_polish)*state.q0 + d_polish*Ey0,
                mx=mx, my=my,
            )
            pd = max(
                abs(state_new.Qx - state.Qx) / (1.0 + abs(state.Qx)),
                abs(state_new.qx - state.qx) / (1.0 + abs(state.qx)),
                abs(state_new.Qy - state.Qy) / (1.0 + abs(state.Qy)),
                abs(state_new.q1 - state.q1) / (1.0 + abs(state.q1)),
                abs(state_new.q0 - state.q0) / (1.0 + abs(state.q0)),
            )
            p_res = pd / max(d_polish, 1e-15)
            state = state_new
            if p_res < 1e-8:
                converged = True
                break
        if verbose:
            print(f"[polish] {pit} iters, final res={p_res:.3e}")
        it += pit

    return _build_result(state, p, device, gh_z, gh_w, n_iter=it, converged=converged)


def _solve_anderson_single(
    p: FiniteTRSParams,
    init: FiniteTRSState,
    device: torch.device,
    verbose: bool,
    anderson_m: int = 5,
    anderson_beta: float = 1.0,
    gh_n_final: int = 0,
) -> FiniteTRSResult:
    gh_z, gh_w = _get_gh(p.gh_n, device)
    gh_z_fine, gh_w_fine = None, None
    if gh_n_final > 0 and gh_n_final != p.gh_n:
        gh_z_fine, gh_w_fine = _get_gh(gh_n_final, device)
    use_fine = False

    theta = np.array([init.Qx, init.qx, init.Qy, init.q1, init.q0])
    current_mx, current_my = init.mx, init.my

    G_hist = []
    theta_hist = []

    converged = False
    best_res = float('inf')

    def _eval_G(th, mx_hint, my_hint):
        st = FiniteTRSState(Qx=th[0], qx=th[1], Qy=th[2], q1=th[3], q0=th[4],
                            mx=mx_hint, my=my_hint)
        z, w = (gh_z_fine, gh_w_fine) if use_fine else (gh_z, gh_w)
        mx, Ex, Ex2, Exm2, logZx = _find_mx_gpu(st, p, z, w, tol=max(1e-13, p.tol),
                                                  prev_mx=mx_hint)
        my, Ey, Ey2, Eym2, Ey0, logPsi = _find_my_gpu(st, p, z, w, z, w,
                                                         tol=max(1e-13, p.tol),
                                                         prev_my=my_hint)
        return np.array([Ex2, Exm2, Ey2, Eym2, Ey0]), mx, my, logZx, logPsi

    for it in range(1, p.max_iter + 1):
        g, current_mx, current_my, logZx_int, logPsi_int = _eval_G(
            theta, current_mx, current_my)

        if np.any(np.isnan(g)):
            G_hist.clear()
            theta_hist.clear()
            continue

        r = g - theta

        max_residual = float(np.max(np.abs(r) / (1.0 + np.abs(theta))))

        if verbose and (it == 1 or it % 20 == 0):
            print(f"[AA iter {it:4d}] res={max_residual:.3e} "
                  f"Qx={theta[0]:.4f} qx={theta[1]:.4f} Qy={theta[2]:.4f}", flush=True)

        if max_residual < best_res:
            best_res = max_residual

        if max_residual < max(p.tol, 1e-10):
            converged = True
            break

        if not use_fine and gh_z_fine is not None and max_residual < 1e-4:
            use_fine = True
            G_hist.clear()
            theta_hist.clear()
            if verbose:
                print(f"  [AA] Switching to gh_n={gh_n_final}", flush=True)

        G_hist.append(g.copy())
        theta_hist.append(theta.copy())
        if len(G_hist) > anderson_m + 1:
            G_hist.pop(0)
            theta_hist.pop(0)

        mk = len(G_hist) - 1

        if mk == 0:
            theta = (1.0 - anderson_beta) * theta + anderson_beta * g
        else:
            DG = np.column_stack([G_hist[-1] - G_hist[i] for i in range(mk)])
            Dtheta = np.column_stack([theta_hist[-1] - theta_hist[i] for i in range(mk)])
            DR = DG - Dtheta

            if np.any(np.isnan(DR)) or np.any(np.isnan(r)):
                theta = (1.0 - 0.3) * theta + 0.3 * g
                G_hist.clear()
                theta_hist.clear()
                continue
            try:
                alpha, _, _, _ = np.linalg.lstsq(DR, r, rcond=1e-10)
            except (np.linalg.LinAlgError, ValueError):
                theta = (1.0 - anderson_beta) * theta + anderson_beta * g
                continue
            if np.any(np.isnan(alpha)):
                theta = (1.0 - 0.3) * theta + 0.3 * g
                G_hist.clear()
                theta_hist.clear()
                continue

            theta_new = g - DG @ alpha

            if anderson_beta < 1.0:
                theta_aa = theta - Dtheta @ alpha
                theta_new = (1.0 - anderson_beta) * theta_aa + anderson_beta * theta_new

            if np.any(theta_new[:2] < 0) or np.any(theta_new[2:5] < 0):
                theta = (1.0 - 0.3) * theta + 0.3 * g
                theta = np.maximum(theta, 1e-10)
                G_hist.clear()
                theta_hist.clear()
                continue

            theta_new = np.maximum(theta_new, 1e-10)
            theta = theta_new

    state = FiniteTRSState(Qx=theta[0], qx=theta[1], Qy=theta[2],
                           q1=theta[3], q0=theta[4],
                           mx=current_mx, my=current_my)
    z, w = (gh_z_fine, gh_w_fine) if use_fine else (gh_z, gh_w)

    return _build_result(state, p, device, z, w, n_iter=it, converged=converged)


def _solve_lbfgs_single(
    p: FiniteTRSParams,
    init: FiniteTRSState,
    device: torch.device,
    verbose: bool,
    gh_n_final: int = 0,
) -> FiniteTRSResult:
    from scipy.optimize import least_squares

    gh_z, gh_w = _get_gh(p.gh_n, device)
    gh_z_use, gh_w_use = gh_z, gh_w
    if gh_n_final > 0 and gh_n_final != p.gh_n:
        gh_z_use, gh_w_use = _get_gh(gh_n_final, device)

    n_eval = [0]

    def residual_fn(params_flat):
        Qx, qx, Qy, q1, q0, mx, my = params_flat
        n_eval[0] += 1

        state = FiniteTRSState(Qx=Qx, qx=qx, Qy=Qy, q1=q1, q0=q0, mx=mx, my=my)

        mx_t = torch.tensor([mx], dtype=DTYPE, device=device)
        Ex_t, Ex2_t, Exm2_t, _ = _x_moments_gpu(mx_t, state, p, gh_z_use, gh_w_use)
        Ex = Ex_t[0].item()
        Ex2 = Ex2_t[0].item()
        Exm2 = Exm2_t[0].item()

        my_t = torch.tensor([my], dtype=DTYPE, device=device)
        Ey_t, Ey2_t, Eym2_t, Ey0_t, _ = _y_moments_gpu(
            my_t, state, p, gh_z_use, gh_w_use, gh_z_use, gh_w_use)
        Ey = Ey_t[0].item()
        Ey2 = Ey2_t[0].item()
        Eym2 = Eym2_t[0].item()
        Ey0 = Ey0_t[0].item()

        return np.array([
            Ex - 1.0,
            Qx - Ex2,
            qx - Exm2,
            Ey - 1.0,
            Qy - Ey2,
            q1 - Eym2,
            q0 - Ey0,
        ])

    x0 = np.array([init.Qx, init.qx, init.Qy, init.q1, init.q0, init.mx, init.my])

    lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, -np.inf, -np.inf])
    ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    result = least_squares(
        residual_fn, x0,
        method='trf',
        bounds=(lb, ub),
        ftol=p.tol,
        xtol=p.tol,
        gtol=p.tol,
        max_nfev=p.max_iter * 15,
        verbose=2 if verbose else 0,
        diff_step=1e-8,
    )

    Qx, qx, Qy, q1, q0, mx, my = result.x
    state = FiniteTRSState(Qx=Qx, qx=qx, Qy=Qy, q1=q1, q0=q0, mx=mx, my=my)

    return _build_result(
        state, p, device, gh_z_use, gh_w_use,
        n_iter=n_eval[0], converged=(result.cost < p.tol),
        residuals_override=result.fun,
    )


def _make_default_inits(p: FiniteTRSParams) -> list[FiniteTRSState]:
    inits = [
        FiniteTRSState(Qx=2.0, qx=1.5, Qy=2.0, q1=1.0, q0=0.5, mx=0.0, my=-1.0),
        FiniteTRSState(Qx=3.0, qx=2.0, Qy=3.0, q1=1.5, q0=0.8, mx=-0.5, my=-1.5),
    ]

    xm = p.x_max if np.isfinite(p.x_max) else 100.0
    ym = p.y_max if np.isfinite(p.y_max) else 100.0
    Qy_init = max(0.1 * ym, 2.0)
    Qx_init = max(0.1 * xm, 2.0)
    inits.append(FiniteTRSState(
        Qx=Qx_init,
        qx=0.8 * Qx_init,
        Qy=Qy_init,
        q1=0.9 * Qy_init,
        q0=0.6 * Qy_init,
        mx=-1.0,
        my=-max(2.0, math.sqrt(2.0 * math.log(max(Qy_init, 2.0)))),
    ))
    return inits


def solve_rs_finiteT(
    params: FiniteTRSParams,
    *,
    init: Optional[FiniteTRSState] = None,
    verbose: bool = False,
    device: Optional[torch.device] = None,
    method: str = "picard",
    anderson_m: int = 5,
    anderson_beta: float = 1.0,
    gh_n_final: int = 0,
) -> FiniteTRSResult:
    if device is None:
        device = DEVICE

    is_warmstart = (init is not None)
    if is_warmstart:
        inits = [init]
        early_stop = 0
    else:
        inits = _make_default_inits(params)
        early_stop = max(500, params.max_iter // 4)

    def _run_single(init_state, es):
        if method == "anderson":
            return _solve_anderson_single(
                params, init_state, device, verbose=verbose,
                anderson_m=anderson_m, anderson_beta=anderson_beta,
                gh_n_final=gh_n_final,
            )
        elif method == "lbfgs":
            return _solve_lbfgs_single(
                params, init_state, device, verbose=verbose,
                gh_n_final=gh_n_final,
            )
        else:  # "picard"
            return _solve_rs_finiteT_single(
                params, init_state, device, verbose=verbose,
                early_stop_iter=es,
                skip_warmup=is_warmstart,
            )

    best = None
    for init_state in inits:
        try:
            res = _run_single(init_state, early_stop)
            if best is None or res.g > best.g:
                best = res
                if res.converged and float(np.max(np.abs(res.residuals))) < 1e-6:
                    break
        except Exception:
            continue

    if best is not None and not best.converged and early_stop > 0:
        try:
            res = _run_single(best.state, 0)
            if res.g > best.g:
                best = res
        except Exception:
            pass

    if best is None:
        raise RuntimeError("All initial states failed.")
    return best


def solve_rs_finiteT_curve(
    gamma_array: np.ndarray,
    *,
    M: int,
    sigma: float,
    beta_max: float,
    beta_min: float,
    gh_n: int = 200,
    tol: float = 1e-15,
    max_iter: int = 5000,
    damping: float = 0.1,
    verbose: bool = False,
    device: Optional[torch.device] = None,
    method: str = "picard",
    anderson_m: int = 5,
    anderson_beta: float = 1.0,
    gh_n_final: int = 0,
) -> list[FiniteTRSResult]:
    if device is None:
        device = DEVICE

    n = len(gamma_array)
    results_fwd = [None] * n
    results_rev = [None] * n

    def _make_params(g):
        N_g = int(round(g * M))
        return FiniteTRSParams(
            sigma=sigma, gamma=g,
            beta_max=beta_max, beta_min=beta_min,
            x_max=float(N_g), y_max=float(M),
            gh_n=gh_n, tol=tol, max_iter=max_iter, damping=damping,
        )

    prev_state = None
    for i, g in enumerate(gamma_array):
        params = _make_params(g)
        try:
            res = solve_rs_finiteT(params, init=prev_state, device=device,
                                   method=method, anderson_m=anderson_m,
                                   anderson_beta=anderson_beta, gh_n_final=gh_n_final)
            results_fwd[i] = res
            prev_state = res.state
            if verbose:
                conv = "CONV" if res.converged else f"NOT_CONV({res.n_iter})"
                print(f"  [fwd {i+1:2d}/{n}] γ={g:.3f}: v={res.v:.6f} {conv}", flush=True)
        except Exception as e:
            if verbose:
                print(f"  [fwd {i+1:2d}/{n}] γ={g:.3f}: FAILED: {e}", flush=True)
            prev_state = None

    prev_state = None
    for i in range(n - 1, -1, -1):
        g = gamma_array[i]
        params = _make_params(g)
        try:
            res = solve_rs_finiteT(params, init=prev_state, device=device,
                                   method=method, anderson_m=anderson_m,
                                   anderson_beta=anderson_beta, gh_n_final=gh_n_final)
            results_rev[i] = res
            prev_state = res.state
            if verbose:
                conv = "CONV" if res.converged else f"NOT_CONV({res.n_iter})"
                print(f"  [rev {n-i:2d}/{n}] γ={g:.3f}: v={res.v:.6f} {conv}", flush=True)
        except Exception as e:
            if verbose:
                print(f"  [rev {n-i:2d}/{n}] γ={g:.3f}: FAILED: {e}", flush=True)
            prev_state = None

    results = []
    n_improved = 0
    for i in range(n):
        fwd = results_fwd[i]
        rev = results_rev[i]
        if fwd is None and rev is None:
            results.append(None)
        elif fwd is None:
            results.append(rev)
        elif rev is None:
            results.append(fwd)
        else:
            if rev.g > fwd.g:
                n_improved += 1
                results.append(rev)
            else:
                results.append(fwd)

    if verbose and n_improved > 0:
        print(f"  Reverse pass improved {n_improved}/{n} points", flush=True)

    return results
