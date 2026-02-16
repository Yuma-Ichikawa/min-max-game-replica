from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import torch

from .divided_differences import batch_simplex_exp_integral_divdiff_gpu
from .simplex import sample_uniform_simplex, log_simplex_volume
from .special import DEVICE, DTYPE


def _schedule_linear(T: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, T + 1)


def _schedule_sigmoid(T: int, steepness: float = 4.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, T + 1)
    raw = 1.0 / (1.0 + np.exp(-steepness * (2.0 * t - 1.0)))
    raw = (raw - raw[0]) / (raw[-1] - raw[0])
    return raw


def _schedule_power(T: int, power: float = 2.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, T + 1)
    return t ** power


def make_schedule(T: int, name: str = "sigmoid", **kwargs) -> np.ndarray:
    if name == "linear":
        return _schedule_linear(T)
    elif name == "sigmoid":
        return _schedule_sigmoid(T, steepness=kwargs.get("steepness", 4.0))
    elif name == "power":
        return _schedule_power(T, power=kwargs.get("power", 2.0))
    else:
        raise ValueError(f"Unknown schedule: {name}")


def _compute_logZy_batch(
    X: torch.Tensor,
    C: torch.Tensor,
    scale: float,
    M: int,
) -> torch.Tensor:
    A = scale * torch.matmul(X, C)
    res = batch_simplex_exp_integral_divdiff_gpu(A, mass=float(M))
    return res.logZ


def _saddlepoint_solve_lambda(
    A: torch.Tensor,
    mass: float,
    n_newton: int = 50,
    tol: float = 1e-14,
) -> torch.Tensor:
    a_max = A.max(dim=1).values
    lam = a_max + mass / A.shape[1] + 0.1

    for _ in range(n_newton):
        inv = 1.0 / (lam.unsqueeze(1) - A)
        f = inv.sum(dim=1) - mass
        fp = -(inv ** 2).sum(dim=1)
        delta = f / fp
        lam = lam - delta
        lam = torch.maximum(lam, a_max + 1e-10)
        if delta.abs().max().item() < tol:
            break

    return lam


def _compute_logZy_saddlepoint_batch(
    X: torch.Tensor,
    C: torch.Tensor,
    scale: float,
    M: int,
) -> torch.Tensor:
    A = scale * torch.matmul(X, C)
    mass = float(M)

    lam = _saddlepoint_solve_lambda(A, mass)

    diff = lam.unsqueeze(1) - A
    log_diff = torch.log(diff)

    S2 = (1.0 / diff ** 2).sum(dim=1)

    logZ = mass * lam - log_diff.sum(dim=1) \
        - 0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(S2)

    return logZ


def compute_y_moments_saddlepoint_gpu(
    X: torch.Tensor,
    C: torch.Tensor,
    scale: float,
    M: int,
    logZy_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = scale * torch.matmul(X, C)
    mass = float(M)

    lam = _saddlepoint_solve_lambda(A, mass)

    inv = 1.0 / (lam.unsqueeze(1) - A)

    inv2 = inv ** 2
    inv3 = inv2 * inv
    S2 = inv2.sum(dim=1, keepdim=True)
    S3 = inv3.sum(dim=1, keepdim=True)

    y_mean = inv - inv3 / S2 + inv2 * S3 / (S2 ** 2)

    var = inv2 - inv2 ** 2 / S2
    y_second = y_mean ** 2 + var

    y_mean = y_mean.clamp(min=0.0, max=mass)
    y_second = y_second.clamp(min=0.0, max=mass * mass)

    return y_mean, y_second


def _sample_trunc_exp_gpu(rate: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(rate)
    ru = rate * upper

    y_exact = torch.log1p(u * torch.expm1(ru)) / rate

    y_taylor = u * upper * (1.0 + 0.5 * ru * (1.0 - u) +
                             ru * ru * (1.0 - u) * (1.0 - 2.0 * u) / 6.0)

    small = ru.abs() < 1e-6
    y = torch.where(small, y_taylor, y_exact)

    upper_safe = upper.clamp(min=0.0)
    y = torch.clamp(torch.clamp(y, min=0.0), max=upper_safe)
    return y


def compute_y_moments_gibbs_gpu(
    X: torch.Tensor,
    C: torch.Tensor,
    scale: float,
    M: int,
    logZy_base: torch.Tensor,
    n_burnin: int = 100,
    n_sweeps: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_chains = X.shape[0]
    M_dim = M
    mass = float(M)
    A = scale * torch.matmul(X, C)
    device = X.device

    y = torch.full((n_chains, M_dim), mass / M_dim, dtype=DTYPE, device=device)

    sum_y = torch.zeros_like(y)
    sum_y2 = torch.zeros_like(y)

    total = n_burnin + n_sweeps
    n_pairs = M_dim // 2

    for sweep in range(total):
        perm = torch.randperm(M_dim, device=device)
        for p in range(n_pairs):
            i = perm[2 * p].item()
            j = perm[2 * p + 1].item()

            c = y[:, i] + y[:, j]
            rate = A[:, i] - A[:, j]
            t = _sample_trunc_exp_gpu(rate, c)
            y[:, i] = t
            y[:, j] = c - t

        if sweep >= n_burnin:
            sum_y += y
            sum_y2 += y ** 2

    y_mean = sum_y / n_sweeps
    y_second = sum_y2 / n_sweeps

    y_mean = y_mean.clamp(min=0.0, max=mass)
    y_second = y_second.clamp(min=0.0, max=mass * mass)

    return y_mean, y_second


def _log_dirichlet_density_batch(
    p: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    log_B = torch.lgamma(alpha).sum(dim=1) - torch.lgamma(alpha.sum(dim=1))
    return ((alpha - 1.0) * torch.log(p.clamp(min=1e-300))).sum(dim=1) - log_B


def _mcmc_step_block_dirichlet_batch(
    X: torch.Tensor,
    logZy: torch.Tensor,
    log_target_coeff: float,
    C: torch.Tensor,
    scale: float,
    M: int,
    kappa: float = 100.0,
    epsilon: float = 0.01,
    block_size: int = 5,
    logZy_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logZy_fn is None:
        logZy_fn = _compute_logZy_batch

    n_chains, N = X.shape
    device = X.device
    bs = min(block_size, N - 1)
    D = bs + 1

    rand_keys = torch.rand(n_chains, N, dtype=DTYPE, device=device)
    idx = rand_keys.argsort(dim=1)[:, :D]

    row_idx = torch.arange(n_chains, device=device).unsqueeze(1).expand(-1, D)
    x_sub = X[row_idx, idx]

    sub_mass = x_sub.sum(dim=1, keepdim=True)
    valid = sub_mass[:, 0] > 1e-15

    p_sub = x_sub / sub_mass.clamp(min=1e-15)

    alpha_fwd = kappa * p_sub + epsilon

    p_sub_prop = torch.zeros_like(alpha_fwd)
    gamma_samples = torch._standard_gamma(alpha_fwd)
    gamma_sum = gamma_samples.sum(dim=1, keepdim=True)
    p_sub_prop = gamma_samples / gamma_sum.clamp(min=1e-300)

    x_sub_prop = sub_mass * p_sub_prop

    X_prop = X.clone()
    X_prop[row_idx, idx] = x_sub_prop

    logZy_prop = logZy_fn(X_prop, C, scale, M)

    alpha_rev = kappa * p_sub_prop + epsilon
    log_target_ratio = log_target_coeff * (logZy_prop - logZy)
    log_proposal_ratio = (
        _log_dirichlet_density_batch(p_sub, alpha_rev)
        - _log_dirichlet_density_batch(p_sub_prop, alpha_fwd)
    )
    log_alpha = log_target_ratio + log_proposal_ratio

    log_u = torch.log(torch.rand(n_chains, dtype=DTYPE, device=device))
    accept = (log_u < log_alpha) & valid

    X_new = torch.where(accept.unsqueeze(1), X_prop, X)
    logZy_new = torch.where(accept, logZy_prop, logZy)

    return X_new, logZy_new, accept


def _mcmc_step_pairwise_batch(
    X: torch.Tensor,
    logZy: torch.Tensor,
    log_target_coeff: float,
    C: torch.Tensor,
    scale: float,
    M: int,
    step_size: float = 0.5,
    logZy_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logZy_fn is None:
        logZy_fn = _compute_logZy_batch

    n_chains, N = X.shape
    device = X.device

    i_idx = torch.randint(0, N, (n_chains,), device=device)
    j_idx = torch.randint(0, N - 1, (n_chains,), device=device)
    j_idx = torch.where(j_idx >= i_idx, j_idx + 1, j_idx)

    delta = torch.empty(n_chains, dtype=DTYPE, device=device).uniform_(-step_size, step_size)

    arange = torch.arange(n_chains, device=device)
    xi = X[arange, i_idx]
    xj = X[arange, j_idx]
    xi_new = xi + delta
    xj_new = xj - delta

    valid = (xi_new >= 0.0) & (xj_new >= 0.0)

    X_prop = X.clone()
    X_prop[arange, i_idx] = xi_new
    X_prop[arange, j_idx] = xj_new

    logZy_prop = logZy_fn(X_prop, C, scale, M)

    log_alpha_val = log_target_coeff * (logZy_prop - logZy)
    log_u = torch.log(torch.rand(n_chains, dtype=DTYPE, device=device))
    accept = (log_u < log_alpha_val) & valid

    X_new = X.clone()
    X_new[arange, i_idx] = torch.where(accept, xi_new, xi)
    X_new[arange, j_idx] = torch.where(accept, xj_new, xj)
    logZy_new = torch.where(accept, logZy_prop, logZy)

    return X_new, logZy_new, accept


@dataclass(frozen=True)
class AISResult:
    Phi: float
    Phi_over_L: float
    logZ_outer: float
    mc_log_mean_weight: float
    log_ess: float
    ess_fraction: float
    mean_acceptance_rate: float
    n_chains: int
    n_temps: int
    n_mcmc: int
    Qx: Optional[float] = None
    qx: Optional[float] = None
    Qy: Optional[float] = None
    q1: Optional[float] = None
    energy: Optional[float] = None


def _divdiff_logZ(a: torch.Tensor, mass: float) -> torch.Tensor:
    B, M_dim = a.shape

    a_sorted, _ = torch.sort(a, dim=1)

    if M_dim >= 2:
        diffs = a_sorted[:, 1:] - a_sorted[:, :-1]
        min_diffs = diffs.abs().min(dim=1).values
        needs_jitter = min_diffs < 1e-12
        if needs_jitter.any().item():
            stds = a_sorted.std(dim=1, keepdim=True).clamp(min=1.0)
            eps_j = 1e-9 * stds
            jv = torch.arange(M_dim, dtype=a.dtype, device=a.device)
            jv = eps_j * (jv - (M_dim - 1) / 2.0) / M_dim
            mask = needs_jitter.unsqueeze(1)
            a_sorted = a_sorted + torch.where(mask, jv, torch.zeros_like(jv))

    a_max = a_sorted.max(dim=1, keepdim=True).values
    a_shift = a_sorted - a_max

    d = torch.exp(mass * a_shift)

    DENOM_MIN = 1e-12
    for r in range(1, M_dim):
        denom = a_sorted[:, r:] - a_sorted[:, :M_dim - r]
        safe_denom = torch.where(denom.abs() < DENOM_MIN,
                                 torch.full_like(denom, DENOM_MIN), denom)
        num = d[:, 1:] - d[:, :-1]
        d = num / safe_denom

    Z_scaled = d[:, 0]
    logZ = mass * a_max[:, 0] + torch.log(Z_scaled.abs().clamp(min=1e-300))
    return logZ


def compute_y_moments_autograd_gpu(
    X: torch.Tensor,
    C: torch.Tensor,
    scale: float,
    M: int,
    logZy_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_chains = X.shape[0]
    mass = float(M)

    A = (scale * torch.matmul(X.detach(), C)).detach().requires_grad_(True)

    logZ = _divdiff_logZ(A, mass)

    grad1 = torch.autograd.grad(
        logZ.sum(), A, create_graph=True, retain_graph=True,
    )[0]

    hess_diag = torch.zeros_like(A)
    for j in range(M):
        g2 = torch.autograd.grad(
            grad1[:, j].sum(), A, retain_graph=(j < M - 1),
        )[0]
        hess_diag[:, j] = g2[:, j]

    y_mean = grad1.detach()
    var = hess_diag.detach()
    y_second = var + y_mean ** 2

    y_mean.clamp_(min=0.0, max=mass)
    y_second.clamp_(min=0.0, max=mass * mass)

    return y_mean, y_second


def _fd_at_eps(
    A_base: torch.Tensor,
    M: int,
    n_chains: int,
    mass: float,
    logZ0: torch.Tensor,
    eps: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    A_base_rep = A_base.unsqueeze(0).expand(M, -1, -1).reshape(M * n_chains, M)

    eps_perturbation = (eps * torch.eye(M, dtype=DTYPE, device=device)
                        .unsqueeze(1)
                        .expand(-1, n_chains, -1)
                        .reshape(M * n_chains, M))

    A_plus_all = A_base_rep + eps_perturbation
    A_minus_all = A_base_rep - eps_perturbation

    logZ_plus_all = batch_simplex_exp_integral_divdiff_gpu(A_plus_all, mass=mass).logZ
    logZ_minus_all = batch_simplex_exp_integral_divdiff_gpu(A_minus_all, mass=mass).logZ

    logZ_plus = logZ_plus_all.reshape(M, n_chains).t()
    logZ_minus = logZ_minus_all.reshape(M, n_chains).t()

    y_mean = (logZ_plus - logZ_minus) / (2.0 * eps)
    logZ0_expanded = logZ0.unsqueeze(1)
    var = (logZ_plus - 2.0 * logZ0_expanded + logZ_minus) / (eps * eps)
    y_second = var + y_mean ** 2

    return y_mean, y_second


def compute_y_moments_batch_gpu(
    X: torch.Tensor,
    C: torch.Tensor,
    scale: float,
    M: int,
    logZy_base: torch.Tensor,
    eps: float = 0.01,
    richardson: bool = True,
    use_autograd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_autograd:
        try:
            return compute_y_moments_autograd_gpu(
                X, C, scale, M, logZy_base,
            )
        except Exception:
            pass

    n_chains = X.shape[0]
    mass = float(M)
    device = X.device
    A_base = scale * torch.matmul(X, C)

    logZ0 = batch_simplex_exp_integral_divdiff_gpu(A_base, mass=mass).logZ

    y_mean_h, y_sec_h = _fd_at_eps(A_base, M, n_chains, mass, logZ0, eps, device)

    if richardson:
        y_mean_h2, y_sec_h2 = _fd_at_eps(A_base, M, n_chains, mass, logZ0, eps / 2.0, device)
        y_mean = (4.0 * y_mean_h2 - y_mean_h) / 3.0
        y_second = (4.0 * y_sec_h2 - y_sec_h) / 3.0
    else:
        y_mean = y_mean_h
        y_second = y_sec_h

    y_mean.clamp_(min=0.0, max=mass)
    y_second.clamp_(min=0.0, max=mass * mass)

    return y_mean, y_second


def estimate_Phi_ais_for_C(
    C: np.ndarray,
    *,
    sigma: float,
    beta_max: float,
    beta_min: float,
    n_chains: int = 500,
    n_temps: int = 500,
    n_mcmc: int = 3,
    schedule: str = "sigmoid",
    proposal: str = "block_dirichlet",
    kappa: Optional[float] = None,
    epsilon: float = 0.01,
    block_size: int = 5,
    pairwise_step: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    use_longdouble: bool = True,
    verbose: bool = False,
    device: torch.device = None,
    logZ_method: str = "auto",
    moment_method: str = "auto",
) -> AISResult:
    if device is None:
        device = DEVICE

    C_np = np.asarray(C, dtype=np.float64)
    N, M_dim = C_np.shape
    L = float(np.sqrt(N * M_dim))
    k = -beta_min / beta_max
    scale_coeff = beta_max * np.sqrt(sigma / L)

    if logZ_method == "auto":
        a_scale = scale_coeff * math.sqrt(M_dim)
        logZ_method = "saddlepoint" if a_scale < 0.5 else "dd"
    if moment_method == "auto":
        moment_method = "saddlepoint"

    if logZ_method == "saddlepoint":
        _logZy_fn = _compute_logZy_saddlepoint_batch
    else:
        _logZy_fn = _compute_logZy_batch

    if kappa is None:
        if proposal == "block_dirichlet":
            kappa = 5.0 * (block_size + 1)
        else:
            kappa = float(N * N)

    betas = make_schedule(n_temps, schedule)

    C_gpu = torch.tensor(C_np, dtype=DTYPE, device=device)

    if rng is not None:
        torch.manual_seed(int(rng.integers(0, 2**31)))

    X = sample_uniform_simplex(dim=N, mass=float(N), n_samples=n_chains, device=device)
    logZy = _logZy_fn(X, C_gpu, scale_coeff, M_dim)

    log_ais_weights = torch.zeros(n_chains, dtype=DTYPE, device=device)
    gpu_total_accepted = torch.zeros(1, dtype=DTYPE, device=device)
    total_proposed = 0

    for t_idx in range(1, n_temps + 1):
        db = float(betas[t_idx] - betas[t_idx - 1])
        log_ais_weights += (k * db) * logZy

        log_tc = k * float(betas[t_idx])

        for _ in range(n_mcmc):
            if proposal == "pairwise":
                X, logZy, acc = _mcmc_step_pairwise_batch(
                    X, logZy, log_tc, C_gpu, scale_coeff, M_dim,
                    step_size=pairwise_step,
                    logZy_fn=_logZy_fn,
                )
                gpu_total_accepted += acc.sum()
                total_proposed += n_chains
            elif proposal == "block_dirichlet":
                n_blocks = int(math.ceil(N / (block_size + 1)))
                for _ in range(n_blocks):
                    X, logZy, acc = _mcmc_step_block_dirichlet_batch(
                        X, logZy, log_tc, C_gpu, scale_coeff, M_dim,
                        kappa=kappa, epsilon=epsilon, block_size=block_size,
                        logZy_fn=_logZy_fn,
                    )
                    gpu_total_accepted += acc.sum()
                    total_proposed += n_chains
            elif proposal == "mixed":
                n_blocks = int(math.ceil(N / (block_size + 1)))
                for _ in range(n_blocks):
                    X, logZy, acc = _mcmc_step_block_dirichlet_batch(
                        X, logZy, log_tc, C_gpu, scale_coeff, M_dim,
                        kappa=kappa, epsilon=epsilon, block_size=block_size,
                        logZy_fn=_logZy_fn,
                    )
                    gpu_total_accepted += acc.sum()
                    total_proposed += n_chains
                for _ in range(N):
                    X, logZy, acc = _mcmc_step_pairwise_batch(
                        X, logZy, log_tc, C_gpu, scale_coeff, M_dim,
                        step_size=pairwise_step,
                        logZy_fn=_logZy_fn,
                    )
                    gpu_total_accepted += acc.sum()
                    total_proposed += n_chains
            elif proposal == "dirichlet":
                X, logZy, acc = _mcmc_step_block_dirichlet_batch(
                    X, logZy, log_tc, C_gpu, scale_coeff, M_dim,
                    kappa=kappa, epsilon=epsilon, block_size=N - 1,
                    logZy_fn=_logZy_fn,
                )
                gpu_total_accepted += acc.sum()
                total_proposed += n_chains
            else:
                raise ValueError(f"Unknown proposal: {proposal}")

        if verbose and t_idx % max(1, n_temps // 10) == 0:
            mean_w = log_ais_weights.mean().item()
            print(f"  AIS temp {t_idx}/{n_temps}, mean log_w={mean_w:.4f}", flush=True)

    total_accepted = int(gpu_total_accepted.item())

    log_w_cpu = log_ais_weights.cpu().numpy()
    from scipy.special import logsumexp as logsumexp_cpu

    logVolX = log_simplex_volume(dim=N, mass=float(N))
    mc_log_mean_w = float(logsumexp_cpu(log_w_cpu) - np.log(n_chains))
    logZ_outer = float(logVolX + mc_log_mean_w)

    Phi = -(1.0 / beta_min) * logZ_outer
    Phi_over_L = Phi / L

    log_ess = float(
        2.0 * logsumexp_cpu(log_w_cpu) - logsumexp_cpu(2.0 * log_w_cpu)
    )
    ess_fraction = float(np.exp(log_ess) / n_chains) if n_chains > 0 else 0.0
    mean_acc_rate = total_accepted / max(total_proposed, 1)

    log_w_norm = log_ais_weights - torch.logsumexp(log_ais_weights, dim=0)
    w_norm = torch.exp(log_w_norm)

    x2_per_chain = torch.mean(X ** 2, dim=1)
    Qx_t = torch.dot(w_norm, x2_per_chain)

    x_mean = torch.sum(w_norm.unsqueeze(1) * X, dim=0)
    qx_t = torch.mean(x_mean ** 2)

    if moment_method == "saddlepoint":
        y_mean_all, y_second_all = compute_y_moments_saddlepoint_gpu(
            X, C_gpu, scale_coeff, M_dim, logZy,
        )
    elif moment_method == "gibbs":
        y_mean_all, y_second_all = compute_y_moments_gibbs_gpu(
            X, C_gpu, scale_coeff, M_dim, logZy,
        )
    elif moment_method == "autograd":
        y_mean_all, y_second_all = compute_y_moments_autograd_gpu(
            X, C_gpu, scale_coeff, M_dim, logZy,
        )
    else:
        y_mean_all, y_second_all = compute_y_moments_batch_gpu(
            X, C_gpu, scale_coeff, M_dim, logZy,
        )
    Qy_per_chain = torch.mean(y_second_all, dim=1)
    Qy_t = torch.dot(w_norm, Qy_per_chain)

    q1_per_chain = torch.mean(y_mean_all ** 2, dim=1)
    q1_t = torch.dot(w_norm, q1_per_chain)

    XC = torch.matmul(X, C_gpu)
    energy_coeff = math.sqrt(sigma) / (L ** 1.5)
    energy_per_chain = (XC * y_mean_all).sum(dim=1) * energy_coeff
    energy_t = torch.dot(w_norm, energy_per_chain)

    op_vals = torch.stack([Qx_t, qx_t, Qy_t, q1_t, energy_t]).tolist()
    Qx_est, qx_est, Qy_est, q1_est, energy_est = op_vals

    return AISResult(
        Phi=Phi, Phi_over_L=Phi_over_L,
        logZ_outer=logZ_outer, mc_log_mean_weight=mc_log_mean_w,
        log_ess=log_ess, ess_fraction=ess_fraction,
        mean_acceptance_rate=mean_acc_rate,
        n_chains=n_chains, n_temps=n_temps, n_mcmc=n_mcmc,
        Qx=Qx_est, qx=qx_est, Qy=Qy_est, q1=q1_est,
        energy=energy_est,
    )
