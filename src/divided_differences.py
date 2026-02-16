from __future__ import annotations

from dataclasses import dataclass

import torch

from .special import DEVICE, DTYPE


@dataclass(frozen=True)
class BatchDivDiffResult:
    logZ: torch.Tensor

    used_jitter: bool


def batch_simplex_exp_integral_divdiff_gpu(
    a_batch: torch.Tensor,
    mass: float,
    *,
    jitter_tol: float = 1e-12,
    jitter_scale: float = 1e-9,
) -> BatchDivDiffResult:
    if a_batch.ndim != 2:
        raise ValueError("a_batch must be 2D (B, M)")
    B, M = a_batch.shape
    mass = float(mass)

    a, _ = torch.sort(a_batch, dim=1)

    used_jitter = False
    if M >= 2:
        diffs = a[:, 1:] - a[:, :-1]  # (B, M-1)
        min_diffs = diffs.abs().min(dim=1).values  # (B,)
        row_needs_jitter = min_diffs < jitter_tol
        if row_needs_jitter.any().item():
            used_jitter = True

        stds = a.std(dim=1, keepdim=True).clamp(min=1.0)
        eps_large = jitter_scale * stds
        eps_small = 1e-14 * stds
        mask = row_needs_jitter.unsqueeze(1)
        eps = torch.where(mask, eps_large, eps_small)
        jitter_idx = torch.arange(M, dtype=DTYPE, device=a.device)
        jitter_vec = eps * (jitter_idx - (M - 1) / 2.0) / M
        a = a + jitter_vec

    a_max, _ = a.max(dim=1, keepdim=True)
    a_shift = a - a_max

    d = torch.exp(mass * a_shift)

    DENOM_MIN = 1e-30
    denom_min_t = torch.tensor(DENOM_MIN, dtype=DTYPE, device=a.device)
    for r in range(1, M):
        denom = a[:, r:] - a[:, :M - r]
        denom = torch.where(denom.abs() < DENOM_MIN, denom_min_t, denom)
        num = d[:, 1:] - d[:, :-1]
        d = num / denom

    Z_scaled = d[:, 0].abs().clamp(min=1e-300)
    logZ = mass * a_max[:, 0] + torch.log(Z_scaled)

    return BatchDivDiffResult(logZ=logZ, used_jitter=used_jitter)
