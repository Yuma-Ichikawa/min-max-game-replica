from __future__ import annotations

import math

import numpy as np
import torch

from .special import DEVICE, DTYPE


def log_simplex_volume(dim: int, mass: float) -> float:
    if dim < 1:
        raise ValueError("dim must be >= 1")
    mass = float(mass)
    if mass <= 0:
        raise ValueError("mass must be > 0")
    return (dim - 1) * math.log(mass) - math.lgamma(float(dim))


def sample_uniform_simplex(
    dim: int,
    mass: float,
    n_samples: int,
    device: torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = DEVICE
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    mass = float(mass)
    if mass <= 0:
        raise ValueError("mass must be > 0")

    alpha = torch.ones(dim, dtype=DTYPE, device=device)
    dist = torch.distributions.Dirichlet(alpha)
    p = dist.sample((n_samples,))
    return mass * p
