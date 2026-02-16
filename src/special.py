from __future__ import annotations

import math
import torch

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def phi(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x - LOG_SQRT_2PI)


def Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-x / _SQRT2)


def logPhi(x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=DTYPE)
    if hasattr(torch.special, 'log_ndtr'):
        return torch.special.log_ndtr(x)
    result = torch.log(torch.clamp(Phi(x), min=1e-300))
    mask = x < -20.0
    if mask.any():
        xm = x[mask]
        result[mask] = (-0.5 * xm * xm - torch.log(-xm) - LOG_SQRT_2PI
                        + torch.log1p(-1.0 / (xm * xm)))
    return result


def logdiffexp(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    x = log_b - log_a
    safe = log_a > log_b
    out = torch.full_like(log_a, -math.inf)
    if safe.any():
        xs = x[safe]
        tiny = xs > -1e-8
        result = torch.empty_like(xs)
        if tiny.any():
            xt = xs[tiny]
            result[tiny] = torch.log(-xt) + xt * 0.5
        big = ~tiny
        if big.any():
            xb = xs[big].clamp(min=-745.0)
            result[big] = torch.log1p(-torch.exp(xb))
        out[safe] = log_a[safe] + result
    return out
