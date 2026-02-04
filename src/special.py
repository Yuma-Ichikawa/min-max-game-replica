"""
Special functions and numerically stable helpers.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import log_ndtr


LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def phi(x: np.ndarray) -> np.ndarray:
    """Standard normal pdf."""
    x = np.asarray(x)
    return np.exp(-0.5 * x * x - LOG_SQRT_2PI)


def Phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    # scipy.special.ndtr would also work; using erf is fine, but keep SciPy consistent
    from scipy.special import ndtr
    return ndtr(x)


def logPhi(x: np.ndarray) -> np.ndarray:
    """log Φ(x) computed stably."""
    x = np.asarray(x)
    return log_ndtr(x)


def logdiffexp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return log(exp(a) - exp(b)) for a>b elementwise, stably.

    Raises if a <= b (up to tolerance).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if np.any(a <= b):
        raise ValueError("logdiffexp requires a > b elementwise.")
    # log(exp(a) - exp(b)) = a + log(1 - exp(b-a))
    return a + np.log1p(-np.exp(b - a))


def safe_logsumexp(logw: np.ndarray, axis=None) -> np.ndarray:
    """
    A thin wrapper around scipy.special.logsumexp.
    """
    from scipy.special import logsumexp
    return logsumexp(logw, axis=axis)
