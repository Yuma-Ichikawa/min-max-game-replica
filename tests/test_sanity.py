"""
Minimal sanity checks.

Run:
    python -m tests.test_sanity
"""
from __future__ import annotations

import numpy as np

from src.divided_differences import simplex_exp_integral_divdiff
from src.simplex import log_simplex_volume


def test_divdiff_M2():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a1, a2 = rng.normal(size=2)
        M = 2.0
        res = simplex_exp_integral_divdiff(np.array([a1, a2]), mass=M)
        exact = (np.exp(M*a1) - np.exp(M*a2)) / (a1 - a2)
        rel = abs(res.Z - exact) / (1.0 + abs(exact))
        assert rel < 1e-10, (res.Z, exact, rel)


def test_simplex_volume_small():
    # For dim=2, mass=2: simplex is a line segment from (0,2) to (2,0), length sqrt(8)?
    # But our "volume" is the Lebesgue measure on the constraint hyperplane with delta,
    # which gives Vol = mass^(dim-1)/(dim-1)! = 2^1/1 = 2.
    lv = log_simplex_volume(dim=2, mass=2.0)
    assert abs(np.exp(lv) - 2.0) < 1e-12


def main():
    test_divdiff_M2()
    test_simplex_volume_small()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
