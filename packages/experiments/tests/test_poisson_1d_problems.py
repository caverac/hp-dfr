"""Unit tests for 1D Poisson benchmark problems."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from hp_dfr.problems.poisson_1d import ArcTanProblem


def test_arctan_exact_solution_satisfies_homogeneous_dirichlet_bcs() -> None:
    prob = ArcTanProblem(steepness=10.0)
    a, b = prob.domain

    u_a = prob.exact_solution(np.array([a], dtype=np.float64))[0]
    u_b = prob.exact_solution(np.array([b], dtype=np.float64))[0]

    assert_allclose(u_a, 0.0, atol=1e-12, rtol=0.0)
    assert_allclose(u_b, 0.0, atol=1e-12, rtol=0.0)


def test_arctan_forcing_matches_second_derivative() -> None:
    """Check that u'' + f ≈ 0 via a finite difference stencil."""

    prob = ArcTanProblem(steepness=10.0)
    a, b = prob.domain

    n = 5001
    x = np.linspace(a, b, n, dtype=np.float64)
    h = (b - a) / (n - 1)

    u = prob.exact_solution(x)
    f = prob.forcing_fn(x[1:-1])

    # Central second-difference approximation of u'' on interior points.
    d2u = (u[:-2] - 2.0 * u[1:-1] + u[2:]) / (h**2)

    residual = d2u + f

    # Finite-difference truncation error is O(h^2). Use a conservative tolerance.
    assert float(np.max(np.abs(residual))) < 5e-4
