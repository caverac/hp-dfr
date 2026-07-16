"""The discrete DFR loss must vanish at the exact solution of every swept problem.

This is the correctness check the manuscript relies on: the global minimum of the
DFR loss sits at ``u*``, so any failure to reach it is an optimization failure
rather than a wrongly assembled loss.

The Laplacian of the exact solution is computed by finite differences, which is
*independent* of the closed form used by each problem's ``forcing_fn``. A test
that differentiated the same expression twice would be trivially satisfied and
could not detect a mismatch between ``exact_solution`` and ``forcing_fn``.

Absolute tolerances would be meaningless here because the loss carries the scale
of the problem, so each case is compared against the loss of a perturbed
solution: the exact solution must sit orders of magnitude below it.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from hp_dfr.fourier import dst_matrix_nd_full, h_minus_1_weights
from hp_dfr.problems.poisson_1d import ArcTanProblem, SineProblem
from hp_dfr.problems.poisson_2d import ArcTanBump2D

Float64Array = NDArray[np.float64]
Domain = Tuple[Tuple[float, float], ...]

# Balances central-difference truncation (order h^2) against float64 roundoff
# (order eps / h^2); both land near 1e-8 at this value.
_FD_STEP = 1.0e-4


def _midpoint_grid(domain: Domain, n_quadrature: int) -> Float64Array:
    """Build the tensor midpoint-rule grid the DFR models quadrature on."""
    axes = []
    for lo, hi in domain:
        edges = np.linspace(lo, hi, n_quadrature + 1)
        axes.append(edges[:-1] + (edges[1] - edges[0]) / 2.0)
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def _fd_laplacian(u_fn: Callable[[Float64Array], Float64Array], pts: Float64Array) -> Float64Array:
    """Central-difference Laplacian of ``u_fn`` at ``pts``, independent of forcing_fn."""
    dim = pts.shape[1]
    laplacian = np.zeros(pts.shape[0], dtype=np.float64)
    for d in range(dim):
        step = np.zeros(dim, dtype=np.float64)
        step[d] = _FD_STEP
        laplacian += (u_fn(pts + step) - 2.0 * u_fn(pts) + u_fn(pts - step)) / _FD_STEP**2
    return laplacian


def _dfr_loss(residual: Float64Array, domain: Domain, n_quadrature: int, n_modes: int) -> float:
    """Discrete DFR loss ``sum_k |R_k|^2 / lambda_k`` of a residual on the grid."""
    dim = len(domain)
    dst = dst_matrix_nd_full((n_quadrature,) * dim, domain, n_modes)
    weights = h_minus_1_weights((n_modes,) * dim, domain).ravel()
    return float(np.sum((dst @ residual * weights) ** 2))


def _weak_residual(
    u_fn: Callable[[Float64Array], Float64Array],
    f_fn: Callable[[Float64Array], Float64Array],
    pts: Float64Array,
) -> Float64Array:
    """Weak residual ``f + Laplacian(u)``, which vanishes at the exact solution."""
    return np.asarray(f_fn(pts)).ravel() + _fd_laplacian(u_fn, pts)


def _as_1d(fn: Callable[[Float64Array], Float64Array]) -> Callable[[Float64Array], Float64Array]:
    """Adapt a 1D problem callable to the (N, 1) point layout used here."""
    return lambda pts: np.asarray(fn(pts[:, 0])).ravel()


# Each case: name, exact solution, forcing, domain, quadrature and mode counts.
# The 1D counts mirror data.py's sweep (60 modes, 150 points); the 2D counts
# mirror its 2D sweep (16 modes and 32 points per dimension).
_CASES = [
    pytest.param(
        _as_1d(SineProblem().exact_solution),
        _as_1d(SineProblem().forcing_fn),
        (SineProblem().domain,),
        150,
        60,
        id="sine-1d",
    ),
    pytest.param(
        _as_1d(ArcTanProblem(8.0).exact_solution),
        _as_1d(ArcTanProblem(8.0).forcing_fn),
        (ArcTanProblem(8.0).domain,),
        150,
        60,
        id="arctan8-1d",
    ),
    pytest.param(
        ArcTanBump2D(8.0).exact_solution,
        ArcTanBump2D(8.0).forcing_fn,
        ArcTanBump2D(8.0).domain,
        32,
        16,
        id="arctanbump8-2d",
    ),
]


@pytest.mark.parametrize("u_fn, f_fn, domain, n_quadrature, n_modes", _CASES)
def test_dfr_loss_vanishes_at_exact_solution(
    u_fn: Callable[[Float64Array], Float64Array],
    f_fn: Callable[[Float64Array], Float64Array],
    domain: Domain,
    n_quadrature: int,
    n_modes: int,
) -> None:
    """The exact solution must sit far below a perturbed one in the DFR loss."""
    pts = _midpoint_grid(domain, n_quadrature)

    exact_loss = _dfr_loss(_weak_residual(u_fn, f_fn, pts), domain, n_quadrature, n_modes)

    # A 1% multiplicative perturbation still satisfies the boundary conditions,
    # so any gap comes from the PDE residual rather than the cutoff.
    perturbed = _dfr_loss(_weak_residual(lambda x: 1.01 * u_fn(x), f_fn, pts), domain, n_quadrature, n_modes)

    assert exact_loss < 1.0e-8 * perturbed, f"DFR loss at u* is {exact_loss:.3e}, not negligible against {perturbed:.3e}"


@pytest.mark.parametrize("u_fn, f_fn, domain, n_quadrature, n_modes", _CASES)
def test_dfr_loss_grows_with_perturbation_size(
    u_fn: Callable[[Float64Array], Float64Array],
    f_fn: Callable[[Float64Array], Float64Array],
    domain: Domain,
    n_quadrature: int,
    n_modes: int,
) -> None:
    """The loss must increase with the perturbation, so u* is a genuine minimum."""
    pts = _midpoint_grid(domain, n_quadrature)

    losses = [
        _dfr_loss(_weak_residual(lambda x, s=scale: s * u_fn(x), f_fn, pts), domain, n_quadrature, n_modes) for scale in (1.001, 1.01, 1.1)
    ]

    assert losses[0] < losses[1] < losses[2]
