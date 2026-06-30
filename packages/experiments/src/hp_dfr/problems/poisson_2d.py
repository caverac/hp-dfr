"""2D Poisson problems on the unit box for goal-oriented DFR experiments.

Provides manufactured solutions on [0, 1]^2 with homogeneous Dirichlet boundary
conditions, exposing the same interface the models expect:
``domain`` (tuple of per-axis bounds), ``forcing_fn(xy)`` and
``exact_solution(xy)`` taking arrays of shape (N, 2).
"""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]


class ArcTanBump2D:
    r"""Separable sharp-bump manufactured solution on [0, 1]^2.

    The exact solution is ``u(x, y) = A(x) A(y)`` where ``A`` is the 1D arctan
    profile ``A(t) = arctan(k (t - 1/2)) - L(t)`` with ``L`` the affine function
    making ``A(0) = A(1) = 0``. For large ``k`` the solution has a sharp,
    localized feature near the center, so resolving it globally with a limited
    Fourier-mode budget is hard. The forcing is ``f = -Laplacian(u)``.
    """

    def __init__(self, steepness: float = 8.0) -> None:
        self.k = float(steepness)
        self.domain: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.0, 1.0),
            (0.0, 1.0),
        )
        # Affine correction so the 1D profile vanishes at t = 0 and t = 1.
        k = self.k
        self._ua = float(np.arctan(k * (0.0 - 0.5)))
        self._ub = float(np.arctan(k * (1.0 - 0.5)))
        self._slope = self._ub - self._ua  # over [0, 1]

    def _a1d(self, t: Float64Array) -> Float64Array:
        k = self.k
        raw = np.arctan(k * (t - 0.5))
        affine = self._ua + self._slope * t
        return cast(Float64Array, raw - affine)

    def _a1d_second(self, t: Float64Array) -> Float64Array:
        # d^2/dt^2 arctan(k (t - 1/2)); the affine part has zero second derivative.
        k = self.k
        z = k * (t - 0.5)
        return -2.0 * k**2 * z / (1.0 + z**2) ** 2

    def exact_solution(self, xy: Float64Array) -> Float64Array:
        """Evaluate the exact solution at points ``xy`` of shape (N, 2)."""
        xy = np.atleast_2d(xy)
        x, y = xy[:, 0], xy[:, 1]
        return self._a1d(x) * self._a1d(y)

    def forcing_fn(self, xy: Float64Array) -> Float64Array:
        """Evaluate the forcing ``f = -Laplacian(u)`` at points ``xy`` of shape (N, 2)."""
        # f = -(u_xx + u_yy) = -(A''(x) A(y) + A(x) A''(y)).
        xy = np.atleast_2d(xy)
        x, y = xy[:, 0], xy[:, 1]
        lap = self._a1d_second(x) * self._a1d(y) + self._a1d(x) * self._a1d_second(y)
        return -lap
