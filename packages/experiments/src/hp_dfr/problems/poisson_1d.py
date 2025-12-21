"""1D Poisson problem definitions.

This module provides problem classes for the 1D Poisson equation used in
benchmarking PINNs and DFR methods. Each problem defines a forcing function
and (when available) an exact solution.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, cast

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]

T = TypeVar("T")


class Poisson1D(ABC, Generic[T]):
    """Abstract base class for 1D Poisson problems.

    Defines the 1D Poisson equation -u''(x) = f(x) on an interval [a, b]
    with homogeneous Dirichlet boundary conditions u(a) = u(b) = 0.

    Parameters
    ----------
    params : T
        Problem-specific parameters (e.g., steepness, coefficients).
    domain : tuple[float, float], optional
        The spatial domain [a, b], by default (0.0, 1.0).

    Attributes
    ----------
    domain : tuple[float, float]
        The spatial domain [a, b].
    params : T
        Problem-specific parameters.

    Notes
    -----
    Subclasses must implement `forcing_fn` and `exact_solution` methods.
    """

    domain: tuple[float, float]
    params: T

    def __init__(self, params: T, domain: tuple[float, float] = (0.0, 1.0)) -> None:
        self.params = params
        self.domain = domain

    @abstractmethod
    def forcing_fn(self, x: Float64Array) -> Float64Array:
        """Evaluate the forcing function f(x).

        Parameters
        ----------
        x : Float64Array
            Points at which to evaluate f(x).

        Returns
        -------
        Float64Array
            Values of f(x) at the given points.
        """

    @abstractmethod
    def exact_solution(self, x: Float64Array) -> Float64Array:
        """Evaluate the exact solution u(x).

        Parameters
        ----------
        x : Float64Array
            Points at which to evaluate u(x).

        Returns
        -------
        Float64Array | None
            Values of u(x) at the given points, or None if unknown.
        """


class SineProblem(Poisson1D[None]):
    """Model Problem 1: Smooth sine solution.

    Solves -u'' = 4*sin(2x) on [0, pi] with u(0) = u(pi) = 0.
    The exact solution is u(x) = sin(2x).

    This problem has a smooth solution and serves as a baseline test.
    Both PINNs and DFR should achieve high accuracy.

    Examples
    --------
    >>> problem = SineProblem()
    >>> x = np.linspace(0, np.pi, 100)
    >>> f = problem.forcing_fn(x)
    >>> u = problem.exact_solution(x)
    """

    def __init__(self) -> None:
        super().__init__(params=None, domain=(0.0, np.pi))

    def forcing_fn(self, x: Float64Array) -> Float64Array:
        """Evaluate f(x) = 4*sin(2x).

        Parameters
        ----------
        x : Float64Array
            Points at which to evaluate.

        Returns
        -------
        Float64Array
            Values of 4*sin(2x).
        """
        return cast(Float64Array, 4.0 * np.sin(2.0 * x))

    def exact_solution(self, x: Float64Array) -> Float64Array:
        """Evaluate u(x) = sin(2x).

        Parameters
        ----------
        x : Float64Array
            Points at which to evaluate.

        Returns
        -------
        Float64Array
            Values of sin(2x).
        """
        return cast(Float64Array, np.sin(2.0 * x))


class ArcTanProblem(Poisson1D[float]):
    """Model Problem 2: Solution with large gradients.

    Solves -u'' = f(x) on [0, 1] where the exact solution is
    u(x) = arctan(k*(x - 0.5)), with k controlling the steepness.

    This problem tests methods' ability to resolve sharp gradients.
    Standard PINNs struggle; DFR typically performs better.

    Parameters
    ----------
    steepness : float, optional
        Controls gradient steepness, by default 100.0.
        Higher values create sharper transitions.

    Examples
    --------
    >>> problem = ArcTanProblem(steepness=50.0)
    >>> x = np.linspace(0, 1, 100)
    >>> u = problem.exact_solution(x)
    """

    def __init__(self, steepness: float = 100.0) -> None:
        super().__init__(params=steepness, domain=(0.0, 1.0))

    def forcing_fn(self, x: Float64Array) -> Float64Array:
        """Evaluate the forcing function.

        Computed as f = -u'' for u = arctan(k*(x - 0.5)).

        Parameters
        ----------
        x : Float64Array
            Points at which to evaluate.

        Returns
        -------
        Float64Array
            Values of the forcing function.
        """
        k = self.params
        return cast(
            Float64Array,
            2.0 * k**3 * (x - 0.5) / (1.0 + k**2 * (x - 0.5) ** 2) ** 2,
        )

    def exact_solution(self, x: Float64Array) -> Float64Array:
        """Evaluate u(x) = arctan(k*(x - 0.5)).

        Parameters
        ----------
        x : Float64Array
            Points at which to evaluate.

        Returns
        -------
        Float64Array
            Values of the exact solution.
        """
        k = self.params
        return cast(Float64Array, np.arctan(k * (x - 0.5)))


class DiscontinuousProblem(Poisson1D[tuple[float, float]]):
    """Model Problem 3: Discontinuous coefficients.

    Solves -u'' = sigma(x) on [0, 1] where sigma has a jump:

        sigma(x) = sigma_left   if x < 0.5
        sigma(x) = sigma_right  if x >= 0.5

    This problem tests hp-adaptivity. The solution has a kink at x=0.5.

    Parameters
    ----------
    sigma_left : float, optional
        Coefficient for x < 0.5, by default 1.0.
    sigma_right : float, optional
        Coefficient for x >= 0.5, by default 100.0.
    """

    def __init__(self, sigma_left: float = 1.0, sigma_right: float = 100.0) -> None:
        super().__init__(params=(sigma_left, sigma_right), domain=(0.0, 1.0))

    def forcing_fn(self, x: Float64Array) -> Float64Array:
        """Evaluate f(x) = 1 (constant forcing).

        Parameters
        ----------
        x : Float64Array
            Points at which to evaluate.

        Returns
        -------
        Float64Array
            Array of ones with same shape as x.
        """
        sigma_left, sigma_right = self.params
        return np.where(x < 0.5, sigma_left, sigma_right).astype(np.float64)

    def exact_solution(self, x: Float64Array) -> Float64Array:
        """Exact solution.

        Parameters
        ----------
        x : Float64Array
            Points (unused).
        """
        sigma_left, sigma_right = self.params
        return np.where(
            x < 0.5,
            (0.5 * sigma_left) * x * (1.0 - x),
            (0.5 * sigma_right) * x * (1.0 - x) + (sigma_left - sigma_right) * 0.25 * (x - 0.5),
        )


class DeltaProblem(Poisson1D[float]):
    """Model Problem 4: Point source (Dirac delta).

    Solves -u'' = delta(x - x0) on [0, pi] with u(0) = u(pi) = 0.

    The exact solution is the Green's function, which has a kink at x0.
    The forcing is approximated by a narrow Gaussian for numerical methods.

    Parameters
    ----------
    x0 : float, optional
        Location of the point source, by default 0.5.

    Notes
    -----
    The strong form is undefined at x0. DFR handles this via weak formulation.
    The Gaussian approximation uses eps=0.01 for the standard deviation.

    Examples
    --------
    >>> problem = DeltaProblem(x0=0.5)
    >>> x = np.linspace(0, np.pi, 100)
    >>> u = problem.exact_solution(x)  # Green's function
    """

    def __init__(self, x0: float = 0.5) -> None:
        self.x0 = x0
        length = np.pi
        super().__init__(params=x0, domain=(0.0, length))

    def forcing_fn(self, x: Float64Array) -> Float64Array:
        """Evaluate Gaussian approximation to delta(x - x0).

        Uses a narrow Gaussian with eps=0.01 to approximate the Dirac delta.

        Parameters
        ----------
        x : Float64Array
            Points at which to evaluate.

        Returns
        -------
        Float64Array
            Gaussian approximation to delta function.
        """
        eps = 0.01
        return cast(
            Float64Array,
            np.exp(-((x - self.x0) ** 2) / (2.0 * eps**2)) / (eps * np.sqrt(2.0 * np.pi)),
        )

    def exact_solution(self, x: Float64Array) -> Float64Array:
        """Evaluate the Green's function.

        The exact solution for -u'' = delta(x - x0) on [0, L] is:

            u(x) = x*(L - x0)/L      if x < x0
            u(x) = x0*(L - x)/L      if x >= x0

        Parameters
        ----------
        x : Float64Array
            Points at which to evaluate.

        Returns
        -------
        Float64Array
            Values of the Green's function.
        """
        length = self.domain[1]
        x0 = self.x0
        return np.where(
            x < x0,
            x * (length - x0) / length,
            x0 * (length - x) / length,
        )


def get_problem(name: str) -> Poisson1D[Any]:
    """Get a problem instance by name.

    Parameters
    ----------
    name : str
        Problem identifier. One of: 'sine', 'arctan', 'discontinuous', 'delta'.

    Returns
    -------
    Poisson1D[Any]
        The requested problem instance with default parameters.

    Raises
    ------
    ValueError
        If the problem name is not recognized.

    Examples
    --------
    >>> problem = get_problem("sine")
    >>> x = np.linspace(*problem.domain, 100)
    >>> u = problem.exact_solution(x)
    """
    problems: dict[str, Poisson1D[Any]] = {
        "sine": SineProblem(),
        "arctan": ArcTanProblem(),
        "discontinuous": DiscontinuousProblem(),
        "delta": DeltaProblem(),
    }

    if name not in problems:
        raise ValueError(f"Unknown problem: {name}. Available: {list(problems.keys())}")
    return problems[name]
