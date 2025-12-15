"""1D Poisson problem definitions."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


@dataclass
class Poisson1D:
    """1D Poisson equation: -u''(x) = f(x) with Dirichlet BCs.

    This class defines a 1D Poisson problem on the interval [a, b]
    with homogeneous Dirichlet boundary conditions u(a) = u(b) = 0.

    Attributes:
        domain: Tuple (a, b) specifying the spatial domain.
        forcing_fn: Right-hand side function f(x).
        exact_solution: Optional exact solution for error computation.
        name: Problem name for identification.

    Example:
        >>> # Problem with u(x) = sin(2x) as exact solution
        >>> problem = Poisson1D(
        ...     domain=(0, np.pi),
        ...     forcing_fn=lambda x: 4 * np.sin(2 * x),
        ...     exact_solution=lambda x: np.sin(2 * x),
        ...     name="sine"
        ... )
    """

    domain: Tuple[float, float] = (0.0, np.pi)
    forcing_fn: Callable[[np.ndarray], np.ndarray] = lambda x: 4 * np.sin(2 * x)
    exact_solution: Optional[Callable[[np.ndarray], np.ndarray]] = None
    name: str = "poisson1d"

    def __post_init__(self):
        """Set default exact solution if not provided."""
        if self.exact_solution is None and self.name == "poisson1d":
            # Default: sin(2x) problem
            self.exact_solution = lambda x: np.sin(2 * x)


# Predefined benchmark problems from the paper


def sine_problem() -> Poisson1D:
    """Model Problem 1: Smooth sine solution.

    -u'' = 4sin(2x) on [0, pi]
    u(0) = u(pi) = 0
    Exact: u(x) = sin(2x)
    """
    return Poisson1D(
        domain=(0, np.pi),
        forcing_fn=lambda x: 4 * np.sin(2 * x),
        exact_solution=lambda x: np.sin(2 * x),
        name="sine",
    )


def arctan_problem(steepness: float = 100.0) -> Poisson1D:
    """Model Problem 2: Large gradients (arctan).

    Solution with steep gradient, challenging for standard methods.

    Args:
        steepness: Controls the steepness of the gradient.
    """

    def exact(x):
        return np.arctan(steepness * (x - 0.5))

    def forcing(x):
        # f = -u'' for u = arctan(k(x - 0.5))
        k = steepness
        return 2 * k ** 3 * (x - 0.5) / (1 + k ** 2 * (x - 0.5) ** 2) ** 2

    return Poisson1D(
        domain=(0, 1),
        forcing_fn=forcing,
        exact_solution=exact,
        name=f"arctan_{steepness}",
    )


def discontinuous_problem(sigma_left: float = 1.0, sigma_right: float = 100.0) -> Poisson1D:
    """Model Problem 3: Discontinuous coefficients.

    -(sigma(x) u')' = f(x)
    where sigma has a jump discontinuity.

    Note: This requires modified weak formulation for accurate solution.

    Args:
        sigma_left: Coefficient value for x < 0.5.
        sigma_right: Coefficient value for x >= 0.5.
    """
    # Simplified forcing for demonstration
    def forcing(x):
        return np.ones_like(x)

    return Poisson1D(
        domain=(0, 1),
        forcing_fn=forcing,
        exact_solution=None,  # Exact solution depends on jump location
        name=f"discontinuous_{sigma_left}_{sigma_right}",
    )


def delta_problem(x0: float = 0.5) -> Poisson1D:
    """Model Problem 4: Point source (Dirac delta).

    -u'' = delta(x - x0)

    Note: Requires weak formulation; strong form is undefined.

    Args:
        x0: Location of the point source.
    """
    L = np.pi

    def exact(x):
        # Green's function for -u'' = delta(x - x0) on [0, L]
        return np.where(
            x < x0,
            x * (L - x0) / L,
            x0 * (L - x) / L,
        )

    # For DFR, we handle this in the weak formulation
    # The forcing function is formally a delta, approximated here
    def forcing(x):
        # Approximation using narrow Gaussian
        eps = 0.01
        return np.exp(-((x - x0) ** 2) / (2 * eps ** 2)) / (eps * np.sqrt(2 * np.pi))

    return Poisson1D(
        domain=(0, L),
        forcing_fn=forcing,
        exact_solution=exact,
        name=f"delta_{x0}",
    )


# Registry of all problems
PROBLEMS = {
    "sine": sine_problem,
    "arctan": arctan_problem,
    "discontinuous": discontinuous_problem,
    "delta": delta_problem,
}


def get_problem(name: str, **kwargs) -> Poisson1D:
    """Get a problem by name.

    Args:
        name: Problem name from PROBLEMS registry.
        **kwargs: Additional arguments passed to problem constructor.

    Returns:
        Poisson1D problem instance.

    Raises:
        ValueError: If problem name is not found.
    """
    if name not in PROBLEMS:
        raise ValueError(f"Unknown problem: {name}. Available: {list(PROBLEMS.keys())}")
    return PROBLEMS[name](**kwargs)
