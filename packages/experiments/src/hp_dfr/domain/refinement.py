"""Refinement indicators for hp-adaptive methods.

This module provides error indicators to guide adaptive refinement:
- Residual-based indicators
- Gradient-based indicators
- Goal-oriented indicators (via adjoint)

The indicators help decide:
- WHERE to refine (h-refinement)
- HOW to refine (h vs p)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from hp_dfr.domain.partitioning import BoundingBox
from hp_dfr.domain.subdomain import Subdomain, SubdomainNetwork


class RefinementIndicator(ABC):
    """Abstract base class for refinement indicators."""

    @abstractmethod
    def compute(
        self,
        subdomain: Subdomain,
        network: SubdomainNetwork,
        problem: "Problem",
    ) -> float:
        """Compute error indicator for a subdomain.

        Args:
            subdomain: Subdomain to evaluate.
            network: Associated neural network.
            problem: PDE problem definition.

        Returns:
            Error indicator value (larger = more error).
        """
        pass

    def compute_all(
        self,
        subdomains: List[Subdomain],
        networks: List[SubdomainNetwork],
        problem: "Problem",
    ) -> NDArray[np.floating]:
        """Compute indicators for all subdomains.

        Returns:
            Array of indicator values.
        """
        return np.array([
            self.compute(sd, net, problem)
            for sd, net in zip(subdomains, networks)
        ])


class ResidualIndicator(RefinementIndicator):
    """Residual-based error indicator.

    Estimates error by computing the PDE residual:
        eta_K = ||R(u_h)||_{L^2(K)} or ||R(u_h)||_{H^{-1}(K)}

    where R(u) = Lu - f is the PDE residual.
    """

    def __init__(
        self,
        n_quadrature: int = 16,
        norm: str = "l2",
        weighted: bool = True,
    ):
        """Initialize residual indicator.

        Args:
            n_quadrature: Quadrature points per dimension.
            norm: 'l2' or 'h_minus_1'.
            weighted: Whether to weight by subdomain size.
        """
        self.n_quadrature = n_quadrature
        self.norm = norm
        self.weighted = weighted

    def compute(
        self,
        subdomain: Subdomain,
        network: SubdomainNetwork,
        problem: "Problem",
    ) -> float:
        """Compute residual-based indicator."""
        # Get quadrature points
        x = subdomain.quadrature_points(self.n_quadrature)
        n_points = x.shape[0]

        # Compute solution
        u = network.forward(x, apply_cutoff=True).flatten()

        # Compute Laplacian via finite differences
        # (In production, use autodiff)
        laplacian = self._compute_laplacian_fd(x, network, subdomain)

        # Compute residual: -Delta u - f
        f = problem.forcing_fn(x)
        if f.ndim > 1:
            f = f.flatten()
        residual = -laplacian - f

        # Compute norm
        if self.norm == "l2":
            indicator = np.sqrt(np.mean(residual**2))
        else:
            # Simplified H^{-1} (would use Fourier in production)
            indicator = np.sqrt(np.mean(residual**2))

        # Weight by subdomain size
        if self.weighted:
            indicator *= np.sqrt(subdomain.volume)

        return indicator

    def _compute_laplacian_fd(
        self,
        x: NDArray[np.floating],
        network: SubdomainNetwork,
        subdomain: Subdomain,
    ) -> NDArray[np.floating]:
        """Compute Laplacian via finite differences."""
        dim = subdomain.dim
        eps = 1e-5
        laplacian = np.zeros(x.shape[0])

        for d in range(dim):
            # Offset in dimension d
            offset = np.zeros(dim)
            offset[d] = eps

            x_plus = x + offset
            x_minus = x - offset

            u_plus = network.forward(x_plus, apply_cutoff=True).flatten()
            u_minus = network.forward(x_minus, apply_cutoff=True).flatten()
            u_center = network.forward(x, apply_cutoff=True).flatten()

            # Second derivative
            d2u = (u_plus - 2 * u_center + u_minus) / eps**2
            laplacian += d2u

        return laplacian


class GradientIndicator(RefinementIndicator):
    """Gradient-based error indicator.

    Estimates error by solution gradient magnitude:
        eta_K = ||grad u_h||_{L^2(K)}

    High gradients suggest need for refinement.
    """

    def __init__(
        self,
        n_quadrature: int = 16,
        weighted: bool = True,
    ):
        self.n_quadrature = n_quadrature
        self.weighted = weighted

    def compute(
        self,
        subdomain: Subdomain,
        network: SubdomainNetwork,
        problem: "Problem",
    ) -> float:
        """Compute gradient-based indicator."""
        x = subdomain.quadrature_points(self.n_quadrature)
        dim = subdomain.dim
        eps = 1e-5

        # Compute gradient magnitude
        grad_sq = np.zeros(x.shape[0])

        for d in range(dim):
            offset = np.zeros(dim)
            offset[d] = eps

            u_plus = network.forward(x + offset, apply_cutoff=True).flatten()
            u_minus = network.forward(x - offset, apply_cutoff=True).flatten()

            du_dx = (u_plus - u_minus) / (2 * eps)
            grad_sq += du_dx**2

        indicator = np.sqrt(np.mean(grad_sq))

        if self.weighted:
            indicator *= np.sqrt(subdomain.volume)

        return indicator


class JumpIndicator(RefinementIndicator):
    """Jump indicator across subdomain interfaces.

    Estimates error by solution jumps at interfaces:
        eta_K = sum_e ||[u_h]||_{L^2(e)}

    where [u] is the jump across edge e.
    """

    def __init__(
        self,
        interfaces: List[Tuple[int, int, BoundingBox]],
        networks: List[SubdomainNetwork],
        n_interface_points: int = 16,
    ):
        self.interfaces = interfaces
        self.networks = networks
        self.n_interface_points = n_interface_points

    def compute(
        self,
        subdomain: Subdomain,
        network: SubdomainNetwork,
        problem: "Problem",
    ) -> float:
        """Compute jump indicator for subdomain."""
        total_jump = 0.0
        n_interfaces = 0

        for i, j, box in self.interfaces:
            if i != subdomain.index and j != subdomain.index:
                continue

            # Get interface points
            x = box.quadrature_points(self.n_interface_points)

            # Evaluate both networks
            u_i = self.networks[i].forward(x, apply_cutoff=False).flatten()
            u_j = self.networks[j].forward(x, apply_cutoff=False).flatten()

            jump = np.sqrt(np.mean((u_i - u_j)**2))
            total_jump += jump
            n_interfaces += 1

        if n_interfaces > 0:
            return total_jump / n_interfaces
        return 0.0


@dataclass
class RefinementDecision:
    """Decision about how to refine a subdomain.

    Attributes:
        subdomain_index: Index of subdomain to refine.
        refine_type: 'h' for spatial, 'p' for polynomial.
        direction: For h-refinement, which axis to split.
        indicator_value: Error indicator that triggered refinement.
    """

    subdomain_index: int
    refine_type: str  # 'h' or 'p'
    direction: Optional[int] = None
    indicator_value: float = 0.0


def mark_for_refinement(
    indicators: NDArray[np.floating],
    strategy: str = "maximum",
    theta: float = 0.5,
    max_marked: Optional[int] = None,
) -> List[int]:
    """Mark subdomains for refinement based on indicators.

    Args:
        indicators: Array of error indicators per subdomain.
        strategy: Marking strategy.
            - 'maximum': Mark if indicator > theta * max(indicators)
            - 'mean': Mark if indicator > theta * mean(indicators)
            - 'doerfler': Bulk marking to capture theta fraction of total
            - 'fixed_fraction': Mark top theta fraction of subdomains
        theta: Strategy-dependent threshold (0 < theta <= 1).
        max_marked: Maximum number of subdomains to mark.

    Returns:
        List of subdomain indices to refine.
    """
    n = len(indicators)
    marked = []

    if strategy == "maximum":
        threshold = theta * np.max(indicators)
        marked = [i for i in range(n) if indicators[i] > threshold]

    elif strategy == "mean":
        threshold = theta * np.mean(indicators)
        marked = [i for i in range(n) if indicators[i] > threshold]

    elif strategy == "doerfler":
        # Bulk marking: mark smallest set capturing theta of total indicator
        total = np.sum(indicators)
        target = theta * total

        # Sort by indicator (descending)
        sorted_idx = np.argsort(indicators)[::-1]

        cumsum = 0.0
        for i in sorted_idx:
            marked.append(i)
            cumsum += indicators[i]
            if cumsum >= target:
                break

    elif strategy == "fixed_fraction":
        # Mark top theta fraction
        n_mark = max(1, int(theta * n))
        sorted_idx = np.argsort(indicators)[::-1]
        marked = list(sorted_idx[:n_mark])

    # Apply max_marked limit
    if max_marked is not None and len(marked) > max_marked:
        # Keep highest indicators
        marked_indicators = [(i, indicators[i]) for i in marked]
        marked_indicators.sort(key=lambda x: x[1], reverse=True)
        marked = [x[0] for x in marked_indicators[:max_marked]]

    return marked


def decide_refinement_type(
    subdomain: Subdomain,
    network: SubdomainNetwork,
    problem: "Problem",
    smoothness_threshold: float = 0.5,
) -> str:
    """Decide between h-refinement and p-refinement.

    Uses solution smoothness to decide:
    - Smooth solution -> p-refinement (increase network capacity)
    - Non-smooth solution -> h-refinement (subdivide domain)

    Args:
        subdomain: Subdomain to evaluate.
        network: Associated network.
        problem: PDE problem.
        smoothness_threshold: Threshold for smoothness metric.

    Returns:
        'h' or 'p' indicating refinement type.
    """
    # Compute smoothness indicator
    # (Ratio of high-frequency to low-frequency content)

    x = subdomain.quadrature_points(32)
    u = network.forward(x, apply_cutoff=True).flatten()

    # Simple smoothness: variance of second derivative
    dim = subdomain.dim
    eps = 1e-4

    second_deriv_var = 0.0
    for d in range(dim):
        offset = np.zeros(dim)
        offset[d] = eps

        u_plus = network.forward(x + offset, apply_cutoff=True).flatten()
        u_minus = network.forward(x - offset, apply_cutoff=True).flatten()

        d2u = (u_plus - 2 * u + u_minus) / eps**2
        second_deriv_var += np.var(d2u)

    # Normalize by solution variance
    u_var = np.var(u) + 1e-10
    smoothness = 1.0 / (1.0 + second_deriv_var / u_var)

    if smoothness > smoothness_threshold:
        return "p"  # Smooth -> increase network capacity
    else:
        return "h"  # Non-smooth -> subdivide domain
