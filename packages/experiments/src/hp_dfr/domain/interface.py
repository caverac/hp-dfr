"""Interface coupling for domain decomposition methods.

This module provides methods to couple solutions across subdomain
interfaces, ensuring continuity of the solution and its fluxes.

Coupling Methods:
    - Penalty: Add penalty terms for solution jumps
    - Mortar: Project solutions onto interface space
    - Overlapping Schwarz: Iterate with overlapping domains
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from hp_dfr.domain.partitioning import BoundingBox
from hp_dfr.domain.subdomain import Subdomain, SubdomainNetwork


@dataclass
class Interface:
    """Interface between two subdomains.

    Attributes:
        subdomain_i: First subdomain index.
        subdomain_j: Second subdomain index.
        box: Bounding box of the interface.
        normal: Outward normal from subdomain_i to j.
        quadrature_points: Points on the interface for integration.
    """

    subdomain_i: int
    subdomain_j: int
    box: BoundingBox
    normal: NDArray[np.floating]
    quadrature_points: Optional[NDArray[np.floating]] = None

    def setup_quadrature(self, n_points: int = 16) -> None:
        """Set up quadrature points on the interface.

        Args:
            n_points: Number of points per dimension.
        """
        dim = self.box.dim

        # For interface, one dimension is degenerate
        grids_1d = []
        for d in range(dim):
            lo, hi = self.box.bounds[d]
            if abs(hi - lo) < 1e-10:
                # Degenerate dimension - single point
                grids_1d.append(np.array([lo]))
            else:
                x = np.linspace(lo, hi, n_points + 1)
                dx = x[1] - x[0]
                x_mid = x[:-1] + dx / 2
                grids_1d.append(x_mid)

        grids = np.meshgrid(*grids_1d, indexing="ij")
        self.quadrature_points = np.stack([g.ravel() for g in grids], axis=1)


class InterfaceCoupling(ABC):
    """Abstract base class for interface coupling methods."""

    @abstractmethod
    def compute_coupling_loss(
        self,
        interface: Interface,
        network_i: SubdomainNetwork,
        network_j: SubdomainNetwork,
    ) -> float:
        """Compute coupling loss for an interface.

        Args:
            interface: Interface definition.
            network_i: Network for subdomain i.
            network_j: Network for subdomain j.

        Returns:
            Coupling loss value.
        """
        pass

    @abstractmethod
    def compute_coupling_gradient(
        self,
        interface: Interface,
        network_i: SubdomainNetwork,
        network_j: SubdomainNetwork,
    ) -> Tuple[List, List]:
        """Compute gradients of coupling loss.

        Returns:
            Tuple of (gradients_i, gradients_j) for each network.
        """
        pass


class PenaltyCoupling(InterfaceCoupling):
    """Penalty-based interface coupling.

    Adds penalty terms to enforce continuity:
        L_interface = beta * integral |u_i - u_j|^2 ds
                    + gamma * integral |du_i/dn - du_j/dn|^2 ds

    where beta and gamma are penalty parameters.
    """

    def __init__(
        self,
        beta: float = 100.0,
        gamma: float = 10.0,
        n_interface_points: int = 16,
    ):
        """Initialize penalty coupling.

        Args:
            beta: Penalty for solution discontinuity.
            gamma: Penalty for flux discontinuity.
            n_interface_points: Quadrature points per interface.
        """
        self.beta = beta
        self.gamma = gamma
        self.n_interface_points = n_interface_points

    def compute_coupling_loss(
        self,
        interface: Interface,
        network_i: SubdomainNetwork,
        network_j: SubdomainNetwork,
    ) -> float:
        """Compute penalty coupling loss."""
        if interface.quadrature_points is None:
            interface.setup_quadrature(self.n_interface_points)

        x = interface.quadrature_points
        n_points = x.shape[0]

        # Evaluate both networks at interface (without cutoff!)
        u_i = network_i.forward(x, apply_cutoff=False).flatten()
        u_j = network_j.forward(x, apply_cutoff=False).flatten()

        # Solution jump
        jump = u_i - u_j
        solution_penalty = self.beta * np.mean(jump**2)

        # Flux continuity (if gamma > 0)
        flux_penalty = 0.0
        if self.gamma > 0:
            # Approximate normal derivatives
            eps = 1e-6
            normal = interface.normal
            x_plus = x + eps * normal
            x_minus = x - eps * normal

            # du/dn for subdomain i
            u_i_plus = network_i.forward(x_plus, apply_cutoff=False).flatten()
            u_i_minus = network_i.forward(x_minus, apply_cutoff=False).flatten()
            du_dn_i = (u_i_plus - u_i_minus) / (2 * eps)

            # du/dn for subdomain j (note: opposite normal)
            u_j_plus = network_j.forward(x_plus, apply_cutoff=False).flatten()
            u_j_minus = network_j.forward(x_minus, apply_cutoff=False).flatten()
            du_dn_j = (u_j_plus - u_j_minus) / (2 * eps)

            flux_jump = du_dn_i + du_dn_j  # Should sum to zero
            flux_penalty = self.gamma * np.mean(flux_jump**2)

        return solution_penalty + flux_penalty

    def compute_coupling_gradient(
        self,
        interface: Interface,
        network_i: SubdomainNetwork,
        network_j: SubdomainNetwork,
    ) -> Tuple[List, List]:
        """Compute gradients using automatic differentiation.

        Note: This is a placeholder. Actual implementation depends
        on the backend and is handled in the training loop.
        """
        # In practice, gradients are computed via autodiff in the training loop
        return [], []


class MortarCoupling(InterfaceCoupling):
    """Mortar element coupling for non-matching grids.

    Projects solutions onto a common interface space before
    comparing, allowing for different mesh resolutions.
    """

    def __init__(
        self,
        n_mortar_modes: int = 8,
        penalty: float = 100.0,
    ):
        """Initialize mortar coupling.

        Args:
            n_mortar_modes: Number of Fourier modes on interface.
            penalty: Penalty parameter.
        """
        self.n_mortar_modes = n_mortar_modes
        self.penalty = penalty

    def compute_coupling_loss(
        self,
        interface: Interface,
        network_i: SubdomainNetwork,
        network_j: SubdomainNetwork,
    ) -> float:
        """Compute mortar coupling loss.

        Projects solutions onto interface Fourier space and
        penalizes differences in projection.
        """
        if interface.quadrature_points is None:
            interface.setup_quadrature(32)

        x = interface.quadrature_points

        # Evaluate networks
        u_i = network_i.forward(x, apply_cutoff=False).flatten()
        u_j = network_j.forward(x, apply_cutoff=False).flatten()

        # Project onto Fourier basis
        # For simplicity, use direct comparison (full mortar would project)
        jump = u_i - u_j
        return self.penalty * np.mean(jump**2)

    def compute_coupling_gradient(
        self,
        interface: Interface,
        network_i: SubdomainNetwork,
        network_j: SubdomainNetwork,
    ) -> Tuple[List, List]:
        return [], []


class OverlappingSchwarz:
    """Overlapping Schwarz iteration for domain decomposition.

    Iteratively solves on each subdomain using boundary conditions
    from neighboring subdomains until convergence.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
    ):
        """Initialize Schwarz iteration.

        Args:
            max_iterations: Maximum number of Schwarz iterations.
            tolerance: Convergence tolerance.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def iterate(
        self,
        subdomains: List[Subdomain],
        networks: List[SubdomainNetwork],
        local_solver: Callable,
    ) -> int:
        """Perform Schwarz iteration.

        Args:
            subdomains: List of subdomains.
            networks: List of subdomain networks.
            local_solver: Function to solve on each subdomain.

        Returns:
            Number of iterations performed.
        """
        for iteration in range(self.max_iterations):
            max_change = 0.0

            for i, (sd, net) in enumerate(zip(subdomains, networks)):
                # Get boundary values from neighbors
                # This is a placeholder - actual implementation would
                # sample from neighboring networks

                # Solve locally
                old_solution = net.forward(
                    sd.quadrature_points(16), apply_cutoff=True
                )

                # local_solver would update the network here

                new_solution = net.forward(
                    sd.quadrature_points(16), apply_cutoff=True
                )

                change = np.max(np.abs(new_solution - old_solution))
                max_change = max(max_change, change)

            if max_change < self.tolerance:
                return iteration + 1

        return self.max_iterations


def compute_total_interface_loss(
    interfaces: List[Interface],
    networks: List[SubdomainNetwork],
    coupling: InterfaceCoupling,
) -> float:
    """Compute total interface coupling loss.

    Args:
        interfaces: List of interface definitions.
        networks: List of subdomain networks.
        coupling: Coupling method.

    Returns:
        Total coupling loss.
    """
    total_loss = 0.0
    for interface in interfaces:
        loss = coupling.compute_coupling_loss(
            interface,
            networks[interface.subdomain_i],
            networks[interface.subdomain_j],
        )
        total_loss += loss
    return total_loss
