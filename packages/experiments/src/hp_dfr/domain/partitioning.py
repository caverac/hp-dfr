"""Domain partitioning for hp-adaptive methods.

This module provides classes for partitioning computational domains
into subdomains for local neural network approximation.

Partitioning Strategies:
    - Uniform: Equal-sized rectangular subdomains
    - Adaptive: Residual-based refinement
    - Overlapping: For Schwarz-type methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterator, Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for a subdomain.

    Attributes:
        bounds: Tuple of (min, max) for each dimension.
                For 2D: ((x_min, x_max), (y_min, y_max))
    """

    bounds: Tuple[Tuple[float, float], ...]

    @property
    def dim(self) -> int:
        """Spatial dimension."""
        return len(self.bounds)

    @property
    def volume(self) -> float:
        """Volume (area in 2D, length in 1D)."""
        vol = 1.0
        for lo, hi in self.bounds:
            vol *= (hi - lo)
        return vol

    @property
    def center(self) -> NDArray[np.floating]:
        """Center point of the box."""
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds])

    @property
    def size(self) -> NDArray[np.floating]:
        """Size in each dimension."""
        return np.array([hi - lo for lo, hi in self.bounds])

    def contains(self, point: NDArray[np.floating]) -> bool:
        """Check if point is inside the box."""
        for i, (lo, hi) in enumerate(self.bounds):
            if point[i] < lo or point[i] > hi:
                return False
        return True

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this box overlaps with another."""
        for i in range(self.dim):
            if (self.bounds[i][1] < other.bounds[i][0] or
                other.bounds[i][1] < self.bounds[i][0]):
                return False
        return True

    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """Compute intersection with another box."""
        if not self.overlaps(other):
            return None

        new_bounds = []
        for i in range(self.dim):
            lo = max(self.bounds[i][0], other.bounds[i][0])
            hi = min(self.bounds[i][1], other.bounds[i][1])
            new_bounds.append((lo, hi))

        return BoundingBox(bounds=tuple(new_bounds))

    def split(self, axis: int, position: Optional[float] = None) -> Tuple["BoundingBox", "BoundingBox"]:
        """Split box along an axis.

        Args:
            axis: Dimension to split along.
            position: Split position (default: midpoint).

        Returns:
            Tuple of two child boxes.
        """
        if position is None:
            position = (self.bounds[axis][0] + self.bounds[axis][1]) / 2

        bounds1 = list(self.bounds)
        bounds2 = list(self.bounds)

        bounds1[axis] = (self.bounds[axis][0], position)
        bounds2[axis] = (position, self.bounds[axis][1])

        return BoundingBox(tuple(bounds1)), BoundingBox(tuple(bounds2))

    def quadrature_points(self, n_points: int) -> NDArray[np.floating]:
        """Generate quadrature points inside the box.

        Args:
            n_points: Number of points per dimension.

        Returns:
            Array of shape (n_points^dim, dim).
        """
        grids_1d = []
        for lo, hi in self.bounds:
            x = np.linspace(lo, hi, n_points + 1)
            dx = x[1] - x[0]
            x_mid = x[:-1] + dx / 2
            grids_1d.append(x_mid)

        grids = np.meshgrid(*grids_1d, indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=1)


class DomainPartition(ABC):
    """Abstract base class for domain partitions.

    A partition divides a domain into non-overlapping (or overlapping)
    subdomains, each of which will have its own local neural network.
    """

    @abstractmethod
    def __init__(self, domain: BoundingBox):
        """Initialize partition for a domain.

        Args:
            domain: Global domain bounding box.
        """
        self.domain = domain

    @property
    @abstractmethod
    def n_subdomains(self) -> int:
        """Number of subdomains."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        """Iterate over subdomain bounding boxes."""
        pass

    @abstractmethod
    def get_subdomain(self, index: int) -> BoundingBox:
        """Get subdomain by index."""
        pass

    @abstractmethod
    def find_subdomain(self, point: NDArray[np.floating]) -> int:
        """Find which subdomain contains a point."""
        pass

    @abstractmethod
    def get_interfaces(self) -> List[Tuple[int, int, BoundingBox]]:
        """Get interfaces between adjacent subdomains.

        Returns:
            List of (subdomain_i, subdomain_j, interface_box) tuples.
        """
        pass


class RectangularPartition(DomainPartition):
    """Uniform rectangular partition of a domain.

    Divides domain into n_x × n_y × ... equal-sized rectangles.

    Example:
        >>> domain = BoundingBox(((0, 1), (0, 1)))
        >>> partition = RectangularPartition(domain, divisions=(2, 2))
        >>> print(f"Created {partition.n_subdomains} subdomains")
        Created 4 subdomains
    """

    def __init__(
        self,
        domain: BoundingBox,
        divisions: Tuple[int, ...],
        overlap: float = 0.0,
    ):
        """Initialize rectangular partition.

        Args:
            domain: Global domain bounding box.
            divisions: Number of divisions per dimension.
            overlap: Overlap ratio (0 = no overlap, 0.1 = 10% overlap).
        """
        self.domain = domain
        self.divisions = divisions
        self.overlap = overlap

        if len(divisions) != domain.dim:
            raise ValueError(
                f"Divisions {divisions} must match domain dimension {domain.dim}"
            )

        self._subdomains = self._create_subdomains()
        self._interfaces = self._find_interfaces()

    def _create_subdomains(self) -> List[BoundingBox]:
        """Create all subdomain bounding boxes."""
        subdomains = []

        # Compute base sizes
        sizes = []
        for d in range(self.domain.dim):
            lo, hi = self.domain.bounds[d]
            size = (hi - lo) / self.divisions[d]
            sizes.append(size)

        # Generate all subdomains
        indices = [range(n) for n in self.divisions]
        import itertools
        for idx in itertools.product(*indices):
            bounds = []
            for d in range(self.domain.dim):
                lo_base = self.domain.bounds[d][0] + idx[d] * sizes[d]
                hi_base = lo_base + sizes[d]

                # Add overlap
                if self.overlap > 0:
                    overlap_size = sizes[d] * self.overlap
                    lo = lo_base - overlap_size if idx[d] > 0 else lo_base
                    hi = hi_base + overlap_size if idx[d] < self.divisions[d] - 1 else hi_base
                else:
                    lo, hi = lo_base, hi_base

                bounds.append((lo, hi))

            subdomains.append(BoundingBox(tuple(bounds)))

        return subdomains

    def _find_interfaces(self) -> List[Tuple[int, int, BoundingBox]]:
        """Find interfaces between adjacent subdomains."""
        interfaces = []

        for i in range(len(self._subdomains)):
            for j in range(i + 1, len(self._subdomains)):
                box_i = self._subdomains[i]
                box_j = self._subdomains[j]

                # Check for adjacency (shared face)
                intersection = box_i.intersection(box_j)
                if intersection is not None:
                    # Check if it's a proper interface (not just a point)
                    if intersection.volume > 0 or self._is_face_interface(box_i, box_j):
                        interfaces.append((i, j, intersection))

        return interfaces

    def _is_face_interface(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two boxes share a face (not just overlap)."""
        # Count dimensions where they share a boundary
        shared_dims = 0
        for d in range(box1.dim):
            if (abs(box1.bounds[d][1] - box2.bounds[d][0]) < 1e-10 or
                abs(box2.bounds[d][1] - box1.bounds[d][0]) < 1e-10):
                shared_dims += 1

        return shared_dims == 1

    @property
    def n_subdomains(self) -> int:
        return len(self._subdomains)

    def __iter__(self) -> Iterator[BoundingBox]:
        return iter(self._subdomains)

    def __len__(self) -> int:
        return len(self._subdomains)

    def get_subdomain(self, index: int) -> BoundingBox:
        return self._subdomains[index]

    def find_subdomain(self, point: NDArray[np.floating]) -> int:
        """Find subdomain containing point (returns first match for overlapping)."""
        for i, box in enumerate(self._subdomains):
            if box.contains(point):
                return i
        return -1

    def get_interfaces(self) -> List[Tuple[int, int, BoundingBox]]:
        return self._interfaces

    def get_neighbors(self, index: int) -> List[int]:
        """Get indices of neighboring subdomains."""
        neighbors = []
        for i, j, _ in self._interfaces:
            if i == index:
                neighbors.append(j)
            elif j == index:
                neighbors.append(i)
        return neighbors


class AdaptivePartition(DomainPartition):
    """Adaptively refined domain partition using quad/octree structure.

    Starts with a coarse partition and refines based on error indicators.

    Example:
        >>> domain = BoundingBox(((0, 1), (0, 1)))
        >>> partition = AdaptivePartition(domain, max_level=4)
        >>> # Refine based on residual
        >>> partition.refine(indicator_fn, threshold=0.1)
    """

    def __init__(
        self,
        domain: BoundingBox,
        max_level: int = 5,
        initial_divisions: Optional[Tuple[int, ...]] = None,
    ):
        """Initialize adaptive partition.

        Args:
            domain: Global domain bounding box.
            max_level: Maximum refinement level.
            initial_divisions: Initial uniform divisions (default: 1 per dim).
        """
        self.domain = domain
        self.max_level = max_level

        if initial_divisions is None:
            initial_divisions = tuple([1] * domain.dim)

        # Start with uniform partition
        self._base = RectangularPartition(domain, initial_divisions)
        self._subdomains: List[BoundingBox] = list(self._base)
        self._levels: List[int] = [0] * len(self._subdomains)
        self._interfaces: List[Tuple[int, int, BoundingBox]] = []
        self._update_interfaces()

    def _update_interfaces(self) -> None:
        """Recompute interfaces after refinement."""
        self._interfaces = []
        for i in range(len(self._subdomains)):
            for j in range(i + 1, len(self._subdomains)):
                box_i = self._subdomains[i]
                box_j = self._subdomains[j]
                if self._are_adjacent(box_i, box_j):
                    # Create interface box
                    interface = self._compute_interface(box_i, box_j)
                    if interface is not None:
                        self._interfaces.append((i, j, interface))

    def _are_adjacent(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two boxes are adjacent (share a face)."""
        for d in range(box1.dim):
            # Check if they touch along this dimension
            touch = (abs(box1.bounds[d][1] - box2.bounds[d][0]) < 1e-10 or
                     abs(box2.bounds[d][1] - box1.bounds[d][0]) < 1e-10)

            if touch:
                # Check overlap in other dimensions
                overlap = True
                for d2 in range(box1.dim):
                    if d2 != d:
                        if (box1.bounds[d2][1] <= box2.bounds[d2][0] or
                            box2.bounds[d2][1] <= box1.bounds[d2][0]):
                            overlap = False
                            break
                if overlap:
                    return True
        return False

    def _compute_interface(
        self, box1: BoundingBox, box2: BoundingBox
    ) -> Optional[BoundingBox]:
        """Compute interface between two adjacent boxes."""
        bounds = []
        for d in range(box1.dim):
            if abs(box1.bounds[d][1] - box2.bounds[d][0]) < 1e-10:
                # box1 is to the left of box2
                bounds.append((box1.bounds[d][1], box1.bounds[d][1]))
            elif abs(box2.bounds[d][1] - box1.bounds[d][0]) < 1e-10:
                # box2 is to the left of box1
                bounds.append((box2.bounds[d][1], box2.bounds[d][1]))
            else:
                # Overlap in this dimension
                lo = max(box1.bounds[d][0], box2.bounds[d][0])
                hi = min(box1.bounds[d][1], box2.bounds[d][1])
                bounds.append((lo, hi))

        return BoundingBox(tuple(bounds))

    def refine(
        self,
        indicator: Callable[[BoundingBox], float],
        threshold: float,
        max_refinements: int = 100,
    ) -> int:
        """Refine subdomains based on indicator values.

        Args:
            indicator: Function mapping subdomain to error indicator.
            threshold: Refine if indicator > threshold.
            max_refinements: Maximum number of refinements.

        Returns:
            Number of refinements performed.
        """
        n_refined = 0

        for _ in range(max_refinements):
            # Find subdomain with largest indicator above threshold
            max_indicator = threshold
            refine_idx = -1

            for i, box in enumerate(self._subdomains):
                if self._levels[i] >= self.max_level:
                    continue
                ind = indicator(box)
                if ind > max_indicator:
                    max_indicator = ind
                    refine_idx = i

            if refine_idx < 0:
                break

            # Refine this subdomain
            self._refine_subdomain(refine_idx)
            n_refined += 1

        self._update_interfaces()
        return n_refined

    def _refine_subdomain(self, index: int) -> None:
        """Refine a subdomain by splitting it."""
        box = self._subdomains[index]
        level = self._levels[index]

        # Split along longest dimension
        sizes = box.size
        split_axis = int(np.argmax(sizes))

        child1, child2 = box.split(split_axis)

        # Replace parent with children
        self._subdomains[index] = child1
        self._levels[index] = level + 1
        self._subdomains.append(child2)
        self._levels.append(level + 1)

    def coarsen(
        self,
        indicator: Callable[[BoundingBox], float],
        threshold: float,
    ) -> int:
        """Coarsen subdomains with low indicator values.

        Note: This is a simplified implementation that doesn't
        actually merge subdomains, just marks them for potential merging.

        Args:
            indicator: Function mapping subdomain to error indicator.
            threshold: Coarsen if indicator < threshold.

        Returns:
            Number of subdomains that could be coarsened.
        """
        n_coarsen = 0
        for i, box in enumerate(self._subdomains):
            if self._levels[i] > 0 and indicator(box) < threshold:
                n_coarsen += 1
        return n_coarsen

    @property
    def n_subdomains(self) -> int:
        return len(self._subdomains)

    def __iter__(self) -> Iterator[BoundingBox]:
        return iter(self._subdomains)

    def get_subdomain(self, index: int) -> BoundingBox:
        return self._subdomains[index]

    def get_level(self, index: int) -> int:
        """Get refinement level of a subdomain."""
        return self._levels[index]

    def find_subdomain(self, point: NDArray[np.floating]) -> int:
        for i, box in enumerate(self._subdomains):
            if box.contains(point):
                return i
        return -1

    def get_interfaces(self) -> List[Tuple[int, int, BoundingBox]]:
        return self._interfaces
