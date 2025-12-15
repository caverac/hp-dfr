"""Unit tests for hp-adaptive domain decomposition methods.

Tests cover:
1. Domain partitioning (rectangular, adaptive)
2. Subdomain networks
3. Interface coupling
4. Refinement indicators
5. hp-DFR model
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hp_dfr.domain.partitioning import (
    BoundingBox,
    RectangularPartition,
    AdaptivePartition,
)
from hp_dfr.domain.subdomain import (
    Subdomain,
    SubdomainNetwork,
    SubdomainCollection,
)
from hp_dfr.domain.interface import (
    Interface,
    PenaltyCoupling,
)
from hp_dfr.domain.refinement import (
    ResidualIndicator,
    GradientIndicator,
    mark_for_refinement,
)


# =============================================================================
# BoundingBox Tests
# =============================================================================


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_1d_box(self):
        """Test 1D bounding box."""
        box = BoundingBox(bounds=((0.0, 1.0),))
        assert box.dim == 1
        assert box.volume == 1.0
        assert_allclose(box.center, [0.5])
        assert_allclose(box.size, [1.0])

    def test_2d_box(self):
        """Test 2D bounding box."""
        box = BoundingBox(bounds=((0.0, 2.0), (0.0, 3.0)))
        assert box.dim == 2
        assert box.volume == 6.0
        assert_allclose(box.center, [1.0, 1.5])
        assert_allclose(box.size, [2.0, 3.0])

    def test_contains_point(self):
        """Test point containment."""
        box = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        assert box.contains(np.array([0.5, 0.5]))
        assert box.contains(np.array([0.0, 0.0]))
        assert not box.contains(np.array([1.5, 0.5]))
        assert not box.contains(np.array([-0.1, 0.5]))

    def test_overlaps(self):
        """Test box overlap detection."""
        box1 = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        box2 = BoundingBox(bounds=((0.5, 1.5), (0.5, 1.5)))
        box3 = BoundingBox(bounds=((2.0, 3.0), (2.0, 3.0)))

        assert box1.overlaps(box2)
        assert box2.overlaps(box1)
        assert not box1.overlaps(box3)

    def test_intersection(self):
        """Test box intersection."""
        box1 = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        box2 = BoundingBox(bounds=((0.5, 1.5), (0.5, 1.5)))

        intersection = box1.intersection(box2)
        assert intersection is not None
        assert_allclose(intersection.bounds[0], (0.5, 1.0))
        assert_allclose(intersection.bounds[1], (0.5, 1.0))

    def test_split(self):
        """Test box splitting."""
        box = BoundingBox(bounds=((0.0, 2.0), (0.0, 1.0)))
        left, right = box.split(axis=0)

        assert_allclose(left.bounds[0], (0.0, 1.0))
        assert_allclose(right.bounds[0], (1.0, 2.0))
        assert_allclose(left.bounds[1], (0.0, 1.0))
        assert_allclose(right.bounds[1], (0.0, 1.0))

    def test_quadrature_points(self):
        """Test quadrature point generation."""
        box = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        pts = box.quadrature_points(n_points=4)

        assert pts.shape == (16, 2)
        assert np.all(pts >= 0)
        assert np.all(pts <= 1)


# =============================================================================
# Partitioning Tests
# =============================================================================


class TestRectangularPartition:
    """Tests for RectangularPartition class."""

    def test_uniform_2x2(self):
        """Test 2x2 partition."""
        domain = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        partition = RectangularPartition(domain, divisions=(2, 2))

        assert partition.n_subdomains == 4
        assert len(list(partition)) == 4

    def test_uniform_3x2(self):
        """Test 3x2 partition."""
        domain = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        partition = RectangularPartition(domain, divisions=(3, 2))

        assert partition.n_subdomains == 6

    def test_subdomain_coverage(self):
        """Test subdomains cover the domain."""
        domain = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        partition = RectangularPartition(domain, divisions=(2, 2))

        # Sample points and check coverage
        test_points = np.array([
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ])

        for pt in test_points:
            idx = partition.find_subdomain(pt)
            assert idx >= 0, f"Point {pt} not found in any subdomain"

    def test_interfaces(self):
        """Test interface detection."""
        domain = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        partition = RectangularPartition(domain, divisions=(2, 2))

        interfaces = partition.get_interfaces()
        # 2x2 grid has 4 internal edges
        assert len(interfaces) == 4

    def test_neighbors(self):
        """Test neighbor detection."""
        domain = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        partition = RectangularPartition(domain, divisions=(3, 3))

        # Center subdomain should have 4 neighbors
        center_idx = 4  # Center of 3x3 grid
        neighbors = partition.get_neighbors(center_idx)
        assert len(neighbors) == 4

        # Corner subdomain should have 2 neighbors
        corner_idx = 0
        neighbors = partition.get_neighbors(corner_idx)
        assert len(neighbors) == 2


class TestAdaptivePartition:
    """Tests for AdaptivePartition class."""

    def test_initial_partition(self):
        """Test initial partition creation."""
        domain = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        partition = AdaptivePartition(domain, initial_divisions=(2, 2))

        assert partition.n_subdomains == 4

    def test_refinement(self):
        """Test adaptive refinement."""
        domain = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        partition = AdaptivePartition(domain, initial_divisions=(1, 1))

        # Indicator that triggers refinement
        def indicator(box):
            return 1.0 if box.center[0] < 0.5 else 0.0

        n_refined = partition.refine(indicator, threshold=0.5, max_refinements=1)
        assert n_refined == 1
        assert partition.n_subdomains == 2  # Split into 2


# =============================================================================
# Subdomain Tests
# =============================================================================


class TestSubdomain:
    """Tests for Subdomain class."""

    def test_creation(self):
        """Test subdomain creation."""
        box = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        sd = Subdomain(index=0, box=box, level=0)

        assert sd.index == 0
        assert sd.dim == 2
        assert sd.volume == 1.0

    def test_cutoff_function(self):
        """Test cutoff function values."""
        box = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        sd = Subdomain(index=0, box=box)

        # Center point: maximum cutoff
        x_center = np.array([[0.5, 0.5]])
        cutoff_center = sd.cutoff_function(x_center)
        assert cutoff_center[0, 0] > 0

        # Boundary point: zero cutoff
        x_boundary = np.array([[0.0, 0.5]])
        cutoff_boundary = sd.cutoff_function(x_boundary)
        assert_allclose(cutoff_boundary[0, 0], 0.0, atol=1e-10)


class TestSubdomainCollection:
    """Tests for SubdomainCollection class."""

    def test_creation(self):
        """Test collection creation."""
        box1 = BoundingBox(bounds=((0.0, 0.5), (0.0, 1.0)))
        box2 = BoundingBox(bounds=((0.5, 1.0), (0.0, 1.0)))

        sd1 = Subdomain(index=0, box=box1)
        sd2 = Subdomain(index=1, box=box2)

        collection = SubdomainCollection(
            subdomains=[sd1, sd2],
            hidden_layers=[10, 10],
            backend="tensorflow",
        )

        assert len(collection) == 2

    def test_build_networks(self):
        """Test network building."""
        box = BoundingBox(bounds=((0.0, 1.0), (0.0, 1.0)))
        sd = Subdomain(index=0, box=box)

        collection = SubdomainCollection(
            subdomains=[sd],
            hidden_layers=[10, 10],
            backend="tensorflow",
        )
        collection.build_all()

        assert collection.networks[0]._built


# =============================================================================
# Interface Tests
# =============================================================================


class TestInterface:
    """Tests for Interface class."""

    def test_interface_creation(self):
        """Test interface creation."""
        box = BoundingBox(bounds=((0.5, 0.5), (0.0, 1.0)))
        interface = Interface(
            subdomain_i=0,
            subdomain_j=1,
            box=box,
            normal=np.array([1.0, 0.0]),
        )

        assert interface.subdomain_i == 0
        assert interface.subdomain_j == 1

    def test_quadrature_setup(self):
        """Test interface quadrature."""
        box = BoundingBox(bounds=((0.5, 0.5), (0.0, 1.0)))
        interface = Interface(
            subdomain_i=0,
            subdomain_j=1,
            box=box,
            normal=np.array([1.0, 0.0]),
        )
        interface.setup_quadrature(n_points=8)

        assert interface.quadrature_points is not None
        # All points should have x = 0.5
        assert_allclose(interface.quadrature_points[:, 0], 0.5)


class TestPenaltyCoupling:
    """Tests for PenaltyCoupling class."""

    def test_coupling_creation(self):
        """Test coupling creation."""
        coupling = PenaltyCoupling(beta=100.0, gamma=10.0)
        assert coupling.beta == 100.0
        assert coupling.gamma == 10.0


# =============================================================================
# Refinement Indicator Tests
# =============================================================================


class TestResidualIndicator:
    """Tests for ResidualIndicator class."""

    def test_indicator_creation(self):
        """Test indicator creation."""
        indicator = ResidualIndicator(n_quadrature=16, norm="l2")
        assert indicator.n_quadrature == 16
        assert indicator.norm == "l2"


class TestMarkForRefinement:
    """Tests for mark_for_refinement function."""

    def test_maximum_strategy(self):
        """Test maximum marking strategy."""
        indicators = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        marked = mark_for_refinement(indicators, strategy="maximum", theta=0.5)

        # Should mark indicator > 0.5 * 0.8 = 0.4
        assert 1 in marked  # 0.5 > 0.4
        assert 3 in marked  # 0.8 > 0.4
        assert 0 not in marked

    def test_doerfler_strategy(self):
        """Test Doerfler bulk marking."""
        indicators = np.array([0.1, 0.2, 0.3, 0.4])
        marked = mark_for_refinement(indicators, strategy="doerfler", theta=0.5)

        # Should mark enough to capture 50% of total (1.0)
        total_marked = sum(indicators[i] for i in marked)
        assert total_marked >= 0.5

    def test_fixed_fraction_strategy(self):
        """Test fixed fraction marking."""
        indicators = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        marked = mark_for_refinement(
            indicators, strategy="fixed_fraction", theta=0.4
        )

        # Should mark top 40% = 2 subdomains
        assert len(marked) == 2
        assert 4 in marked  # Highest
        assert 3 in marked  # Second highest

    def test_max_marked_limit(self):
        """Test max_marked limit."""
        indicators = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        marked = mark_for_refinement(
            indicators, strategy="maximum", theta=0.1, max_marked=2
        )

        assert len(marked) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
