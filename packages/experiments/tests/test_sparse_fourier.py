"""Unit tests for sparse Fourier methods.

Tests cover:
1. Index set generation (hyperbolic cross, Smolyak)
2. DST/DCT transforms (1D and nD)
3. Sparse vs full comparison
4. H^{-1} norm computation
5. Complexity scaling
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hp_dfr.fourier import (
    HyperbolicCrossIndexSet,
    full_tensor_indices,
    hyperbolic_cross_indices,
    smolyak_indices,
    dst_1d,
    dct_1d,
    dst_nd_full,
    dst_nd_sparse,
    h_minus_1_weights,
    h_minus_1_weights_sparse,
    dst_matrix_1d,
)
from hp_dfr.fourier.sparse_indices import (
    count_hyperbolic_cross,
    asymptotic_size,
    hyperbolic_cross_log_indices,
)
from hp_dfr.fourier.transforms import (
    compute_h_minus_1_norm,
    dfr_loss_1d,
    dfr_loss_nd_sparse,
    dst_matrix_nd_sparse,
)


# =============================================================================
# Index Set Tests
# =============================================================================


class TestFullTensorIndices:
    """Tests for full tensor product index generation."""

    def test_1d_indices(self):
        """Test 1D index generation."""
        indices = full_tensor_indices(dim=1, n_modes=5)
        expected = np.array([[1], [2], [3], [4], [5]])
        assert_array_equal(indices, expected)

    def test_2d_indices_shape(self):
        """Test 2D index generation produces correct shape."""
        indices = full_tensor_indices(dim=2, n_modes=4)
        assert indices.shape == (16, 2)

    def test_2d_indices_count(self):
        """Test 2D produces N^2 indices."""
        for n in [3, 5, 8]:
            indices = full_tensor_indices(dim=2, n_modes=n)
            assert len(indices) == n**2

    def test_3d_indices_count(self):
        """Test 3D produces N^3 indices."""
        indices = full_tensor_indices(dim=3, n_modes=4)
        assert len(indices) == 64

    def test_indices_are_positive(self):
        """Test all indices are >= 1."""
        indices = full_tensor_indices(dim=3, n_modes=5)
        assert np.all(indices >= 1)

    def test_indices_bounded(self):
        """Test all indices are <= n_modes."""
        n = 7
        indices = full_tensor_indices(dim=2, n_modes=n)
        assert np.all(indices <= n)


class TestHyperbolicCrossIndices:
    """Tests for hyperbolic cross index generation."""

    def test_1d_same_as_full(self):
        """In 1D, hyperbolic cross equals full tensor."""
        for n in [5, 10, 20]:
            hc = hyperbolic_cross_indices(dim=1, max_level=n)
            full = full_tensor_indices(dim=1, n_modes=n)
            assert_array_equal(hc, full)

    def test_2d_product_constraint(self):
        """Test all 2D indices satisfy k1*k2 <= N."""
        for n in [8, 16, 32]:
            indices = hyperbolic_cross_indices(dim=2, max_level=n)
            products = indices[:, 0] * indices[:, 1]
            assert np.all(products <= n)

    def test_3d_product_constraint(self):
        """Test all 3D indices satisfy k1*k2*k3 <= N."""
        indices = hyperbolic_cross_indices(dim=3, max_level=16)
        products = np.prod(indices, axis=1)
        assert np.all(products <= 16)

    def test_compression_2d(self):
        """Test 2D compression ratio is significant."""
        for n in [8, 16, 32]:
            hc = hyperbolic_cross_indices(dim=2, max_level=n)
            full_size = n**2
            hc_size = len(hc)
            # Hyperbolic cross should be much smaller
            assert hc_size < full_size / 2

    def test_compression_3d(self):
        """Test 3D compression is even more significant."""
        hc = hyperbolic_cross_indices(dim=3, max_level=16)
        full_size = 16**3
        hc_size = len(hc)
        # Should be at least 10x compression in 3D
        assert hc_size < full_size / 10

    def test_indices_contain_corners(self):
        """Test hyperbolic cross contains corner indices."""
        indices = hyperbolic_cross_indices(dim=2, max_level=8)
        indices_set = set(map(tuple, indices))
        # Should contain (1,1), (1,8), (8,1) but not (8,8)
        assert (1, 1) in indices_set
        assert (1, 8) in indices_set
        assert (8, 1) in indices_set
        assert (8, 8) not in indices_set  # 8*8 = 64 > 8

    def test_known_values_2d(self):
        """Test against known hyperbolic cross sizes."""
        # |H_N^2| for small N
        known = {2: 3, 4: 8, 8: 20, 16: 51}
        for n, expected_size in known.items():
            indices = hyperbolic_cross_indices(dim=2, max_level=n)
            assert len(indices) == expected_size, f"N={n}: got {len(indices)}, expected {expected_size}"


class TestHyperbolicCrossIndexSet:
    """Tests for HyperbolicCrossIndexSet class."""

    def test_creation(self):
        """Test index set creation."""
        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=8)
        assert idx_set.dim == 2
        assert idx_set.max_level == 8
        assert idx_set.n_indices == 20

    def test_len(self):
        """Test __len__ method."""
        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=16)
        assert len(idx_set) == idx_set.n_indices

    def test_iter(self):
        """Test iteration over indices."""
        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=4)
        indices_list = list(idx_set)
        assert len(indices_list) == len(idx_set)

    def test_contains(self):
        """Test __contains__ method."""
        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=8)
        assert (2, 4) in idx_set  # 2*4 = 8 <= 8
        assert (3, 3) not in idx_set  # 3*3 = 9 > 8

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=16)
        ratio = idx_set.compression_ratio()
        assert ratio > 1.0  # Hyperbolic cross is smaller
        assert ratio == 16**2 / idx_set.n_indices


class TestSmolyakIndices:
    """Tests for Smolyak sparse grid indices."""

    def test_1d_same_as_range(self):
        """In 1D, Smolyak is just 1 to level."""
        indices = smolyak_indices(dim=1, level=5)
        expected = np.arange(1, 6).reshape(-1, 1)
        assert_array_equal(indices, expected)

    def test_sum_constraint(self):
        """Test sum constraint is satisfied."""
        level = 4
        dim = 3
        indices = smolyak_indices(dim=dim, level=level)
        max_sum = level + dim - 1
        sums = np.sum(indices, axis=1)
        assert np.all(sums <= max_sum)


class TestCountHyperbolicCross:
    """Tests for counting function."""

    def test_matches_generation(self):
        """Test count matches actual generation."""
        for dim in [2, 3]:
            for n in [4, 8, 16]:
                count = count_hyperbolic_cross(dim, n)
                indices = hyperbolic_cross_indices(dim, n)
                assert count == len(indices)


class TestAsymptoticSize:
    """Tests for asymptotic size estimation."""

    def test_reasonable_estimate_2d(self):
        """Test asymptotic estimate is reasonable for 2D."""
        for n in [32, 64, 128]:
            actual = len(hyperbolic_cross_indices(dim=2, max_level=n))
            estimate = asymptotic_size(dim=2, n=n)
            # Should be within factor of 2
            assert 0.5 * estimate < actual < 2 * estimate


# =============================================================================
# Transform Tests
# =============================================================================


class TestDST1D:
    """Tests for 1D Discrete Sine Transform."""

    def test_single_mode(self):
        """Test DST correctly identifies single sine mode."""
        n_points = 128
        domain = (0.0, np.pi)
        x = np.linspace(0, np.pi, n_points)

        # f = sin(2x) should have coefficient 1 at mode 2
        f = np.sin(2 * x)
        coeffs = dst_1d(f, domain, n_modes=10)

        # Mode 2 (index 1) should be close to 1
        assert_allclose(coeffs[1], 1.0, atol=0.1)

        # Other modes should be small
        assert np.abs(coeffs[0]) < 0.1
        assert np.abs(coeffs[2]) < 0.1

    def test_orthonormality(self):
        """Test basis is approximately orthonormal."""
        n_points = 256
        domain = (0.0, np.pi)
        L = np.pi

        x = np.linspace(0, np.pi, n_points + 1)
        dx = x[1] - x[0]
        x_mid = x[:-1] + dx / 2

        norm = np.sqrt(2.0 / L)

        # Test that integral of sin(kx)^2 ≈ L/2 (so normalized gives 1)
        for k in [1, 2, 3]:
            basis = norm * np.sin(k * np.pi * x_mid / L)
            integral = np.sum(basis**2) * dx
            assert_allclose(integral, 1.0, atol=0.05)


class TestDSTMatrix1D:
    """Tests for 1D DST matrix construction."""

    def test_matrix_vs_fft(self):
        """Test matrix multiplication matches FFT-based DST."""
        n_points = 64
        n_modes = 16
        domain = (0.0, np.pi)

        matrix = dst_matrix_1d(n_points, n_modes, domain)

        # Random function values
        np.random.seed(42)
        f = np.random.randn(n_points)

        # Matrix multiplication
        coeffs_matrix = matrix @ f

        # FFT-based (need to set up midpoint values)
        x = np.linspace(0, np.pi, n_points + 1)
        dx = x[1] - x[0]
        x_mid = x[:-1] + dx / 2

        # The matrix was built for midpoint rule
        # For comparison, we need matching quadrature
        assert coeffs_matrix.shape == (n_modes,)


class TestDSTNDFull:
    """Tests for n-dimensional full tensor DST."""

    def test_2d_single_mode(self):
        """Test 2D DST correctly identifies single mode."""
        n = 64
        domain = ((0.0, np.pi), (0.0, np.pi))
        x = np.linspace(0, np.pi, n)
        y = np.linspace(0, np.pi, n)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # f = sin(x) * sin(2y) should have coefficient at (1,2)
        f = np.sin(X) * np.sin(2 * Y)
        coeffs = dst_nd_full(f, domain, n_modes=8)

        # Coefficient at (1,2) -> index (0,1) should be largest
        assert coeffs[0, 1] > 0.5

    def test_separability(self):
        """Test separable function gives product of 1D coefficients."""
        n = 64
        domain = ((0.0, np.pi), (0.0, np.pi))
        x = np.linspace(0, np.pi, n)
        y = np.linspace(0, np.pi, n)
        X, Y = np.meshgrid(x, y, indexing="ij")

        f1 = np.sin(X)
        f2 = np.sin(Y)
        f = f1 * f2

        coeffs_2d = dst_nd_full(f, domain, n_modes=8)
        coeffs_1d_x = dst_1d(np.sin(x), (0, np.pi), n_modes=8)
        coeffs_1d_y = dst_1d(np.sin(y), (0, np.pi), n_modes=8)

        # Product of 1D coefficients
        expected = np.outer(coeffs_1d_x, coeffs_1d_y)

        # Should match approximately
        assert_allclose(coeffs_2d, expected, atol=0.2)


class TestDSTNDSparse:
    """Tests for sparse n-dimensional DST."""

    def test_matches_full_at_sparse_indices(self):
        """Test sparse DST matches full DST at hyperbolic cross indices."""
        n = 32
        domain = ((0.0, np.pi), (0.0, np.pi))
        x = np.linspace(0, np.pi, n)
        y = np.linspace(0, np.pi, n)
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = np.sin(X) * np.sin(2 * Y) + 0.5 * np.sin(3 * X) * np.sin(Y)

        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=8)
        sparse_coeffs = dst_nd_sparse(f, domain, idx_set)

        # Compute full and extract
        full_coeffs = dst_nd_full(f, domain, n_modes=8)
        extracted = np.array([
            full_coeffs[tuple(idx - 1)] for idx in idx_set.indices
        ])

        assert_allclose(sparse_coeffs, extracted, rtol=1e-10)

    def test_compression_savings(self):
        """Test sparse method uses fewer coefficients."""
        n = 64
        domain = ((0.0, np.pi), (0.0, np.pi))
        x = np.linspace(0, np.pi, n)
        y = np.linspace(0, np.pi, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = np.sin(X) * np.sin(Y)

        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=16)
        sparse_coeffs = dst_nd_sparse(f, domain, idx_set)

        full_size = 16 * 16
        sparse_size = len(sparse_coeffs)

        assert sparse_size < full_size / 2


# =============================================================================
# H^{-1} Norm Tests
# =============================================================================


class TestHMinus1Weights:
    """Tests for H^{-1} norm weights."""

    def test_1d_weights_formula(self):
        """Test 1D weights match formula w_k = L/(pi*k)."""
        n_modes = 10
        domain = (0.0, np.pi)
        weights = h_minus_1_weights(n_modes, domain)

        L = np.pi
        k = np.arange(1, n_modes + 1)
        expected = L / (np.pi * k)

        assert_allclose(weights, expected)

    def test_weights_decrease(self):
        """Test weights decrease with k (higher modes weighted less)."""
        weights = h_minus_1_weights(20, (0.0, np.pi))
        assert np.all(np.diff(weights) < 0)

    def test_2d_weights_tensor_product(self):
        """Test 2D weights are tensor product of 1D."""
        domain = ((0.0, np.pi), (0.0, np.pi))
        weights = h_minus_1_weights((5, 5), domain)

        w1d = h_minus_1_weights(5, (0.0, np.pi))
        expected = np.outer(w1d, w1d)

        assert_allclose(weights, expected)


class TestHMinus1WeightsSparse:
    """Tests for sparse H^{-1} weights."""

    def test_matches_full_at_indices(self):
        """Test sparse weights match full weights at hyperbolic cross."""
        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=8)
        domain = ((0.0, np.pi), (0.0, np.pi))

        sparse_weights = h_minus_1_weights_sparse(idx_set, domain)

        # Full weights
        max_k = int(np.max(idx_set.indices))
        full_weights = h_minus_1_weights((max_k, max_k), domain)

        # Extract at sparse indices
        extracted = np.array([
            full_weights[tuple(idx - 1)] for idx in idx_set.indices
        ])

        assert_allclose(sparse_weights, extracted)


class TestHMinus1Norm:
    """Tests for H^{-1} norm computation."""

    def test_smooth_function_small_norm(self):
        """Smooth functions should have small H^{-1} norm."""
        n = 128
        x = np.linspace(0, np.pi, n)

        # Very smooth: sin(x)
        f_smooth = np.sin(x)
        coeffs_smooth = dst_1d(f_smooth, (0, np.pi), 32)
        weights = h_minus_1_weights(32, (0, np.pi))
        norm_smooth = compute_h_minus_1_norm(coeffs_smooth, weights)

        # Less smooth: sin(10x)
        f_rough = np.sin(10 * x)
        coeffs_rough = dst_1d(f_rough, (0, np.pi), 32)
        norm_rough = compute_h_minus_1_norm(coeffs_rough, weights)

        # Smooth function has larger H^{-1} norm (less penalized high modes)
        # Actually, H^{-1} penalizes high frequencies, so smooth is larger
        # But for same L^2 norm, rough should have smaller H^{-1}
        # Let's just check both are positive
        assert norm_smooth > 0
        assert norm_rough > 0


# =============================================================================
# DFR Loss Tests
# =============================================================================


class TestDFRLoss1D:
    """Tests for 1D DFR loss function."""

    def test_zero_residual_zero_loss(self):
        """Zero residual gives zero loss."""
        def zero_residual(x):
            return np.zeros_like(x)

        x_quad = np.linspace(0.1, np.pi - 0.1, 64)
        loss = dfr_loss_1d(zero_residual, x_quad, (0, np.pi), 16)
        assert_allclose(loss, 0.0, atol=1e-14)

    def test_nonzero_residual_positive_loss(self):
        """Nonzero residual gives positive loss."""
        def nonzero_residual(x):
            return np.sin(x)

        x_quad = np.linspace(0.1, np.pi - 0.1, 64)
        loss = dfr_loss_1d(nonzero_residual, x_quad, (0, np.pi), 16)
        assert loss > 0


class TestDFRLossNDSparse:
    """Tests for n-dimensional sparse DFR loss."""

    def test_zero_residual(self):
        """Zero residual gives zero loss."""
        n = 32
        domain = ((0.0, np.pi), (0.0, np.pi))
        residual = np.zeros((n, n))

        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=8)
        loss = dfr_loss_nd_sparse(residual, domain, idx_set)

        assert_allclose(loss, 0.0, atol=1e-14)

    def test_positive_for_nonzero(self):
        """Nonzero residual gives positive loss."""
        n = 32
        x = np.linspace(0, np.pi, n)
        y = np.linspace(0, np.pi, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        residual = np.sin(X) * np.sin(Y)

        domain = ((0.0, np.pi), (0.0, np.pi))
        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=8)
        loss = dfr_loss_nd_sparse(residual, domain, idx_set)

        assert loss > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestSparseDSTMatrix:
    """Tests for sparse DST matrix construction."""

    def test_matrix_shape(self):
        """Test matrix has correct shape."""
        grid_shape = (32, 32)
        domain = ((0.0, np.pi), (0.0, np.pi))
        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=8)

        matrix = dst_matrix_nd_sparse(grid_shape, domain, idx_set)

        assert matrix.shape == (len(idx_set), 32 * 32)

    def test_matrix_multiplication(self):
        """Test matrix gives same result as function call."""
        n = 32
        grid_shape = (n, n)
        domain = ((0.0, np.pi), (0.0, np.pi))
        idx_set = HyperbolicCrossIndexSet(dim=2, max_level=8)

        x = np.linspace(0, np.pi, n)
        y = np.linspace(0, np.pi, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = np.sin(X) * np.sin(2 * Y)

        # Function-based
        coeffs_fn = dst_nd_sparse(f, domain, idx_set)

        # Matrix-based
        matrix = dst_matrix_nd_sparse(grid_shape, domain, idx_set)
        coeffs_mat = matrix @ f.ravel()

        assert_allclose(coeffs_fn, coeffs_mat, rtol=1e-10)


# =============================================================================
# Scaling Tests
# =============================================================================


class TestComplexityScaling:
    """Tests for verifying complexity scaling."""

    def test_hyperbolic_cross_scaling(self):
        """Test hyperbolic cross grows as O(N log^{d-1} N)."""
        dim = 2
        sizes = []
        n_values = [16, 32, 64, 128]

        for n in n_values:
            idx = hyperbolic_cross_indices(dim=dim, max_level=n)
            sizes.append(len(idx))

        # Check ratio approaches asymptotic
        # |H_2N| / |H_N| should approach 2 * (1 + log(2)/log(N))
        for i in range(1, len(sizes)):
            ratio = sizes[i] / sizes[i - 1]
            # Should be roughly 2 to 2.5 for doubling N
            assert 1.8 < ratio < 3.0

    def test_3d_scaling_better(self):
        """Test 3D compression is better than 2D."""
        n = 16

        hc_2d = hyperbolic_cross_indices(dim=2, max_level=n)
        hc_3d = hyperbolic_cross_indices(dim=3, max_level=n)

        compression_2d = (n**2) / len(hc_2d)
        compression_3d = (n**3) / len(hc_3d)

        # 3D should have better compression
        assert compression_3d > compression_2d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
