"""Discrete Sine/Cosine Transforms for DFR method.

This module implements DST/DCT transforms for computing the H^{-1} dual norm
of PDE residuals. Both full tensor product and sparse implementations are
provided.

Transform Types:
    - DST-I: For homogeneous Dirichlet BCs (u = 0 on boundary)
    - DCT-I: For homogeneous Neumann BCs (du/dn = 0 on boundary)

The H^{-1} norm is computed as:
    ||f||_{H^{-1}}^2 = sum_k |f_k|^2 / (1 + |k|^2 * pi^2 / L^2)

where f_k are the Fourier coefficients.

References:
    - Taylor et al. (2022). A Deep Fourier Residual Method.
    - Briggs & Henson (1995). The DFT: An Owner's Manual.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy import fft as scipy_fft

from hp_dfr.fourier.sparse_indices import (
    HyperbolicCrossIndexSet,
    full_tensor_indices,
    hyperbolic_cross_indices,
)


# =============================================================================
# 1D Transforms
# =============================================================================


def dst_1d(
    f: NDArray[np.floating],
    domain: Tuple[float, float] = (0.0, np.pi),
    n_modes: Optional[int] = None,
) -> NDArray[np.floating]:
    """Compute 1D Discrete Sine Transform (DST-I).

    Computes the sine series coefficients:
        f_k = sqrt(2/L) * integral_a^b f(x) * sin(k*pi*(x-a)/L) dx

    Uses scipy's DST-I implementation with FFT.

    Args:
        f: Function values at quadrature points, shape (n_points,).
        domain: Spatial domain (a, b).
        n_modes: Number of Fourier modes (default: len(f) - 1).

    Returns:
        Sine coefficients of shape (n_modes,).

    Example:
        >>> x = np.linspace(0, np.pi, 100)
        >>> f = np.sin(2 * x)  # f_2 = 1, others = 0
        >>> coeffs = dst_1d(f, domain=(0, np.pi))
        >>> print(f"Mode 2 coefficient: {coeffs[1]:.4f}")
    """
    a, b = domain
    L = b - a
    n = len(f)

    if n_modes is None:
        n_modes = n - 1

    # Normalization factor for orthonormal basis
    norm = np.sqrt(2.0 / L)

    # scipy.fft.dst computes unnormalized DST-I
    # Need to scale by dx for quadrature approximation
    dx = L / n
    coeffs = scipy_fft.dst(f, type=1, norm=None)[:n_modes]

    # Scale to orthonormal coefficients
    return norm * dx * coeffs / 2


def dct_1d(
    f: NDArray[np.floating],
    domain: Tuple[float, float] = (0.0, np.pi),
    n_modes: Optional[int] = None,
) -> NDArray[np.floating]:
    """Compute 1D Discrete Cosine Transform (DCT-I).

    Computes the cosine series coefficients:
        f_k = sqrt(2/L) * integral_a^b f(x) * cos(k*pi*(x-a)/L) dx

    Args:
        f: Function values at quadrature points, shape (n_points,).
        domain: Spatial domain (a, b).
        n_modes: Number of Fourier modes (default: len(f)).

    Returns:
        Cosine coefficients of shape (n_modes,).
    """
    a, b = domain
    L = b - a
    n = len(f)

    if n_modes is None:
        n_modes = n

    norm = np.sqrt(2.0 / L)
    dx = L / n
    coeffs = scipy_fft.dct(f, type=1, norm=None)[:n_modes]

    # Scale to orthonormal coefficients
    # DCT-I has different normalization at endpoints
    return norm * dx * coeffs / 2


def dst_matrix_1d(
    n_points: int,
    n_modes: int,
    domain: Tuple[float, float] = (0.0, np.pi),
) -> NDArray[np.floating]:
    """Construct 1D DST matrix for direct multiplication.

    The matrix M satisfies: coeffs = M @ f_values

    Args:
        n_points: Number of quadrature points.
        n_modes: Number of Fourier modes.
        domain: Spatial domain (a, b).

    Returns:
        DST matrix of shape (n_modes, n_points).
    """
    a, b = domain
    L = b - a

    # Quadrature points (midpoint rule)
    x = np.linspace(a, b, n_points + 1)
    dx = x[1] - x[0]
    x_mid = x[:-1] + dx / 2

    # Fourier modes
    k = np.arange(1, n_modes + 1)

    # Basis functions: sqrt(2/L) * sin(k*pi*(x-a)/L)
    norm = np.sqrt(2.0 / L)
    # Shape: (n_modes, n_points)
    basis = norm * np.sin(np.outer(k, (x_mid - a) * np.pi / L))

    # Include quadrature weight
    return basis * dx


def dct_derivative_matrix_1d(
    n_points: int,
    n_modes: int,
    domain: Tuple[float, float] = (0.0, np.pi),
) -> NDArray[np.floating]:
    """Construct matrix for DCT of derivative (gradient term in DFR).

    For the weak form, we need integral of du/dx * test_function.
    Using integration by parts with sine test functions:
        integral(du/dx * sin) = -integral(u * d(sin)/dx) = -integral(u * k*pi/L * cos)

    This matrix computes: integral(du/dx * sin_k) from function values.

    Args:
        n_points: Number of quadrature points.
        n_modes: Number of Fourier modes.
        domain: Spatial domain (a, b).

    Returns:
        Matrix of shape (n_modes, n_points).
    """
    a, b = domain
    L = b - a

    # Quadrature points
    x = np.linspace(a, b, n_points + 1)
    dx = x[1] - x[0]
    x_mid = x[:-1] + dx / 2

    # Fourier modes
    k = np.arange(1, n_modes + 1)

    # Derivative of basis: sqrt(2/L) * k*pi/L * cos(k*pi*(x-a)/L)
    norm = np.sqrt(2.0 / L)
    deriv_factor = k * np.pi / L
    basis_deriv = norm * np.outer(deriv_factor, np.cos(np.outer(k, (x_mid - a) * np.pi / L))[0])

    # Recompute properly
    basis_deriv = np.zeros((n_modes, n_points))
    for i, ki in enumerate(k):
        basis_deriv[i, :] = norm * (ki * np.pi / L) * np.cos(ki * np.pi * (x_mid - a) / L)

    return basis_deriv * dx


# =============================================================================
# Multi-dimensional Transforms (Full Tensor)
# =============================================================================


def dst_nd_full(
    f: NDArray[np.floating],
    domain: Tuple[Tuple[float, float], ...],
    n_modes: Union[int, Tuple[int, ...]],
) -> NDArray[np.floating]:
    """Compute n-dimensional DST using full tensor product.

    Applies DST-I along each dimension using separability:
        f_{k1,...,kd} = DST_1(...DST_d(f)...)

    Args:
        f: Function values on tensor grid, shape (n1, ..., nd).
        domain: Tuple of (a_i, b_i) for each dimension.
        n_modes: Number of modes (int for uniform, tuple for varying).

    Returns:
        Fourier coefficients of shape (m1, ..., md).

    Example:
        >>> # 2D Poisson on [0,pi]^2
        >>> x = np.linspace(0, np.pi, 32)
        >>> X, Y = np.meshgrid(x, x, indexing='ij')
        >>> f = np.sin(X) * np.sin(2*Y)  # f_{1,2} = 1
        >>> coeffs = dst_nd_full(f, ((0,np.pi), (0,np.pi)), n_modes=16)
    """
    dim = f.ndim

    if isinstance(n_modes, int):
        n_modes = tuple([n_modes] * dim)

    result = f.copy()

    for axis in range(dim):
        a, b = domain[axis]
        L = b - a
        norm = np.sqrt(2.0 / L)
        n = f.shape[axis]
        dx = L / n

        # Apply DST along this axis
        result = scipy_fft.dst(result, type=1, axis=axis, norm=None)

        # Truncate to n_modes
        slices = [slice(None)] * dim
        slices[axis] = slice(0, n_modes[axis])
        result = result[tuple(slices)]

        # Normalize
        result = result * (norm * dx / 2)

    return result


# =============================================================================
# Sparse Transforms (Hyperbolic Cross)
# =============================================================================


def dst_nd_sparse(
    f: NDArray[np.floating],
    domain: Tuple[Tuple[float, float], ...],
    index_set: HyperbolicCrossIndexSet,
) -> NDArray[np.floating]:
    """Compute n-dimensional DST at sparse index set only.

    Instead of computing all O(N^d) coefficients, computes only
    coefficients at indices in the hyperbolic cross.

    Args:
        f: Function values on tensor grid, shape (n1, ..., nd).
        domain: Tuple of (a_i, b_i) for each dimension.
        index_set: Hyperbolic cross index set.

    Returns:
        Sparse Fourier coefficients of shape (n_indices,).

    Note:
        For large problems, this is more efficient than computing
        the full transform and extracting sparse indices.
    """
    dim = f.ndim
    n_indices = len(index_set)

    # For small problems, compute full transform and extract
    # For large problems, would use hierarchical evaluation
    if np.prod(f.shape) < 1e6:
        return _dst_sparse_via_full(f, domain, index_set)
    else:
        return _dst_sparse_hierarchical(f, domain, index_set)


def _dst_sparse_via_full(
    f: NDArray[np.floating],
    domain: Tuple[Tuple[float, float], ...],
    index_set: HyperbolicCrossIndexSet,
) -> NDArray[np.floating]:
    """Compute sparse DST by extracting from full transform."""
    dim = f.ndim

    # Compute full transform
    max_modes = tuple(int(np.max(index_set.indices[:, i])) for i in range(dim))
    full_coeffs = dst_nd_full(f, domain, max_modes)

    # Extract sparse indices (convert to 0-based)
    sparse_coeffs = np.zeros(len(index_set))
    for i, idx in enumerate(index_set.indices):
        idx_0based = tuple(idx - 1)
        sparse_coeffs[i] = full_coeffs[idx_0based]

    return sparse_coeffs


def _dst_sparse_hierarchical(
    f: NDArray[np.floating],
    domain: Tuple[Tuple[float, float], ...],
    index_set: HyperbolicCrossIndexSet,
) -> NDArray[np.floating]:
    """Compute sparse DST using hierarchical evaluation.

    For very large problems, this avoids computing all full coefficients
    by using the tensor product structure more efficiently.
    """
    # For now, fall back to full computation
    # TODO: Implement true hierarchical sparse evaluation
    return _dst_sparse_via_full(f, domain, index_set)


def dst_matrix_nd_sparse(
    grid_shape: Tuple[int, ...],
    domain: Tuple[Tuple[float, float], ...],
    index_set: HyperbolicCrossIndexSet,
) -> NDArray[np.floating]:
    """Construct sparse DST matrix for direct multiplication.

    Creates matrix M such that: sparse_coeffs = M @ f.ravel()

    This is useful when the same transform is applied many times
    (e.g., during neural network training).

    Args:
        grid_shape: Shape of the function grid (n1, ..., nd).
        domain: Tuple of (a_i, b_i) for each dimension.
        index_set: Hyperbolic cross index set.

    Returns:
        Sparse DST matrix of shape (n_indices, prod(grid_shape)).
    """
    dim = len(grid_shape)
    n_grid = int(np.prod(grid_shape))
    n_indices = len(index_set)

    # Build 1D basis functions for each dimension
    basis_1d = []
    for d in range(dim):
        a, b = domain[d]
        L = b - a
        n = grid_shape[d]

        x = np.linspace(a, b, n + 1)
        dx = x[1] - x[0]
        x_mid = x[:-1] + dx / 2

        norm = np.sqrt(2.0 / L)
        max_k = int(np.max(index_set.indices[:, d]))

        # basis_1d[d][k, x] = norm * sin(k*pi*(x-a)/L) * dx
        k_vals = np.arange(1, max_k + 1)
        basis = np.zeros((max_k, n))
        for i, k in enumerate(k_vals):
            basis[i, :] = norm * np.sin(k * np.pi * (x_mid - a) / L) * dx

        basis_1d.append(basis)

    # Build sparse matrix using tensor product of 1D bases
    matrix = np.zeros((n_indices, n_grid))

    for i, idx in enumerate(index_set.indices):
        # Tensor product of 1D basis functions
        row = np.ones(n_grid)
        for d in range(dim):
            k = idx[d] - 1  # 0-based index into basis
            # Expand 1D basis to full grid using outer product structure
            basis_d = basis_1d[d][k, :]

            # Reshape for broadcasting
            shape = [1] * dim
            shape[d] = grid_shape[d]
            basis_expanded = basis_d.reshape(shape)

            # Tile to full grid
            tile_shape = list(grid_shape)
            tile_shape[d] = 1
            basis_full = np.tile(basis_expanded, tile_shape).ravel()

            row *= basis_full

        matrix[i, :] = row

    return matrix


# =============================================================================
# H^{-1} Norm Weights
# =============================================================================


def h_minus_1_weights(
    n_modes: Union[int, Tuple[int, ...]],
    domain: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]],
) -> NDArray[np.floating]:
    """Compute H^{-1} norm weights for full tensor Fourier modes.

    The H^{-1} norm is:
        ||f||_{H^{-1}}^2 = sum_k |f_k|^2 * w_k^2

    where w_k = (pi^2 * |k|^2 / L^2)^{-1/2} = L / (pi * |k|)

    For multi-dimensional case with k = (k1, ..., kd):
        w_k = prod_i (L_i / (pi * k_i))

    Args:
        n_modes: Number of modes (int for 1D, tuple for nD).
        domain: Domain bounds.

    Returns:
        Array of weights, shape (n_modes,) for 1D or (m1,...,md) for nD.
    """
    if isinstance(n_modes, int):
        # 1D case
        a, b = domain
        L = b - a
        k = np.arange(1, n_modes + 1)
        return L / (np.pi * k)

    # nD case
    dim = len(n_modes)
    weights_1d = []

    for d in range(dim):
        a, b = domain[d]
        L = b - a
        k = np.arange(1, n_modes[d] + 1)
        weights_1d.append(L / (np.pi * k))

    # Tensor product of weights
    grids = np.meshgrid(*weights_1d, indexing="ij")
    weights = np.ones(n_modes)
    for g in grids:
        weights *= g

    return weights


def h_minus_1_weights_sparse(
    index_set: HyperbolicCrossIndexSet,
    domain: Tuple[Tuple[float, float], ...],
) -> NDArray[np.floating]:
    """Compute H^{-1} norm weights for sparse index set.

    Args:
        index_set: Hyperbolic cross index set.
        domain: Tuple of (a_i, b_i) for each dimension.

    Returns:
        Array of weights, shape (n_indices,).
    """
    dim = index_set.dim
    n_indices = len(index_set)

    weights = np.ones(n_indices)

    for d in range(dim):
        a, b = domain[d]
        L = b - a
        k_d = index_set.indices[:, d]
        weights *= L / (np.pi * k_d)

    return weights


def compute_h_minus_1_norm(
    coeffs: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> float:
    """Compute H^{-1} norm from Fourier coefficients.

    ||f||_{H^{-1}}^2 = sum_k |f_k|^2 * w_k^2

    Args:
        coeffs: Fourier coefficients (flattened for nD).
        weights: H^{-1} weights (flattened for nD).

    Returns:
        The H^{-1} norm.
    """
    coeffs_flat = coeffs.ravel()
    weights_flat = weights.ravel()

    norm_sq = np.sum((coeffs_flat * weights_flat) ** 2)
    return np.sqrt(norm_sq)


# =============================================================================
# Convenience Functions
# =============================================================================


def dfr_loss_1d(
    residual_fn: Callable[[NDArray], NDArray],
    x_quad: NDArray[np.floating],
    domain: Tuple[float, float],
    n_modes: int,
) -> float:
    """Compute 1D DFR loss (H^{-1} norm of residual).

    Args:
        residual_fn: Function computing PDE residual at points.
        x_quad: Quadrature points.
        domain: Spatial domain (a, b).
        n_modes: Number of Fourier modes.

    Returns:
        DFR loss value (H^{-1} norm squared).
    """
    residual = residual_fn(x_quad)
    coeffs = dst_1d(residual, domain, n_modes)
    weights = h_minus_1_weights(n_modes, domain)
    return float(np.sum((coeffs * weights) ** 2))


def dfr_loss_nd_sparse(
    residual_values: NDArray[np.floating],
    domain: Tuple[Tuple[float, float], ...],
    index_set: HyperbolicCrossIndexSet,
) -> float:
    """Compute n-dimensional DFR loss using sparse Fourier modes.

    Args:
        residual_values: Residual on tensor grid, shape (n1, ..., nd).
        domain: Tuple of (a_i, b_i) for each dimension.
        index_set: Hyperbolic cross index set.

    Returns:
        DFR loss value (H^{-1} norm squared).
    """
    coeffs = dst_nd_sparse(residual_values, domain, index_set)
    weights = h_minus_1_weights_sparse(index_set, domain)
    return float(np.sum((coeffs * weights) ** 2))
