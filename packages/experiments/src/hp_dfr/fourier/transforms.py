"""Discrete Sine/Cosine Transforms for DFR method.

This module implements DST/DCT transforms for computing the H^{-1} dual norm
of PDE residuals. Both full tensor product and sparse implementations are
provided.

Transform Types:
    - DST-I: For homogeneous Dirichlet BCs (u = 0 on boundary)
    - DCT-I: For homogeneous Neumann BCs (du/dn = 0 on boundary)

The H^{-1} norm uses the energy inner product on H^1_0, so it weights by the
inverse Dirichlet-Laplacian eigenvalues:
    ||f||_{H^{-1}}^2 = sum_k |f_k|^2 / lambda_k,   lambda_k = sum_i (pi k_i / L_i)^2

where f_k are the sine coefficients. Note lambda_k couples the axes through the
sum of squared frequencies; it is not the product of the one-dimensional weights.

References:
    - Taylor et al. (2023). A Deep Fourier Residual Method.
    - Briggs & Henson (1995). The DFT: An Owner's Manual.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy import fft as scipy_fft

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
    return cast(NDArray[np.floating], norm * dx * coeffs / 2)


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
    return cast(NDArray[np.floating], norm * dx * coeffs / 2)


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
    return cast(NDArray[np.floating], basis * dx)


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

    return cast(NDArray[np.floating], basis_deriv * dx)


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

    result: NDArray[np.floating] = f.copy()

    for axis in range(dim):
        a, b = domain[axis]
        L = b - a
        norm = np.sqrt(2.0 / L)
        n = f.shape[axis]
        dx = L / n

        # Apply DST along this axis
        result = scipy_fft.dst(result, type=1, axis=axis, norm=None)

        # Truncate to n_modes along this axis
        result = cast(NDArray[np.floating], np.take(result, range(n_modes[axis]), axis=axis))

        # Normalize
        result = result * (norm * dx / 2)

    return result


def dst_matrix_nd_full(
    grid_shape: Tuple[int, ...],
    domain: Tuple[Tuple[float, float], ...],
    n_modes: Union[int, Tuple[int, ...]],
) -> NDArray[np.floating]:
    """Construct full-tensor DST matrix for direct multiplication.

    Creates matrix M such that ``coeffs = M @ f.ravel()``, where ``f`` is the
    function sampled on a tensor quadrature grid of shape ``grid_shape`` and
    ``coeffs`` are the full tensor-product DST-I coefficients raveled in C order
    over the multi-index (k_1, ..., k_d), matching the ordering of both
    :func:`dst_nd_full` and :func:`h_minus_1_weights`.

    The matrix is the Kronecker product of the per-dimension 1D DST matrices,
    which is exact because the sine basis is separable. This precomputed form is
    useful when the same transform is applied many times (e.g., to the network
    residual during training).

    Args:
        grid_shape: Shape of the function grid (n1, ..., nd).
        domain: Tuple of (a_i, b_i) for each dimension.
        n_modes: Number of modes per dimension (int for uniform, tuple for
            varying).

    Returns:
        DST matrix of shape (prod(n_modes_per_dim), prod(grid_shape)).
    """
    dim = len(grid_shape)
    if isinstance(n_modes, int):
        modes_per_dim: Tuple[int, ...] = (n_modes,) * dim
    else:
        modes_per_dim = n_modes

    matrix = dst_matrix_1d(grid_shape[0], modes_per_dim[0], domain[0])
    for d in range(1, dim):
        matrix = np.kron(matrix, dst_matrix_1d(grid_shape[d], modes_per_dim[d], domain[d]))

    return matrix


# =============================================================================
# H^{-1} Norm Weights
# =============================================================================


def h_minus_1_weights(
    n_modes: Union[int, Tuple[int, ...]],
    domain: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]],
) -> NDArray[np.floating]:
    r"""Compute H^{-1} norm weights for full tensor Fourier modes.

    The H^{-1} norm uses the eigenvalues of the Dirichlet Laplacian. For a
    multi-index k = (k_1, ..., k_d) on the box (0, L_1) x ... x (0, L_d), the
    sine eigenfunction has eigenvalue

        lambda_k = sum_i (pi * k_i / L_i)^2,

    and the dual (H^{-1}) weight is the Riesz factor

        w_k = lambda_k^{-1/2} = ( sum_i (pi * k_i / L_i)^2 )^{-1/2},

    so that ||f||_{H^{-1}}^2 = sum_k |f_k|^2 * w_k^2. In 1D this reduces to
    w_k = L / (pi * k). Note this is *not* the separable product of 1D weights:
    in d >= 2 the weight couples the axes through the sum of squared frequencies.

    Args:
        n_modes: Number of modes (int for 1D, tuple for nD).
        domain: Domain bounds.

    Returns:
        Array of weights, shape (n_modes,) for 1D or (m1,...,md) for nD.
    """
    if isinstance(n_modes, int):
        # 1D case
        a, b = cast(Tuple[float, float], domain)
        L = b - a
        k = np.arange(1, n_modes + 1)
        return cast(NDArray[np.floating], L / (np.pi * k))

    # nD case: w_k = ( sum_i (pi k_i / L_i)^2 )^{-1/2}.
    nd_domain = cast(Tuple[Tuple[float, float], ...], domain)
    dim = len(n_modes)
    freq_sq_1d = []

    for d in range(dim):
        a, b = nd_domain[d]
        L = b - a
        k = np.arange(1, n_modes[d] + 1)
        freq_sq_1d.append((np.pi * k / L) ** 2)

    # Sum the per-axis squared frequencies over the tensor grid, then invert sqrt.
    grids = np.meshgrid(*freq_sq_1d, indexing="ij")
    eigenvalues = np.zeros(n_modes)
    for g in grids:
        eigenvalues = eigenvalues + g

    return cast(NDArray[np.floating], 1.0 / np.sqrt(eigenvalues))


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
    return float(np.sqrt(norm_sq))


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
