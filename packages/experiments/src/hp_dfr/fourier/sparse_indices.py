"""Sparse index set generation for high-dimensional Fourier methods.

This module provides functions to generate sparse index sets that reduce
the curse of dimensionality from O(N^d) to O(N (log N)^{d-1}).

Index Set Types:
    - Full tensor: All (k1, ..., kd) with 1 <= ki <= N
    - Hyperbolic cross: Product constraint k1 * k2 * ... * kd <= N
    - Smolyak: Sum of logs constraint sum(log(1 + ki)) <= M

References:
    - Bungartz, H.J., Griebel, M. (2004). Sparse grids. Acta Numerica.
    - Garcke, J. (2013). Sparse Grids in a Nutshell. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class HyperbolicCrossIndexSet:
    """Hyperbolic cross index set for sparse Fourier methods.

    The hyperbolic cross contains all multi-indices (k1, ..., kd) with
    ki >= 1 satisfying the constraint:

        prod(ki) <= N  (product form)

    or equivalently:

        sum(log(ki)) <= log(N)

    This reduces cardinality from O(N^d) to O(N (log N)^{d-1}).

    Attributes:
        dim: Spatial dimension d.
        max_level: Maximum refinement level N.
        indices: Array of shape (n_indices, dim) containing all indices.
        n_indices: Number of indices in the set.

    Example:
        >>> idx_set = HyperbolicCrossIndexSet(dim=2, max_level=8)
        >>> print(f"Full tensor: {8**2}, Hyperbolic: {idx_set.n_indices}")
        Full tensor: 64, Hyperbolic: 20
    """

    dim: int
    max_level: int
    indices: NDArray[np.int64] = field(init=False, repr=False)
    n_indices: int = field(init=False)

    def __post_init__(self) -> None:
        """Generate the hyperbolic cross indices."""
        self.indices = hyperbolic_cross_indices(self.dim, self.max_level)
        self.n_indices = len(self.indices)

    def __len__(self) -> int:
        """Return number of indices."""
        return self.n_indices

    def __iter__(self) -> Iterator[NDArray[np.int64]]:
        """Iterate over indices."""
        return iter(self.indices)

    def __contains__(self, index: Tuple[int, ...]) -> bool:
        """Check if index is in the set."""
        return np.prod(index) <= self.max_level

    def compression_ratio(self) -> float:
        """Compute compression ratio vs full tensor product.

        Returns:
            Ratio of full tensor size to hyperbolic cross size.
        """
        full_size = self.max_level**self.dim
        return full_size / self.n_indices

    def to_linear_indices(self, shape: Tuple[int, ...]) -> NDArray[np.int64]:
        """Convert multi-indices to linear indices for array access.

        Args:
            shape: Shape of the full tensor array.

        Returns:
            Array of linear indices.
        """
        # Convert from 1-based to 0-based indexing
        zero_based = self.indices - 1
        strides = np.cumprod([1] + list(shape[:-1]))
        return np.sum(zero_based * strides, axis=1)


def full_tensor_indices(dim: int, n_modes: int) -> NDArray[np.int64]:
    """Generate full tensor product indices.

    Creates all multi-indices (k1, ..., kd) with 1 <= ki <= n_modes.

    Args:
        dim: Spatial dimension.
        n_modes: Number of modes per dimension.

    Returns:
        Array of shape (n_modes^dim, dim) with all indices.

    Example:
        >>> indices = full_tensor_indices(dim=2, n_modes=3)
        >>> print(indices)
        [[1 1]
         [1 2]
         [1 3]
         [2 1]
         ...
         [3 3]]
    """
    ranges = [np.arange(1, n_modes + 1) for _ in range(dim)]
    grids = np.meshgrid(*ranges, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def hyperbolic_cross_indices(dim: int, max_level: int) -> NDArray[np.int64]:
    """Generate hyperbolic cross indices using product constraint.

    The hyperbolic cross H_N^d contains all (k1, ..., kd) with ki >= 1
    satisfying prod(ki) <= N.

    Complexity: O(N (log N)^{d-1}) indices instead of O(N^d).

    Args:
        dim: Spatial dimension d.
        max_level: Maximum product level N.

    Returns:
        Array of shape (n_indices, dim) with hyperbolic cross indices.

    Example:
        >>> indices = hyperbolic_cross_indices(dim=2, max_level=8)
        >>> print(f"Number of indices: {len(indices)}")
        Number of indices: 20

        >>> # Compare to full tensor
        >>> full = full_tensor_indices(dim=2, n_modes=8)
        >>> print(f"Full tensor: {len(full)}, Compression: {len(full)/len(indices):.1f}x")
        Full tensor: 64, Compression: 3.2x
    """
    if dim == 1:
        return np.arange(1, max_level + 1).reshape(-1, 1)

    indices: List[Tuple[int, ...]] = []
    _generate_hyperbolic_recursive(dim, max_level, (), 1, indices)
    return np.array(indices, dtype=np.int64)


def _generate_hyperbolic_recursive(
    dim: int,
    max_level: int,
    current: Tuple[int, ...],
    current_product: int,
    result: List[Tuple[int, ...]],
) -> None:
    """Recursively generate hyperbolic cross indices.

    Args:
        dim: Total dimension.
        max_level: Maximum product constraint.
        current: Current partial index tuple.
        current_product: Product of indices so far.
        result: List to append complete indices to.
    """
    depth = len(current)

    if depth == dim:
        result.append(current)
        return

    # Maximum value for this dimension given product constraint
    max_k = max_level // current_product

    for k in range(1, max_k + 1):
        new_product = current_product * k
        if new_product <= max_level:
            _generate_hyperbolic_recursive(
                dim, max_level, current + (k,), new_product, result
            )


def smolyak_indices(dim: int, level: int) -> NDArray[np.int64]:
    """Generate Smolyak sparse grid indices using sum constraint.

    The Smolyak index set S_L^d contains all (k1, ..., kd) with ki >= 1
    satisfying sum(ki) <= L + d - 1.

    This is equivalent to choosing indices where the sum of "levels"
    is bounded, useful for combining univariate quadrature rules.

    Args:
        dim: Spatial dimension d.
        level: Smolyak level L.

    Returns:
        Array of shape (n_indices, dim) with Smolyak indices.

    Example:
        >>> indices = smolyak_indices(dim=3, level=4)
        >>> print(f"Number of indices: {len(indices)}")
    """
    if dim == 1:
        return np.arange(1, level + 1).reshape(-1, 1)

    max_sum = level + dim - 1
    indices: List[Tuple[int, ...]] = []
    _generate_smolyak_recursive(dim, max_sum, (), 0, indices)
    return np.array(indices, dtype=np.int64)


def _generate_smolyak_recursive(
    dim: int,
    max_sum: int,
    current: Tuple[int, ...],
    current_sum: int,
    result: List[Tuple[int, ...]],
) -> None:
    """Recursively generate Smolyak indices.

    Args:
        dim: Total dimension.
        max_sum: Maximum sum constraint.
        current: Current partial index tuple.
        current_sum: Sum of indices so far.
        result: List to append complete indices to.
    """
    depth = len(current)

    if depth == dim:
        result.append(current)
        return

    remaining_dims = dim - depth
    # Reserve at least 1 for each remaining dimension
    max_k = max_sum - current_sum - (remaining_dims - 1)

    for k in range(1, max_k + 1):
        _generate_smolyak_recursive(
            dim, max_sum, current + (k,), current_sum + k, result
        )


def hyperbolic_cross_log_indices(
    dim: int, max_log_sum: float
) -> NDArray[np.int64]:
    """Generate hyperbolic cross indices using log-sum constraint.

    Contains all (k1, ..., kd) with ki >= 1 satisfying:
        sum(log(1 + ki)) <= max_log_sum

    This variant is useful for Sobolev regularity analysis.

    Args:
        dim: Spatial dimension d.
        max_log_sum: Maximum sum of log(1 + ki).

    Returns:
        Array of shape (n_indices, dim) with indices.
    """
    if dim == 1:
        max_k = int(np.exp(max_log_sum) - 1)
        return np.arange(1, max_k + 1).reshape(-1, 1)

    indices: List[Tuple[int, ...]] = []
    _generate_log_hyperbolic_recursive(dim, max_log_sum, (), 0.0, indices)
    return np.array(indices, dtype=np.int64)


def _generate_log_hyperbolic_recursive(
    dim: int,
    max_log_sum: float,
    current: Tuple[int, ...],
    current_log_sum: float,
    result: List[Tuple[int, ...]],
) -> None:
    """Recursively generate log-hyperbolic cross indices."""
    depth = len(current)

    if depth == dim:
        result.append(current)
        return

    remaining = max_log_sum - current_log_sum
    max_k = int(np.exp(remaining) - 1)

    for k in range(1, max(2, max_k + 1)):
        log_k = np.log(1 + k)
        if current_log_sum + log_k <= max_log_sum + 1e-10:
            _generate_log_hyperbolic_recursive(
                dim, max_log_sum, current + (k,), current_log_sum + log_k, result
            )


def count_hyperbolic_cross(dim: int, max_level: int) -> int:
    """Count indices in hyperbolic cross without generating them.

    Uses the formula: |H_N^d| = sum_{k=1}^{N} |H_{N/k}^{d-1}|

    Args:
        dim: Spatial dimension.
        max_level: Maximum product level.

    Returns:
        Number of indices in the hyperbolic cross.
    """
    if dim == 1:
        return max_level

    count = 0
    for k in range(1, max_level + 1):
        count += count_hyperbolic_cross(dim - 1, max_level // k)
    return count


def asymptotic_size(dim: int, n: int) -> float:
    """Estimate asymptotic size of hyperbolic cross.

    The hyperbolic cross H_N^d has asymptotic size:
        |H_N^d| ~ (2/d!) * N * (log N)^{d-1}

    Args:
        dim: Spatial dimension.
        n: Maximum level.

    Returns:
        Asymptotic estimate of index set size.
    """
    from math import factorial, log

    if n <= 1:
        return 1.0
    return (2.0 / factorial(dim)) * n * (log(n) ** (dim - 1))
