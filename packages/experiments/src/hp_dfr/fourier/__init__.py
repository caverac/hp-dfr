"""Fourier methods for DFR including sparse tensor approaches.

This module implements:
- Sparse index set generation (hyperbolic cross)
- Full and sparse DST/DCT transforms
- H^{-1} norm computation with sparse Fourier modes

The sparse methods reduce complexity from O(N^d) to O(N (log N)^{d-1})
for d-dimensional problems, addressing the curse of dimensionality.
"""

from hp_dfr.fourier.sparse_indices import (
    HyperbolicCrossIndexSet,
    full_tensor_indices,
    hyperbolic_cross_indices,
    smolyak_indices,
)
from hp_dfr.fourier.transforms import (
    dst_1d,
    dct_1d,
    dst_nd_full,
    dst_nd_sparse,
    dst_matrix_nd_sparse,
    h_minus_1_weights,
    h_minus_1_weights_sparse,
)

__all__ = [
    # Index sets
    "HyperbolicCrossIndexSet",
    "full_tensor_indices",
    "hyperbolic_cross_indices",
    "smolyak_indices",
    # Transforms
    "dst_1d",
    "dct_1d",
    "dst_nd_full",
    "dst_nd_sparse",
    "dst_matrix_nd_sparse",
    # Weights
    "h_minus_1_weights",
    "h_minus_1_weights_sparse",
]
