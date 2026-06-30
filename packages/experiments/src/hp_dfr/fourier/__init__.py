"""Fourier methods for the Deep Fourier Residual (DFR) loss.

This module implements:
- Full tensor-product DST/DCT transforms (1D and nD).
- H^{-1} norm weights for the dual-norm residual loss.

The DFR loss is the H^{-1} (dual) norm of the PDE residual, computed by
projecting the residual onto a sine/cosine spectral basis via the Discrete
Sine/Cosine Transform.
"""

from hp_dfr.fourier.transforms import (
    compute_h_minus_1_norm,
    dct_1d,
    dct_derivative_matrix_1d,
    dfr_loss_1d,
    dst_1d,
    dst_matrix_1d,
    dst_matrix_nd_full,
    dst_nd_full,
    h_minus_1_weights,
)

__all__ = [
    # Transforms
    "dst_1d",
    "dct_1d",
    "dst_matrix_1d",
    "dct_derivative_matrix_1d",
    "dst_nd_full",
    "dst_matrix_nd_full",
    # Weights
    "h_minus_1_weights",
    "compute_h_minus_1_norm",
    # Loss
    "dfr_loss_1d",
]
