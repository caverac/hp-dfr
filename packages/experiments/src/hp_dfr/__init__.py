"""
DFR-PINNs: Physics-Informed Neural Networks and Deep Fourier Residual Methods

A comprehensive implementation of PINNs and DFR methods for solving PDEs,
with support for TensorFlow, JAX, and PyTorch backends.

Reference:
    Taylor, Pardo, Muga (2023). "A Deep Fourier Residual Method for solving
    PDEs using Neural Networks". Computer Methods in Applied Mechanics and
    Engineering, 405, 115850.
    https://arxiv.org/abs/2210.14129
"""

__version__ = "0.1.0"
__author__ = "MATHMODE Group"

from hp_dfr.models import PINNsModel, DFRModel
from hp_dfr.problems import Poisson1D

__all__ = [
    "PINNsModel",
    "DFRModel",
    "Poisson1D",
    "__version__",
]
