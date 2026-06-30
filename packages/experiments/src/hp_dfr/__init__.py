"""Physics-Informed Neural Networks and Deep Fourier Residual methods for PDEs.

A comprehensive implementation of PINNs and DFR methods for solving PDEs,
with support for TensorFlow, JAX, and PyTorch backends.

Reference:
    Taylor, Pardo, Muga (2023). "A Deep Fourier Residual Method for solving
    PDEs using Neural Networks". Computer Methods in Applied Mechanics and
    Engineering, 405, 115850.
    https://arxiv.org/abs/2210.14129
"""

__version__ = "0.1.0"
__author__ = "Carlos Vera-Ciro"

import matplotlib

from hp_dfr.models import DFRModel, PINNsModel
from hp_dfr.problems import Poisson1D

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"


__all__ = [
    "PINNsModel",
    "DFRModel",
    "Poisson1D",
    "__version__",
]
