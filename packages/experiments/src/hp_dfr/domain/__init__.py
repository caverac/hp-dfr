"""Domain decomposition for hp-adaptive DFR methods.

This module provides infrastructure for:
- Domain partitioning into subdomains
- Local neural networks per subdomain
- Interface coupling between subdomains
- Residual-based refinement indicators
- Automatic h-refinement strategies

The hp-adaptive approach enables:
- h-refinement: subdivide domains in high-error regions
- p-refinement: increase network capacity in smooth regions
- Local error control with domain-specific networks

References:
    - Taylor et al. (2024). Adaptive Deep Fourier Residual method.
    - Kharazmi et al. (2021). hp-VPINNs.
"""

from hp_dfr.domain.partitioning import (
    DomainPartition,
    RectangularPartition,
    AdaptivePartition,
)
from hp_dfr.domain.subdomain import (
    Subdomain,
    SubdomainNetwork,
)
from hp_dfr.domain.interface import (
    InterfaceCoupling,
    MortarCoupling,
    PenaltyCoupling,
)
from hp_dfr.domain.refinement import (
    RefinementIndicator,
    ResidualIndicator,
    GradientIndicator,
    mark_for_refinement,
)

__all__ = [
    # Partitioning
    "DomainPartition",
    "RectangularPartition",
    "AdaptivePartition",
    # Subdomains
    "Subdomain",
    "SubdomainNetwork",
    # Interface coupling
    "InterfaceCoupling",
    "MortarCoupling",
    "PenaltyCoupling",
    # Refinement
    "RefinementIndicator",
    "ResidualIndicator",
    "GradientIndicator",
    "mark_for_refinement",
]
