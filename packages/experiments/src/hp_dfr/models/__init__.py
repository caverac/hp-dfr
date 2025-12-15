"""Neural network models for PINNs and DFR methods."""

from hp_dfr.models.base import BaseModel
from hp_dfr.models.pinns import PINNsModel
from hp_dfr.models.dfr import DFRModel
from hp_dfr.models.sparse_dfr import SparseDFRModel
from hp_dfr.models.hp_dfr import HPDFRModel
from hp_dfr.models.goal_oriented_dfr import (
    GoalOrientedDFRModel,
    QuantityOfInterest,
    PointEvaluationQoI,
    AverageValueQoI,
    BoundaryFluxQoI,
)

__all__ = [
    "BaseModel",
    "PINNsModel",
    "DFRModel",
    "SparseDFRModel",
    "HPDFRModel",
    "GoalOrientedDFRModel",
    "QuantityOfInterest",
    "PointEvaluationQoI",
    "AverageValueQoI",
    "BoundaryFluxQoI",
]
