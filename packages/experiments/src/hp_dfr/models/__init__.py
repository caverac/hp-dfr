"""Neural network models for PINNs and DFR methods."""

from hp_dfr.models.base import BaseModel
from hp_dfr.models.dfr import DFRModel
from hp_dfr.models.goal_oriented_dfr import (
    AverageValueQoI,
    BoundaryFluxQoI,
    GoalOrientedDFRModel,
    PointEvaluationQoI,
    QuantityOfInterest,
)
from hp_dfr.models.pinns import PINNsModel

__all__ = [
    "BaseModel",
    "PINNsModel",
    "DFRModel",
    "GoalOrientedDFRModel",
    "QuantityOfInterest",
    "PointEvaluationQoI",
    "AverageValueQoI",
    "BoundaryFluxQoI",
]
