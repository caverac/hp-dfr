"""Unit tests for goal-oriented DFR methods.

Tests cover:
1. Quantity of Interest (QoI) classes
2. Goal-Oriented DFR model structure
3. Adjoint network behavior
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hp_dfr.models.goal_oriented_dfr import (
    QuantityOfInterest,
    PointEvaluationQoI,
    AverageValueQoI,
    BoundaryFluxQoI,
    GoalOrientedDFRModel,
)


# =============================================================================
# PointEvaluationQoI Tests
# =============================================================================


class TestPointEvaluationQoI:
    """Tests for PointEvaluationQoI class."""

    def test_creation_1d(self):
        """Test 1D point evaluation QoI."""
        qoi = PointEvaluationQoI(point=np.array([0.5]))
        assert_allclose(qoi.point, [0.5])
        assert qoi.sigma == 0.01

    def test_creation_2d(self):
        """Test 2D point evaluation QoI."""
        qoi = PointEvaluationQoI(point=np.array([0.5, 0.5]), sigma=0.02)
        assert_allclose(qoi.point, [0.5, 0.5])
        assert qoi.sigma == 0.02

    def test_evaluate(self):
        """Test evaluation of QoI."""
        qoi = PointEvaluationQoI(point=np.array([0.5]))

        # Simple test function: u(x) = x^2
        def u_fn(x):
            return x[:, 0] ** 2

        result = qoi.evaluate(u_fn, domain=((0.0, 1.0),))
        assert_allclose(result, 0.25, rtol=1e-10)

    def test_evaluate_2d(self):
        """Test 2D evaluation."""
        qoi = PointEvaluationQoI(point=np.array([0.5, 0.3]))

        # u(x, y) = x + y
        def u_fn(x):
            return x[:, 0] + x[:, 1]

        result = qoi.evaluate(u_fn, domain=((0.0, 1.0), (0.0, 1.0)))
        assert_allclose(result, 0.8, rtol=1e-10)

    def test_adjoint_rhs_peak_at_point(self):
        """Test adjoint RHS peaks at the evaluation point."""
        point = np.array([0.5, 0.5])
        qoi = PointEvaluationQoI(point=point, sigma=0.01)

        # Evaluate at the point and nearby
        x = np.array([
            [0.5, 0.5],    # At point
            [0.6, 0.5],    # Nearby
            [0.9, 0.9],    # Far away
        ])

        rhs = qoi.adjoint_rhs(x)

        # Peak should be at the evaluation point
        assert rhs[0] > rhs[1]
        assert rhs[1] > rhs[2]

    def test_adjoint_rhs_narrow_gaussian(self):
        """Test adjoint RHS has proper Gaussian shape."""
        point = np.array([0.0])
        qoi = PointEvaluationQoI(point=point, sigma=0.1)

        # Points along a line
        x = np.linspace(-0.5, 0.5, 101).reshape(-1, 1)
        rhs = qoi.adjoint_rhs(x)

        # Maximum at center
        max_idx = np.argmax(rhs)
        assert max_idx == 50  # Middle point

        # Symmetric
        assert_allclose(rhs[:50], rhs[100:50:-1], rtol=1e-10)


# =============================================================================
# AverageValueQoI Tests
# =============================================================================


class TestAverageValueQoI:
    """Tests for AverageValueQoI class."""

    def test_creation_full_domain(self):
        """Test creation for full domain."""
        qoi = AverageValueQoI()
        assert qoi.subdomain is None

    def test_creation_subdomain(self):
        """Test creation for subdomain."""
        subdomain = ((0.25, 0.75), (0.25, 0.75))
        qoi = AverageValueQoI(subdomain=subdomain)
        assert qoi.subdomain == subdomain

    def test_evaluate_constant(self):
        """Test evaluation with constant function."""
        qoi = AverageValueQoI()

        def u_fn(x):
            return np.ones(x.shape[0]) * 5.0

        result = qoi.evaluate(u_fn, domain=((0.0, 1.0), (0.0, 1.0)))
        assert_allclose(result, 5.0, rtol=1e-10)

    def test_evaluate_linear(self):
        """Test evaluation with linear function."""
        qoi = AverageValueQoI()

        # u(x, y) = x
        def u_fn(x):
            return x[:, 0]

        # Average of x over [0,1]x[0,1] is 0.5
        result = qoi.evaluate(u_fn, domain=((0.0, 1.0), (0.0, 1.0)))
        assert_allclose(result, 0.5, rtol=0.05)  # Quadrature error

    def test_evaluate_subdomain(self):
        """Test evaluation over subdomain."""
        subdomain = ((0.5, 1.0), (0.0, 1.0))
        qoi = AverageValueQoI(subdomain=subdomain)

        # u(x, y) = x
        def u_fn(x):
            return x[:, 0]

        # Average of x over [0.5,1]x[0,1] is 0.75
        result = qoi.evaluate(u_fn, domain=((0.0, 1.0), (0.0, 1.0)))
        assert_allclose(result, 0.75, rtol=0.05)

    def test_adjoint_rhs_full_domain(self):
        """Test adjoint RHS for full domain is constant."""
        qoi = AverageValueQoI()

        x = np.array([
            [0.1, 0.2],
            [0.5, 0.5],
            [0.9, 0.8],
        ])

        rhs = qoi.adjoint_rhs(x)
        # All ones for full domain
        assert_allclose(rhs, np.ones(3))

    def test_adjoint_rhs_subdomain(self):
        """Test adjoint RHS for subdomain is indicator."""
        subdomain = ((0.5, 1.0), (0.0, 1.0))
        qoi = AverageValueQoI(subdomain=subdomain)

        x = np.array([
            [0.25, 0.5],  # Outside
            [0.75, 0.5],  # Inside
        ])

        rhs = qoi.adjoint_rhs(x)
        assert rhs[0] == 0.0  # Outside
        assert rhs[1] > 0.0   # Inside


# =============================================================================
# BoundaryFluxQoI Tests
# =============================================================================


class TestBoundaryFluxQoI:
    """Tests for BoundaryFluxQoI class."""

    def test_creation(self):
        """Test creation of boundary flux QoI."""
        start = np.array([0.0, 0.0])
        end = np.array([0.0, 1.0])
        normal = np.array([1.0, 0.0])

        qoi = BoundaryFluxQoI(
            boundary_segment=(start, end),
            normal=normal,
        )

        assert_allclose(qoi.start, start)
        assert_allclose(qoi.end, end)
        assert_allclose(qoi.normal, [1.0, 0.0])

    def test_normal_normalization(self):
        """Test that normal is normalized."""
        qoi = BoundaryFluxQoI(
            boundary_segment=(np.array([0.0, 0.0]), np.array([1.0, 0.0])),
            normal=np.array([3.0, 4.0]),  # Not unit
        )

        # Should be normalized
        assert_allclose(np.linalg.norm(qoi.normal), 1.0)
        assert_allclose(qoi.normal, [0.6, 0.8])

    def test_evaluate_constant_gradient(self):
        """Test flux with constant gradient."""
        # Left boundary x=0, normal pointing right
        qoi = BoundaryFluxQoI(
            boundary_segment=(np.array([0.0, 0.0]), np.array([0.0, 1.0])),
            normal=np.array([1.0, 0.0]),
        )

        # u(x, y) = x -> du/dn = 1
        def u_fn(x):
            return x[:, 0]

        result = qoi.evaluate(u_fn, domain=((0.0, 1.0), (0.0, 1.0)))
        # Flux = integral of 1 over segment of length 1 = 1
        assert_allclose(result, 1.0, rtol=0.1)


# =============================================================================
# GoalOrientedDFRModel Tests
# =============================================================================


class TestGoalOrientedDFRModel:
    """Tests for GoalOrientedDFRModel class."""

    def test_creation_default(self):
        """Test default model creation."""
        model = GoalOrientedDFRModel()

        assert model.dim == 2
        assert model.hidden_layers == [20, 20, 20]
        assert model.qoi is None
        assert model.adjoint_weight == 1.0
        assert model.primal_weight == 0.1

    def test_creation_with_qoi(self):
        """Test creation with QoI."""
        qoi = PointEvaluationQoI(point=np.array([0.5, 0.5]))
        model = GoalOrientedDFRModel(dim=2, qoi=qoi)

        assert model.qoi is qoi

    def test_creation_custom_params(self):
        """Test creation with custom parameters."""
        model = GoalOrientedDFRModel(
            dim=3,
            hidden_layers=[32, 32],
            activation="relu",
            n_quadrature=16,
            max_level=8,
            use_sparse=False,
            adjoint_weight=2.0,
            primal_weight=0.5,
        )

        assert model.dim == 3
        assert model.hidden_layers == [32, 32]
        assert model.activation == "relu"
        assert model.n_quadrature == 16
        assert model.max_level == 8
        assert model.use_sparse is False
        assert model.adjoint_weight == 2.0
        assert model.primal_weight == 0.5

    def test_build_tensorflow(self):
        """Test building TensorFlow models."""
        pytest.importorskip("tensorflow")

        model = GoalOrientedDFRModel(
            dim=2,
            hidden_layers=[10, 10],
            backend="tensorflow",
        )
        model.build()

        assert model._primal_model is not None
        assert model._adjoint_model is not None

    def test_build_pytorch(self):
        """Test building PyTorch models."""
        pytest.importorskip("torch")

        model = GoalOrientedDFRModel(
            dim=2,
            hidden_layers=[10, 10],
            backend="pytorch",
        )
        model.build()

        assert model._primal_model is not None
        assert model._adjoint_model is not None

    def test_invalid_backend(self):
        """Test invalid backend raises error."""
        model = GoalOrientedDFRModel(backend="invalid")

        with pytest.raises(ValueError, match="not yet implemented"):
            model.build()

    def test_summary(self, capsys):
        """Test summary output."""
        qoi = PointEvaluationQoI(point=np.array([0.5, 0.5]))
        model = GoalOrientedDFRModel(dim=2, qoi=qoi)
        model.summary()

        captured = capsys.readouterr()
        assert "Goal-Oriented DFR" in captured.out
        assert "Dimension: 2" in captured.out
        assert "PointEvaluationQoI" in captured.out


# =============================================================================
# Integration Tests
# =============================================================================


class TestGoalOrientedIntegration:
    """Integration tests for goal-oriented components."""

    def test_qoi_with_model_tensorflow(self):
        """Test QoI evaluation with TensorFlow model."""
        pytest.importorskip("tensorflow")

        qoi = PointEvaluationQoI(point=np.array([0.5, 0.5]))
        model = GoalOrientedDFRModel(
            dim=2,
            qoi=qoi,
            hidden_layers=[10],
            backend="tensorflow",
        )
        model.build()

        # Model should be ready to train
        assert model._primal_model is not None
        assert model._adjoint_model is not None

    def test_qoi_with_model_pytorch(self):
        """Test QoI evaluation with PyTorch model."""
        pytest.importorskip("torch")

        qoi = AverageValueQoI(subdomain=((0.25, 0.75), (0.25, 0.75)))
        model = GoalOrientedDFRModel(
            dim=2,
            qoi=qoi,
            hidden_layers=[10],
            backend="pytorch",
        )
        model.build()

        assert model._primal_model is not None
        assert model._adjoint_model is not None

    def test_multiple_qoi_types(self):
        """Test that all QoI types can be created."""
        qois = [
            PointEvaluationQoI(point=np.array([0.5, 0.5])),
            AverageValueQoI(),
            AverageValueQoI(subdomain=((0.0, 0.5), (0.0, 0.5))),
            BoundaryFluxQoI(
                boundary_segment=(np.array([0.0, 0.0]), np.array([0.0, 1.0])),
                normal=np.array([1.0, 0.0]),
            ),
        ]

        for qoi in qois:
            assert isinstance(qoi, QuantityOfInterest)

            # All should have required methods
            x_test = np.array([[0.5, 0.5], [0.3, 0.7]])
            rhs = qoi.adjoint_rhs(x_test)
            assert rhs.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
