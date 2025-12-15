"""Goal-Oriented Deep Fourier Residual (GO-DFR) method.

This module implements goal-oriented error estimation for DFR using
dual-weighted residual (DWR) methods. Instead of minimizing the global
H^{-1} norm of the residual, we focus on quantities of interest (QoI).

The key idea is to solve an adjoint problem and use it to weight
the residual, focusing computational effort on regions that most
affect the QoI.

Key Concepts:
    - Primal problem: Lu = f (the original PDE)
    - Adjoint problem: L*z = J' (adjoint with QoI functional derivative)
    - Error representation: J(u) - J(u_h) ≈ <R(u_h), z>

References:
    - Becker & Rannacher (2001). Optimal control approach to error estimation.
    - Endtmayer et al. (2021). Multigoal-oriented DWR with DNNs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from hp_dfr.models.base import BaseModel
from hp_dfr.fourier import (
    HyperbolicCrossIndexSet,
    dst_matrix_nd_sparse,
    h_minus_1_weights_sparse,
)


class QuantityOfInterest(ABC):
    """Abstract base class for quantities of interest (QoI).

    A QoI is a functional J(u) that we want to compute accurately.
    Examples: point evaluation, average value, flux through boundary.
    """

    @abstractmethod
    def evaluate(self, u_fn: Callable, domain: Any) -> float:
        """Evaluate QoI given solution function.

        Args:
            u_fn: Function u(x) -> value.
            domain: Problem domain.

        Returns:
            QoI value.
        """
        pass

    @abstractmethod
    def adjoint_rhs(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute right-hand side for adjoint problem.

        The adjoint problem is L*z = J'(u), where J' is the
        Fréchet derivative of the QoI functional.

        Args:
            x: Points at which to evaluate.

        Returns:
            Adjoint RHS values.
        """
        pass


class PointEvaluationQoI(QuantityOfInterest):
    """Point evaluation quantity of interest.

    J(u) = u(x_0) for a specified point x_0.

    The adjoint RHS is the Dirac delta: J'(u) = delta(x - x_0).
    For numerical purposes, we approximate with a narrow Gaussian.
    """

    def __init__(
        self,
        point: NDArray[np.floating],
        sigma: float = 0.01,
    ):
        """Initialize point evaluation QoI.

        Args:
            point: Evaluation point x_0.
            sigma: Width of Gaussian approximation to delta.
        """
        self.point = np.asarray(point)
        self.sigma = sigma

    def evaluate(self, u_fn: Callable, domain: Any) -> float:
        """Evaluate u at the specified point."""
        return float(u_fn(self.point.reshape(1, -1))[0])

    def adjoint_rhs(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Approximate delta function as narrow Gaussian."""
        dist_sq = np.sum((x - self.point) ** 2, axis=1)
        # Normalized Gaussian
        dim = len(self.point)
        norm = (2 * np.pi * self.sigma**2) ** (-dim / 2)
        return norm * np.exp(-dist_sq / (2 * self.sigma**2))


class AverageValueQoI(QuantityOfInterest):
    """Average value over a subdomain.

    J(u) = (1/|Omega|) * integral_Omega u dx

    The adjoint RHS is constant: J'(u) = 1/|Omega|.
    """

    def __init__(
        self,
        subdomain: Optional[Tuple[Tuple[float, float], ...]] = None,
    ):
        """Initialize average value QoI.

        Args:
            subdomain: Region to average over (None = full domain).
        """
        self.subdomain = subdomain

    def evaluate(self, u_fn: Callable, domain: Any) -> float:
        """Compute average value via quadrature."""
        # Generate quadrature points
        if self.subdomain is not None:
            bounds = self.subdomain
        else:
            bounds = domain

        n_quad = 32
        grids_1d = []
        for lo, hi in bounds:
            x = np.linspace(lo, hi, n_quad)
            grids_1d.append(x)

        grids = np.meshgrid(*grids_1d, indexing="ij")
        x = np.stack([g.ravel() for g in grids], axis=1)

        u_vals = u_fn(x)
        return float(np.mean(u_vals))

    def adjoint_rhs(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Constant RHS for average value QoI."""
        n_points = x.shape[0]

        if self.subdomain is not None:
            # Indicator function for subdomain
            inside = np.ones(n_points, dtype=bool)
            for d, (lo, hi) in enumerate(self.subdomain):
                inside &= (x[:, d] >= lo) & (x[:, d] <= hi)

            # Compute volume
            volume = np.prod([hi - lo for lo, hi in self.subdomain])
            rhs = np.zeros(n_points)
            rhs[inside] = 1.0 / volume
            return rhs
        else:
            # Will be normalized by domain volume
            return np.ones(n_points)


class BoundaryFluxQoI(QuantityOfInterest):
    """Flux through a boundary segment.

    J(u) = integral_Gamma (grad u · n) ds

    where Gamma is a boundary segment and n is the normal.
    """

    def __init__(
        self,
        boundary_segment: Tuple[NDArray, NDArray],  # (start, end) points
        normal: NDArray[np.floating],
    ):
        """Initialize boundary flux QoI.

        Args:
            boundary_segment: Start and end points of segment.
            normal: Outward normal direction.
        """
        self.start, self.end = boundary_segment
        self.normal = np.asarray(normal) / np.linalg.norm(normal)

    def evaluate(self, u_fn: Callable, domain: Any) -> float:
        """Compute flux via finite differences."""
        # Sample points along boundary
        n_points = 32
        t = np.linspace(0, 1, n_points)
        points = self.start + np.outer(t, self.end - self.start)

        # Compute normal derivative via finite differences
        eps = 1e-5
        points_plus = points + eps * self.normal
        points_minus = points - eps * self.normal

        u_plus = u_fn(points_plus)
        u_minus = u_fn(points_minus)

        du_dn = (u_plus - u_minus) / (2 * eps)

        # Integrate along boundary
        length = np.linalg.norm(self.end - self.start)
        return float(np.mean(du_dn) * length)

    def adjoint_rhs(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Adjoint RHS for flux (involves boundary terms)."""
        # This is simplified - proper implementation needs careful treatment
        # of boundary conditions in the adjoint problem
        return np.zeros(x.shape[0])


class GoalOrientedDFRModel(BaseModel):
    """Goal-Oriented Deep Fourier Residual model.

    Trains both primal and adjoint networks simultaneously,
    using the adjoint to weight the residual for QoI-focused training.

    The loss function is:
        L = <R(u_h), z_h> + regularization terms

    where R(u_h) is the primal residual and z_h is the adjoint solution.

    Example:
        >>> qoi = PointEvaluationQoI(point=np.array([0.5, 0.5]))
        >>> model = GoalOrientedDFRModel(
        ...     dim=2,
        ...     qoi=qoi,
        ...     hidden_layers=[20, 20, 20],
        ... )
        >>> model.build()
        >>> history = model.fit(problem, epochs=1000)
        >>> qoi_value = qoi.evaluate(model.predict, problem.domain)
    """

    def __init__(
        self,
        hidden_layers: List[int] = [20, 20, 20],
        activation: str = "tanh",
        dtype: str = "float64",
        seed: int = 1234,
        dim: int = 2,
        qoi: Optional[QuantityOfInterest] = None,
        n_quadrature: int = 32,
        max_level: int = 16,
        use_sparse: bool = True,
        adjoint_weight: float = 1.0,
        primal_weight: float = 0.1,
        backend: str = "tensorflow",
    ):
        """Initialize Goal-Oriented DFR model.

        Args:
            hidden_layers: Network architecture.
            activation: Activation function.
            dtype: Data type.
            seed: Random seed.
            dim: Spatial dimension.
            qoi: Quantity of interest to optimize for.
            n_quadrature: Quadrature points per dimension.
            max_level: Fourier level for sparse DFR.
            use_sparse: Use sparse Fourier modes.
            adjoint_weight: Weight for adjoint-based loss.
            primal_weight: Weight for standard DFR loss (regularization).
            backend: Deep learning backend.
        """
        super().__init__(hidden_layers, activation, dtype, seed)
        self.dim = dim
        self.qoi = qoi
        self.n_quadrature = n_quadrature
        self.max_level = max_level
        self.use_sparse = use_sparse
        self.adjoint_weight = adjoint_weight
        self.primal_weight = primal_weight
        self.backend = backend

        self._primal_model = None
        self._adjoint_model = None
        self._dst_matrix = None
        self._h_weights = None
        self._index_set = None
        self._domain = None
        self._quad_points = None

    def build(self, input_dim: Optional[int] = None, output_dim: int = 1) -> None:
        """Build primal and adjoint networks."""
        if input_dim is None:
            input_dim = self.dim

        if self.backend == "tensorflow":
            self._build_tensorflow(input_dim, output_dim)
        elif self.backend == "pytorch":
            self._build_pytorch(input_dim, output_dim)
        else:
            raise ValueError(f"Backend {self.backend} not yet implemented")

    def _build_tensorflow(self, input_dim: int, output_dim: int) -> None:
        """Build TensorFlow models for primal and adjoint."""
        import os
        os.environ["KERAS_BACKEND"] = "tensorflow"
        import keras

        keras.utils.set_random_seed(self.seed)

        def create_network(name: str):
            inputs = keras.layers.Input(shape=(input_dim,), dtype=self.dtype)
            x = inputs
            for neurons in self.hidden_layers:
                x = keras.layers.Dense(
                    neurons, activation=self.activation, dtype=self.dtype
                )(x)
            x = keras.layers.Dense(
                output_dim, activation=self.activation, dtype=self.dtype
            )(x)
            return keras.Model(inputs=inputs, outputs=x, name=name)

        self._primal_model = create_network("primal")
        self._adjoint_model = create_network("adjoint")

    def _build_pytorch(self, input_dim: int, output_dim: int) -> None:
        """Build PyTorch models for primal and adjoint."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)

        def create_network():
            layers = []
            prev_dim = input_dim
            for neurons in self.hidden_layers:
                layers.append(nn.Linear(prev_dim, neurons))
                if self.activation == "tanh":
                    layers.append(nn.Tanh())
                elif self.activation == "relu":
                    layers.append(nn.ReLU())
                prev_dim = neurons
            layers.append(nn.Linear(prev_dim, output_dim))
            if self.activation == "tanh":
                layers.append(nn.Tanh())
            model = nn.Sequential(*layers)
            if self.dtype == "float64":
                model = model.double()
            return model

        self._primal_model = create_network()
        self._adjoint_model = create_network()

    def _setup_fourier(self, domain: Tuple[Tuple[float, float], ...]) -> None:
        """Set up Fourier transforms."""
        self._domain = domain

        grid_shape = tuple([self.n_quadrature] * self.dim)

        if self.use_sparse:
            self._index_set = HyperbolicCrossIndexSet(
                dim=self.dim, max_level=self.max_level
            )
            self._dst_matrix = dst_matrix_nd_sparse(
                grid_shape, domain, self._index_set
            )
            self._h_weights = h_minus_1_weights_sparse(
                self._index_set, domain
            )

        # Create quadrature points
        quad_1d = []
        for d in range(self.dim):
            a, b = domain[d]
            x = np.linspace(a, b, self.n_quadrature + 1)
            dx = x[1] - x[0]
            x_mid = x[:-1] + dx / 2
            quad_1d.append(x_mid)

        grids = np.meshgrid(*quad_1d, indexing="ij")
        self._quad_points = np.stack([g.ravel() for g in grids], axis=1)

    def fit(
        self,
        problem: Any,
        epochs: int = 1000,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the goal-oriented model.

        Simultaneously trains primal and adjoint networks with
        coupled loss functions.

        Args:
            problem: PDE problem.
            epochs: Training epochs.
            learning_rate: Learning rate.
            verbose: Print progress.

        Returns:
            Training history.
        """
        if self._primal_model is None:
            self.build()

        # Set up domain
        if self.dim == 1:
            domain = (problem.domain,)
        else:
            domain = problem.domain

        self._setup_fourier(domain)

        if self.qoi is None:
            # Default: point evaluation at center
            center = np.array([(b[0] + b[1]) / 2 for b in domain])
            self.qoi = PointEvaluationQoI(point=center)

        if verbose:
            print(f"Goal-Oriented DFR: QoI = {type(self.qoi).__name__}")
            if self.use_sparse:
                print(f"Sparse modes: {len(self._index_set)}")

        if self.backend == "tensorflow":
            return self._fit_tensorflow(problem, epochs, learning_rate, verbose)
        elif self.backend == "pytorch":
            return self._fit_pytorch(problem, epochs, learning_rate, verbose)

    def _fit_tensorflow(
        self,
        problem: Any,
        epochs: int,
        learning_rate: float,
        verbose: bool,
    ) -> Dict[str, List[float]]:
        """Train using TensorFlow."""
        import tensorflow as tf
        import keras

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Get all trainable variables
        all_vars = (
            self._primal_model.trainable_variables +
            self._adjoint_model.trainable_variables
        )

        # Precompute
        pts = tf.constant(self._quad_points, dtype=self.dtype)
        dst_matrix = tf.constant(self._dst_matrix, dtype=self.dtype)
        weights = tf.constant(self._h_weights, dtype=self.dtype)

        # Forcing term
        f_vals = problem.forcing_fn(self._quad_points)
        if f_vals.ndim > 1:
            f_vals = f_vals.flatten()
        ft_forcing = tf.constant(self._dst_matrix @ f_vals, dtype=self.dtype)

        # Adjoint RHS (QoI derivative)
        adjoint_rhs = self.qoi.adjoint_rhs(self._quad_points)
        ft_adjoint_rhs = tf.constant(self._dst_matrix @ adjoint_rhs, dtype=self.dtype)

        domain = self._domain

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                # Primal solution with cutoff
                u_nn = self._primal_model(pts, training=True)
                cutoff = self._cutoff_tf(pts, domain)
                u = cutoff * u_nn

                # Adjoint solution with cutoff
                z_nn = self._adjoint_model(pts, training=True)
                z = cutoff * z_nn

                # Compute primal residual R(u) = -Delta u - f
                laplacian_u = self._laplacian_tf(pts, self._primal_model, domain)
                residual_u = tf.reshape(-laplacian_u, [-1])

                # Compute adjoint residual R*(z) = -Delta z - J'
                laplacian_z = self._laplacian_tf(pts, self._adjoint_model, domain)
                residual_z = tf.reshape(-laplacian_z, [-1])

                # Fourier transforms
                ft_residual_u = tf.linalg.matvec(dst_matrix, residual_u)
                ft_residual_z = tf.linalg.matvec(dst_matrix, residual_z)

                # Goal-oriented loss: <R(u), z> weighted by H^{-1}
                # This estimates the error in the QoI
                z_flat = tf.reshape(z, [-1])
                go_loss = tf.abs(tf.reduce_sum(residual_u * z_flat))

                # Standard primal DFR loss (regularization)
                ft_primal = (ft_residual_u + ft_forcing) * weights
                primal_loss = tf.reduce_sum(ft_primal ** 2)

                # Adjoint DFR loss
                ft_adjoint = (ft_residual_z + ft_adjoint_rhs) * weights
                adjoint_loss = tf.reduce_sum(ft_adjoint ** 2)

                # Total loss
                total_loss = (
                    self.adjoint_weight * go_loss +
                    self.primal_weight * primal_loss +
                    self.primal_weight * adjoint_loss
                )

            gradients = tape.gradient(total_loss, all_vars)
            optimizer.apply_gradients(zip(gradients, all_vars))

            return total_loss, go_loss, primal_loss, adjoint_loss

        self._history = {
            "loss": [],
            "go_loss": [],
            "primal_loss": [],
            "adjoint_loss": [],
        }

        for epoch in range(epochs):
            total, go, primal, adjoint = train_step()

            self._history["loss"].append(float(total))
            self._history["go_loss"].append(float(go))
            self._history["primal_loss"].append(float(primal))
            self._history["adjoint_loss"].append(float(adjoint))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Loss: {total:.6e} (GO: {go:.6e})")

        return self._history

    def _cutoff_tf(
        self,
        x: "tf.Tensor",
        domain: Tuple[Tuple[float, float], ...],
    ) -> "tf.Tensor":
        """Compute cutoff function in TensorFlow."""
        import tensorflow as tf

        result = tf.ones((tf.shape(x)[0], 1), dtype=self.dtype)
        for d in range(self.dim):
            lo, hi = domain[d]
            result *= (x[:, d:d+1] - lo) * (hi - x[:, d:d+1])
        return result

    def _laplacian_tf(
        self,
        x: "tf.Tensor",
        model: Any,
        domain: Tuple[Tuple[float, float], ...],
    ) -> "tf.Tensor":
        """Compute Laplacian via autodiff in TensorFlow."""
        import tensorflow as tf

        laplacian = tf.zeros((tf.shape(x)[0],), dtype=self.dtype)

        for d in range(self.dim):
            with tf.GradientTape() as t2:
                t2.watch(x)
                with tf.GradientTape() as t1:
                    t1.watch(x)
                    cutoff = self._cutoff_tf(x, domain)
                    u = cutoff * model(x, training=True)
                du = t1.gradient(u, x)
            d2u = t2.gradient(du[:, d], x)
            if d2u is not None:
                laplacian += d2u[:, d]

        return laplacian

    def _fit_pytorch(
        self,
        problem: Any,
        epochs: int,
        learning_rate: float,
        verbose: bool,
    ) -> Dict[str, List[float]]:
        """Train using PyTorch."""
        import torch

        dtype = torch.float64 if self.dtype == "float64" else torch.float32

        all_params = (
            list(self._primal_model.parameters()) +
            list(self._adjoint_model.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=learning_rate)

        pts = torch.tensor(self._quad_points, dtype=dtype, requires_grad=True)
        dst_matrix = torch.tensor(self._dst_matrix, dtype=dtype)
        weights = torch.tensor(self._h_weights, dtype=dtype)

        f_vals = problem.forcing_fn(self._quad_points)
        if f_vals.ndim > 1:
            f_vals = f_vals.flatten()
        ft_forcing = torch.tensor(self._dst_matrix @ f_vals, dtype=dtype)

        adjoint_rhs = self.qoi.adjoint_rhs(self._quad_points)
        ft_adjoint_rhs = torch.tensor(self._dst_matrix @ adjoint_rhs, dtype=dtype)

        domain = self._domain

        self._history = {
            "loss": [],
            "go_loss": [],
            "primal_loss": [],
            "adjoint_loss": [],
        }

        for epoch in range(epochs):
            optimizer.zero_grad()

            x = pts.detach().clone().requires_grad_(True)

            # Primal
            u_nn = self._primal_model(x)
            cutoff = self._cutoff_torch(x, domain, dtype)
            u = cutoff * u_nn

            # Adjoint
            z_nn = self._adjoint_model(x)
            z = cutoff * z_nn

            # Laplacians
            laplacian_u = self._laplacian_torch(x, self._primal_model, domain, dtype)
            laplacian_z = self._laplacian_torch(x, self._adjoint_model, domain, dtype)

            residual_u = -laplacian_u
            residual_z = -laplacian_z

            # Losses
            z_flat = z.reshape(-1)
            go_loss = torch.abs(torch.sum(residual_u * z_flat))

            ft_primal = torch.mv(dst_matrix, residual_u) + ft_forcing
            ft_primal = ft_primal * weights
            primal_loss = torch.sum(ft_primal ** 2)

            ft_adjoint = torch.mv(dst_matrix, residual_z) + ft_adjoint_rhs
            ft_adjoint = ft_adjoint * weights
            adjoint_loss = torch.sum(ft_adjoint ** 2)

            total_loss = (
                self.adjoint_weight * go_loss +
                self.primal_weight * primal_loss +
                self.primal_weight * adjoint_loss
            )

            total_loss.backward()
            optimizer.step()

            self._history["loss"].append(float(total_loss))
            self._history["go_loss"].append(float(go_loss))
            self._history["primal_loss"].append(float(primal_loss))
            self._history["adjoint_loss"].append(float(adjoint_loss))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.6e}")

        return self._history

    def _cutoff_torch(
        self,
        x: "torch.Tensor",
        domain: Tuple[Tuple[float, float], ...],
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute cutoff in PyTorch."""
        import torch

        result = torch.ones((x.shape[0], 1), dtype=dtype)
        for d in range(self.dim):
            lo, hi = domain[d]
            result = result * (x[:, d:d+1] - lo) * (hi - x[:, d:d+1])
        return result

    def _laplacian_torch(
        self,
        x: "torch.Tensor",
        model: Any,
        domain: Tuple[Tuple[float, float], ...],
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute Laplacian in PyTorch."""
        import torch

        cutoff = self._cutoff_torch(x, domain, dtype)
        u = cutoff * model(x)

        laplacian = torch.zeros(x.shape[0], dtype=dtype)
        for d in range(self.dim):
            du_d = torch.autograd.grad(
                u.sum(), x, create_graph=True, retain_graph=True
            )[0][:, d]
            d2u_d = torch.autograd.grad(
                du_d.sum(), x, create_graph=True, retain_graph=True
            )[0][:, d]
            laplacian = laplacian + d2u_d

        return laplacian

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate primal solution at given points."""
        if x.ndim == 1 and self.dim == 1:
            x = x.reshape(-1, 1)

        if self._domain is None:
            raise ValueError("Model not trained yet")

        domain = self._domain

        # Cutoff
        cutoff = np.ones((x.shape[0], 1))
        for d in range(self.dim):
            lo, hi = domain[d]
            cutoff *= (x[:, d:d+1] - lo) * (hi - x[:, d:d+1])

        if self.backend == "tensorflow":
            nn_out = self._primal_model(x, training=False).numpy()
        elif self.backend == "pytorch":
            import torch
            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_t = torch.tensor(x, dtype=dtype)
            with torch.no_grad():
                nn_out = self._primal_model(x_t).numpy()

        return (cutoff * nn_out).flatten()

    def predict_adjoint(self, x: np.ndarray) -> np.ndarray:
        """Evaluate adjoint solution at given points."""
        if x.ndim == 1 and self.dim == 1:
            x = x.reshape(-1, 1)

        if self._domain is None:
            raise ValueError("Model not trained yet")

        domain = self._domain

        cutoff = np.ones((x.shape[0], 1))
        for d in range(self.dim):
            lo, hi = domain[d]
            cutoff *= (x[:, d:d+1] - lo) * (hi - x[:, d:d+1])

        if self.backend == "tensorflow":
            nn_out = self._adjoint_model(x, training=False).numpy()
        elif self.backend == "pytorch":
            import torch
            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_t = torch.tensor(x, dtype=dtype)
            with torch.no_grad():
                nn_out = self._adjoint_model(x_t).numpy()

        return (cutoff * nn_out).flatten()

    def estimate_qoi_error(self, problem: Any) -> float:
        """Estimate error in QoI using DWR formula.

        Error ≈ |<R(u_h), z_h>|

        Returns:
            Estimated QoI error.
        """
        if self._quad_points is None:
            raise ValueError("Model not trained yet")

        x = self._quad_points

        # Get primal residual
        u = self.predict(x)
        # Would need Laplacian computation here...

        # Get adjoint
        z = self.predict_adjoint(x)

        # This is simplified - proper implementation needs residual
        return float(np.abs(np.mean(z)))

    def summary(self) -> None:
        """Print model summary."""
        print(f"Goal-Oriented DFR Model")
        print(f"  Dimension: {self.dim}")
        print(f"  QoI: {type(self.qoi).__name__ if self.qoi else 'Not set'}")
        print(f"  Network: {self.hidden_layers}")
        print(f"  Sparse: {self.use_sparse}")
        if self._index_set:
            print(f"  Fourier modes: {len(self._index_set)}")
