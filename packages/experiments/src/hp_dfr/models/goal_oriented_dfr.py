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
    - Error representation: J(u) - J(u_h) = <R(u_h), z>

References:
    - Becker & Rannacher (2001). Optimal control approach to error estimation.
    - Chakraborty, Wick, Rabczuk, Zhuang (2025). Multigoal-oriented DWR error
      estimation using PINNs. Machine Learning for Computational Science and
      Engineering 1(1):13. doi:10.1007/s44379-025-00012-4.
    - Roth, Schroder, Wick (2022). Neural network guided adjoint computations in
      dual weighted residual error estimation. SN Applied Sciences 4(2):62.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from hp_dfr.fourier import (
    dst_matrix_nd_full,
    h_minus_1_weights,
)
from hp_dfr.models.base import BaseModel, Problem
from hp_dfr.utils import console

ArrayF = NDArray[np.floating]
Domain = Tuple[Tuple[float, float], ...]

# Optional deep-learning backends. They are heavy and only one is needed at a
# time, so they are imported lazily here (guarded) rather than at first use.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
try:
    import keras
    import tensorflow as tf
except ImportError:  # backend not installed
    keras = None
    tf = None

try:
    import torch
    from torch import nn
except ImportError:  # backend not installed
    torch = None
    nn = None


class QuantityOfInterest(ABC):
    """Abstract base class for quantities of interest (QoI).

    A QoI is a functional J(u) that we want to compute accurately.
    Examples: point evaluation, average value, flux through boundary.
    """

    @abstractmethod
    def evaluate(self, u_fn: Callable[[ArrayF], ArrayF], domain: tuple) -> float:
        """Evaluate QoI given solution function.

        Args:
            u_fn: Function u(x) -> value.
            domain: Problem domain.

        Returns:
            QoI value.
        """

    @abstractmethod
    def adjoint_rhs(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute right-hand side for adjoint problem.

        The adjoint problem is L*z = J'(u), where J' is the
        Frechet derivative of the QoI functional.

        Args:
            x: Points at which to evaluate.

        Returns:
            Adjoint RHS values.
        """


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

    def evaluate(self, u_fn: Callable[[ArrayF], ArrayF], domain: tuple) -> float:
        """Evaluate u at the specified point."""
        return float(u_fn(self.point.reshape(1, -1))[0])

    def adjoint_rhs(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Approximate delta function as narrow Gaussian."""
        dist_sq = np.sum((x - self.point) ** 2, axis=1)
        # Normalized Gaussian
        dim = len(self.point)
        norm = (2 * np.pi * self.sigma**2) ** (-dim / 2)
        return cast(ArrayF, norm * np.exp(-dist_sq / (2 * self.sigma**2)))


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

    def evaluate(self, u_fn: Callable[[ArrayF], ArrayF], domain: tuple) -> float:
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
        # Will be normalized by domain volume
        return np.ones(n_points)


class BoundaryFluxQoI(QuantityOfInterest):
    """Flux through a boundary segment.

    J(u) = integral_Gamma (grad u . n) ds

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

    def evaluate(self, u_fn: Callable[[ArrayF], ArrayF], domain: tuple) -> float:
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
        hidden_layers: Tuple[int, ...] = (20, 20, 20),
        activation: str = "tanh",
        dtype: str = "float64",
        seed: int = 1234,
        dim: int = 2,
        qoi: Optional[QuantityOfInterest] = None,
        n_quadrature: int = 32,
        n_modes: int = 16,
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
            n_modes: Number of Fourier modes per dimension.
            adjoint_weight: Weight for adjoint-based loss.
            primal_weight: Weight for standard DFR loss (regularization).
            backend: Deep learning backend.
        """
        super().__init__(hidden_layers=hidden_layers, activation=activation, dtype=dtype, seed=seed)
        self.dim = dim
        self.qoi = qoi
        self.n_quadrature = n_quadrature
        self.n_modes = n_modes
        self.adjoint_weight = adjoint_weight
        self.primal_weight = primal_weight
        self.backend = backend

        # Backend network objects and precomputed arrays are set lazily in
        # build()/fit().
        self._primal_model: keras.Model | torch.nn.Module | None = None
        self._adjoint_model: keras.Model | torch.nn.Module | None = None
        self._dst_matrix: NDArray[np.floating] | None = None
        self._h_weights: NDArray[np.floating] | None = None
        self._domain: Tuple[Tuple[float, float], ...] | None = None
        self._quad_points: NDArray[np.floating] | None = None

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
        if keras is None:
            raise RuntimeError("TensorFlow backend requires tensorflow and keras.")

        keras.utils.set_random_seed(self.seed)

        def create_network(name: str) -> keras.Model:
            inputs = keras.layers.Input(shape=(input_dim,), dtype=self.dtype)
            x = inputs
            for neurons in self.hidden_layers:
                x = keras.layers.Dense(neurons, activation=self.activation, dtype=self.dtype)(x)
            # Linear output (no activation: solution/adjoint values are unbounded).
            x = keras.layers.Dense(output_dim, activation=None, dtype=self.dtype)(x)
            return keras.Model(inputs=inputs, outputs=x, name=name)

        self._primal_model = create_network("primal")
        self._adjoint_model = create_network("adjoint")

    def _build_pytorch(self, input_dim: int, output_dim: int) -> None:
        """Build PyTorch models for primal and adjoint."""
        if torch is None:
            raise RuntimeError("PyTorch backend requires torch.")

        torch.manual_seed(self.seed)

        def create_network() -> torch.nn.Module:
            layers = []
            prev_dim = input_dim
            for neurons in self.hidden_layers:
                layers.append(nn.Linear(prev_dim, neurons))
                if self.activation == "tanh":
                    layers.append(nn.Tanh())
                elif self.activation == "relu":
                    layers.append(nn.ReLU())
                prev_dim = neurons
            # Linear output (no activation: solution/adjoint values are unbounded).
            layers.append(nn.Linear(prev_dim, output_dim))
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

        # Full tensor-product DST matrix and H^{-1} weights. Coefficients and
        # weights share the same C-order ravel over the multi-index, so the
        # weighted dual norm is a plain elementwise product downstream.
        self._dst_matrix = dst_matrix_nd_full(grid_shape, domain, self.n_modes)
        self._h_weights = h_minus_1_weights(tuple([self.n_modes] * self.dim), domain).ravel()

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
        problem: Problem,
        epochs: int = 1000,
        learning_rate: float = 1e-3,
        verbose: bool = True,
        lbfgs_iters: int = 0,
    ) -> Dict[str, List[float]]:
        """Train the goal-oriented model.

        Simultaneously trains primal and adjoint networks with
        coupled loss functions.

        Args:
            problem: PDE problem.
            epochs: Training epochs.
            learning_rate: Learning rate.
            verbose: Print progress.
            lbfgs_iters: If > 0, run an LBFGS polishing phase for this many
                iterations after Adam (PyTorch backend only).

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
            console.print(f"Goal-Oriented DFR: QoI = {type(self.qoi).__name__}", markup=False)
            console.print(f"Fourier modes per dim: {self.n_modes}", markup=False)

        if self.backend == "tensorflow":
            return self._fit_tensorflow(problem, epochs, learning_rate, verbose)
        if self.backend == "pytorch":
            return self._fit_pytorch(problem, epochs, learning_rate, verbose, lbfgs_iters)
        raise ValueError(f"Backend {self.backend} not yet implemented")

    def _fit_tensorflow(
        self,
        problem: Problem,
        epochs: int,
        learning_rate: float,
        verbose: bool,
    ) -> Dict[str, List[float]]:
        """Train using TensorFlow."""
        if keras is None:
            raise RuntimeError("TensorFlow backend requires tensorflow and keras.")
        assert self._primal_model is not None and self._adjoint_model is not None
        assert self._dst_matrix is not None and self._h_weights is not None
        assert self._domain is not None and self._quad_points is not None
        assert self.qoi is not None
        primal_model = self._primal_model
        adjoint_model = self._adjoint_model

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Get all trainable variables
        all_vars = self._primal_model.trainable_variables + self._adjoint_model.trainable_variables

        # Precompute
        pts = tf.constant(self._quad_points, dtype=self.dtype)
        dst_matrix = tf.constant(self._dst_matrix, dtype=self.dtype)
        weights = tf.constant(self._h_weights, dtype=self.dtype)

        # Forcing term (PDE: -Delta u = f)
        f_vals = problem.forcing_fn(self._quad_points)
        if f_vals.ndim > 1:
            f_vals = f_vals.flatten()
        f_vals_t = tf.constant(f_vals, dtype=self.dtype)

        # Adjoint RHS (Riesz representative of the QoI derivative)
        adjoint_rhs = self.qoi.adjoint_rhs(self._quad_points)
        adjoint_rhs_t = tf.constant(adjoint_rhs, dtype=self.dtype)

        domain = self._domain

        # Quadrature cell measure for the L2 dual pairing <R(u), z>.
        quad_weight = 1.0
        for d in range(self.dim):
            lo, hi = domain[d]
            quad_weight *= (hi - lo) / self.n_quadrature

        @tf.function
        def train_step() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            with tf.GradientTape() as tape:
                # Adjoint solution with cutoff. The primal solution itself does
                # not enter the loss, only its Laplacian (via the residual).
                cutoff = self._cutoff_tf(pts, domain)
                z = cutoff * adjoint_model(pts, training=True)

                # Laplacians of the cutoff-enforced primal and adjoint networks.
                laplacian_u = tf.reshape(self._laplacian_tf(pts, primal_model, domain), [-1])
                laplacian_z = tf.reshape(self._laplacian_tf(pts, adjoint_model, domain), [-1])

                # Weak residuals, which vanish at the exact solutions:
                #   primal:  f + Delta u           (PDE: -Delta u = f)
                #   adjoint: J' + Delta z          (J' is the QoI Riesz rep)
                weak_residual_u = f_vals_t + laplacian_u
                weak_residual_z = adjoint_rhs_t + laplacian_z

                # Goal-oriented loss: |<R(u), z>| = |int (f + Delta u) z dx|.
                # The adjoint is held fixed here (stop_gradient) so the goal term
                # drives only the primal network; the adjoint is trained solely by
                # its own residual below. See the PyTorch path for the rationale.
                z_flat = tf.reshape(z, [-1])
                go_loss = tf.abs(tf.reduce_sum(weak_residual_u * tf.stop_gradient(z_flat)) * quad_weight)

                # DFR regularization: H^{-1} dual norm of each weak residual.
                ft_primal = tf.linalg.matvec(dst_matrix, weak_residual_u) * weights
                primal_loss = tf.reduce_sum(ft_primal**2)

                ft_adjoint = tf.linalg.matvec(dst_matrix, weak_residual_z) * weights
                adjoint_loss = tf.reduce_sum(ft_adjoint**2)

                # Total loss
                total_loss = self.adjoint_weight * go_loss + self.primal_weight * primal_loss + self.primal_weight * adjoint_loss

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
                console.print(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {total:.6e} (GO: {go:.6e})",
                    markup=False,
                )

        return self._history

    def _cutoff_tf(
        self,
        x: tf.Tensor,
        domain: Tuple[Tuple[float, float], ...],
    ) -> tf.Tensor:
        """Compute cutoff function in TensorFlow."""
        result = tf.ones((tf.shape(x)[0], 1), dtype=self.dtype)
        for d in range(self.dim):
            lo, hi = domain[d]
            result *= (x[:, d : d + 1] - lo) * (hi - x[:, d : d + 1])
        return result

    def _laplacian_tf(
        self,
        x: tf.Tensor,
        model: keras.Model,
        domain: Tuple[Tuple[float, float], ...],
    ) -> tf.Tensor:
        """Compute Laplacian via autodiff in TensorFlow."""
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
        problem: Problem,
        epochs: int,
        learning_rate: float,
        verbose: bool,
        lbfgs_iters: int = 0,
    ) -> Dict[str, List[float]]:
        """Train using PyTorch (Adam, optional LBFGS polish)."""
        if torch is None:
            raise RuntimeError("PyTorch backend requires torch.")
        assert self._primal_model is not None and self._adjoint_model is not None
        assert self._dst_matrix is not None and self._h_weights is not None
        assert self._domain is not None and self._quad_points is not None
        assert self.qoi is not None
        primal_model = self._primal_model
        adjoint_model = self._adjoint_model

        dtype = torch.float64 if self.dtype == "float64" else torch.float32

        all_params = list(primal_model.parameters()) + list(adjoint_model.parameters())

        pts = torch.tensor(self._quad_points, dtype=dtype, requires_grad=True)
        dst_matrix = torch.tensor(self._dst_matrix, dtype=dtype)
        weights = torch.tensor(self._h_weights, dtype=dtype)

        f_vals = problem.forcing_fn(self._quad_points)
        if f_vals.ndim > 1:
            f_vals = f_vals.flatten()
        f_vals_t = torch.tensor(f_vals, dtype=dtype)

        adjoint_rhs = self.qoi.adjoint_rhs(self._quad_points)
        adjoint_rhs_t = torch.tensor(adjoint_rhs, dtype=dtype)

        domain = self._domain

        # Quadrature cell measure for the L2 dual pairing <R(u), z>.
        quad_weight = 1.0
        for d in range(self.dim):
            lo, hi = domain[d]
            quad_weight *= (hi - lo) / self.n_quadrature

        def compute_losses() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            x = pts.detach().clone().requires_grad_(True)
            # Adjoint solution with cutoff. The primal solution itself does not
            # enter the loss, only its Laplacian (via the residual).
            cutoff = self._cutoff_torch(x, domain, dtype)
            z = cutoff * adjoint_model(x)

            # Laplacians of the cutoff-enforced primal and adjoint networks.
            laplacian_u = self._laplacian_torch(x, primal_model, domain, dtype)
            laplacian_z = self._laplacian_torch(x, adjoint_model, domain, dtype)

            # Weak residuals, which vanish at the exact solutions:
            #   primal:  f + Delta u           (PDE: -Delta u = f)
            #   adjoint: J' + Delta z          (adjoint RHS J' is the QoI Riesz rep)
            weak_residual_u = f_vals_t + laplacian_u
            weak_residual_z = adjoint_rhs_t + laplacian_z

            # Goal-oriented loss: |<R(u), z>| = |int (f + Delta u) z dx|.
            # The adjoint is detached here so the goal term drives only the primal
            # network: it makes u reduce the residual where the (independently
            # trained) adjoint is large. Without this, minimizing the pairing over
            # the adjoint parameters lets z go orthogonal to the residual -- the
            # pairing collapses to ~0 without the adjoint solving its own problem.
            z_flat = z.reshape(-1)
            go_loss = torch.abs(torch.sum(weak_residual_u * z_flat.detach()) * quad_weight)

            # DFR regularization: H^{-1} dual norm of each weak residual.
            ft_primal = torch.mv(dst_matrix, weak_residual_u) * weights
            primal_loss = torch.sum(ft_primal**2)

            ft_adjoint = torch.mv(dst_matrix, weak_residual_z) * weights
            adjoint_loss = torch.sum(ft_adjoint**2)

            total = self.adjoint_weight * go_loss + self.primal_weight * primal_loss + self.primal_weight * adjoint_loss
            return total, go_loss, primal_loss, adjoint_loss

        self._history = {
            "loss": [],
            "go_loss": [],
            "primal_loss": [],
            "adjoint_loss": [],
        }

        def record(
            total: torch.Tensor,
            go_loss: torch.Tensor,
            primal_loss: torch.Tensor,
            adjoint_loss: torch.Tensor,
        ) -> None:
            self._history["loss"].append(float(total))
            self._history["go_loss"].append(float(go_loss))
            self._history["primal_loss"].append(float(primal_loss))
            self._history["adjoint_loss"].append(float(adjoint_loss))

        optimizer = torch.optim.Adam(all_params, lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            total, go_loss, primal_loss, adjoint_loss = compute_losses()
            total.backward()
            optimizer.step()
            record(total, go_loss, primal_loss, adjoint_loss)
            if verbose and (epoch + 1) % 100 == 0:
                console.print(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {float(total):.6e}",
                    markup=False,
                )

        if lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(
                all_params,
                max_iter=lbfgs_iters,
                line_search_fn="strong_wolfe",
                history_size=50,
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                total, _, _, _ = compute_losses()
                total.backward()
                return total

            lbfgs.step(closure)
            total, go_loss, primal_loss, adjoint_loss = compute_losses()
            record(total, go_loss, primal_loss, adjoint_loss)
            if verbose:
                console.print(f"LBFGS polish - Loss: {float(total):.6e}", markup=False)

        return self._history

    def _cutoff_torch(
        self,
        x: torch.Tensor,
        domain: Tuple[Tuple[float, float], ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute cutoff in PyTorch."""
        result = torch.ones((x.shape[0], 1), dtype=dtype)
        for d in range(self.dim):
            lo, hi = domain[d]
            result = result * (x[:, d : d + 1] - lo) * (hi - x[:, d : d + 1])
        return result

    def _laplacian_torch(
        self,
        x: torch.Tensor,
        model: torch.nn.Module,
        domain: Tuple[Tuple[float, float], ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute Laplacian in PyTorch."""
        cutoff = self._cutoff_torch(x, domain, dtype)
        u = cutoff * model(x)

        laplacian = torch.zeros(x.shape[0], dtype=dtype)
        for d in range(self.dim):
            du_d = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0][:, d]
            d2u_d = torch.autograd.grad(du_d.sum(), x, create_graph=True, retain_graph=True)[0][:, d]
            laplacian = laplacian + d2u_d

        return laplacian

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate primal solution at given points."""
        if x.ndim == 1 and self.dim == 1:
            x = x.reshape(-1, 1)

        if self._domain is None or self._primal_model is None:
            raise ValueError("Model not trained yet")

        domain = self._domain
        primal_model = self._primal_model

        # Cutoff
        cutoff = np.ones((x.shape[0], 1))
        for d in range(self.dim):
            lo, hi = domain[d]
            cutoff *= (x[:, d : d + 1] - lo) * (hi - x[:, d : d + 1])

        if self.backend == "tensorflow":
            nn_out = primal_model(x, training=False).numpy()
        else:
            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_t = torch.tensor(x, dtype=dtype)
            with torch.no_grad():
                nn_out = primal_model(x_t).numpy()

        return cast(np.ndarray, (cutoff * nn_out).flatten())

    def predict_adjoint(self, x: np.ndarray) -> np.ndarray:
        """Evaluate adjoint solution at given points."""
        if x.ndim == 1 and self.dim == 1:
            x = x.reshape(-1, 1)

        if self._domain is None or self._adjoint_model is None:
            raise ValueError("Model not trained yet")

        domain = self._domain
        adjoint_model = self._adjoint_model

        cutoff = np.ones((x.shape[0], 1))
        for d in range(self.dim):
            lo, hi = domain[d]
            cutoff *= (x[:, d : d + 1] - lo) * (hi - x[:, d : d + 1])

        if self.backend == "tensorflow":
            nn_out = adjoint_model(x, training=False).numpy()
        else:
            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_t = torch.tensor(x, dtype=dtype)
            with torch.no_grad():
                nn_out = adjoint_model(x_t).numpy()

        return cast(np.ndarray, (cutoff * nn_out).flatten())

    def estimate_qoi_error(self, inf_sup_constant: float = 1.0) -> float:
        """Bound the QoI error by the computable terms of the error-control theorem.

        Returns ``L_QoI + ||R(u_h)|| ||R*(z_h)|| / gamma``, the full right-hand
        side of the bound, evaluated after the last training step.

        The goal term ``L_QoI = |<R(u_h), z_h>|`` alone is *not* an estimate of
        the QoI error, even though the theorem is often read that way. Training
        minimizes it directly, and it is a single scalar condition, so the
        optimizer drives it to roughly ``1e-10`` while the true QoI error remains
        near ``1e-4``. A quantity that has been minimized to zero cannot measure
        what error is left; the bound is carried entirely by the remainder term,
        which is why that term is included here.

        Args:
            inf_sup_constant: The inf-sup constant ``gamma`` of (A2). Defaults to
                ``1.0``, its value for the symmetric coercive Poisson problem in
                the energy norm.

        Returns:
            An upper bound on ``|J(u*) - J(u_h)|``.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self._history.get("go_loss"):
            raise ValueError("Model not trained yet; call fit() first.")
        goal_term = float(self._history["go_loss"][-1])
        # The history stores the squared dual norms.
        res_primal = float(np.sqrt(self._history["primal_loss"][-1]))
        res_adjoint = float(np.sqrt(self._history["adjoint_loss"][-1]))
        return goal_term + res_primal * res_adjoint / inf_sup_constant

    def summary(self) -> None:
        """Print model summary."""
        console.print("Goal-Oriented DFR Model", markup=False, highlight=False)
        console.print(f"  Dimension: {self.dim}", markup=False, highlight=False)
        console.print(f"  QoI: {type(self.qoi).__name__ if self.qoi else 'Not set'}", markup=False, highlight=False)
        console.print(f"  Network: {self.hidden_layers}", markup=False, highlight=False)
        console.print(f"  Fourier modes per dim: {self.n_modes}", markup=False, highlight=False)
