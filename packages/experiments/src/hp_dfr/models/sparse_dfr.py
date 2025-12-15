"""Sparse Deep Fourier Residual (Sparse-DFR) method implementation.

This module extends the DFR method to use sparse Fourier modes via
hyperbolic cross index sets, reducing complexity from O(N^d) to
O(N (log N)^{d-1}) for d-dimensional problems.

The sparse approach maintains error-loss equivalence while dramatically
reducing computational cost in high dimensions.

References:
    - Taylor et al. (2022). A Deep Fourier Residual Method.
    - Taylor et al. (2024). Adaptive Deep Fourier Residual method.
    - Bungartz & Griebel (2004). Sparse grids. Acta Numerica.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from hp_dfr.models.base import BaseModel
from hp_dfr.fourier import (
    HyperbolicCrossIndexSet,
    dst_matrix_nd_sparse,
    h_minus_1_weights_sparse,
    hyperbolic_cross_indices,
    full_tensor_indices,
)


class SparseDFRModel(BaseModel):
    """Sparse Deep Fourier Residual model for high-dimensional PDEs.

    Uses hyperbolic cross Fourier modes instead of full tensor product,
    reducing complexity from O(N^d) to O(N (log N)^{d-1}).

    The loss function remains equivalent to the H^1 error for well-posed
    problems, while requiring far fewer Fourier modes.

    Example:
        >>> # 2D problem with sparse modes
        >>> model = SparseDFRModel(
        ...     hidden_layers=[20, 20, 20],
        ...     dim=2,
        ...     max_level=16,  # Hyperbolic cross level
        ...     use_sparse=True,
        ... )
        >>> model.build()
        >>> # Full tensor: 16^2 = 256 modes
        >>> # Sparse: ~50 modes (5x compression)
        >>> history = model.fit(problem, epochs=1000)

    Attributes:
        dim: Spatial dimension of the problem.
        max_level: Maximum level for hyperbolic cross.
        use_sparse: Whether to use sparse (True) or full (False) modes.
        index_set: HyperbolicCrossIndexSet if sparse, None if full.
    """

    def __init__(
        self,
        hidden_layers: List[int] = [20, 20, 20, 20],
        activation: str = "tanh",
        dtype: str = "float64",
        seed: int = 1234,
        dim: int = 1,
        n_quadrature: int = 32,
        max_level: int = 16,
        use_sparse: bool = True,
        backend: str = "tensorflow",
    ):
        """Initialize Sparse DFR model.

        Args:
            hidden_layers: Number of neurons in each hidden layer.
            activation: Activation function for hidden layers.
            dtype: Floating point precision.
            seed: Random seed for reproducibility.
            dim: Spatial dimension of the problem.
            n_quadrature: Number of quadrature points per dimension.
            max_level: Maximum level N for hyperbolic cross (product <= N).
            use_sparse: If True, use hyperbolic cross. If False, use full tensor.
            backend: Deep learning backend ('tensorflow', 'jax', or 'pytorch').
        """
        super().__init__(hidden_layers, activation, dtype, seed)
        self.dim = dim
        self.n_quadrature = n_quadrature
        self.max_level = max_level
        self.use_sparse = use_sparse
        self.backend = backend

        self._backend_module = None
        self._dst_matrix: Optional[NDArray] = None
        self._h_minus_1_weights: Optional[NDArray] = None
        self._index_set: Optional[HyperbolicCrossIndexSet] = None
        self._domain: Optional[Tuple[Tuple[float, float], ...]] = None
        self._quad_points: Optional[NDArray] = None

    @property
    def n_modes(self) -> int:
        """Return number of Fourier modes used."""
        if self.use_sparse and self._index_set is not None:
            return len(self._index_set)
        else:
            return self.max_level**self.dim

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio vs full tensor."""
        if self.use_sparse and self._index_set is not None:
            return self._index_set.compression_ratio()
        return 1.0

    def _setup_fourier(
        self, domain: Tuple[Tuple[float, float], ...]
    ) -> None:
        """Set up Fourier transform matrices and weights.

        Args:
            domain: Tuple of (a_i, b_i) for each dimension.
        """
        self._domain = domain

        # Create index set
        if self.use_sparse:
            self._index_set = HyperbolicCrossIndexSet(
                dim=self.dim, max_level=self.max_level
            )
            n_indices = len(self._index_set)
        else:
            self._index_set = None
            n_indices = self.max_level**self.dim

        # Grid shape for quadrature
        grid_shape = tuple([self.n_quadrature] * self.dim)

        # Create quadrature points
        quad_1d = []
        for d in range(self.dim):
            a, b = domain[d]
            x = np.linspace(a, b, self.n_quadrature + 1)
            dx = x[1] - x[0]
            x_mid = x[:-1] + dx / 2
            quad_1d.append(x_mid)

        # Create meshgrid for quadrature points
        grids = np.meshgrid(*quad_1d, indexing="ij")
        # Shape: (n_quad^dim, dim)
        self._quad_points = np.stack([g.ravel() for g in grids], axis=1)

        # Create DST matrix
        if self.use_sparse:
            self._dst_matrix = dst_matrix_nd_sparse(
                grid_shape, domain, self._index_set
            )
            self._h_minus_1_weights = h_minus_1_weights_sparse(
                self._index_set, domain
            )
        else:
            # For full tensor, create matrix from full index set
            self._dst_matrix = self._create_full_dst_matrix(grid_shape, domain)
            self._h_minus_1_weights = self._create_full_weights(domain)

    def _create_full_dst_matrix(
        self,
        grid_shape: Tuple[int, ...],
        domain: Tuple[Tuple[float, float], ...],
    ) -> NDArray:
        """Create DST matrix for full tensor product."""
        indices = full_tensor_indices(self.dim, self.max_level)
        n_grid = int(np.prod(grid_shape))
        n_modes = len(indices)

        # Build 1D basis functions
        basis_1d = []
        for d in range(self.dim):
            a, b = domain[d]
            L = b - a
            n = grid_shape[d]

            x = np.linspace(a, b, n + 1)
            dx = x[1] - x[0]
            x_mid = x[:-1] + dx / 2

            norm = np.sqrt(2.0 / L)
            basis = np.zeros((self.max_level, n))
            for k in range(1, self.max_level + 1):
                basis[k - 1, :] = norm * np.sin(k * np.pi * (x_mid - a) / L) * dx

            basis_1d.append(basis)

        # Build matrix via tensor product
        matrix = np.zeros((n_modes, n_grid))

        for i, idx in enumerate(indices):
            row = np.ones(n_grid)
            for d in range(self.dim):
                k = idx[d] - 1
                basis_d = basis_1d[d][k, :]
                shape = [1] * self.dim
                shape[d] = grid_shape[d]
                basis_expanded = basis_d.reshape(shape)
                tile_shape = list(grid_shape)
                tile_shape[d] = 1
                basis_full = np.tile(basis_expanded, tile_shape).ravel()
                row *= basis_full
            matrix[i, :] = row

        return matrix

    def _create_full_weights(
        self, domain: Tuple[Tuple[float, float], ...]
    ) -> NDArray:
        """Create H^{-1} weights for full tensor product."""
        indices = full_tensor_indices(self.dim, self.max_level)
        n_modes = len(indices)

        weights = np.ones(n_modes)
        for d in range(self.dim):
            a, b = domain[d]
            L = b - a
            k_d = indices[:, d]
            weights *= L / (np.pi * k_d)

        return weights

    def build(self, input_dim: Optional[int] = None, output_dim: int = 1) -> None:
        """Build the neural network.

        Args:
            input_dim: Input dimension (defaults to self.dim).
            output_dim: Output dimension (solution components).
        """
        if input_dim is None:
            input_dim = self.dim

        if self.backend == "tensorflow":
            self._build_tensorflow(input_dim, output_dim)
        elif self.backend == "jax":
            self._build_jax(input_dim, output_dim)
        elif self.backend == "pytorch":
            self._build_pytorch(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _build_tensorflow(self, input_dim: int, output_dim: int) -> None:
        """Build model using TensorFlow/Keras."""
        import os
        os.environ["KERAS_BACKEND"] = "tensorflow"

        import keras
        keras.utils.set_random_seed(self.seed)
        keras.backend.set_floatx(self.dtype)

        inputs = keras.layers.Input(shape=(input_dim,), dtype=self.dtype)
        x = inputs
        for neurons in self.hidden_layers:
            x = keras.layers.Dense(
                neurons, activation=self.activation, dtype=self.dtype
            )(x)
        x = keras.layers.Dense(
            output_dim, activation=self.activation, dtype=self.dtype
        )(x)

        self._model = keras.Model(inputs=inputs, outputs=x)
        self._backend_module = keras

    def _build_jax(self, input_dim: int, output_dim: int) -> None:
        """Build model using JAX."""
        import jax
        import jax.numpy as jnp
        from jax import random

        jax.config.update("jax_enable_x64", self.dtype == "float64")

        layer_sizes = [input_dim] + self.hidden_layers + [output_dim]
        key = random.PRNGKey(self.seed)

        params = []
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, w_key, b_key = random.split(key, 3)
            w = random.normal(w_key, (m, n)) * jnp.sqrt(2.0 / m)
            b = jnp.zeros(n)
            params.append({"w": w, "b": b})

        self._model = params
        self._backend_module = jax

    def _build_pytorch(self, input_dim: int, output_dim: int) -> None:
        """Build model using PyTorch."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)

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

        self._model = nn.Sequential(*layers)
        if self.dtype == "float64":
            self._model = self._model.double()

        self._backend_module = torch

    def fit(
        self,
        problem: Any,
        epochs: int = 1000,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model using Sparse DFR method.

        Args:
            problem: Problem instance with domain, forcing_fn, boundary conditions.
            epochs: Number of training iterations.
            learning_rate: Learning rate for optimizer.
            verbose: Whether to print progress.

        Returns:
            Dictionary containing training history.
        """
        if self._model is None:
            self.build()

        # Convert domain to tuple of tuples for multi-D
        if self.dim == 1:
            domain = (problem.domain,)
        else:
            domain = problem.domain

        # Set up Fourier transforms
        self._setup_fourier(domain)

        if verbose:
            full_modes = self.max_level**self.dim
            print(f"Sparse DFR: {self.n_modes} modes "
                  f"(vs {full_modes} full, {self.compression_ratio:.1f}x compression)")

        if self.backend == "tensorflow":
            return self._fit_tensorflow(problem, epochs, learning_rate, verbose)
        elif self.backend == "jax":
            return self._fit_jax(problem, epochs, learning_rate, verbose)
        elif self.backend == "pytorch":
            return self._fit_pytorch(problem, epochs, learning_rate, verbose)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _fit_tensorflow(
        self, problem: Any, epochs: int, learning_rate: float, verbose: bool
    ) -> Dict[str, List[float]]:
        """Train using TensorFlow with Sparse DFR loss."""
        import tensorflow as tf
        import keras

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        if self.dim == 1:
            domain = (problem.domain,)
        else:
            domain = problem.domain

        # Convert to tensors
        pts = tf.constant(self._quad_points, dtype=self.dtype)
        dst_matrix = tf.constant(self._dst_matrix, dtype=self.dtype)
        weights = tf.constant(self._h_minus_1_weights, dtype=self.dtype)

        # Precompute forcing term transform
        f_vals = problem.forcing_fn(self._quad_points)
        if f_vals.ndim == 1:
            f_vals = f_vals.reshape(-1)
        ft_forcing = tf.constant(
            self._dst_matrix @ f_vals, dtype=self.dtype
        )

        # Cutoff function for Dirichlet BCs
        def cutoff(x: tf.Tensor) -> tf.Tensor:
            result = tf.ones((tf.shape(x)[0], 1), dtype=self.dtype)
            for d in range(self.dim):
                a, b = domain[d]
                result *= (x[:, d:d+1] - a) * (b - x[:, d:d+1])
            return result

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                # Forward pass with cutoff
                nn_out = self._model(pts, training=True)
                u = cutoff(pts) * nn_out

                # Compute Laplacian via second derivatives
                # For simplicity, compute -Delta u at quadrature points
                residual = self._compute_residual_tf(pts, u, problem, domain)

                # Fourier transform of residual
                ft_residual = tf.linalg.matvec(dst_matrix, residual)

                # Weighted coefficients for H^{-1} norm
                ft_weighted = ft_residual * weights

                # DFR loss = ||residual||_{H^{-1}}^2
                loss = tf.reduce_sum(ft_weighted ** 2)

            gradients = tape.gradient(loss, self._model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
            return loss

        self._history = {"loss": []}

        for epoch in range(epochs):
            loss = train_step()
            self._history["loss"].append(float(loss))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}")

        return self._history

    def _compute_residual_tf(
        self,
        pts: "tf.Tensor",
        u: "tf.Tensor",
        problem: Any,
        domain: Tuple[Tuple[float, float], ...],
    ) -> "tf.Tensor":
        """Compute PDE residual for Poisson equation using TensorFlow.

        For -Delta u = f, the residual is: -Delta u - f

        This requires computing the Laplacian via automatic differentiation.
        """
        import tensorflow as tf

        # For 1D: -u'' - f = 0
        if self.dim == 1:
            with tf.GradientTape() as t2:
                t2.watch(pts)
                with tf.GradientTape() as t1:
                    t1.watch(pts)
                    a, b = domain[0]
                    cutoff = (pts[:, 0:1] - a) * (b - pts[:, 0:1])
                    nn_out = self._model(pts, training=True)
                    u_val = cutoff * nn_out
                du = t1.gradient(u_val, pts)
            d2u = t2.gradient(du, pts)

            f_vals = problem.forcing_fn(pts.numpy())
            f_tensor = tf.constant(f_vals.reshape(-1), dtype=self.dtype)

            # Residual: -u'' - f (for -u'' = f)
            laplacian = d2u[:, 0]
            return -laplacian - f_tensor

        # For nD: -sum(d^2u/dx_i^2) - f = 0
        else:
            # Compute Laplacian as sum of second derivatives
            laplacian = tf.zeros(tf.shape(pts)[0], dtype=self.dtype)

            for d in range(self.dim):
                with tf.GradientTape() as t2:
                    t2.watch(pts)
                    with tf.GradientTape() as t1:
                        t1.watch(pts)
                        # Cutoff function
                        cutoff_val = tf.ones((tf.shape(pts)[0], 1), dtype=self.dtype)
                        for dd in range(self.dim):
                            a, b = domain[dd]
                            cutoff_val *= (pts[:, dd:dd+1] - a) * (b - pts[:, dd:dd+1])
                        nn_out = self._model(pts, training=True)
                        u_val = cutoff_val * nn_out
                    du = t1.gradient(u_val, pts)
                d2u = t2.gradient(du[:, d], pts)
                laplacian += d2u[:, d]

            f_vals = problem.forcing_fn(self._quad_points)
            f_tensor = tf.constant(f_vals.reshape(-1), dtype=self.dtype)

            return -laplacian - f_tensor

    def _fit_jax(
        self, problem: Any, epochs: int, learning_rate: float, verbose: bool
    ) -> Dict[str, List[float]]:
        """Train using JAX with Sparse DFR loss."""
        import jax
        import jax.numpy as jnp
        from jax import grad, jit, vmap
        import optax

        params = self._model
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        if self.dim == 1:
            domain = (problem.domain,)
        else:
            domain = problem.domain

        pts = jnp.array(self._quad_points)
        dst_matrix = jnp.array(self._dst_matrix)
        weights = jnp.array(self._h_minus_1_weights)

        f_vals = problem.forcing_fn(self._quad_points)
        f_array = jnp.array(f_vals.reshape(-1))

        def forward(params, x):
            for layer in params[:-1]:
                x = jnp.tanh(x @ layer["w"] + layer["b"])
            return jnp.tanh(x @ params[-1]["w"] + params[-1]["b"])

        def cutoff_fn(x):
            result = jnp.ones(x.shape[0])
            for d in range(self.dim):
                a, b = domain[d]
                result *= (x[:, d] - a) * (b - x[:, d])
            return result.reshape(-1, 1)

        def loss_fn(params):
            # Compute u with cutoff
            nn_out = forward(params, pts)
            u = cutoff_fn(pts) * nn_out

            # For simplicity, use finite differences for Laplacian
            # In production, would use autodiff
            residual = -f_array  # Simplified: just forcing for now

            ft_residual = dst_matrix @ residual
            ft_weighted = ft_residual * weights

            return jnp.sum(ft_weighted ** 2)

        @jit
        def train_step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        self._history = {"loss": []}

        for epoch in range(epochs):
            params, opt_state, loss = train_step(params, opt_state)
            self._history["loss"].append(float(loss))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}")

        self._model = params
        return self._history

    def _fit_pytorch(
        self, problem: Any, epochs: int, learning_rate: float, verbose: bool
    ) -> Dict[str, List[float]]:
        """Train using PyTorch with Sparse DFR loss."""
        import torch

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        if self.dim == 1:
            domain = (problem.domain,)
        else:
            domain = problem.domain

        dtype = torch.float64 if self.dtype == "float64" else torch.float32

        pts = torch.tensor(self._quad_points, dtype=dtype, requires_grad=True)
        dst_matrix = torch.tensor(self._dst_matrix, dtype=dtype)
        weights = torch.tensor(self._h_minus_1_weights, dtype=dtype)

        f_vals = problem.forcing_fn(self._quad_points)
        f_tensor = torch.tensor(f_vals.reshape(-1), dtype=dtype)

        def cutoff(x):
            result = torch.ones(x.shape[0], 1, dtype=dtype)
            for d in range(self.dim):
                a, b = domain[d]
                result *= (x[:, d:d+1] - a) * (b - x[:, d:d+1])
            return result

        self._history = {"loss": []}

        for epoch in range(epochs):
            optimizer.zero_grad()

            x = pts.detach().clone().requires_grad_(True)

            # Forward pass
            nn_out = self._model(x)
            u = cutoff(x) * nn_out

            # Compute Laplacian
            laplacian = torch.zeros(x.shape[0], dtype=dtype)
            for d in range(self.dim):
                du_d = torch.autograd.grad(
                    u.sum(), x, create_graph=True, retain_graph=True
                )[0][:, d]
                d2u_d = torch.autograd.grad(
                    du_d.sum(), x, create_graph=True, retain_graph=True
                )[0][:, d]
                laplacian += d2u_d

            # Residual: -Delta u - f
            residual = -laplacian - f_tensor

            # DFR loss
            ft_residual = torch.mv(dst_matrix, residual)
            ft_weighted = ft_residual * weights
            loss = torch.sum(ft_weighted ** 2)

            loss.backward()
            optimizer.step()

            self._history["loss"].append(float(loss))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}")

        return self._history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the trained model at given points.

        Args:
            x: Input points of shape (n_points, dim).

        Returns:
            Predicted solution values with boundary conditions enforced.
        """
        if x.ndim == 1 and self.dim == 1:
            x = x.reshape(-1, 1)

        if self._domain is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        domain = self._domain

        # Cutoff function
        cutoff = np.ones((x.shape[0], 1))
        for d in range(self.dim):
            a, b = domain[d]
            cutoff *= (x[:, d:d+1] - a) * (b - x[:, d:d+1])

        if self.backend == "tensorflow":
            nn_out = self._model(x, training=False).numpy()
            return (cutoff * nn_out).flatten()
        elif self.backend == "jax":
            import jax.numpy as jnp

            def forward(params, x):
                for layer in params[:-1]:
                    x = jnp.tanh(x @ layer["w"] + layer["b"])
                return jnp.tanh(x @ params[-1]["w"] + params[-1]["b"])

            nn_out = np.array(forward(self._model, x))
            return (cutoff * nn_out).flatten()
        elif self.backend == "pytorch":
            import torch

            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_tensor = torch.tensor(x, dtype=dtype)
            with torch.no_grad():
                nn_out = self._model(x_tensor).numpy()
            return (cutoff * nn_out).flatten()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def summary(self) -> None:
        """Print model summary with sparse Fourier info."""
        super().summary()
        if self._index_set is not None:
            print(f"Dimension: {self.dim}")
            print(f"Max level: {self.max_level}")
            print(f"Sparse modes: {self.use_sparse}")
            print(f"Number of modes: {self.n_modes}")
            print(f"Full tensor modes: {self.max_level**self.dim}")
            print(f"Compression ratio: {self.compression_ratio:.2f}x")
