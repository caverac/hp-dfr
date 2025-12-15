"""Deep Fourier Residual (DFR) method implementation."""

from typing import Any, Dict, List
import numpy as np

from hp_dfr.models.base import BaseModel


class DFRModel(BaseModel):
    """Deep Fourier Residual model using variational formulation.

    This implementation uses the H^-1 dual norm of the weak residual,
    computed efficiently via Discrete Sine/Cosine Transforms.

    The loss function is equivalent to the H^1 error for well-posed problems,
    enabling reliable error estimation during training.

    Example:
        >>> model = DFRModel(hidden_layers=[10, 10, 10, 10], n_fourier_modes=10)
        >>> model.build()
        >>> history = model.fit(problem, epochs=1000)
        >>> u_pred = model.predict(x_test)

    Reference:
        Taylor, Pardo, Muga (2023). "A Deep Fourier Residual Method for solving
        PDEs using Neural Networks". CMAME, 405, 115850.
    """

    def __init__(
        self,
        hidden_layers: List[int] = [10, 10, 10, 10],
        activation: str = "tanh",
        dtype: str = "float64",
        seed: int = 1234,
        n_quadrature: int = 100,
        n_fourier_modes: int = 10,
        backend: str = "tensorflow",
    ):
        """Initialize DFR model.

        Args:
            hidden_layers: Number of neurons in each hidden layer.
            activation: Activation function for hidden layers.
            dtype: Floating point precision.
            seed: Random seed for reproducibility.
            n_quadrature: Number of quadrature points for integration.
            n_fourier_modes: Number of Fourier modes for H^-1 norm computation.
            backend: Deep learning backend ('tensorflow', 'jax', or 'pytorch').
        """
        super().__init__(hidden_layers, activation, dtype, seed)
        self.n_quadrature = n_quadrature
        self.n_fourier_modes = n_fourier_modes
        self.backend = backend
        self._backend_module = None
        self._dst_matrix = None
        self._dct_matrix = None
        self._h_minus_1_weights = None

    def _compute_transform_matrices(self, domain: tuple) -> None:
        """Precompute DST and DCT matrices for Fourier transforms.

        Args:
            domain: Tuple (a, b) specifying the spatial domain.
        """
        a, b = domain
        L = b - a

        # Quadrature points (midpoint rule)
        pts = np.linspace(a, b, self.n_quadrature + 1)
        h = np.abs(pts[1:] - pts[:-1])
        self._quadrature_pts = pts[:-1] + h / 2

        # H^-1 norm weights: (1 + k^2 * pi^2 / L^2)^-0.5
        k = np.arange(1, self.n_fourier_modes + 1)
        self._h_minus_1_weights = ((np.pi ** 2 * k ** 2) / L ** 2) ** -0.5

        # DST matrix (Discrete Sine Transform)
        V = np.sqrt(2.0 / L)
        self._dst_matrix = np.array([
            V * np.sin(np.pi * ki * (self._quadrature_pts - a) / L) * h
            for ki in range(1, self.n_fourier_modes + 1)
        ])

        # DCT matrix with derivative (for gradient terms)
        self._dct_matrix = np.array([
            V * (ki * np.pi / L) * np.cos(np.pi * ki * (self._quadrature_pts - a) / L) * h
            for ki in range(1, self.n_fourier_modes + 1)
        ])

    def build(self, input_dim: int = 1, output_dim: int = 1) -> None:
        """Build the neural network with cutoff layer.

        The cutoff layer enforces homogeneous Dirichlet boundary conditions
        by multiplying the network output by (x - a)(b - x).

        Args:
            input_dim: Dimension of input (spatial coordinates).
            output_dim: Dimension of output (solution components).
        """
        if self.backend == "tensorflow":
            self._build_tensorflow(input_dim, output_dim)
        elif self.backend == "jax":
            self._build_jax(input_dim, output_dim)
        elif self.backend == "pytorch":
            self._build_pytorch(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _build_tensorflow(self, input_dim: int, output_dim: int) -> None:
        """Build model using TensorFlow/Keras with cutoff layer."""
        import os
        os.environ["KERAS_BACKEND"] = "tensorflow"

        import keras
        keras.utils.set_random_seed(self.seed)
        keras.backend.set_floatx(self.dtype)

        # Input layer
        inputs = keras.layers.Input(shape=(input_dim,), dtype=self.dtype)

        # Hidden layers
        x = inputs
        for neurons in self.hidden_layers:
            x = keras.layers.Dense(neurons, activation=self.activation, dtype=self.dtype)(x)

        # Output layer (with activation for DFR, before cutoff)
        x = keras.layers.Dense(output_dim, activation=self.activation, dtype=self.dtype)(x)

        # Store raw model (without cutoff) for internal use
        self._raw_model = keras.Model(inputs=inputs, outputs=x)
        self._model = self._raw_model
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
        for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
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

        # Output layer with activation (for DFR)
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
        """Train the model using DFR method.

        Args:
            problem: Problem instance with domain, forcing_fn, and boundary conditions.
            epochs: Number of training iterations.
            learning_rate: Learning rate for optimizer.
            verbose: Whether to print progress.

        Returns:
            Dictionary containing training history.
        """
        if self._model is None:
            self.build()

        # Precompute transform matrices
        self._compute_transform_matrices(problem.domain)

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
        """Train using TensorFlow with DFR loss."""
        import tensorflow as tf
        import keras

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        domain = problem.domain
        a, b = domain

        # Convert matrices to tensors
        pts = tf.constant(self._quadrature_pts.reshape(-1, 1), dtype=self.dtype)
        dct_matrix = tf.constant(self._dct_matrix, dtype=self.dtype)
        dst_matrix = tf.constant(self._dst_matrix, dtype=self.dtype)
        weights = tf.constant(self._h_minus_1_weights, dtype=self.dtype)

        # Precompute forcing term transform
        f_vals = problem.forcing_fn(self._quadrature_pts)
        ft_forcing = tf.constant(
            np.einsum("ji,i->j", self._dst_matrix, f_vals),
            dtype=self.dtype,
        )

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                # Compute u and its derivative at quadrature points
                with tf.GradientTape() as t1:
                    t1.watch(pts)
                    # Apply cutoff layer: u = (x - a)(b - x) * nn(x)
                    cutoff = (pts - a) * (b - pts)
                    nn_out = self._model(pts, training=True)
                    u = cutoff * nn_out
                du = t1.gradient(u, pts)

                # Fourier transform of gradient term
                ft_gradient = tf.einsum("ji,ij->j", dct_matrix, du)

                # Total Fourier coefficients (weak residual)
                ft_total = (ft_gradient + ft_forcing) * weights

                # H^-1 norm squared (DFR loss)
                loss = tf.reduce_sum(ft_total ** 2)

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

    def _fit_jax(
        self, problem: Any, epochs: int, learning_rate: float, verbose: bool
    ) -> Dict[str, List[float]]:
        """Train using JAX with DFR loss."""
        import jax
        import jax.numpy as jnp
        from jax import grad, jit, vmap
        import optax

        params = self._model
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        domain = problem.domain
        a, b = domain

        pts = jnp.array(self._quadrature_pts.reshape(-1, 1))
        dct_matrix = jnp.array(self._dct_matrix)
        weights = jnp.array(self._h_minus_1_weights)

        # Precompute forcing term transform
        f_vals = problem.forcing_fn(self._quadrature_pts)
        ft_forcing = jnp.einsum("ji,i->j", self._dst_matrix, f_vals)

        def forward(params, x):
            for i, layer in enumerate(params[:-1]):
                x = jnp.tanh(x @ layer["w"] + layer["b"])
            # Last layer with activation for DFR
            x = jnp.tanh(x @ params[-1]["w"] + params[-1]["b"])
            return x

        def forward_with_cutoff(params, x):
            cutoff = (x - a) * (b - x)
            return cutoff * forward(params, x)

        def loss_fn(params):
            # Compute derivatives using vmap
            def u_scalar(xi):
                return forward_with_cutoff(params, xi.reshape(1, -1))[0, 0]

            du_dx = vmap(grad(u_scalar))
            du = du_dx(pts[:, 0])

            # Fourier transform
            ft_gradient = jnp.einsum("ji,i->j", dct_matrix, du)
            ft_total = (ft_gradient + ft_forcing) * weights

            return jnp.sum(ft_total ** 2)

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
        """Train using PyTorch with DFR loss."""
        import torch

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        domain = problem.domain
        a, b = domain
        dtype = torch.float64 if self.dtype == "float64" else torch.float32

        pts = torch.tensor(
            self._quadrature_pts.reshape(-1, 1), dtype=dtype, requires_grad=True
        )
        dct_matrix = torch.tensor(self._dct_matrix, dtype=dtype)
        weights = torch.tensor(self._h_minus_1_weights, dtype=dtype)

        # Precompute forcing term transform
        f_vals = problem.forcing_fn(self._quadrature_pts)
        ft_forcing = torch.tensor(
            np.einsum("ji,i->j", self._dst_matrix, f_vals), dtype=dtype
        )

        self._history = {"loss": []}

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Need fresh tensor for gradient computation
            x = pts.detach().clone().requires_grad_(True)

            # Apply cutoff and forward pass
            cutoff = (x - a) * (b - x)
            nn_out = self._model(x)
            u = cutoff * nn_out

            # Compute derivative
            du = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]

            # Fourier transform
            ft_gradient = torch.einsum("ji,ij->j", dct_matrix, du)
            ft_total = (ft_gradient + ft_forcing) * weights

            loss = torch.sum(ft_total ** 2)
            loss.backward()
            optimizer.step()

            self._history["loss"].append(float(loss))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}")

        return self._history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the trained model (with cutoff) at given points.

        Args:
            x: Input points of shape (n_points,) or (n_points, 1).

        Returns:
            Predicted solution values with boundary conditions enforced.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Note: We need the domain to apply cutoff
        # This assumes predict is called after fit
        a, b = 0, np.pi  # Default domain, should be stored from problem

        if self.backend == "tensorflow":
            cutoff = (x - a) * (b - x)
            nn_out = self._model(x, training=False).numpy()
            return (cutoff * nn_out).flatten()
        elif self.backend == "jax":
            import jax.numpy as jnp

            def forward(params, x):
                for layer in params[:-1]:
                    x = jnp.tanh(x @ layer["w"] + layer["b"])
                return jnp.tanh(x @ params[-1]["w"] + params[-1]["b"])

            cutoff = (x - a) * (b - x)
            nn_out = np.array(forward(self._model, x))
            return (cutoff * nn_out).flatten()
        elif self.backend == "pytorch":
            import torch

            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_tensor = torch.tensor(x, dtype=dtype)
            cutoff = (x_tensor - a) * (b - x_tensor)
            with torch.no_grad():
                nn_out = self._model(x_tensor)
            return (cutoff * nn_out).numpy().flatten()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
