"""Physics-Informed Neural Networks (PINNs) implementation."""

from typing import Any, Callable, Dict, List, Optional
import numpy as np

from hp_dfr.models.base import BaseModel


class PINNsModel(BaseModel):
    """Physics-Informed Neural Network using collocation method.

    This implementation uses strong-form PDE residuals as the loss function,
    with boundary conditions enforced via penalty terms.

    Example:
        >>> model = PINNsModel(hidden_layers=[10, 10, 10, 10])
        >>> model.build()
        >>> history = model.fit(problem, epochs=1000)
        >>> u_pred = model.predict(x_test)
    """

    def __init__(
        self,
        hidden_layers: List[int] = [10, 10, 10, 10],
        activation: str = "tanh",
        dtype: str = "float64",
        seed: int = 1234,
        n_collocation: int = 1000,
        bc_weight: float = 1.0,
        backend: str = "tensorflow",
    ):
        """Initialize PINNs model.

        Args:
            hidden_layers: Number of neurons in each hidden layer.
            activation: Activation function for hidden layers.
            dtype: Floating point precision.
            seed: Random seed for reproducibility.
            n_collocation: Number of collocation points for PDE residual.
            bc_weight: Weight for boundary condition penalty.
            backend: Deep learning backend ('tensorflow', 'jax', or 'pytorch').
        """
        super().__init__(hidden_layers, activation, dtype, seed)
        self.n_collocation = n_collocation
        self.bc_weight = bc_weight
        self.backend = backend
        self._backend_module = None

    def build(self, input_dim: int = 1, output_dim: int = 1) -> None:
        """Build the neural network architecture.

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
        """Build model using TensorFlow/Keras."""
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

        # Output layer (no activation)
        outputs = keras.layers.Dense(output_dim, activation=None, dtype=self.dtype)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs)
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

        layers.append(nn.Linear(prev_dim, output_dim))

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
        """Train the model using collocation method.

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
        """Train using TensorFlow."""
        import tensorflow as tf
        import keras

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        domain = problem.domain

        @tf.function
        def train_step():
            # Random collocation points
            x = tf.random.uniform(
                [self.n_collocation, 1],
                minval=domain[0],
                maxval=domain[1],
                dtype=self.dtype,
            )

            with tf.GradientTape() as tape:
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch(x)
                    with tf.GradientTape() as t2:
                        t2.watch(x)
                        u = self._model(x, training=True)
                    du_dx = t2.gradient(u, x)
                d2u_dx2 = t1.gradient(du_dx, x)

                # PDE residual: u'' + f = 0
                f = problem.forcing_fn(x)
                residual = d2u_dx2 + f
                pde_loss = tf.reduce_mean(residual ** 2)

                # Boundary conditions
                x_bc = tf.constant([[domain[0]], [domain[1]]], dtype=self.dtype)
                u_bc = self._model(x_bc, training=True)
                bc_loss = tf.reduce_mean(u_bc ** 2)

                loss = pde_loss + self.bc_weight * bc_loss

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
        """Train using JAX."""
        import jax
        import jax.numpy as jnp
        from jax import grad, jit, random, vmap
        import optax

        params = self._model
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        domain = problem.domain

        def forward(params, x):
            for i, layer in enumerate(params[:-1]):
                x = jnp.tanh(x @ layer["w"] + layer["b"])
            x = x @ params[-1]["w"] + params[-1]["b"]
            return x

        def loss_fn(params, key):
            # Random collocation points
            x = random.uniform(key, (self.n_collocation, 1), minval=domain[0], maxval=domain[1])

            # Compute second derivative using vmap
            def u_scalar(xi):
                return forward(params, xi.reshape(1, -1))[0, 0]

            du_dx = vmap(grad(u_scalar))
            d2u_dx2 = vmap(grad(grad(u_scalar)))

            residual = d2u_dx2(x[:, 0]) + problem.forcing_fn(x[:, 0])
            pde_loss = jnp.mean(residual ** 2)

            # Boundary conditions
            x_bc = jnp.array([[domain[0]], [domain[1]]])
            u_bc = forward(params, x_bc)
            bc_loss = jnp.mean(u_bc ** 2)

            return pde_loss + self.bc_weight * bc_loss

        @jit
        def train_step(params, opt_state, key):
            loss, grads = jax.value_and_grad(loss_fn)(params, key)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        key = random.PRNGKey(self.seed)
        self._history = {"loss": []}

        for epoch in range(epochs):
            key, subkey = random.split(key)
            params, opt_state, loss = train_step(params, opt_state, subkey)
            self._history["loss"].append(float(loss))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}")

        self._model = params
        return self._history

    def _fit_pytorch(
        self, problem: Any, epochs: int, learning_rate: float, verbose: bool
    ) -> Dict[str, List[float]]:
        """Train using PyTorch."""
        import torch

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        domain = problem.domain
        dtype = torch.float64 if self.dtype == "float64" else torch.float32

        self._history = {"loss": []}

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Random collocation points
            x = torch.rand(self.n_collocation, 1, dtype=dtype, requires_grad=True)
            x = x * (domain[1] - domain[0]) + domain[0]

            u = self._model(x)

            # Compute second derivative
            du_dx = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]
            d2u_dx2 = torch.autograd.grad(
                du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
            )[0]

            # PDE residual
            f = problem.forcing_fn(x.detach().numpy())
            f = torch.tensor(f, dtype=dtype)
            residual = d2u_dx2 + f
            pde_loss = torch.mean(residual ** 2)

            # Boundary conditions
            x_bc = torch.tensor([[domain[0]], [domain[1]]], dtype=dtype)
            u_bc = self._model(x_bc)
            bc_loss = torch.mean(u_bc ** 2)

            loss = pde_loss + self.bc_weight * bc_loss
            loss.backward()
            optimizer.step()

            self._history["loss"].append(float(loss))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}")

        return self._history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the trained model at given points.

        Args:
            x: Input points of shape (n_points,) or (n_points, 1).

        Returns:
            Predicted solution values.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if self.backend == "tensorflow":
            return self._model(x, training=False).numpy().flatten()
        elif self.backend == "jax":
            import jax.numpy as jnp

            def forward(params, x):
                for layer in params[:-1]:
                    x = jnp.tanh(x @ layer["w"] + layer["b"])
                return x @ params[-1]["w"] + params[-1]["b"]

            return np.array(forward(self._model, x)).flatten()
        elif self.backend == "pytorch":
            import torch

            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_tensor = torch.tensor(x, dtype=dtype)
            with torch.no_grad():
                return self._model(x_tensor).numpy().flatten()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
