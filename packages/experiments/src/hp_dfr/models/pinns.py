# pylint: disable=import-outside-toplevel, import-error
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportCallIssue=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
# type: ignore[attr-defined]
"""Physics-Informed Neural Networks (PINNs) implementation.

This module provides a multi-backend PINNs implementation supporting TensorFlow,
JAX, and PyTorch. The backends are optional dependencies - only the backend you
choose to use needs to be installed.

Linter Exceptions
-----------------
The pylint and pyright directives at the top of this file are required because:

1. **Optional backends**: TensorFlow, JAX, and PyTorch are optional dependencies.
   Users only need to install the backend they intend to use. This means imports
   like ``import keras`` or ``import torch`` may fail on systems where that
   backend is not installed.

2. **Deferred imports**: Backend imports are done inside methods (not at module
   level) to avoid import errors when the backend is not installed. This triggers
   ``import-outside-toplevel`` warnings.

3. **Missing type stubs**: These deep learning frameworks either lack type stubs
   or have incomplete stubs, causing ``reportUnknownMemberType`` and similar
   warnings from Pyright.

Without these exceptions, the linter would report errors for any backend that
is not currently installed, even though the code is correct and will work
when the appropriate backend is available.

Examples
--------
>>> from hp_dfr.models.pinns import PINNsModel
>>> from hp_dfr.problems import get_problem
>>> problem = get_problem("sine")
>>> model = PINNsModel(backend="tensorflow")
>>> model.build()
>>> history = model.fit(problem, epochs=1000)
"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from hp_dfr.models.base import BaseModel
from hp_dfr.problems.poisson_1d import Poisson1D


class PINNsModel(BaseModel):
    """Physics-Informed Neural Network using the collocation method.

    This implementation uses strong-form PDE residuals as the loss function,
    with boundary conditions enforced via penalty terms. The model supports
    multiple deep learning backends (TensorFlow, JAX, PyTorch).

    Parameters
    ----------
    hidden_layers : tuple[int, ...], optional
        Number of neurons in each hidden layer, by default (10, 10, 10, 10).
    activation : str, optional
        Activation function for hidden layers ('tanh', 'relu'), by default 'tanh'.
    dtype : str, optional
        Floating point precision ('float32', 'float64'), by default 'float64'.
    seed : int, optional
        Random seed for reproducibility, by default 1234.
    n_collocation : int, optional
        Number of collocation points for PDE residual evaluation, by default 1000.
    bc_weight : float, optional
        Weight for boundary condition penalty term, by default 1.0.
    backend : {'tensorflow', 'jax', 'pytorch'}, optional
        Deep learning backend to use, by default 'tensorflow'.

    Attributes
    ----------
    n_collocation : int
        Number of collocation points.
    bc_weight : float
        Boundary condition penalty weight.
    backend : str
        Selected deep learning backend.
    history : dict[str, list[float]]
        Training history with loss values.

    Notes
    -----
    The collocation method minimizes the PDE residual at randomly sampled
    interior points. For the Poisson equation -u'' = f, the residual is:

        R(x) = u''(x) + f(x)

    The total loss combines the PDE residual and boundary condition penalty:

        L = mean(R^2) + bc_weight * mean(u_bc^2)

    Examples
    --------
    >>> model = PINNsModel(hidden_layers=(64, 64, 64), backend="tensorflow")
    >>> model.build()
    >>> history = model.fit(problem, epochs=5000)
    >>> u_pred = model.predict(x_test)
    """

    def __init__(
        self,
        /,
        *,
        hidden_layers: tuple[int, ...] = (10, 10, 10, 10),
        activation: Literal["tanh", "relu"] = "tanh",
        dtype: str = "float64",
        seed: int = 1234,
        n_collocation: int = 1000,
        bc_weight: float = 1.0,
        backend: Literal["tensorflow", "jax", "pytorch"] = "tensorflow",
    ):
        """Initialize PINNs model.

        See class docstring for parameter descriptions.
        """
        super().__init__(hidden_layers=hidden_layers, activation=activation, dtype=dtype, seed=seed)
        self.n_collocation = n_collocation
        self.bc_weight = bc_weight
        self.backend = backend
        self._backend_module: Any = None

    def build(self, input_dim: int = 1, output_dim: int = 1) -> None:
        """Build the neural network architecture.

        Constructs the neural network using the selected backend. This method
        must be called before ``fit()`` or ``predict()``.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of input (spatial coordinates), by default 1.
        output_dim : int, optional
            Dimension of output (solution components), by default 1.

        Raises
        ------
        ValueError
            If an unknown backend is specified.
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
        """Build model using TensorFlow/Keras.

        Creates a fully-connected neural network using the Keras functional API.
        Sets the Keras backend to TensorFlow and configures the random seed and
        floating point precision.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        output_dim : int
            Output dimension.
        """
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
        """Build model using JAX.

        Creates network parameters as a list of weight/bias dictionaries.
        Uses He initialization for weights and zeros for biases. Configures
        JAX for 64-bit precision if requested.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        output_dim : int
            Output dimension.
        """
        import jax
        import jax.numpy as jnp
        from jax import random

        jax.config.update("jax_enable_x64", self.dtype == "float64")

        layer_sizes = [input_dim, *self.hidden_layers, output_dim]
        key = random.PRNGKey(self.seed)

        params = []
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, w_key, _ = random.split(key, 3)
            w = random.normal(w_key, (m, n)) * jnp.sqrt(2.0 / m)
            b = jnp.zeros(n)
            params.append({"w": w, "b": b})

        self._model = params  # type: ignore
        self._backend_module = jax

    def _build_pytorch(self, input_dim: int, output_dim: int) -> None:
        """Build model using PyTorch.

        Creates a Sequential model with Linear layers and activation functions.
        Converts to double precision if dtype is 'float64'.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        output_dim : int
            Output dimension.
        """
        import torch
        from torch import nn

        torch.manual_seed(self.seed)

        layers: list[Any] = []
        prev_dim = input_dim

        for neurons in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, neurons))
            if self.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.activation == "relu":
                layers.append(nn.ReLU())
            prev_dim = neurons

        layers.append(nn.Linear(prev_dim, output_dim))

        self._model = nn.Sequential(*layers)  # type: ignore
        if self.dtype == "float64":
            self._model = self._model.double()  # type: ignore[attr-defined]

        self._backend_module = torch

    def fit(
        self,
        problem: Poisson1D[Any],
        epochs: int = 1000,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the model using the collocation method.

        Minimizes the PDE residual at randomly sampled collocation points plus
        a boundary condition penalty term. Uses the Adam optimizer.

        Parameters
        ----------
        problem : Poisson1D[Any]
            Problem instance defining the PDE (domain, forcing function, BCs).
        epochs : int, optional
            Number of training iterations, by default 1000.
        learning_rate : float, optional
            Learning rate for the Adam optimizer, by default 1e-3.
        verbose : bool, optional
            Whether to print progress every 100 epochs, by default True.

        Returns
        -------
        dict[str, list[float]]
            Training history with key 'loss' containing per-epoch loss values.

        Raises
        ------
        ValueError
            If an unknown backend is specified.
        """
        if self._model is None:
            self.build()

        if self.backend == "tensorflow":
            return self._fit_tensorflow(problem, epochs, learning_rate, verbose)
        if self.backend == "jax":
            return self._fit_jax(problem, epochs, learning_rate, verbose)
        if self.backend == "pytorch":
            return self._fit_pytorch(problem, epochs, learning_rate, verbose)

        raise ValueError(f"Unknown backend: {self.backend}")

    def _fit_tensorflow(self, problem: Poisson1D[Any], epochs: int, learning_rate: float, verbose: bool) -> dict[str, list[float]]:
        """Train using TensorFlow.

        Uses tf.GradientTape for automatic differentiation to compute second
        derivatives of the network output. The training step is compiled with
        @tf.function for performance.

        Parameters
        ----------
        problem : Poisson1D[Any]
            Problem instance.
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate for Adam optimizer.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        dict[str, list[float]]
            Training history.
        """
        import tensorflow as tf
        import keras

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        domain = problem.domain

        @tf.function
        def train_step() -> Any:
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
                pde_loss = tf.reduce_mean(residual**2)

                # Boundary conditions
                x_bc = tf.constant([[domain[0]], [domain[1]]], dtype=self.dtype)
                u_bc = self._model(x_bc, training=True)
                bc_loss = tf.reduce_mean(u_bc**2)

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

    def _fit_jax(self, problem: Poisson1D[Any], epochs: int, learning_rate: float, verbose: bool) -> dict[str, list[float]]:
        """Train using JAX.

        Uses JAX's functional autodiff (grad) with vmap for vectorization.
        The training step is JIT-compiled for performance. Uses optax for
        the Adam optimizer.

        Parameters
        ----------
        problem : Poisson1D[Any]
            Problem instance.
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate for Adam optimizer.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        dict[str, list[float]]
            Training history.
        """
        import jax
        import jax.numpy as jnp
        from jax import grad, jit, random, vmap
        import optax

        params = self._model
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        domain = problem.domain

        def forward(params: Any, x: Any) -> Any:
            for layer in params[:-1]:
                x = jnp.tanh(x @ layer["w"] + layer["b"])
            return x @ params[-1]["w"] + params[-1]["b"]

        def loss_fn(params: Any, key: Any) -> Any:
            # Random collocation points
            x = random.uniform(key, (self.n_collocation, 1), minval=domain[0], maxval=domain[1])

            # Compute second derivative using vmap
            def u_scalar(xi: Any) -> Any:
                return forward(params, xi.reshape(1, -1))[0, 0]

            # du_dx = vmap(grad(u_scalar))
            d2u_dx2 = vmap(grad(grad(u_scalar)))

            residual = d2u_dx2(x[:, 0]) + problem.forcing_fn(x[:, 0])
            pde_loss = jnp.mean(residual**2)

            # Boundary conditions
            x_bc = jnp.array([[domain[0]], [domain[1]]])
            u_bc = forward(params, x_bc)
            bc_loss = jnp.mean(u_bc**2)

            return pde_loss + self.bc_weight * bc_loss

        @jit
        def train_step(params: Any, opt_state: Any, key: Any) -> tuple[Any, Any, Any]:
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

    def _fit_pytorch(self, problem: Poisson1D[Any], epochs: int, learning_rate: float, verbose: bool) -> dict[str, list[float]]:
        """Train using PyTorch.

        Uses torch.autograd.grad for computing second derivatives with
        create_graph=True to allow backpropagation through the derivative
        computation.

        Parameters
        ----------
        problem : Poisson1D[Any]
            Problem instance.
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate for Adam optimizer.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        dict[str, list[float]]
            Training history.
        """
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
            du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

            # PDE residual
            f = problem.forcing_fn(x.detach().numpy())
            f = torch.tensor(f, dtype=dtype)
            residual = d2u_dx2 + f
            pde_loss = torch.mean(residual**2)

            # Boundary conditions
            x_bc = torch.tensor([[domain[0]], [domain[1]]], dtype=dtype)
            u_bc = self._model(x_bc)
            bc_loss = torch.mean(u_bc**2)

            loss = pde_loss + self.bc_weight * bc_loss
            loss.backward()
            optimizer.step()

            self._history["loss"].append(float(loss))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}")

        return self._history

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the trained model at given points.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Input points of shape (n_points,) or (n_points, 1).

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted solution values of shape (n_points,).

        Raises
        ------
        ValueError
            If an unknown backend is specified.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if self.backend == "tensorflow":
            return self._model(x, training=False).numpy().flatten()

        if self.backend == "jax":
            import jax.numpy as jnp

            def forward(params: Any, x: Any) -> Any:
                for layer in params[:-1]:
                    x = jnp.tanh(x @ layer["w"] + layer["b"])
                return x @ params[-1]["w"] + params[-1]["b"]

            return np.array(forward(self._model, x)).flatten()

        if self.backend == "pytorch":
            import torch

            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_tensor = torch.tensor(x, dtype=dtype)
            with torch.no_grad():
                return self._model(x_tensor).numpy().flatten()

        raise ValueError(f"Unknown backend: {self.backend}")
