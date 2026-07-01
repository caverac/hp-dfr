"""Deep Fourier Residual (DFR) method implementation."""

import os
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from hp_dfr.fourier.transforms import dst_matrix_nd_full, h_minus_1_weights
from hp_dfr.models.base import BaseModel, Problem
from hp_dfr.utils import console

# Optional deep-learning backends, imported lazily (guarded) since only one is
# needed at a time and each is heavy.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
try:
    import keras
    import tensorflow as tf
except ImportError:  # backend not installed
    keras = None
    tf = None

try:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import grad, jit, random, vmap
except ImportError:  # backend not installed
    jax = None
    jnp = None
    optax = None
    grad = jit = random = vmap = None

try:
    import torch
    from torch import nn
except ImportError:  # backend not installed
    torch = None
    nn = None

ArrayF = NDArray[np.floating]


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
        hidden_layers: Tuple[int, ...] = (10, 10, 10, 10),
        activation: str = "tanh",
        dtype: str = "float64",
        seed: int = 1234,
        n_quadrature: int = 100,
        n_fourier_modes: int = 10,
        backend: str = "tensorflow",
        dim: int = 1,
    ):
        """Initialize DFR model.

        Args:
            hidden_layers: Number of neurons in each hidden layer.
            activation: Activation function for hidden layers.
            dtype: Floating point precision.
            seed: Random seed for reproducibility.
            n_quadrature: Number of quadrature points (per axis) for integration.
            n_fourier_modes: Number of Fourier modes (per axis) for H^-1 norm.
            backend: Deep learning backend ('tensorflow', 'jax', or 'pytorch').
            dim: Spatial dimension of the problem (1 for an interval, 2 for a box).
        """
        super().__init__(hidden_layers=hidden_layers, activation=activation, dtype=dtype, seed=seed)
        self.n_quadrature = n_quadrature
        self.n_fourier_modes = n_fourier_modes
        self.backend = backend
        self.dim = dim
        # Precomputed arrays / backend handles, set lazily in build()/fit().
        self._backend_module: object | None = None
        self._domain: Optional[Tuple[float, float]] = None
        self._dst_matrix: Optional[ArrayF] = None
        self._dct_matrix: Optional[ArrayF] = None
        self._h_minus_1_weights: Optional[ArrayF] = None
        self._quadrature_pts: Optional[ArrayF] = None
        self._raw_model: object | None = None
        # nD (dim >= 2) path: full-tensor DST machinery shared with the
        # goal-oriented model, set lazily in _setup_fourier_nd().
        self._nd_domain: Optional[Tuple[Tuple[float, float], ...]] = None
        self._nd_quad_points: Optional[ArrayF] = None

    def _compute_transform_matrices(self, domain: tuple) -> None:
        """Precompute DST and DCT matrices for Fourier transforms.

        Args:
            domain: Tuple (a, b) specifying the spatial domain.
        """
        self._domain = domain
        a, b = domain
        L = b - a

        # Quadrature points (midpoint rule)
        pts = np.linspace(a, b, self.n_quadrature + 1)
        h = np.abs(pts[1:] - pts[:-1])
        self._quadrature_pts = pts[:-1] + h / 2

        # H^-1 norm weights: (1 + k^2 * pi^2 / L^2)^-0.5
        k = np.arange(1, self.n_fourier_modes + 1)
        self._h_minus_1_weights = ((np.pi**2 * k**2) / L**2) ** -0.5

        # DST matrix (Discrete Sine Transform)
        V = np.sqrt(2.0 / L)
        self._dst_matrix = np.array(
            [V * np.sin(np.pi * ki * (self._quadrature_pts - a) / L) * h for ki in range(1, self.n_fourier_modes + 1)]
        )

        # DCT matrix with derivative (for gradient terms)
        self._dct_matrix = np.array(
            [V * (ki * np.pi / L) * np.cos(np.pi * ki * (self._quadrature_pts - a) / L) * h for ki in range(1, self.n_fourier_modes + 1)]
        )

    def build(self, input_dim: Optional[int] = None, output_dim: int = 1) -> None:
        """Build the neural network with cutoff layer.

        The cutoff layer enforces homogeneous Dirichlet boundary conditions
        by multiplying the network output by the product of (x_d - a_d)(b_d - x_d)
        over the spatial axes.

        Args:
            input_dim: Dimension of input (spatial coordinates). Defaults to the
                model's ``dim``.
            output_dim: Dimension of output (solution components).
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
        """Build model using TensorFlow/Keras with cutoff layer."""
        if keras is None:
            raise RuntimeError("TensorFlow backend requires tensorflow and keras.")

        keras.utils.set_random_seed(self.seed)
        keras.backend.set_floatx(self.dtype)

        # Input layer
        inputs = keras.layers.Input(shape=(input_dim,), dtype=self.dtype)

        # Hidden layers
        x = inputs
        for neurons in self.hidden_layers:
            x = keras.layers.Dense(neurons, activation=self.activation, dtype=self.dtype)(x)

        # Linear output layer (no activation: the solution value is unbounded).
        x = keras.layers.Dense(output_dim, activation=None, dtype=self.dtype)(x)

        # Store raw model (without cutoff) for internal use
        self._raw_model = keras.Model(inputs=inputs, outputs=x)
        self._model = self._raw_model
        self._backend_module = keras

    def _build_jax(self, input_dim: int, output_dim: int) -> None:
        """Build model using JAX."""
        if jax is None:
            raise RuntimeError("JAX backend requires jax.")

        jax.config.update("jax_enable_x64", self.dtype == "float64")

        layer_sizes = [input_dim, *self.hidden_layers, output_dim]
        key = random.PRNGKey(self.seed)

        params = []
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, w_key, _b_key = random.split(key, 3)
            w = random.normal(w_key, (m, n)) * jnp.sqrt(2.0 / m)
            b = jnp.zeros(n)
            params.append({"w": w, "b": b})

        self._model = params
        self._backend_module = jax

    def _build_pytorch(self, input_dim: int, output_dim: int) -> None:
        """Build model using PyTorch."""
        if torch is None:
            raise RuntimeError("PyTorch backend requires torch.")

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

        # Linear output layer: the solution value is unbounded, so no output
        # activation (a bounded activation would cap the representable amplitude).
        layers.append(nn.Linear(prev_dim, output_dim))

        model = nn.Sequential(*layers)
        if self.dtype == "float64":
            model = model.double()
        self._model = model

        self._backend_module = torch

    def fit(
        self,
        problem: Problem,
        epochs: int = 1000,
        learning_rate: float = 1e-3,
        verbose: bool = True,
        lbfgs_iters: int = 0,
    ) -> Dict[str, List[float]]:
        """Train the model using DFR method.

        Args:
            problem: Problem instance with domain, forcing_fn, and boundary conditions.
            epochs: Number of Adam iterations.
            learning_rate: Learning rate for Adam.
            verbose: Whether to print progress.
            lbfgs_iters: If > 0, run an LBFGS polishing phase for this many
                iterations after Adam (PyTorch backend only).

        Returns:
            Dictionary containing training history.
        """
        if self._model is None:
            self.build()

        if self.dim >= 2:
            return self._fit_nd(problem, epochs, learning_rate, verbose, lbfgs_iters)

        # Precompute transform matrices
        self._compute_transform_matrices(problem.domain)

        if self.backend == "tensorflow":
            return self._fit_tensorflow(problem, epochs, learning_rate, verbose)
        if self.backend == "jax":
            return self._fit_jax(problem, epochs, learning_rate, verbose)
        if self.backend == "pytorch":
            return self._fit_pytorch(problem, epochs, learning_rate, verbose, lbfgs_iters)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _fit_nd(
        self,
        problem: Problem,
        epochs: int,
        learning_rate: float,
        verbose: bool,
        lbfgs_iters: int,
    ) -> Dict[str, List[float]]:
        """Dispatch the dim >= 2 DFR training to the requested backend."""
        self._setup_fourier_nd(problem.domain)
        if self.backend == "tensorflow":
            return self._fit_tensorflow_nd(problem, epochs, learning_rate, verbose)
        if self.backend == "pytorch":
            return self._fit_pytorch_nd(problem, epochs, learning_rate, verbose, lbfgs_iters)
        raise ValueError(f"Backend {self.backend} not implemented for dim >= 2")

    def _fit_tensorflow(self, problem: Problem, epochs: int, learning_rate: float, verbose: bool) -> Dict[str, List[float]]:
        """Train using TensorFlow with DFR loss."""
        if keras is None:
            raise RuntimeError("TensorFlow backend requires tensorflow and keras.")
        assert self._model is not None
        assert self._quadrature_pts is not None and self._dct_matrix is not None
        assert self._h_minus_1_weights is not None and self._dst_matrix is not None
        model = cast("keras.Model", self._model)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        domain = problem.domain
        a, b = domain

        # Convert matrices to tensors
        pts = tf.constant(self._quadrature_pts.reshape(-1, 1), dtype=self.dtype)
        dct_matrix = tf.constant(self._dct_matrix, dtype=self.dtype)
        weights = tf.constant(self._h_minus_1_weights, dtype=self.dtype)

        # Precompute forcing term transform
        f_vals = problem.forcing_fn(self._quadrature_pts)
        ft_forcing = tf.constant(
            np.einsum("ji,i->j", self._dst_matrix, f_vals),
            dtype=self.dtype,
        )

        @tf.function
        def train_step() -> tf.Tensor:
            with tf.GradientTape() as tape:
                # Compute u and its derivative at quadrature points
                with tf.GradientTape() as t1:
                    t1.watch(pts)
                    # Apply cutoff layer: u = (x - a)(b - x) * nn(x)
                    cutoff = (pts - a) * (b - pts)
                    nn_out = model(pts, training=True)
                    u = cutoff * nn_out
                du = t1.gradient(u, pts)

                # Fourier transform of gradient term
                ft_gradient = tf.einsum("ji,ij->j", dct_matrix, du)

                # Total Fourier coefficients (weak residual)
                ft_total = (ft_gradient - ft_forcing) * weights

                # H^-1 norm squared (DFR loss)
                loss = tf.reduce_sum(ft_total**2)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        self._history = {"loss": []}

        for epoch in range(epochs):
            loss = train_step()
            self._history["loss"].append(float(loss))

            if verbose and (epoch + 1) % 100 == 0:
                console.print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}", markup=False)

        return self._history

    def _fit_jax(self, problem: Problem, epochs: int, learning_rate: float, verbose: bool) -> Dict[str, List[float]]:
        """Train using JAX with DFR loss."""
        if jax is None:
            raise RuntimeError("JAX backend requires jax and optax.")
        assert self._quadrature_pts is not None and self._dct_matrix is not None
        assert self._h_minus_1_weights is not None and self._dst_matrix is not None

        params = cast(list, self._model)
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

        def forward(params: list, x: "jax.Array") -> "jax.Array":
            for layer in params[:-1]:
                x = jnp.tanh(x @ layer["w"] + layer["b"])
            # Last layer with activation for DFR
            x = jnp.tanh(x @ params[-1]["w"] + params[-1]["b"])
            return x

        def forward_with_cutoff(params: list, x: "jax.Array") -> "jax.Array":
            cutoff = (x - a) * (b - x)
            return cutoff * forward(params, x)

        def loss_fn(params: list) -> "jax.Array":
            # Compute derivatives using vmap
            def u_scalar(xi: "jax.Array") -> "jax.Array":
                return forward_with_cutoff(params, xi.reshape(1, -1))[0, 0]

            du_dx = vmap(grad(u_scalar))
            du = du_dx(pts[:, 0])

            # Fourier transform
            ft_gradient = jnp.einsum("ji,i->j", dct_matrix, du)
            ft_total = (ft_gradient - ft_forcing) * weights

            return jnp.sum(ft_total**2)

        @jit
        def train_step(params: list, opt_state: "optax.OptState") -> tuple:
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        self._history = {"loss": []}

        for epoch in range(epochs):
            params, opt_state, loss = train_step(params, opt_state)
            self._history["loss"].append(float(loss))

            if verbose and (epoch + 1) % 100 == 0:
                console.print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}", markup=False)

        self._model = params
        return self._history

    def _fit_pytorch(
        self,
        problem: Problem,
        epochs: int,
        learning_rate: float,
        verbose: bool,
        lbfgs_iters: int = 0,
    ) -> Dict[str, List[float]]:
        """Train using PyTorch with DFR loss (Adam, optional LBFGS polish)."""
        if torch is None:
            raise RuntimeError("PyTorch backend requires torch.")
        assert self._model is not None
        assert self._quadrature_pts is not None and self._dct_matrix is not None
        assert self._h_minus_1_weights is not None and self._dst_matrix is not None
        model = cast("torch.nn.Module", self._model)

        domain = problem.domain
        a, b = domain
        dtype = torch.float64 if self.dtype == "float64" else torch.float32

        pts = torch.tensor(self._quadrature_pts.reshape(-1, 1), dtype=dtype, requires_grad=True)
        dct_matrix = torch.tensor(self._dct_matrix, dtype=dtype)
        weights = torch.tensor(self._h_minus_1_weights, dtype=dtype)

        # Precompute forcing term transform
        f_vals = problem.forcing_fn(self._quadrature_pts)
        ft_forcing = torch.tensor(np.einsum("ji,i->j", self._dst_matrix, f_vals), dtype=dtype)

        def compute_loss() -> torch.Tensor:
            # Fresh leaf tensor so the input-gradient graph is rebuilt each call.
            x = pts.detach().clone().requires_grad_(True)
            cutoff = (x - a) * (b - x)
            u = cutoff * model(x)
            du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            ft_gradient = torch.einsum("ji,ij->j", dct_matrix, du)
            ft_total = (ft_gradient - ft_forcing) * weights
            return torch.sum(ft_total**2)

        self._history = {"loss": []}

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()
            self._history["loss"].append(float(loss))
            if verbose and (epoch + 1) % 100 == 0:
                console.print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}", markup=False)

        if lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                max_iter=lbfgs_iters,
                line_search_fn="strong_wolfe",
                history_size=50,
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                loss = compute_loss()
                loss.backward()
                return loss

            lbfgs.step(closure)
            final = float(compute_loss())
            self._history["loss"].append(final)
            if verbose:
                console.print(f"LBFGS polish - Loss: {final:.6e}", markup=False)

        return self._history

    # ------------------------------------------------------------------
    # nD (dim >= 2) DFR path. The discretization matches the goal-oriented
    # model's primal half exactly -- full-tensor DST of the weak residual
    # ``f + Delta u`` with an autodiff Laplacian and a product cutoff -- so the
    # plain baseline and the goal-oriented method differ only by the goal term.
    # ------------------------------------------------------------------
    def _setup_fourier_nd(self, domain: Tuple[Tuple[float, float], ...]) -> None:
        """Precompute the full-tensor DST matrix, H^-1 weights, and quadrature."""
        self._nd_domain = domain
        grid_shape = tuple([self.n_quadrature] * self.dim)
        self._dst_matrix = dst_matrix_nd_full(grid_shape, domain, self.n_fourier_modes)
        self._h_minus_1_weights = h_minus_1_weights(tuple([self.n_fourier_modes] * self.dim), domain).ravel()

        quad_1d = []
        for d in range(self.dim):
            a, b = domain[d]
            x = np.linspace(a, b, self.n_quadrature + 1)
            dx = x[1] - x[0]
            quad_1d.append(x[:-1] + dx / 2)
        grids = np.meshgrid(*quad_1d, indexing="ij")
        self._nd_quad_points = np.stack([g.ravel() for g in grids], axis=1)

    def _cutoff_torch(self, x: "torch.Tensor", domain: Tuple[Tuple[float, float], ...], dtype: "torch.dtype") -> "torch.Tensor":
        """Product cutoff enforcing homogeneous Dirichlet conditions (PyTorch)."""
        result = torch.ones((x.shape[0], 1), dtype=dtype)
        for d in range(self.dim):
            lo, hi = domain[d]
            result = result * (x[:, d : d + 1] - lo) * (hi - x[:, d : d + 1])
        return result

    def _laplacian_torch(
        self,
        x: "torch.Tensor",
        model: "torch.nn.Module",
        domain: Tuple[Tuple[float, float], ...],
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Laplacian of the cutoff-enforced network via autodiff (PyTorch)."""
        cutoff = self._cutoff_torch(x, domain, dtype)
        u = cutoff * model(x)
        laplacian = torch.zeros(x.shape[0], dtype=dtype)
        for d in range(self.dim):
            du_d = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0][:, d]
            d2u_d = torch.autograd.grad(du_d.sum(), x, create_graph=True, retain_graph=True)[0][:, d]
            laplacian = laplacian + d2u_d
        return laplacian

    def _cutoff_tf(self, x: "tf.Tensor", domain: Tuple[Tuple[float, float], ...]) -> "tf.Tensor":
        """Product cutoff enforcing homogeneous Dirichlet conditions (TensorFlow)."""
        result = tf.ones((tf.shape(x)[0], 1), dtype=self.dtype)
        for d in range(self.dim):
            lo, hi = domain[d]
            result = result * (x[:, d : d + 1] - lo) * (hi - x[:, d : d + 1])
        return result

    def _laplacian_tf(self, x: "tf.Tensor", model: "keras.Model", domain: Tuple[Tuple[float, float], ...]) -> "tf.Tensor":
        """Laplacian of the cutoff-enforced network via autodiff (TensorFlow)."""
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

    def _fit_pytorch_nd(
        self,
        problem: Problem,
        epochs: int,
        learning_rate: float,
        verbose: bool,
        lbfgs_iters: int,
    ) -> Dict[str, List[float]]:
        """Train the nD DFR model using PyTorch (Adam, optional LBFGS polish)."""
        if torch is None:
            raise RuntimeError("PyTorch backend requires torch.")
        assert self._model is not None
        assert self._dst_matrix is not None and self._h_minus_1_weights is not None
        assert self._nd_domain is not None and self._nd_quad_points is not None
        model = cast("torch.nn.Module", self._model)
        domain = self._nd_domain
        dtype = torch.float64 if self.dtype == "float64" else torch.float32

        pts = torch.tensor(self._nd_quad_points, dtype=dtype, requires_grad=True)
        dst_matrix = torch.tensor(self._dst_matrix, dtype=dtype)
        weights = torch.tensor(self._h_minus_1_weights, dtype=dtype)

        f_vals = problem.forcing_fn(self._nd_quad_points)
        if f_vals.ndim > 1:
            f_vals = f_vals.flatten()
        f_vals_t = torch.tensor(f_vals, dtype=dtype)

        def compute_loss() -> "torch.Tensor":
            x = pts.detach().clone().requires_grad_(True)
            laplacian_u = self._laplacian_torch(x, model, domain, dtype)
            weak_residual = f_vals_t + laplacian_u  # f + Delta u (PDE: -Delta u = f)
            ft_total = torch.mv(dst_matrix, weak_residual) * weights
            return torch.sum(ft_total**2)

        self._history = {"loss": []}

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()
            self._history["loss"].append(float(loss))
            if verbose and (epoch + 1) % 100 == 0:
                console.print(f"Epoch {epoch + 1}/{epochs} - Loss: {float(loss):.6e}", markup=False)

        if lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                max_iter=lbfgs_iters,
                line_search_fn="strong_wolfe",
                history_size=50,
            )

            def closure() -> "torch.Tensor":
                lbfgs.zero_grad()
                loss = compute_loss()
                loss.backward()
                return loss

            lbfgs.step(closure)
            final = float(compute_loss())
            self._history["loss"].append(final)
            if verbose:
                console.print(f"LBFGS polish - Loss: {final:.6e}", markup=False)

        return self._history

    def _fit_tensorflow_nd(self, problem: Problem, epochs: int, learning_rate: float, verbose: bool) -> Dict[str, List[float]]:
        """Train the nD DFR model using TensorFlow."""
        if keras is None:
            raise RuntimeError("TensorFlow backend requires tensorflow and keras.")
        assert self._model is not None
        assert self._dst_matrix is not None and self._h_minus_1_weights is not None
        assert self._nd_domain is not None and self._nd_quad_points is not None
        model = cast("keras.Model", self._model)
        domain = self._nd_domain

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        pts = tf.constant(self._nd_quad_points, dtype=self.dtype)
        dst_matrix = tf.constant(self._dst_matrix, dtype=self.dtype)
        weights = tf.constant(self._h_minus_1_weights, dtype=self.dtype)

        f_vals = problem.forcing_fn(self._nd_quad_points)
        if f_vals.ndim > 1:
            f_vals = f_vals.flatten()
        f_vals_t = tf.constant(f_vals, dtype=self.dtype)

        @tf.function
        def train_step() -> "tf.Tensor":
            with tf.GradientTape() as tape:
                laplacian_u = tf.reshape(self._laplacian_tf(pts, model, domain), [-1])
                weak_residual = f_vals_t + laplacian_u  # f + Delta u (PDE: -Delta u = f)
                ft_total = tf.linalg.matvec(dst_matrix, weak_residual) * weights
                loss = tf.reduce_sum(ft_total**2)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        self._history = {"loss": []}
        for epoch in range(epochs):
            loss = train_step()
            self._history["loss"].append(float(loss))
            if verbose and (epoch + 1) % 100 == 0:
                console.print(f"Epoch {epoch + 1}/{epochs} - Loss: {float(loss):.6e}", markup=False)

        return self._history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the trained model (with cutoff) at given points.

        Args:
            x: Input points of shape (n_points,) or (n_points, dim).

        Returns:
            Predicted solution values with boundary conditions enforced.
        """
        if self.dim >= 2:
            return self._predict_nd(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Cutoff must use the same domain the model was trained on.
        if self._domain is None:
            raise ValueError("Model not trained yet; call fit() before predict().")
        a, b = self._domain

        if self.backend == "tensorflow":
            model = cast("keras.Model", self._model)
            cutoff = (x - a) * (b - x)
            nn_out = model(x, training=False).numpy()
            return cast(np.ndarray, (cutoff * nn_out).flatten())
        if self.backend == "jax":

            def forward(params: list, x: "jax.Array") -> "jax.Array":
                for layer in params[:-1]:
                    x = jnp.tanh(x @ layer["w"] + layer["b"])
                return jnp.tanh(x @ params[-1]["w"] + params[-1]["b"])

            cutoff = (x - a) * (b - x)
            nn_out = np.array(forward(cast(list, self._model), x))
            return cast(np.ndarray, (cutoff * nn_out).flatten())
        if self.backend == "pytorch":
            model = cast("torch.nn.Module", self._model)
            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_tensor = torch.tensor(x, dtype=dtype)
            cutoff = (x_tensor - a) * (b - x_tensor)
            with torch.no_grad():
                nn_out = model(x_tensor)
            return cast(np.ndarray, (cutoff * nn_out).numpy().flatten())
        raise ValueError(f"Unknown backend: {self.backend}")

    def _predict_nd(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the trained nD model (with product cutoff) at points (N, dim)."""
        x = np.atleast_2d(x)
        if self._nd_domain is None or self._model is None:
            raise ValueError("Model not trained yet; call fit() before predict().")
        domain = self._nd_domain

        cutoff = np.ones((x.shape[0], 1))
        for d in range(self.dim):
            lo, hi = domain[d]
            cutoff *= (x[:, d : d + 1] - lo) * (hi - x[:, d : d + 1])

        if self.backend == "tensorflow":
            model = cast("keras.Model", self._model)
            nn_out = model(x, training=False).numpy()
            return cast(np.ndarray, (cutoff * nn_out).flatten())
        if self.backend == "pytorch":
            model = cast("torch.nn.Module", self._model)
            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_tensor = torch.tensor(x, dtype=dtype)
            with torch.no_grad():
                nn_out = model(x_tensor).numpy()
            return cast(np.ndarray, (cutoff * nn_out).flatten())
        raise ValueError(f"Backend {self.backend} not implemented for dim >= 2")
