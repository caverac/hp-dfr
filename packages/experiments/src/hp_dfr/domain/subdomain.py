"""Subdomain with local neural network for hp-adaptive DFR.

Each subdomain has its own neural network that approximates
the solution locally. The networks are coupled through
interface conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from hp_dfr.domain.partitioning import BoundingBox


@dataclass
class Subdomain:
    """A subdomain with associated data for hp-DFR.

    Attributes:
        index: Unique subdomain identifier.
        box: Bounding box defining the subdomain geometry.
        level: Refinement level (0 = coarsest).
        neighbors: Indices of neighboring subdomains.
        network: Local neural network (set after initialization).
    """

    index: int
    box: BoundingBox
    level: int = 0
    neighbors: List[int] = field(default_factory=list)
    network: Optional[Any] = None
    _local_error: float = 0.0

    @property
    def dim(self) -> int:
        """Spatial dimension."""
        return self.box.dim

    @property
    def volume(self) -> float:
        """Subdomain volume."""
        return self.box.volume

    @property
    def local_error(self) -> float:
        """Local error indicator."""
        return self._local_error

    @local_error.setter
    def local_error(self, value: float) -> None:
        self._local_error = value

    def contains(self, point: NDArray[np.floating]) -> bool:
        """Check if point is in this subdomain."""
        return self.box.contains(point)

    def quadrature_points(self, n_points: int) -> NDArray[np.floating]:
        """Get quadrature points for this subdomain."""
        return self.box.quadrature_points(n_points)

    def cutoff_function(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute cutoff function for boundary conditions.

        The cutoff enforces u = 0 on the subdomain boundary.
        For interior subdomains, this is relaxed at interfaces.

        Args:
            x: Points of shape (n_points, dim).

        Returns:
            Cutoff values of shape (n_points, 1).
        """
        result = np.ones((x.shape[0], 1))
        for d in range(self.dim):
            lo, hi = self.box.bounds[d]
            result *= (x[:, d:d+1] - lo) * (hi - x[:, d:d+1])
        return result


class SubdomainNetwork:
    """Neural network for a single subdomain.

    Wraps a neural network with subdomain-specific functionality
    including cutoff layers and local loss computation.
    """

    def __init__(
        self,
        subdomain: Subdomain,
        hidden_layers: List[int] = [20, 20, 20],
        activation: str = "tanh",
        backend: str = "tensorflow",
        dtype: str = "float64",
        seed: Optional[int] = None,
    ):
        """Initialize subdomain network.

        Args:
            subdomain: Associated subdomain.
            hidden_layers: Network architecture.
            activation: Activation function.
            backend: Deep learning backend.
            dtype: Data type.
            seed: Random seed.
        """
        self.subdomain = subdomain
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.backend = backend
        self.dtype = dtype
        self.seed = seed if seed is not None else subdomain.index

        self._model = None
        self._optimizer = None
        self._built = False

    def build(self) -> None:
        """Build the neural network."""
        input_dim = self.subdomain.dim
        output_dim = 1

        if self.backend == "tensorflow":
            self._build_tensorflow(input_dim, output_dim)
        elif self.backend == "pytorch":
            self._build_pytorch(input_dim, output_dim)
        elif self.backend == "jax":
            self._build_jax(input_dim, output_dim)

        self._built = True

    def _build_tensorflow(self, input_dim: int, output_dim: int) -> None:
        """Build TensorFlow model."""
        import os
        os.environ["KERAS_BACKEND"] = "tensorflow"
        import keras

        keras.utils.set_random_seed(self.seed)

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

    def _build_pytorch(self, input_dim: int, output_dim: int) -> None:
        """Build PyTorch model."""
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

    def _build_jax(self, input_dim: int, output_dim: int) -> None:
        """Build JAX model (as parameter dict)."""
        import jax.numpy as jnp
        from jax import random

        layer_sizes = [input_dim] + self.hidden_layers + [output_dim]
        key = random.PRNGKey(self.seed)

        params = []
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, w_key, b_key = random.split(key, 3)
            w = random.normal(w_key, (m, n)) * jnp.sqrt(2.0 / m)
            b = jnp.zeros(n)
            params.append({"w": w, "b": b})

        self._model = params

    def forward(
        self,
        x: NDArray[np.floating],
        apply_cutoff: bool = True,
    ) -> NDArray[np.floating]:
        """Forward pass through the network.

        Args:
            x: Input points of shape (n_points, dim).
            apply_cutoff: Whether to apply cutoff for BCs.

        Returns:
            Network output of shape (n_points, 1).
        """
        if not self._built:
            self.build()

        if self.backend == "tensorflow":
            output = self._model(x, training=False).numpy()
        elif self.backend == "pytorch":
            import torch
            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            x_tensor = torch.tensor(x, dtype=dtype)
            with torch.no_grad():
                output = self._model(x_tensor).numpy()
        elif self.backend == "jax":
            import jax.numpy as jnp
            output = x
            for layer in self._model[:-1]:
                output = jnp.tanh(output @ layer["w"] + layer["b"])
            output = jnp.tanh(output @ self._model[-1]["w"] + self._model[-1]["b"])
            output = np.array(output)

        if apply_cutoff:
            cutoff = self.subdomain.cutoff_function(x)
            output = output * cutoff

        return output

    def get_trainable_variables(self) -> List[Any]:
        """Get trainable parameters."""
        if self.backend == "tensorflow":
            return self._model.trainable_variables
        elif self.backend == "pytorch":
            return list(self._model.parameters())
        elif self.backend == "jax":
            return self._model
        return []


class SubdomainCollection:
    """Collection of subdomains with their networks.

    Manages multiple subdomains and provides methods for
    global operations across all subdomains.
    """

    def __init__(
        self,
        subdomains: List[Subdomain],
        hidden_layers: List[int] = [20, 20, 20],
        activation: str = "tanh",
        backend: str = "tensorflow",
        dtype: str = "float64",
    ):
        """Initialize subdomain collection.

        Args:
            subdomains: List of subdomains.
            hidden_layers: Network architecture for each subdomain.
            activation: Activation function.
            backend: Deep learning backend.
            dtype: Data type.
        """
        self.subdomains = subdomains
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.backend = backend
        self.dtype = dtype

        # Create networks
        self.networks: List[SubdomainNetwork] = []
        for sd in subdomains:
            net = SubdomainNetwork(
                subdomain=sd,
                hidden_layers=hidden_layers,
                activation=activation,
                backend=backend,
                dtype=dtype,
                seed=sd.index,
            )
            self.networks.append(net)
            sd.network = net

    def build_all(self) -> None:
        """Build all subdomain networks."""
        for net in self.networks:
            net.build()

    def __len__(self) -> int:
        return len(self.subdomains)

    def __iter__(self):
        return iter(zip(self.subdomains, self.networks))

    def __getitem__(self, index: int) -> Tuple[Subdomain, SubdomainNetwork]:
        return self.subdomains[index], self.networks[index]

    def evaluate(
        self,
        x: NDArray[np.floating],
        blending: str = "partition_of_unity",
    ) -> NDArray[np.floating]:
        """Evaluate global solution at points.

        Args:
            x: Points of shape (n_points, dim).
            blending: How to blend overlapping subdomain contributions.
                     'partition_of_unity' or 'max'.

        Returns:
            Solution values of shape (n_points,).
        """
        n_points = x.shape[0]
        result = np.zeros(n_points)
        weights = np.zeros(n_points)

        for sd, net in self:
            # Find points in this subdomain
            mask = np.array([sd.contains(x[i]) for i in range(n_points)])
            if not np.any(mask):
                continue

            x_local = x[mask]
            u_local = net.forward(x_local, apply_cutoff=True).flatten()

            if blending == "partition_of_unity":
                # Weight by distance from boundary
                w_local = self._partition_of_unity_weights(x_local, sd)
                result[mask] += u_local * w_local
                weights[mask] += w_local
            else:
                # Simple assignment (last subdomain wins)
                result[mask] = u_local

        # Normalize by weights
        if blending == "partition_of_unity":
            nonzero = weights > 0
            result[nonzero] /= weights[nonzero]

        return result

    def _partition_of_unity_weights(
        self,
        x: NDArray[np.floating],
        subdomain: Subdomain,
    ) -> NDArray[np.floating]:
        """Compute partition of unity weights.

        Weight is based on distance from subdomain boundary.
        """
        weights = np.ones(x.shape[0])
        for d in range(subdomain.dim):
            lo, hi = subdomain.box.bounds[d]
            mid = (lo + hi) / 2
            half_size = (hi - lo) / 2
            # Weight decreases near boundary
            dist = 1.0 - np.abs(x[:, d] - mid) / half_size
            weights *= np.maximum(dist, 0.01)
        return weights

    def get_all_parameters(self) -> List[Any]:
        """Get all trainable parameters from all networks."""
        params = []
        for net in self.networks:
            params.extend(net.get_trainable_variables())
        return params

    def total_parameters(self) -> int:
        """Count total number of trainable parameters."""
        total = 0
        for net in self.networks:
            for var in net.get_trainable_variables():
                if hasattr(var, 'numpy'):
                    total += np.prod(var.numpy().shape)
                elif hasattr(var, 'numel'):
                    total += var.numel()
                else:
                    total += np.prod(np.array(var).shape)
        return total
