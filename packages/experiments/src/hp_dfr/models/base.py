"""Base model class for PINNs and DFR implementations.

This module provides the abstract base class that defines the common interface
for all neural network PDE solvers, including PINNs and DFR variants.
"""

from abc import ABC, abstractmethod
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt

from hp_dfr.types.common import NormType
from hp_dfr.utils import console


class Problem(Protocol):
    """Structural interface of a PDE problem used by the models."""

    domain: tuple

    def forcing_fn(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Evaluate the forcing term at points ``x``."""

    def exact_solution(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Evaluate the exact solution at points ``x``."""


class BaseModel(ABC):
    """Abstract base class for neural network PDE solvers.

    Defines the common interface for PINNs and DFR models across different
    backends (TensorFlow, JAX, PyTorch).

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

    Attributes
    ----------
    hidden_layers : tuple[int, ...]
        Network architecture specification.
    activation : str
        Activation function name.
    dtype : str
        Data type for computations.
    seed : int
        Random seed used for initialization.
    history : dict[str, list[float]]
        Training history containing loss and error metrics.

    Notes
    -----
    Subclasses must implement `build`, `fit`, and `predict` methods.
    """

    def __init__(
        self,
        /,
        *,
        hidden_layers: tuple[int, ...] = (10, 10, 10, 10),
        activation: str = "tanh",
        dtype: str = "float64",
        seed: int = 1234,
    ):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dtype = dtype
        self.seed = seed
        # The built backend network (keras.Model, torch.nn.Module, or JAX
        # params). Backend-agnostic here; subclasses cast to the concrete type.
        self._model: object | None = None
        self._history: dict[str, list[float]] = {"loss": [], "h1_error": []}

    @abstractmethod
    def build(self, input_dim: int = 1, output_dim: int = 1) -> None:
        """Build the neural network architecture.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of input (spatial coordinates), by default 1.
        output_dim : int, optional
            Dimension of output (solution components), by default 1.
        """

    @abstractmethod
    def fit(
        self,
        problem: Problem,
        epochs: int = 1000,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the model on a given problem.

        Parameters
        ----------
        problem : Problem
            Problem instance defining the PDE to solve.
        epochs : int, optional
            Number of training iterations, by default 1000.
        learning_rate : float, optional
            Learning rate for optimizer, by default 1e-3.
        verbose : bool, optional
            Whether to print progress during training, by default True.

        Returns
        -------
        dict[str, list[float]]
            Training history with keys 'loss' and optionally 'h1_error'.
        """

    @abstractmethod
    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the trained model at given points.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Input points of shape (n_points,) or (n_points, input_dim).

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted solution values of shape (n_points,) or (n_points, output_dim).
        """

    def compute_error(
        self,
        x: npt.NDArray[np.float64],
        exact_solution: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        /,
        *,
        norm: NormType = "l2",
    ) -> float:
        """Compute error between prediction and exact solution.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Points at which to evaluate the error.
        exact_solution : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
            Function that returns the exact solution at given points.
        norm : NormType, optional
            Error norm to use, by default 'l2'.

        Returns
        -------
        float
            Computed error value.

        Raises
        ------
        ValueError
            If an unknown norm is specified.

        Notes
        -----
        The H1 norm currently falls back to L2 (gradient computation not implemented).
        """
        u_pred = self.predict(x)
        u_exact = exact_solution(x)

        if norm == "l2":
            return float(np.sqrt(np.mean((u_pred - u_exact) ** 2)))
        if norm == "h1":
            # TODO: For H1 norm, would need gradient computation, using L2 error for now
            return float(np.sqrt(np.mean((u_pred - u_exact) ** 2)))

        raise ValueError(f"Unknown norm: {norm}")

    @property
    def history(self) -> dict[str, list[float]]:
        """Training history dictionary.

        Returns
        -------
        dict[str, list[float]]
            Dictionary with 'loss' and 'h1_error' keys containing per-epoch values.
        """
        return self._history

    def summary(self) -> None:
        """Print a summary of the model configuration.

        Displays the model class name, hidden layer architecture,
        activation function, and data type. If the model has not been
        built yet, prints a message indicating this.
        """
        if self._model is not None:
            console.print(f"Model: {self.__class__.__name__}", markup=False)
            console.print(f"Hidden layers: {self.hidden_layers}", markup=False)
            console.print(f"Activation: {self.activation}", markup=False)
            console.print(f"Dtype: {self.dtype}", markup=False)
        else:
            console.print("Model not built yet. Call build() first.", markup=False)
