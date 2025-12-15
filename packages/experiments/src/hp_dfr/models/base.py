"""Base model class for PINNs and DFR implementations."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Dict, Any
import numpy as np


class BaseModel(ABC):
    """Abstract base class for neural network PDE solvers.

    This class defines the common interface for both PINNs and DFR models
    across different backends (TensorFlow, JAX, PyTorch).

    Attributes:
        hidden_layers: List of integers specifying neurons in each hidden layer.
        activation: Activation function name (e.g., 'tanh', 'relu').
        dtype: Data type for computations ('float32' or 'float64').
    """

    def __init__(
        self,
        hidden_layers: List[int] = [10, 10, 10, 10],
        activation: str = "tanh",
        dtype: str = "float64",
        seed: int = 1234,
    ):
        """Initialize the base model.

        Args:
            hidden_layers: Number of neurons in each hidden layer.
            activation: Activation function for hidden layers.
            dtype: Floating point precision.
            seed: Random seed for reproducibility.
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dtype = dtype
        self.seed = seed
        self._model = None
        self._history: Dict[str, List[float]] = {"loss": [], "h1_error": []}

    @abstractmethod
    def build(self, input_dim: int = 1, output_dim: int = 1) -> None:
        """Build the neural network architecture.

        Args:
            input_dim: Dimension of input (spatial coordinates).
            output_dim: Dimension of output (solution components).
        """
        pass

    @abstractmethod
    def fit(
        self,
        problem: Any,
        epochs: int = 1000,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model on a given problem.

        Args:
            problem: Problem instance defining the PDE.
            epochs: Number of training iterations.
            learning_rate: Learning rate for optimizer.
            verbose: Whether to print progress.

        Returns:
            Dictionary containing training history.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the trained model at given points.

        Args:
            x: Input points of shape (n_points, input_dim).

        Returns:
            Predicted solution values of shape (n_points, output_dim).
        """
        pass

    def compute_error(
        self,
        x: np.ndarray,
        exact_solution: Callable[[np.ndarray], np.ndarray],
        norm: str = "l2",
    ) -> float:
        """Compute error between prediction and exact solution.

        Args:
            x: Points at which to evaluate error.
            exact_solution: Function returning exact solution.
            norm: Error norm ('l2' or 'h1').

        Returns:
            Error value.
        """
        u_pred = self.predict(x)
        u_exact = exact_solution(x)

        if norm == "l2":
            return float(np.sqrt(np.mean((u_pred - u_exact) ** 2)))
        elif norm == "h1":
            # For H1 norm, would need gradient computation
            # Simplified L2 error for now
            return float(np.sqrt(np.mean((u_pred - u_exact) ** 2)))
        else:
            raise ValueError(f"Unknown norm: {norm}")

    @property
    def history(self) -> Dict[str, List[float]]:
        """Return training history."""
        return self._history

    def summary(self) -> None:
        """Print model summary."""
        if self._model is not None:
            print(f"Model: {self.__class__.__name__}")
            print(f"Hidden layers: {self.hidden_layers}")
            print(f"Activation: {self.activation}")
            print(f"Dtype: {self.dtype}")
        else:
            print("Model not built yet. Call build() first.")
