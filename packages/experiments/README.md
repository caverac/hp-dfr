# DFR-PINNs Experiments

Python package implementing Physics-Informed Neural Networks (PINNs) and Deep Fourier Residual (DFR) methods for solving PDEs.

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Install with all backends
uv sync --extra all

# Or install specific backends
uv sync --extra tensorflow
uv sync --extra jax
uv sync --extra pytorch
```

Using pip:

```bash
pip install -e ".[all]"
```

## Quick Start

### Command Line

```bash
# Run DFR on sine problem with PyTorch
hp-dfr dfr run --problem sine --backend pytorch

# Run PINNs on arctan problem with PyTorch
hp-dfr pinns run --problem arctan --backend pytorch

# List available problems
hp-dfr dfr problems
hp-dfr pinns problems

# Check backend availability
hp-dfr backends
```

### Python API

```python
from hp_dfr.models import PINNsModel, DFRModel
from hp_dfr.problems import Poisson1D
import numpy as np

# Define problem
problem = Poisson1D(
    domain=(0, np.pi),
    forcing_fn=lambda x: 4 * np.sin(2 * x),
    exact_solution=lambda x: np.sin(2 * x)
)

# Create and train DFR model
model = DFRModel(
    hidden_layers=[10, 10, 10, 10],
    n_fourier_modes=10,
    backend="tensorflow"
)
model.build()
history = model.fit(problem, epochs=1000)

# Evaluate
x_test = np.linspace(0, np.pi, 100)
u_pred = model.predict(x_test)
```

## Project Structure

```
experiments/
├── src/hp_dfr/
│   ├── models/          # PINNs and DFR implementations
│   ├── problems/        # PDE problem definitions
│   ├── backends/        # Backend-specific code
│   ├── utils/           # Utility functions
│   └── cli.py           # Command-line interface
├── scripts/             # Experiment scripts
├── configs/             # Configuration files
├── tests/               # Unit tests
└── data/                # Paper data for comparison
```

## Reproducing Paper Results

```bash
# Run all paper experiments
hp-dfr dfr reproduce --output-dir results/

# Or run individual experiments
hp-dfr dfr run --problem sine --epochs 1000 --output results/sine_dfr.png
hp-dfr pinns run --problem sine --epochs 1000 --output results/sine_pinns.png
```

## Available Problems

| Problem | Description | Regularity |
|---------|-------------|------------|
| `sine` | Smooth solution u(x) = sin(2x) | H² |
| `arctan` | Large gradients | H² |
| `discontinuous` | Discontinuous coefficients | H¹ |
| `delta` | Point source | H¹ |

## Development

```bash
# Install dev dependencies
uv sync --group dev

# If you're using the PyTorch backend, include it when syncing
uv sync --group dev --extra pytorch

# Run tests
pytest

# Run tests with coverage
pytest --cov=hp_dfr --cov-report=html
```

### Formatting

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Format and sort in one go
black src/ tests/ && isort src/ tests/

# Check formatting without making changes
black --check src/ tests/
isort --check-only src/ tests/
```

### Linting

```bash
# Type checking with mypy
mypy src/

# Lint with flake8
flake8 src/

# Lint with pylint (thorough)
pylint src/hp_dfr/

# Check docstrings with pydocstyle
pydocstyle src/

# Run all linters
flake8 src/ && pylint src/hp_dfr/ && mypy src/ && pydocstyle src/
```

### Pre-commit Workflow

Run this before committing:

```bash
# Format
black src/ tests/ && isort src/ tests/

# Lint
flake8 src/ && mypy src/

# Test
pytest
```

## References

- [Taylor, Pardo, Muga (2023). "A Deep Fourier Residual Method for solving PDEs using Neural Networks"](https://arxiv.org/abs/2210.14129)
