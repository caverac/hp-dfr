---
sidebar_position: 2
---

# Quick Start

This guide walks you through running your first PINNs and DFR experiments using the hp-dfr CLI.

## CLI Overview

The `hp-dfr` CLI provides four command groups and two utility commands:

```bash
hp-dfr pinns --help     # Physics-Informed Neural Networks (baseline)
hp-dfr dfr --help       # Deep Fourier Residual method (reference paper)
hp-dfr research --help  # Goal-Oriented DFR (this project)
hp-dfr data --help      # Generate the preprint sweep data
hp-dfr figures          # Build the preprint figures from saved data
hp-dfr backends         # List available deep-learning backends
```

## Running Your First Experiment

### PINNs Example

```bash
# Run PINNs on the sine problem
hp-dfr pinns run --problem sine --backend pytorch --epochs 1000

# With custom settings
hp-dfr pinns run \
  --problem arctan \
  --backend pytorch \
  --epochs 5000 \
  --hidden-layers "64,64,64" \
  --bc-weight 500 \
  --lr 0.001
```

### DFR Example

```bash
# Run DFR on the sine problem
hp-dfr dfr run --problem sine --backend pytorch --epochs 1000

# With custom Fourier modes
hp-dfr dfr run \
  --problem arctan \
  --backend pytorch \
  --epochs 5000 \
  --n-modes 20 \
  --hidden-layers "20,20,20,20"
```

### Goal-Oriented DFR Example

```bash
# Goal-oriented DFR on the sharp 1D problem, point QoI at the 65% location
hp-dfr research run \
  --problem arctan \
  --backend pytorch \
  --qoi point \
  --qoi-location 0.65 \
  --hidden-layers "16,16" \
  --n-modes 60 \
  --epochs 2000

# Full-domain average QoI instead of a point value
hp-dfr research run \
  --problem sine \
  --backend pytorch \
  --qoi average
```

## Listing Available Options

```bash
# List available problems
hp-dfr pinns problems
hp-dfr dfr problems
hp-dfr research problems

# List available backends
hp-dfr backends

# Get help on any command
hp-dfr --help
hp-dfr pinns run --help
hp-dfr dfr run --help
```

## Python API

### PINNs Example

```python
import numpy as np
from hp_dfr.models import PINNsModel
from hp_dfr.problems import poisson_1d

# Get a predefined problem
problem = poisson_1d.get_problem("sine")

# Create the model
model = PINNsModel(
    hidden_layers=(64, 64, 64),
    backend="pytorch",
    seed=1234,
)

# Build and train
model.build()
history = model.fit(
    problem,
    epochs=1000,
    learning_rate=1e-3,
    verbose=True,
)

# Evaluate
x_test = np.linspace(problem.domain[0], problem.domain[1], 100)
u_pred = model.predict(x_test)
u_exact = problem.exact_solution(x_test)
error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
print(f"L2 Error: {error:.6e}")
```

### DFR Example

```python
import numpy as np
from hp_dfr.models import DFRModel
from hp_dfr.problems import poisson_1d

# Get a predefined problem
problem = poisson_1d.get_problem("sine")

# Create the DFR model
model = DFRModel(
    hidden_layers=(10, 10, 10, 10),
    n_fourier_modes=10,
    backend="pytorch",
    seed=1234,
)

# Build and train
model.build()
history = model.fit(
    problem,
    epochs=1000,
    learning_rate=1e-3,
    verbose=True,
)

# Evaluate
x_test = np.linspace(problem.domain[0], problem.domain[1], 100)
u_pred = model.predict(x_test)
u_exact = problem.exact_solution(x_test)
error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
print(f"L2 Error: {error:.6e}")
```

## Comparing Methods

```python
import matplotlib.pyplot as plt
import numpy as np
from hp_dfr.models import PINNsModel, DFRModel
from hp_dfr.problems import poisson_1d

# Setup
problem = poisson_1d.get_problem("sine")
epochs = 1000

# Train PINNs
pinns = PINNsModel(hidden_layers=(64, 64, 64), backend="pytorch", seed=1234)
pinns.build()
pinns_history = pinns.fit(problem, epochs=epochs, learning_rate=1e-3)

# Train DFR
dfr = DFRModel(hidden_layers=(10, 10, 10, 10), n_fourier_modes=10, backend="pytorch", seed=1234)
dfr.build()
dfr_history = dfr.fit(problem, epochs=epochs, learning_rate=1e-3)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss comparison
axes[0].semilogy(pinns_history["loss"], label="PINNs")
axes[0].semilogy(dfr_history["loss"], label="DFR")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].set_title("Training Loss")

# Solution comparison
x_test = np.linspace(problem.domain[0], problem.domain[1], 100)
axes[1].plot(x_test, pinns.predict(x_test), label="PINNs")
axes[1].plot(x_test, dfr.predict(x_test), label="DFR")
axes[1].plot(x_test, problem.exact_solution(x_test), "--", label="Exact")
axes[1].set_xlabel("x")
axes[1].set_ylabel("u(x)")
axes[1].legend()
axes[1].set_title("Solution Comparison")

plt.tight_layout()
plt.savefig("comparison.png")
```

## Reproducing Paper Results

To reproduce results from the reference DFR paper, run the benchmark problems
individually with the paper hyperparameters (create the output directory first,
since `--output` does not create parent directories):

```bash
mkdir -p results
hp-dfr dfr run --problem sine --n-modes 10 --epochs 10000 --output results/sine.png
hp-dfr dfr run --problem arctan --n-modes 20 --epochs 20000 --output results/arctan.png
```

## Reproducing this project's figures

The goal-oriented figures are built from saved sweep data in two steps: generate the
data (slow; requires a PyTorch backend), then render the figures (fast).

```bash
# Generate the 1D and 2D sweep data into the assets/ directory.
uv run hp-dfr data all

# Or generate a single sweep (defaults to the PyTorch backend):
uv run hp-dfr data 1d
uv run hp-dfr data 2d

# Render the figures (assets/dfr-saturation-1d.* and assets/dfr-vs-go-2d.*).
uv run hp-dfr figures
```

See [Method and Results](/docs/preprint/phase3-goal-oriented#results) for the figures
and their interpretation.

## Available Problems

| Problem         | CLI Name        | Description            | Best Method    |
| --------------- | --------------- | ---------------------- | -------------- |
| Smooth sine     | `sine`          | $-u'' = 4\sin(2x)$     | Both work well |
| Large gradients | `arctan`        | Sharp transition layer | DFR preferred  |
| Discontinuous   | `discontinuous` | Jump in coefficients   | DFR            |
| Point source    | `delta`         | Delta function forcing | DFR            |

## Command Reference

### Common Options (all methods)

| Option             | Default    | Description                            |
| ------------------ | ---------- | -------------------------------------- |
| `--backend`        | tensorflow | Backend: tensorflow, pytorch, jax      |
| `--epochs`         | 1000       | Training epochs                        |
| `--lr`             | 0.001      | Learning rate                          |
| `--hidden-layers`  | varies     | Network architecture (comma-separated) |
| `--seed`           | 1234       | Random seed                            |
| `--output`         | None       | Save plot to file                      |
| `--plot/--no-plot` | True       | Show plots                             |

### PINNs-specific Options

| Option            | Default | Description                       |
| ----------------- | ------- | --------------------------------- |
| `--n-collocation` | 1000    | Number of collocation points      |
| `--bc-weight`     | 100.0   | Boundary condition penalty weight |

### DFR-specific Options

| Option           | Default | Description                       |
| ---------------- | ------- | --------------------------------- |
| `--n-modes`      | 10      | Number of Fourier modes           |
| `--n-quadrature` | 100     | Quadrature points for integration |

### Goal-Oriented DFR Options

| Option           | Default | Description                                |
| ---------------- | ------- | ------------------------------------------ |
| `--qoi`          | point   | Quantity of interest: `point` or `average` |
| `--qoi-location` | 0.5     | Point-QoI location, relative in `[0, 1]`   |
| `--n-modes`      | 20      | Fourier modes per dimension                |
| `--n-quadrature` | 64      | Quadrature points per dimension            |

## Next Steps

- [Theory: PINNs](/docs/theory/pinns) - Understand the collocation method
- [Theory: DFR](/docs/theory/dfr) - Learn about the variational approach
- [Our Research](/docs/preprint) - Goal-Oriented DFR
- [API Reference](/docs/api/pytorch) - Full API documentation
