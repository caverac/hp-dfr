---
sidebar_position: 2
---

# Quick Start

This guide walks you through running your first PINNs and DFR experiments using the hp-dfr CLI.

## CLI Overview

The `hp-dfr` CLI provides three main command groups:

```bash
hp-dfr pinns ...     # Physics-Informed Neural Networks
hp-dfr dfr ...       # Deep Fourier Residual method (reference paper)
hp-dfr research ...  # Adaptive hp-DFR (our research)
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

### hp-DFR Research Example

```bash
# Run adaptive hp-DFR
hp-dfr research run \
  --problem discontinuous \
  --backend pytorch \
  --initial-divisions 4 \
  --adapt-every 100

# Run sparse DFR for higher dimensions
hp-dfr research run-sparse \
  --problem sine \
  --backend pytorch \
  --dim 2 \
  --max-level 16

# Run goal-oriented DFR
hp-dfr research run-goal-oriented \
  --problem sine \
  --backend pytorch \
  --qoi point \
  --qoi-location 0.5
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
    hidden_layers=[64, 64, 64],
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
    hidden_layers=[10, 10, 10, 10],
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
pinns = PINNsModel(hidden_layers=[64, 64, 64], backend="pytorch", seed=1234)
pinns.build()
pinns_history = pinns.fit(problem, epochs=epochs, learning_rate=1e-3)

# Train DFR
dfr = DFRModel(hidden_layers=[10, 10, 10, 10], n_fourier_modes=10, backend="pytorch", seed=1234)
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

To reproduce results from the reference DFR paper:

```bash
# Reproduce all paper results
hp-dfr dfr reproduce --output-dir results/dfr_paper

# Or run specific problems
hp-dfr dfr run --problem sine --n-modes 10 --epochs 10000 --output results/sine.png
hp-dfr dfr run --problem arctan --n-modes 20 --epochs 20000 --output results/arctan.png
```

## Available Problems

| Problem | CLI Name | Description | Best Method |
|---------|----------|-------------|-------------|
| Smooth sine | `sine` | $-u'' = 4\sin(2x)$ | Both work well |
| Large gradients | `arctan` | Sharp transition layer | DFR preferred |
| Discontinuous | `discontinuous` | Jump in coefficients | hp-DFR preferred |
| Point source | `delta` | Delta function forcing | hp-DFR preferred |

## Command Reference

### Common Options (all methods)

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | tensorflow | Backend: tensorflow, pytorch, jax |
| `--epochs` | 1000 | Training epochs |
| `--lr` | 0.001 | Learning rate |
| `--hidden-layers` | varies | Network architecture (comma-separated) |
| `--seed` | 1234 | Random seed |
| `--output` | None | Save plot to file |
| `--plot/--no-plot` | True | Show plots |

### PINNs-specific Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n-collocation` | 1000 | Number of collocation points |
| `--bc-weight` | 100.0 | Boundary condition penalty weight |

### DFR-specific Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n-modes` | 10 | Number of Fourier modes |
| `--n-quadrature` | 100 | Quadrature points for integration |

### hp-DFR Research Options

| Option | Default | Description |
|--------|---------|-------------|
| `--initial-divisions` | 2 | Initial domain partitions |
| `--interface-penalty` | 10.0 | Interface coupling weight |
| `--adapt-every` | 100 | Epochs between adaptations |
| `--max-subdomains` | 16 | Maximum subdomain count |

## Next Steps

- [Theory: PINNs](/docs/theory/pinns) - Understand the collocation method
- [Theory: DFR](/docs/theory/dfr) - Learn about the variational approach
- [Our Research](/docs/preprint) - Adaptive hp-DFR methods
- [API Reference](/docs/api/pytorch) - Full API documentation
