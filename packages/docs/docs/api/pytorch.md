---
sidebar_position: 3
---

# PyTorch Backend

`hp-dfr` exposes a single model API; the compute backend is selected with the
`backend="pytorch"` argument. Internally the PyTorch backend uses `torch.autograd`
for the derivatives in the PDE residual. PyTorch is the backend used for the
experiments in the [preprint](/docs/preprint).

## Installation

The PyTorch backend is an optional dependency group:

```bash
uv pip install 'hp-dfr[pytorch]'
```

Check which backends are importable at any time:

```bash
uv run hp-dfr backends
```

## Selecting the backend

Every model takes a `backend` string, one of `"tensorflow"`, `"jax"`, or
`"pytorch"`; an unknown value raises `ValueError`. The rest of the API
(`build`, `fit`, `predict`) is identical across backends, so switching backends is
a one-argument change.

## Models

### PINNs

```python
import numpy as np
from hp_dfr.models import PINNsModel
from hp_dfr.problems import poisson_1d

problem = poisson_1d.get_problem("sine")

model = PINNsModel(
    hidden_layers=(64, 64, 64),
    activation="tanh",
    n_collocation=1000,
    bc_weight=100.0,
    backend="pytorch",
    seed=1234,
)
model.build()
history = model.fit(problem, epochs=1000, learning_rate=1e-3, verbose=True)

x = np.linspace(problem.domain[0], problem.domain[1], 200)
u = model.predict(x)
```

`fit` returns a history dict with a `"loss"` key; `predict` returns a NumPy array.

### DFR

```python
from hp_dfr.models import DFRModel
from hp_dfr.problems import poisson_1d

problem = poisson_1d.get_problem("arctan")

model = DFRModel(
    hidden_layers=(10, 10, 10, 10),
    n_fourier_modes=10,
    n_quadrature=100,
    backend="pytorch",
    dim=1,
)
model.build()
# DFR supports an optional LBFGS polish after Adam (lbfgs_iters=0 disables it):
history = model.fit(problem, epochs=2000, learning_rate=1e-3, lbfgs_iters=400)
```

For a 2D problem, pass `dim=2` and a 2D problem instance:

```python
from hp_dfr.problems.poisson_2d import ArcTanBump2D

model = DFRModel(hidden_layers=(16, 16), n_fourier_modes=16, n_quadrature=32, backend="pytorch", dim=2)
model.build()  # input_dim defaults to the model's dim
history = model.fit(ArcTanBump2D(steepness=8.0), epochs=3000, learning_rate=3e-3, lbfgs_iters=500)
```

### Goal-oriented DFR

```python
import numpy as np
from hp_dfr.models import GoalOrientedDFRModel, PointEvaluationQoI
from hp_dfr.problems import poisson_1d

problem = poisson_1d.get_problem("arctan")
a, b = problem.domain
qoi = PointEvaluationQoI(point=np.array([a + 0.65 * (b - a)]), sigma=0.04)

model = GoalOrientedDFRModel(
    hidden_layers=(16, 16),
    dim=1,
    qoi=qoi,
    n_modes=60,
    n_quadrature=150,
    adjoint_weight=1.0,
    primal_weight=1.0,
    backend="pytorch",
)
model.build()
history = model.fit(problem, epochs=2000, learning_rate=2e-3)

qoi_value = qoi.evaluate(model.predict, problem.domain)
```

The available quantity-of-interest classes are `PointEvaluationQoI`,
`AverageValueQoI`, and `BoundaryFluxQoI`, all importable from `hp_dfr.models`.

## Differentiation

You do not write the autodiff yourself; the model assembles the residual with
`torch.autograd`. For reference, the derivatives entering the strong residual are
obtained with `torch.autograd.grad`:

```python
import torch

def derivatives(model, x):
    x = x.requires_grad_(True)
    u = model(x)
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]
    return u, du_dx, d2u_dx2
```

## Notes

- **Precision.** Models default to `dtype="float64"`, which PyTorch supports natively.
- **Intel macOS.** Install the backend as `torch>=2.0.0,<2.2.0` (this pin is encoded
  in the package's `pytorch` extra); other platforms use `torch>=2.5.1`.
- **Device.** The current model API does not expose a device argument; training runs
  on CPU.

## See Also

- [TensorFlow Backend](/docs/api/tensorflow)
- [JAX Backend](/docs/api/jax)
