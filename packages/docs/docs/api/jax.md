---
sidebar_position: 2
---

# JAX Backend

`hp-dfr` exposes a single model API; the compute backend is selected with the
`backend="jax"` argument. The JAX backend uses JAX's functional transforms
(`grad`, `jit`) internally and trains with `optax`.

:::caution Backend coverage
The JAX backend covers **PINNs and DFR in one dimension**. DFR in 2D (`dim=2`) is
implemented only for TensorFlow and PyTorch, and **goal-oriented DFR is not available
on JAX at all** &mdash; it runs on TensorFlow and PyTorch only. Requesting an
unsupported combination raises `ValueError`.
:::

## Installation

The JAX backend is an optional dependency group:

```bash
uv pip install 'hp-dfr[jax]'
```

Check which backends are importable at any time:

```bash
uv run hp-dfr backends
```

## Selecting the backend

Every model takes a `backend` string, one of `"tensorflow"`, `"jax"`, or
`"pytorch"`; an unknown value raises `ValueError`. The rest of the API
(`build`, `fit`, `predict`) is identical across backends.

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
    backend="jax",
    seed=1234,
)
model.build()
history = model.fit(problem, epochs=1000, learning_rate=1e-3, verbose=True)

x = np.linspace(problem.domain[0], problem.domain[1], 200)
u = model.predict(x)
```

`fit` returns a history dict with a `"loss"` key; `predict` returns a NumPy array.

### DFR (1D)

```python
from hp_dfr.models import DFRModel
from hp_dfr.problems import poisson_1d

problem = poisson_1d.get_problem("arctan")

model = DFRModel(
    hidden_layers=(10, 10, 10, 10),
    n_fourier_modes=10,
    n_quadrature=100,
    backend="jax",
    dim=1,
)
model.build()
history = model.fit(problem, epochs=2000, learning_rate=1e-3)
```

### Goal-oriented DFR

Goal-oriented DFR is not implemented for the JAX backend. Use the
[PyTorch](/docs/api/pytorch) or [TensorFlow](/docs/api/tensorflow) backend for
`GoalOrientedDFRModel`.

## Differentiation

You do not write the autodiff yourself; the model assembles the residual with JAX's
functional gradients. For reference, given a scalar network `u_scalar(x)`,
higher-order derivatives compose `grad`:

```python
from jax import grad, vmap

d_dx = grad(u_scalar)          # first derivative
d2_dx2 = grad(d_dx)            # second derivative
d2_dx2_batched = vmap(d2_dx2)  # apply across a batch of points
```

## Notes

- **Precision.** JAX defaults to 32-bit floats. The model enables 64-bit
  automatically to match its `dtype`: it calls
  `jax.config.update("jax_enable_x64", dtype == "float64")` when the network is
  built, so `dtype="float64"` (the default) works without extra configuration.
- **Dimensionality.** 1D only, as noted above.
- **Device.** The current model API does not expose a device argument.

## See Also

- [TensorFlow Backend](/docs/api/tensorflow)
- [PyTorch Backend](/docs/api/pytorch)
