---
sidebar_position: 1
---

# TensorFlow Backend

`hp-dfr` exposes a single model API; the compute backend is selected with the
`backend="tensorflow"` argument. The TensorFlow backend builds the networks with
Keras and differentiates the residual with `tf.GradientTape`. It is the default
`dtype`-and-backend combination in the package.

## Installation

The TensorFlow backend is an optional dependency group:

```bash
uv pip install 'hp-dfr[tensorflow]'
```

Check which backends are importable at any time:

```bash
uv run hp-dfr backends
```

## Selecting the backend

Every model takes a `backend` string, one of `"tensorflow"`, `"jax"`, or
`"pytorch"`; an unknown value raises `ValueError`. The rest of the API
(`build`, `fit`, `predict`) is identical across backends. `"tensorflow"` is the
default, so it may be omitted.

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
    backend="tensorflow",
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
    backend="tensorflow",
    dim=1,
)
model.build()
history = model.fit(problem, epochs=2000, learning_rate=1e-3)
```

For a 2D problem, pass `dim=2` and a 2D problem instance:

```python
from hp_dfr.problems.poisson_2d import ArcTanBump2D

model = DFRModel(hidden_layers=(16, 16), n_fourier_modes=16, n_quadrature=32, backend="tensorflow", dim=2)
model.build()  # input_dim defaults to the model's dim
history = model.fit(ArcTanBump2D(steepness=8.0), epochs=3000, learning_rate=3e-3)
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
    backend="tensorflow",
)
model.build()
history = model.fit(problem, epochs=2000, learning_rate=2e-3)

qoi_value = qoi.evaluate(model.predict, problem.domain)
```

The available quantity-of-interest classes are `PointEvaluationQoI`,
`AverageValueQoI`, and `BoundaryFluxQoI`, all importable from `hp_dfr.models`.

## Differentiation

You do not write the autodiff yourself; the model assembles the residual with
`tf.GradientTape`. For reference, the derivatives entering the strong residual use
nested tapes:

```python
import tensorflow as tf

def derivatives(model, x):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            u = model(x)
        du_dx = tape1.gradient(u, x)
    d2u_dx2 = tape2.gradient(du_dx, x)
    return u, du_dx, d2u_dx2
```

## Notes

- **Precision.** Models default to `dtype="float64"`; the backend applies it with
  `keras.backend.set_floatx(dtype)` when the network is built.
- **Keras.** The backend is built on Keras 3; the `KERAS_BACKEND` environment
  variable defaults to `tensorflow`.
- **Device.** The current model API does not expose a device argument.

## See Also

- [JAX Backend](/docs/api/jax)
- [PyTorch Backend](/docs/api/pytorch)
