---
sidebar_position: 2
---

# JAX Backend

API reference for the JAX backend implementation.

## Overview

The JAX backend leverages JAX's functional transformations (`grad`, `vmap`, `jit`) for efficient automatic differentiation and vectorized computation.

## Backend Class

```python
from dfr_pinns.backends import JAXBackend

backend = JAXBackend(
    dtype='float64',  # Note: JAX defaults to float32
    enable_x64=True   # Enable 64-bit precision
)
```

### Parameters

| Parameter    | Type   | Default     | Description                  |
| ------------ | ------ | ----------- | ---------------------------- |
| `dtype`      | `str`  | `'float32'` | Floating point precision     |
| `enable_x64` | `bool` | `True`      | Enable 64-bit floating point |

### Enabling float64

JAX uses float32 by default. To enable float64:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Model Construction

JAX models are typically defined as pure functions with explicit parameters:

```python
import jax.numpy as jnp
from jax import random

def init_network(key, layer_sizes):
    """Initialize network parameters."""
    keys = random.split(key, len(layer_sizes) - 1)
    params = []
    for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w_key, b_key = random.split(k)
        params.append({
            'w': random.normal(w_key, (m, n)) * jnp.sqrt(2.0 / m),
            'b': jnp.zeros(n)
        })
    return params

def forward(params, x):
    """Forward pass through the network."""
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer['w'] + layer['b'])
    # Output layer (no activation)
    x = x @ params[-1]['w'] + params[-1]['b']
    return x
```

## Automatic Differentiation

JAX provides functional gradient computation:

```python
from jax import grad, vmap

# Gradient with respect to input
def u(params, x):
    return forward(params, x)

# First derivative
du_dx = grad(lambda x: u(params, x)[0])

# Second derivative
d2u_dx2 = grad(du_dx)

# Vectorized over batch
du_dx_batched = vmap(grad(lambda x: u(params, x.reshape(1, -1))[0, 0]))
```

### Using `jacfwd` and `jacrev`

```python
from jax import jacfwd, jacrev

# Forward-mode (efficient for few outputs, many inputs)
jacobian_fwd = jacfwd(lambda x: forward(params, x))

# Reverse-mode (efficient for many outputs, few inputs)
jacobian_rev = jacrev(lambda x: forward(params, x))
```

## Loss Computation

### PINNs Loss

```python
from jax import grad, vmap, jit

def pinns_loss(params, x_colloc, x_boundary, forcing_fn, bc_weight=100.0):
    """Compute PINNs loss."""
    # Second derivative computation
    def u_scalar(x):
        return forward(params, x.reshape(1, -1))[0, 0]

    du_dx = grad(u_scalar)
    d2u_dx2 = grad(du_dx)

    # Vectorize over collocation points
    d2u_dx2_batch = vmap(d2u_dx2)

    # PDE residual
    residual = d2u_dx2_batch(x_colloc) + forcing_fn(x_colloc)
    pde_loss = jnp.mean(residual**2)

    # Boundary loss
    u_bc = forward(params, x_boundary)
    bc_loss = jnp.mean(u_bc**2)

    return pde_loss + bc_weight * bc_loss

# JIT compile for speed
pinns_loss_jit = jit(pinns_loss)
```

### DFR Loss

```python
def dfr_loss(params, x_quad, dst_matrix, weights, forcing_fn):
    """Compute DFR loss using H^-1 norm."""
    # Forward pass with cutoff
    def u_with_cutoff(x):
        cutoff = x * (jnp.pi - x)
        return cutoff * forward(params, x.reshape(1, -1))[0, 0]

    # Compute derivatives
    du_dx = vmap(grad(u_with_cutoff))

    # Weak residual
    du = du_dx(x_quad)
    f = forcing_fn(x_quad)

    # Transform to Fourier space
    residual_fourier = dst_matrix.T @ (du - f)

    # H^-1 norm
    h_minus_1_sq = jnp.sum(weights * residual_fourier**2)

    return h_minus_1_sq
```

## Training

### Basic Training Loop

```python
from jax import value_and_grad
import optax

# Initialize optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

@jit
def train_step(params, opt_state, x_colloc, x_boundary):
    loss, grads = value_and_grad(pinns_loss)(
        params, x_colloc, x_boundary, forcing_fn
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
for epoch in range(1000):
    params, opt_state, loss = train_step(
        params, opt_state, x_colloc, x_boundary
    )
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")
```

### With Random Sampling

```python
from jax import random

@jit
def train_step_random(params, opt_state, key):
    # Random collocation points
    x_colloc = random.uniform(key, (100, 1), minval=0.0, maxval=jnp.pi)

    loss, grads = value_and_grad(pinns_loss)(
        params, x_colloc, x_boundary, forcing_fn
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training with different random points each epoch
key = random.PRNGKey(42)
for epoch in range(1000):
    key, subkey = random.split(key)
    params, opt_state, loss = train_step_random(params, opt_state, subkey)
```

## GPU/TPU Configuration

```python
import jax

# Check available devices
print(jax.devices())

# Force computation on specific device
with jax.default_device(jax.devices('gpu')[0]):
    result = forward(params, x)

# Distribute across devices
from jax import pmap
parallel_forward = pmap(forward)
```

## Performance Tips

1. **Always use `@jit`** for functions called repeatedly
2. **Use `vmap`** instead of Python loops
3. **Preallocate** arrays when possible
4. **Use `lax.scan`** for sequential operations

```python
from jax import lax

def scan_training(params, opt_state, keys):
    def step(carry, key):
        params, opt_state = carry
        params, opt_state, loss = train_step_random(params, opt_state, key)
        return (params, opt_state), loss

    (final_params, final_opt_state), losses = lax.scan(
        step, (params, opt_state), keys
    )
    return final_params, final_opt_state, losses
```

## See Also

- [TensorFlow Backend](/docs/api/tensorflow)
- [PyTorch Backend](/docs/api/pytorch)
