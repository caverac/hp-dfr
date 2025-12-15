---
sidebar_position: 1
---

# TensorFlow Backend

API reference for the TensorFlow backend implementation.

## Overview

The TensorFlow backend uses Keras Core for model construction and TensorFlow's `GradientTape` for automatic differentiation.

## Backend Class

```python
from dfr_pinns.backends import TensorFlowBackend

backend = TensorFlowBackend(
    dtype='float64',  # Precision: 'float32' or 'float64'
    device='gpu:0'    # Device placement
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | `str` | `'float64'` | Floating point precision |
| `device` | `str` | `None` | Device for computation |

## Model Construction

### PINNs Model

```python
from dfr_pinns.models import PINNsModel

model = PINNsModel(
    backend=TensorFlowBackend(),
    input_dim=1,
    output_dim=1,
    hidden_layers=[10, 10, 10, 10],
    activation='tanh'
)
```

### DFR Model

```python
from dfr_pinns.models import DFRModel

model = DFRModel(
    backend=TensorFlowBackend(),
    input_dim=1,
    output_dim=1,
    hidden_layers=[10, 10, 10, 10],
    activation='tanh',
    n_fourier_modes=10,
    n_quadrature=100
)
```

## Automatic Differentiation

The TensorFlow backend computes derivatives using `GradientTape`:

```python
import tensorflow as tf

def compute_derivatives(model, x):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            u = model(x)
        du_dx = tape1.gradient(u, x)
    d2u_dx2 = tape2.gradient(du_dx, x)
    return u, du_dx, d2u_dx2
```

## Loss Computation

### PINNs Loss

```python
class PINNsLoss(tf.keras.layers.Layer):
    def __init__(self, n_collocation, domain, forcing_fn, bc_weight=100.0):
        super().__init__()
        self.n_collocation = n_collocation
        self.domain = domain
        self.forcing_fn = forcing_fn
        self.bc_weight = bc_weight

    def call(self, inputs):
        model = inputs

        # Sample collocation points
        x = tf.random.uniform(
            (self.n_collocation, 1),
            self.domain[0],
            self.domain[1]
        )

        # Compute PDE residual
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            with tf.GradientTape() as tape1:
                tape1.watch(x)
                u = model(x)
            du_dx = tape1.gradient(u, x)
        d2u_dx2 = tape.gradient(du_dx, x)

        residual = d2u_dx2 + self.forcing_fn(x)

        # Boundary conditions
        x_bc = tf.constant([[self.domain[0]], [self.domain[1]]])
        u_bc = model(x_bc)

        loss = tf.reduce_mean(residual**2) + self.bc_weight * tf.reduce_mean(u_bc**2)
        return loss
```

### DFR Loss

```python
class DFRLoss(tf.keras.layers.Layer):
    def __init__(self, n_quadrature, n_modes, domain):
        super().__init__()
        self.n_quadrature = n_quadrature
        self.n_modes = n_modes
        self.domain = domain

        # Precompute DST matrix
        self.dst_matrix = self._compute_dst_matrix()

    def _compute_dst_matrix(self):
        # DST-I matrix for weak residual transform
        L = self.domain[1] - self.domain[0]
        x = tf.linspace(self.domain[0], self.domain[1], self.n_quadrature)
        k = tf.range(1, self.n_modes + 1, dtype=tf.float64)
        return tf.sin(tf.tensordot(x, k * np.pi / L, axes=0))

    def call(self, inputs):
        model, forcing_fn = inputs

        # Quadrature points
        x = tf.linspace(self.domain[0], self.domain[1], self.n_quadrature)
        x = tf.reshape(x, (-1, 1))

        # Compute derivatives
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = model(x)
        du_dx = tape.gradient(u, x)

        # Weak residual components
        f = forcing_fn(x)

        # Transform to Fourier space and apply H^-1 weighting
        # ... (implementation details)

        return h_minus_1_norm_squared
```

## Training

```python
# Compile and train
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

history = model.fit(
    x=tf.constant([[1.0]]),  # Dummy input
    y=tf.constant([[1.0]]),  # Dummy target
    epochs=1000,
    verbose=1
)
```

## Custom Training Loop

For more control:

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(model, loss_layer):
    with tf.GradientTape() as tape:
        loss = loss_layer(model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(1000):
    loss = train_step(model, loss_layer)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy():.6f}")
```

## GPU Configuration

```python
# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {gpus}")

# Enable memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Limit GPU memory
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
)
```

## See Also

- [JAX Backend](/docs/api/jax)
- [PyTorch Backend](/docs/api/pytorch)
