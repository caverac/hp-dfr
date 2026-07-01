---
sidebar_position: 3
---

# PyTorch Backend

API reference for the PyTorch backend implementation.

## Overview

The PyTorch backend uses PyTorch's `autograd` system for automatic differentiation and provides a familiar object-oriented API.

## Backend Class

```python
from dfr_pinns.backends import PyTorchBackend

backend = PyTorchBackend(
    dtype='float64',
    device='cuda:0'  # or 'cpu'
)
```

### Parameters

| Parameter | Type  | Default     | Description              |
| --------- | ----- | ----------- | ------------------------ |
| `dtype`   | `str` | `'float64'` | Floating point precision |
| `device`  | `str` | `'cpu'`     | Device for computation   |

## Model Construction

```python
import torch
import torch.nn as nn

class PINNsNetwork(nn.Module):
    def __init__(self, hidden_layers=[10, 10, 10, 10], activation='tanh'):
        super().__init__()

        layers = []
        input_dim = 1

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

### With Cutoff Layer (for DFR)

```python
class DFRNetwork(nn.Module):
    def __init__(self, hidden_layers=[10, 10, 10, 10], domain=(0, torch.pi)):
        super().__init__()
        self.domain = domain
        self.network = PINNsNetwork(hidden_layers)

    def forward(self, x):
        # Cutoff for homogeneous Dirichlet BCs
        a, b = self.domain
        cutoff = (x - a) * (b - x)
        return cutoff * self.network(x)
```

## Automatic Differentiation

PyTorch uses `autograd.grad` for computing derivatives:

```python
def compute_derivatives(model, x):
    """Compute first and second derivatives."""
    x = x.requires_grad_(True)

    u = model(x)

    # First derivative
    du_dx = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # Second derivative
    d2u_dx2 = torch.autograd.grad(
        du_dx, x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0]

    return u, du_dx, d2u_dx2
```

## Loss Computation

### PINNs Loss

```python
def pinns_loss(model, x_colloc, x_boundary, forcing_fn, bc_weight=100.0):
    """Compute PINNs collocation loss."""
    x_colloc = x_colloc.requires_grad_(True)

    # Forward pass
    u = model(x_colloc)

    # Compute second derivative
    du_dx = torch.autograd.grad(
        u, x_colloc,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    d2u_dx2 = torch.autograd.grad(
        du_dx, x_colloc,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0]

    # PDE residual: u'' + f = 0
    f = forcing_fn(x_colloc)
    residual = d2u_dx2 + f
    pde_loss = torch.mean(residual**2)

    # Boundary loss
    u_bc = model(x_boundary)
    bc_loss = torch.mean(u_bc**2)

    return pde_loss + bc_weight * bc_loss
```

### DFR Loss

```python
def dfr_loss(model, x_quad, dst_matrix, h_minus_1_weights, forcing_fn):
    """Compute DFR loss using H^-1 norm."""
    x_quad = x_quad.requires_grad_(True)

    # Forward pass (model includes cutoff)
    u = model(x_quad)

    # Compute derivative
    du_dx = torch.autograd.grad(
        u, x_quad,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # Forcing term
    f = forcing_fn(x_quad)

    # Weak residual (simplified for 1D)
    weak_res = du_dx  # Would include test function integration

    # Transform to Fourier space
    fourier_coeffs = dst_matrix.T @ weak_res

    # H^-1 norm
    loss = torch.sum(h_minus_1_weights * fourier_coeffs**2)

    return loss
```

## Training

### Basic Training Loop

```python
model = PINNsNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Domain
domain = (0, torch.pi)
x_boundary = torch.tensor([[domain[0]], [domain[1]]], dtype=torch.float64)

def forcing_fn(x):
    return 4 * torch.sin(2 * x)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()

    # Random collocation points
    x_colloc = torch.rand(100, 1) * torch.pi

    loss = pinns_loss(model, x_colloc, x_boundary, forcing_fn)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
```

### With Learning Rate Scheduler

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

for epoch in range(2000):
    optimizer.zero_grad()
    x_colloc = torch.rand(100, 1) * torch.pi
    loss = pinns_loss(model, x_colloc, x_boundary, forcing_fn)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

### With LBFGS Optimizer

LBFGS often works better for PINNs:

```python
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=50,
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer.zero_grad()
    loss = pinns_loss(model, x_colloc, x_boundary, forcing_fn)
    loss.backward()
    return loss

for epoch in range(100):
    loss = optimizer.step(closure)
    print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
```

## GPU Configuration

```python
# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# Move model to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Move data to GPU
x_colloc = x_colloc.to(device)
x_boundary = x_boundary.to(device)
```

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(1000):
    optimizer.zero_grad()

    with autocast():
        loss = pinns_loss(model, x_colloc, x_boundary, forcing_fn)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Known Issues

:::caution
The PyTorch backend may produce slightly different results compared to TensorFlow and JAX backends. This is under investigation. See [GitHub Issue #X](https://github.com/mathmode/pinns-dfr/issues/X).
:::

## See Also

- [TensorFlow Backend](/docs/api/tensorflow)
- [JAX Backend](/docs/api/jax)
