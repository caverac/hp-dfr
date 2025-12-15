---
sidebar_position: 4
sidebar_label: 'Phase 3: Goal-Oriented'
---

# Phase 3: Goal-Oriented DFR

This page documents the implementation and results of goal-oriented error estimation for DFR.

## Motivation

In many applications, we are not interested in the global $H^1$ error but rather in a specific **quantity of interest (QoI)**, such as:
- Point evaluation: $J(u) = u(x_0)$
- Average value: $J(u) = \frac{1}{|\omega|} \int_\omega u \, dx$
- Boundary flux: $J(u) = \int_{\Gamma} \nabla u \cdot n \, ds$
- Functional output: $J(u) = \int_\Omega g \cdot u \, dx$

Goal-oriented methods focus computational effort on reducing error in the QoI rather than global error.

## Dual-Weighted Residual Framework

### Adjoint Problem

For a QoI $J(u)$, we define the adjoint problem: Find $z \in V$ such that

$$
a(v, z) = J(v) \quad \forall v \in V
$$

where $a(\cdot, \cdot)$ is the bilinear form from the primal problem.

### Error Representation

The error in the QoI satisfies:

$$
J(u) - J(u_h) = a(u - u_h, z) = \langle R(u_h), z \rangle
$$

This motivates the **goal-oriented loss**:

$$
\mathcal{L}_{\text{QoI}}(u_h) = |\langle R(u_h), z_h \rangle|
$$

where $z_h$ is a neural network approximation of the adjoint.

## Implementation

### Primal-Adjoint Network Architecture

```python
# TODO: Add implementation
class PrimalAdjointDFR:
    """Joint primal-adjoint network for goal-oriented DFR."""

    def __init__(self, primal_config: NetworkConfig, adjoint_config: NetworkConfig):
        self.primal_network = ...
        self.adjoint_network = ...

    def compute_primal_residual(self, x: Tensor) -> Tensor:
        """Compute R(u_h) at points x."""
        pass

    def compute_adjoint_residual(self, x: Tensor) -> Tensor:
        """Compute R*(z_h) at points x."""
        pass

    def compute_goal_oriented_loss(self) -> Tensor:
        """Compute |<R(u_h), z_h>|."""
        pass
```

### Training Loop

```python
# TODO: Add implementation
def train_goal_oriented(
    primal_adjoint: PrimalAdjointDFR,
    qoi: Callable,
    epochs: int
) -> TrainingHistory:
    """
    Train primal and adjoint networks simultaneously.

    Strategy:
    1. Update primal network to minimize <R(u_h), z_h>
    2. Update adjoint network to minimize adjoint residual
    3. Alternate or joint optimization
    """
    pass
```

### QoI Definitions

```python
# TODO: Add implementation
class PointEvaluationQoI:
    """QoI: J(u) = u(x_0)"""
    def __init__(self, x0: np.ndarray):
        self.x0 = x0

    def evaluate(self, u: Callable) -> float:
        return u(self.x0)

    def adjoint_rhs(self, x: np.ndarray) -> np.ndarray:
        """Right-hand side for adjoint problem (Dirac delta at x0)."""
        pass


class AverageValueQoI:
    """QoI: J(u) = (1/|ω|) ∫_ω u dx"""
    def __init__(self, omega: Domain):
        self.omega = omega

    def evaluate(self, u: Callable) -> float:
        pass

    def adjoint_rhs(self, x: np.ndarray) -> np.ndarray:
        """Right-hand side for adjoint problem (characteristic function)."""
        pass
```

## Theoretical Results

### Error Representation Formula

**Theorem (Neural Network Error Representation)**

*Let $u$ be the exact solution, $u_h$ the primal network approximation, and $z_h$ the adjoint network approximation. Then:*

$$
|J(u) - J(u_h)| \leq |\langle R(u_h), z - z_h \rangle| + |\langle R(u_h), z_h \rangle|
$$

*The first term is the adjoint approximation error, the second is the computable goal-oriented loss.*

*Proof:* TODO

### QoI Error Bound

**Theorem (Goal-Oriented Error Control)**

*Under suitable regularity assumptions, minimizing $\mathcal{L}_{\text{QoI}}$ controls the QoI error:*

$$
|J(u) - J(u_h)| \leq C \left( \mathcal{L}_{\text{QoI}}(u_h) + \|R^*(z_h)\|_{H^{-1}} \right)
$$

*Proof:* TODO

## Numerical Experiments

### Point Evaluation QoI (1D)

Poisson equation with QoI $J(u) = u(0.5)$:

| Method | DOF | QoI Error | Global $H^1$ Error | Training Time |
|--------|-----|-----------|-------------------|---------------|
| Standard DFR | TODO | TODO | TODO | TODO |
| Goal-Oriented DFR | TODO | TODO | TODO | TODO |

### Point Evaluation QoI (2D)

| Method | DOF | QoI Error | Global $H^1$ Error | Training Time |
|--------|-----|-----------|-------------------|---------------|
| Standard DFR | TODO | TODO | TODO | TODO |
| Goal-Oriented DFR | TODO | TODO | TODO | TODO |

### Average Flux QoI

QoI: $J(u) = \int_{\Gamma_{\text{out}}} \nabla u \cdot n \, ds$

| Method | DOF | QoI Error | Global $H^1$ Error | Training Time |
|--------|-----|-----------|-------------------|---------------|
| Standard DFR | TODO | TODO | TODO | TODO |
| Goal-Oriented DFR | TODO | TODO | TODO | TODO |

### Efficiency Comparison

Plot: DOF vs. QoI accuracy for global DFR vs. goal-oriented DFR

TODO: Add figure

## Results Summary

TODO: Summarize findings

## Next Steps

- [ ] Complete implementation
- [ ] Run validation experiments
- [ ] Write theoretical proofs
- [ ] Integrate with Phase 2 (hp-adaptivity) for goal-oriented hp-refinement
