---
sidebar_position: 3
sidebar_label: 'Phase 2: hp-Adaptivity'
---

# Phase 2: hp-Adaptive Neural Network Architecture

This page documents the implementation and results of hp-adaptive domain decomposition for DFR.

## Motivation

Problems with localized features (singularities, boundary layers, discontinuous coefficients) benefit from non-uniform resolution. Traditional DFR uses a single network with uniform Fourier resolution everywhere. Our hp-adaptive approach:

- **h-refinement**: Subdivides the domain into smaller subdomains with local networks
- **p-refinement**: Increases network capacity in regions requiring higher accuracy

## Domain Decomposition Framework

### Subdomain Partition

Given domain $\Omega$, we partition it into non-overlapping subdomains:

$$
\Omega = \bigcup_{j=1}^{J} \Omega_j, \quad \Omega_i \cap \Omega_j = \emptyset \text{ for } i \neq j
$$

### Local Networks

Each subdomain $\Omega_j$ has a local network $u_j: \Omega_j \to \mathbb{R}$ with:
- Local DFR loss: $\mathcal{L}_j = \|R(u_j)\|_{H^{-1}(\Omega_j)}$
- Interface coupling via mortar conditions

### Interface Conditions

At interfaces $\Gamma_{ij} = \partial\Omega_i \cap \partial\Omega_j$, we enforce:

$$
\mathcal{L}_{\text{interface}} = \sum_{i < j} \int_{\Gamma_{ij}} |u_i - u_j|^2 \, ds + \int_{\Gamma_{ij}} |\nabla u_i \cdot n - \nabla u_j \cdot n|^2 \, ds
$$

## Implementation

### Domain Partitioning

```python
# TODO: Add implementation
class DomainPartition:
    """Manages domain decomposition for hp-adaptive DFR."""

    def __init__(self, domain: Domain, initial_partitions: int = 1):
        pass

    def refine(self, subdomain_id: int) -> None:
        """h-refine a subdomain into smaller pieces."""
        pass

    def coarsen(self, subdomain_ids: list[int]) -> None:
        """Coarsen subdomains by merging."""
        pass
```

### Local Network Architecture

```python
# TODO: Add implementation
class LocalDFRNetwork:
    """Local neural network for a subdomain."""

    def __init__(self, subdomain: Subdomain, depth: int, width: int):
        pass

    def compute_local_loss(self) -> float:
        """Compute DFR loss on this subdomain."""
        pass
```

### Error Indicators

```python
# TODO: Add implementation
def compute_error_indicator(subdomain: Subdomain, network: LocalDFRNetwork) -> float:
    """Compute residual-based error indicator for refinement decisions."""
    pass
```

## Theoretical Results

### Reliability

**Theorem (Reliability of Local Indicators)**

*The local error indicators $\eta_j$ satisfy:*

$$
\|u - u_h\|_{H^1(\Omega)}^2 \leq C_{\text{rel}} \sum_{j=1}^{J} \eta_j^2
$$

*Proof:* TODO

### Efficiency

**Theorem (Efficiency of Local Indicators)**

*The local error indicators $\eta_j$ satisfy:*

$$
\eta_j \leq C_{\text{eff}} \|u - u_h\|_{H^1(\omega_j)}
$$

*where $\omega_j$ is a neighborhood of $\Omega_j$.*

*Proof:* TODO

## Numerical Experiments

### 1D Internal Layer

Reaction-diffusion equation with $\epsilon = 10^{-3}$:

$$
-\epsilon u'' + u = f, \quad u(0) = u(1) = 0
$$

| Method | DOF | $H^1$ Error | Training Time |
|--------|-----|-------------|---------------|
| Single DFR | TODO | TODO | TODO |
| h-adaptive DFR | TODO | TODO | TODO |

### 2D L-Shaped Domain

Poisson equation on L-shaped domain with corner singularity:

| Method | DOF | $H^1$ Error | Training Time |
|--------|-----|-------------|---------------|
| Single DFR | TODO | TODO | TODO |
| h-adaptive DFR | TODO | TODO | TODO |

### 2D Discontinuous Coefficient

$$
-\nabla \cdot (\sigma \nabla u) = f
$$

with $\sigma$ discontinuous across an interface.

| Method | DOF | $H^1$ Error | Training Time |
|--------|-----|-------------|---------------|
| Single DFR | TODO | TODO | TODO |
| h-adaptive DFR | TODO | TODO | TODO |

## Results Summary

TODO: Summarize findings

## Next Steps

- [ ] Complete implementation
- [ ] Run validation experiments
- [ ] Write theoretical proofs
- [ ] Integrate with Phase 3 (goal-oriented)
