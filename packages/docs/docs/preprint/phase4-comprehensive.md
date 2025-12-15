---
sidebar_position: 5
sidebar_label: 'Phase 4: Comprehensive Study'
---

# Phase 4: Comprehensive Study

This page documents the integration of all methods and comprehensive benchmark experiments.

## Combined Framework

### Adaptive hp-DFR with Goal-Oriented Refinement

The full framework combines:

1. **Sparse Fourier** (Phase 1): Efficient dual norm computation in high dimensions
2. **hp-Adaptivity** (Phase 2): Local networks with residual-based refinement
3. **Goal-Oriented** (Phase 3): Focus refinement on quantities of interest

### Algorithm Overview

```
Algorithm: Goal-Oriented Adaptive hp-DFR

Input: PDE, QoI J, tolerance ε
Output: Approximate solution u_h with |J(u) - J(u_h)| < ε

1. Initialize coarse partition with single subdomain
2. Initialize primal and adjoint networks

3. While QoI error estimate > ε:
   a. Train primal network u_h using sparse DFR loss
   b. Train adjoint network z_h using sparse DFR loss for adjoint
   c. Compute goal-oriented error indicators η_j for each subdomain
   d. Mark subdomains for refinement based on Dörfler marking
   e. Refine marked subdomains (h or p refinement)
   f. Update networks for new partition

4. Return u_h
```

## Benchmark Problems

### Problem 1: High-Frequency Helmholtz

$$
-\Delta u - k^2 u = f \quad \text{in } \Omega
$$

with wavenumber $k \gg 1$.

**Challenge**: The $H^{-1}$ norm does not control the energy-norm error. Goal-oriented estimation is essential.

**Configuration:**
- Domain: $\Omega = (0, 1)^2$
- Wavenumber: $k = 10, 20, 40$
- QoI: Far-field pattern

| k | Method | DOF | QoI Error | Energy Error |
|---|--------|-----|-----------|--------------|
| 10 | Standard DFR | TODO | TODO | TODO |
| 10 | Goal-Oriented hp-DFR | TODO | TODO | TODO |
| 20 | Standard DFR | TODO | TODO | TODO |
| 20 | Goal-Oriented hp-DFR | TODO | TODO | TODO |
| 40 | Standard DFR | TODO | TODO | TODO |
| 40 | Goal-Oriented hp-DFR | TODO | TODO | TODO |

### Problem 2: Reaction-Diffusion with Boundary Layers

$$
-\epsilon \Delta u + u = f \quad \text{in } \Omega
$$

with $\epsilon \ll 1$ creating sharp boundary layers.

**Challenge**: Boundary layers require local refinement; uniform resolution is wasteful.

**Configuration:**
- Domain: $\Omega = (0, 1)^2$
- Parameter: $\epsilon = 10^{-2}, 10^{-3}, 10^{-4}$
- QoI: Average value in interior subdomain

| ε | Method | DOF | QoI Error | Global H¹ Error |
|---|--------|-----|-----------|-----------------|
| 1e-2 | Standard DFR | TODO | TODO | TODO |
| 1e-2 | hp-DFR | TODO | TODO | TODO |
| 1e-3 | Standard DFR | TODO | TODO | TODO |
| 1e-3 | hp-DFR | TODO | TODO | TODO |
| 1e-4 | Standard DFR | TODO | TODO | TODO |
| 1e-4 | hp-DFR | TODO | TODO | TODO |

### Problem 3: Interface Problem

$$
-\nabla \cdot (\sigma \nabla u) = f
$$

with $\sigma$ discontinuous across an interface.

**Challenge**: Kinks in solution across interface; standard Fourier basis is inefficient.

**Configuration:**
- Domain: $\Omega = (0, 1)^2$
- Interface: Circle of radius 0.25 centered at (0.5, 0.5)
- Contrast: $\sigma_{\text{in}}/\sigma_{\text{out}} = 10, 100, 1000$

| Contrast | Method | DOF | H¹ Error | Interface Error |
|----------|--------|-----|----------|-----------------|
| 10 | Standard DFR | TODO | TODO | TODO |
| 10 | hp-DFR | TODO | TODO | TODO |
| 100 | Standard DFR | TODO | TODO | TODO |
| 100 | hp-DFR | TODO | TODO | TODO |
| 1000 | Standard DFR | TODO | TODO | TODO |
| 1000 | hp-DFR | TODO | TODO | TODO |

### Problem 4: High-Dimensional Poisson

$$
-\Delta u = f \quad \text{in } (0, \pi)^d
$$

for $d = 2, 3, 4, 6$.

**Challenge**: Curse of dimensionality for full tensor product Fourier.

| d | Method | Modes | H¹ Error | Memory | Time |
|---|--------|-------|----------|--------|------|
| 2 | Full DFR | TODO | TODO | TODO | TODO |
| 2 | Sparse DFR | TODO | TODO | TODO | TODO |
| 3 | Full DFR | TODO | TODO | TODO | TODO |
| 3 | Sparse DFR | TODO | TODO | TODO | TODO |
| 4 | Full DFR | N/A | N/A | OOM | N/A |
| 4 | Sparse DFR | TODO | TODO | TODO | TODO |
| 6 | Full DFR | N/A | N/A | OOM | N/A |
| 6 | Sparse DFR | TODO | TODO | TODO | TODO |

## Comparison Study

### Methods Compared

1. **Standard PINNs**: Collocation loss $\mathcal{L}_{\text{col}} = \|Lu - f\|_{L^2}^2$
2. **VPINNs**: Variational loss with $L^2$ norm of residual
3. **Standard DFR**: Full tensor product Fourier, $H^{-1}$ loss
4. **Sparse DFR**: Hyperbolic cross Fourier modes
5. **hp-DFR**: Domain decomposition with local networks
6. **Goal-Oriented hp-DFR**: Full adaptive framework

### Metrics

- **DOF**: Total degrees of freedom (network parameters + Fourier modes)
- **Training time**: Wall-clock time to reach tolerance
- **Memory**: Peak GPU/CPU memory usage
- **H¹ error**: Global energy-norm error
- **QoI error**: Error in quantity of interest

### Summary Table

| Problem | Best Method | Speedup vs. Standard DFR | DOF Reduction |
|---------|-------------|--------------------------|---------------|
| Helmholtz | TODO | TODO | TODO |
| Reaction-diffusion | TODO | TODO | TODO |
| Interface | TODO | TODO | TODO |
| High-dimensional | TODO | TODO | TODO |

## Computational Cost Analysis

### Complexity Comparison

| Method | Loss Evaluation | Gradient Computation | Memory |
|--------|-----------------|---------------------|--------|
| Standard DFR | $O(N^d \log N)$ | $O(N^d \log N)$ | $O(N^d)$ |
| Sparse DFR | $O(N (\log N)^d)$ | $O(N (\log N)^d)$ | $O(N (\log N)^{d-1})$ |
| hp-DFR | $O(J \cdot N_j^d \log N_j)$ | $O(J \cdot N_j^d \log N_j)$ | $O(\sum_j N_j^d)$ |

### Scaling Plots

TODO: Add figures showing:
1. Training time vs. dimension
2. Memory usage vs. dimension
3. Error vs. DOF for each method

## Conclusions

TODO: Summarize key findings and recommendations

## Reproducibility

All experiments are reproducible using the code in `packages/experiments`. To run the full benchmark:

```bash
cd packages/experiments
uv sync
uv run python -m dfr_pinns.benchmarks.comprehensive
```

Results are saved to `results/phase4/` with timestamps.
