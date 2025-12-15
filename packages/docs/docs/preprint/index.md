---
sidebar_position: 1
sidebar_label: Overview
---

# Our Research: Adaptive hp-DFR

This section documents our research extending the Deep Fourier Residual method with adaptive hp-refinement and goal-oriented error control.

## The New Idea

We propose extending the Deep Fourier Residual method with **adaptive hp-refinement** driven by **goal-oriented error estimation**. This combines three key innovations:

1. **Adaptive Fourier Mode Selection**: Instead of using a fixed truncation of Fourier modes, dynamically select modes based on their contribution to the residual norm. This addresses the curse of dimensionality mentioned in the original paper.

2. **Hierarchical Neural Network Architecture (hp-refinement)**: Use a multi-scale network where:
   - **h-refinement**: Partition the domain and use local networks in regions requiring higher resolution
   - **p-refinement**: Adaptively increase network depth/width in regions with smooth solutions

3. **Goal-Oriented Error Estimation**: For quantities of interest (QoI), compute the dual-weighted residual to focus computational effort where it affects the output most.

## Why Is It Novel?

The original DFR paper ([arXiv:2210.14129](https://arxiv.org/abs/2210.14129)) establishes that the $H^{-1}$ dual norm loss is equivalent to the $H^1$ error for well-posed problems. However, several limitations remain:

### Limitations Addressed

| Limitation | Original DFR | Our Approach |
|------------|--------------|--------------|
| **Curse of dimensionality** | $O(N^d)$ Fourier modes required | Sparse tensor methods: $O(N(\log N)^{d-1})$ |
| **Uniform refinement** | Same modes everywhere | hp-adaptivity focuses resolution where needed |
| **Energy norm mismatch** | $H^{-1}$ may not control energy error | Goal-oriented estimation targets QoI directly |
| **Fixed architecture** | Single network for entire domain | Domain decomposition with local networks |

### Novel Contributions

- First integration of hp-adaptivity with DFR-style dual norm losses
- Theoretical analysis of how adaptive mode selection affects error-loss equivalence
- Goal-oriented loss functions for physics-informed learning
- Sparse tensor Fourier methods to break the curse of dimensionality

## Key Research Questions

### Question 1: Adaptive Fourier Mode Selection

**Can we maintain error-loss equivalence while using only $O(N \log N)$ modes instead of $O(N^d)$?**

We investigate sparse tensor product Fourier spaces where modes $(k_1, \ldots, k_d)$ satisfy:

$$
\sum_{i=1}^{d} \log(1 + k_i) \leq M
$$

instead of the full tensor product. This reduces complexity from $O(N^d)$ to $O(N(\log N)^{d-1})$.

**Numerical component:**
- Implement sparse DST/DCT via hierarchical evaluation
- Compare accuracy vs. computational cost against full DFR
- Test on 2D, 3D, and higher-dimensional Poisson problems

**Theoretical component:**
- Prove that for solutions in mixed Sobolev spaces, the truncation error is controlled
- Establish error bounds for sparse mode approximation of the dual norm

### Question 2: hp-Adaptive Neural Network Architecture

**Can domain decomposition with local networks improve efficiency for problems with localized features?**

For problems with localized singularities, boundary layers, or discontinuous coefficients:
- Partition $\Omega$ into subdomains $\Omega_j$ with local networks $u_j$
- Use the DFR loss locally: $\mathcal{L}_j = \|R(u_j)\|_{H^{-1}(\Omega_j)}$
- Couple subdomains via mortar/interface conditions

**Numerical component:**
- Implement h-adaptive DFR with automatic subdivision based on local residual indicators
- Test on: (a) L-shaped domain with corner singularity, (b) problems with internal layers, (c) discontinuous diffusion coefficients
- Compare against single-network DFR

**Theoretical component:**
- Analyze stability of the coupled system
- Prove that local error indicators form a reliable and efficient a posteriori estimator

### Question 3: Goal-Oriented DFR

**Can dual-weighted residuals focus the loss on quantities of interest?**

Instead of minimizing $\|R(u)\|_{H^{-1}}$, minimize a weighted loss:

$$
\mathcal{L}_{\text{QoI}}(u) = |\langle R(u), z \rangle|
$$

where $z$ is the adjoint solution corresponding to the QoI.

**Numerical component:**
- Implement simultaneous primal-adjoint neural network training
- Test on: (a) point evaluation problems, (b) average flux computations, (c) eigenvalue problems
- Demonstrate that QoI-targeted training requires fewer DOF than global error minimization

**Theoretical component:**
- Establish that $\mathcal{L}_{\text{QoI}}$ controls the error in the quantity of interest
- Analyze convergence rates for goal-oriented DFR

## Implementation Steps

Use this checklist to track progress as you develop the paper. Each phase builds on the previous one.

### Phase 1: Sparse Fourier Methods

#### 1.1 Literature Review
- [x] Review sparse grid methods (Bungartz & Griebel, Acta Numerica 2004)
- [x] Study hyperbolic cross approximation theory
- [x] Review existing sparse FFT implementations
- [x] Document relevant error estimates for sparse tensor products

See [Literature Review](/docs/preprint/literature-review) for details.

#### 1.2 Implementation
- [x] Implement sparse index set generation for hyperbolic cross
- [x] Implement hierarchical DST/DCT evaluation
- [x] Create unit tests comparing against full DST/DCT
- [ ] Benchmark computational complexity vs. full tensor product

**Implementation files**:
- `dfr_pinns/fourier/sparse_indices.py` - Hyperbolic cross, Smolyak indices
- `dfr_pinns/fourier/transforms.py` - DST/DCT transforms, H^{-1} weights
- `dfr_pinns/models/sparse_dfr.py` - SparseDFRModel class
- `tests/test_sparse_fourier.py` - Comprehensive unit tests

#### 1.3 Validation Experiments
- [ ] 2D Poisson with smooth solution (verify no accuracy loss)
- [ ] 2D Poisson with corner singularity
- [ ] 3D Poisson problem
- [ ] 4D+ problems to demonstrate scaling

#### 1.4 Theory
- [ ] Prove error bound for sparse dual norm approximation
- [ ] Establish conditions for error-loss equivalence preservation
- [ ] Write up theoretical results for paper

### Phase 2: hp-Adaptive Architecture

#### 2.1 Literature Review
- [x] Review domain decomposition methods for PDEs
- [x] Study mortar element methods
- [x] Review hp-FEM error estimation strategies
- [x] Survey neural network domain decomposition approaches (hp-VPINNs, AB-PINNs, AS-PINNs)

See [Literature Review](/docs/preprint/literature-review) for details.

#### 2.2 Implementation
- [x] Implement domain partitioning infrastructure
- [x] Create local network architecture with interface coupling
- [x] Implement local DFR loss computation
- [x] Add residual-based refinement indicators
- [x] Implement automatic h-refinement strategy

**Implementation files**:
- `dfr_pinns/domain/partitioning.py` - BoundingBox, RectangularPartition, AdaptivePartition
- `dfr_pinns/domain/subdomain.py` - Subdomain, SubdomainNetwork, SubdomainCollection
- `dfr_pinns/domain/interface.py` - Interface, PenaltyCoupling, MortarCoupling
- `dfr_pinns/domain/refinement.py` - RefinementIndicator, mark_for_refinement
- `dfr_pinns/models/hp_dfr.py` - HPDFRModel class
- `tests/test_hp_adaptive.py` - Comprehensive unit tests

#### 2.3 Validation Experiments
- [ ] 1D problem with internal layer
- [ ] 2D L-shaped domain (corner singularity)
- [ ] 2D problem with discontinuous diffusion coefficient
- [ ] Compare DOF vs. accuracy against single-network DFR

#### 2.4 Theory
- [ ] Analyze stability of coupled multi-network system
- [ ] Prove reliability of local error indicators
- [ ] Prove efficiency of local error indicators
- [ ] Write up theoretical results for paper

### Phase 3: Goal-Oriented Training

#### 3.1 Literature Review
- [x] Review dual-weighted residual (DWR) methods
- [x] Study goal-oriented adaptivity in FEM
- [x] Review adjoint methods in deep learning (DWR-DNN, E2N, SA-PINN)
- [x] Document relevant error representation formulas

See [Literature Review](/docs/preprint/literature-review) for details.

#### 3.2 Implementation
- [x] Implement adjoint network architecture
- [x] Create simultaneous primal-adjoint training loop
- [x] Implement goal-oriented loss function
- [x] Add QoI error estimation

**Implementation files**:
- `dfr_pinns/models/goal_oriented_dfr.py` - GoalOrientedDFRModel, QoI classes
- `tests/test_goal_oriented.py` - Comprehensive unit tests

**Supported QoI types**:
- PointEvaluationQoI - Point evaluation u(x₀)
- AverageValueQoI - Average over subdomain
- BoundaryFluxQoI - Normal flux through boundary segment

#### 3.3 Validation Experiments
- [ ] Point evaluation QoI (1D and 2D)
- [ ] Average flux QoI
- [ ] Boundary integral QoI
- [ ] Compare DOF vs. QoI accuracy against global DFR

#### 3.4 Theory
- [ ] Prove error representation formula for neural network approximations
- [ ] Establish QoI error bounds
- [ ] Analyze convergence rates
- [ ] Write up theoretical results for paper

### Phase 4: Comprehensive Study

#### 4.1 Combined Methods
- [ ] Integrate sparse Fourier with hp-adaptivity
- [ ] Integrate goal-oriented estimation with hp-adaptivity
- [ ] Test full adaptive hp-DFR framework

#### 4.2 Benchmark Problems
- [ ] High-frequency Helmholtz equation
- [ ] Reaction-diffusion with boundary layers
- [ ] Interface problems with discontinuous coefficients
- [ ] Higher-dimensional problems (d ≥ 4)

#### 4.3 Comparison Study
- [ ] Compare against standard PINNs
- [ ] Compare against standard DFR
- [ ] Compare against VPINNs
- [ ] Document computational cost vs. accuracy trade-offs

### Phase 5: Paper Writing

#### 5.1 Drafting
- [ ] Write introduction and motivation
- [ ] Write methodology section
- [ ] Write theoretical results section
- [ ] Write numerical experiments section
- [ ] Write conclusions

#### 5.2 Figures and Tables
- [ ] Create convergence plots
- [ ] Create architecture diagrams
- [ ] Create comparison tables
- [ ] Create computational cost figures

#### 5.3 Finalization
- [ ] Internal review
- [ ] Address reviewer comments
- [ ] Prepare supplementary material
- [ ] Submit to arXiv

## Numerical Experiments Plan

| Experiment | Dimension | Problem Type | Focus | Phase |
|------------|-----------|--------------|-------|-------|
| Sparse modes validation | 2D, 3D | Smooth Poisson | Curse of dimensionality | 1 |
| High-dimensional scaling | 4D, 6D | Poisson | Sparse tensor efficiency | 1 |
| Corner singularity | 2D | L-domain Poisson | hp-adaptivity | 2 |
| Internal layer | 1D, 2D | Reaction-diffusion | h-adaptivity | 2 |
| Discontinuous σ | 2D | Elliptic interface | Local DFR | 2 |
| Point evaluation | 1D, 2D | Various | Goal-oriented | 3 |
| Average flux | 2D | Diffusion | Goal-oriented | 3 |
| High-frequency Helmholtz | 1D, 2D | Wave equation | Alternative norms | 4 |

## Current Status

| Component | Status |
|-----------|--------|
| Literature review | **Complete** - see [Literature Review](/docs/preprint/literature-review) |
| Sparse Fourier implementation | **Complete** - Phase 1.2 |
| hp-adaptive architecture | **Complete** - Phase 2.2 |
| Goal-oriented training | **Complete** - Phase 3.2 |
| Unit tests | **Complete** - 3 test files |
| Validation experiments | Not started - Phases 1.3, 2.3, 3.3 |
| Theoretical framework | Not started - Phases 1.4, 2.4, 3.4 |
| Paper draft | Not started - Phase 5 |

## Related Pages

- [Original DFR Method](/docs/theory/dfr) - Background on the Deep Fourier Residual approach
- [Background: Original DFR](/docs/paper/summary) - Summary of the original publication we build upon
- [Method Comparison](/docs/theory/comparison) - How DFR compares to PINNs

## References

Key references for this research (see [Literature Review](/docs/preprint/literature-review) for comprehensive list):

### DFR Methods
1. Taylor, J.M., Pardo, D., Muga, I. (2022). A Deep Fourier Residual Method for solving PDEs using Neural Networks. [arXiv:2210.14129](https://arxiv.org/abs/2210.14129)
2. Taylor, J.M., et al. (2024). Adaptive Deep Fourier Residual method via overlapping domain decomposition. [arXiv:2401.04663](https://arxiv.org/abs/2401.04663)

### Adaptive Methods
3. Kharazmi, E., Zhang, Z., Karniadakis, G.E. (2021). hp-VPINNs: Variational physics-informed neural networks with domain decomposition. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045782520307325)
4. Burrows, L., Chen, W., Sherif, M. (2025). AB-PINNs: Adaptive-Basis Physics-Informed Neural Networks. [arXiv:2510.08924](https://arxiv.org/abs/2510.08924)

### Goal-Oriented Error Estimation
5. Endtmayer, B., Langer, U., Wick, T. (2021). Multigoal-oriented dual-weighted-residual error estimation using deep neural networks. [arXiv:2112.11360](https://arxiv.org/abs/2112.11360)
6. Becker, R., Rannacher, R. (2001). An optimal control approach to a posteriori error estimation in finite element methods. Acta Numerica, 10, 1-102.

### Classical References
7. Bungartz, H.J., Griebel, M. (2004). Sparse grids. Acta Numerica, 13, 147-269.
8. Schwab, C. (1998). p- and hp-Finite Element Methods. Oxford University Press.
