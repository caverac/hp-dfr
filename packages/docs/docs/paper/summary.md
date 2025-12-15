---
sidebar_position: 1
sidebar_label: 'Background: Original DFR'
---

# Background: The Original DFR Method

Our adaptive hp-DFR method builds upon the theoretical foundations established in the original Deep Fourier Residual paper. This section summarizes the key concepts from:

> **"A Deep Fourier Residual Method for solving PDEs using Neural Networks"** by Taylor, Pardo, and Muga.

## Citation

```bibtex
@article{taylor2023deep,
  title={A Deep Fourier Residual method for solving PDEs using Neural Networks},
  author={Taylor, Jamie M and Pardo, David and Muga, Ignacio},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={405},
  pages={115850},
  year={2023},
  publisher={Elsevier}
}
```

## Abstract

When using Neural Networks as trial functions to numerically solve PDEs, the choice of loss function is critical. This work proposes the Deep Fourier Residual (DFR) method, which uses a Discrete Sine/Cosine Transform to accurately and efficiently compute the $H^{-1}$ norm of the residual. The resulting loss is equivalent to the $H^1$ error for well-posed problems.

## Key Contributions

### 1. Dual Norm Loss Function

The paper establishes that for well-posed PDEs, the dual norm of the residual provides bounds on the error:

$$
\frac{1}{M}\|\mathcal{R}(u)\|_{V^*} \leq \|u - u^*\|_U \leq \frac{1}{\gamma}\|\mathcal{R}(u)\|_{V^*}
$$

This means minimizing the dual norm loss is equivalent to minimizing the error.

### 2. Efficient Fourier Computation

For rectangular domains, the $H^{-1}$ norm can be computed efficiently using:

$$
\|R\|_{H^{-1}}^2 = \sum_{k} \frac{\hat{R}_k^2}{1 + |k|^2}
$$

where $\hat{R}_k$ are Fourier coefficients obtained via DST/DCT.

### 3. Improved Training Correlation

The paper demonstrates strong correlation between the DFR loss and $H^1$ error during training, enabling reliable error estimation without knowing the exact solution.

## Model Problems

### Problem 1: Smooth Solution (Sine)

- **PDE**: $-u'' = 4\sin(2x)$ on $[0, \pi]$
- **Solution**: $u(x) = \sin(2x)$
- **Result**: All methods perform comparably

### Problem 2: Large Gradients (Arctan)

- **PDE**: Solution with steep gradients
- **Solution**: $u(x) = \arctan(100(x - 0.5))$
- **Result**: DFR significantly outperforms PINNs and VPINNs

### Problem 3: Discontinuous Coefficients

- **PDE**: $-(\sigma u')' = f$ with discontinuous $\sigma$
- **Result**: Only DFR works (PINNs requires $H^2$ regularity)

### Problem 4: Point Source

- **PDE**: $-u'' = \delta(x - x_0)$
- **Result**: DFR handles distributional forcing

### Problem 5: Nonlinear ODE

- **PDE**: $u'' + u^3 = f$
- **Result**: DFR extends to nonlinear problems

### Problem 6: 2D Elliptic

- **PDE**: 2D Poisson on $[0,1]^2$
- **Result**: DFR scales to higher dimensions

## Numerical Results

### Loss-Error Correlation

| Method | Correlation (Smooth) | Correlation (Singular) |
|--------|---------------------|----------------------|
| PINNs (L2) | 0.85 | 0.40 |
| VPINNs | 0.88 | 0.55 |
| DFR (H⁻¹) | **0.99** | **0.98** |

### Final H1 Errors (Arctan Problem)

| Method | H1 Error |
|--------|----------|
| PINNs | 2.3e-1 |
| VPINNs | 1.8e-1 |
| DFR | **4.2e-3** |

## Method Comparison

### PINNs (Physics-Informed Neural Networks)

- Uses strong form of PDE
- Loss: $\|\Delta u + f\|_{L^2}^2$
- Requires $H^2$ regularity
- Simple implementation

### VPINNs (Variational PINNs)

- Uses weak form with $L^2$ loss
- Loss: $\|\int \nabla u \cdot \nabla v - fv\|_{L^2}^2$
- Still doesn't provide error equivalence

### DFR (Deep Fourier Residual)

- Uses weak form with $H^{-1}$ loss
- Loss: $\|\mathcal{R}(u)\|_{H^{-1}}^2$
- Error-equivalent loss
- Works for $H^1$ solutions

## Limitations

1. **Domain restriction**: Rectangular domains only
2. **Boundary conditions**: Dirichlet or Neumann on each face
3. **Computational cost**: Fourier transform overhead

## Limitations Addressed by Our Research

The original DFR paper identified several limitations that our adaptive hp-DFR method directly addresses:

1. **Domain restriction**: Rectangular domains only → We extend via domain decomposition (h-refinement)
2. **Curse of dimensionality**: $O(N^d)$ modes required → Our sparse tensor methods reduce to $O(N(\log N)^{d-1})$
3. **Uniform refinement**: Same resolution everywhere → hp-adaptivity focuses computation where needed
4. **Goal-oriented adaptivity**: Suggested as future work → We implement dual-weighted residual estimation

See the [Introduction](/docs/intro) for details on how our method addresses these challenges.

## Links

- [arXiv Preprint](https://arxiv.org/abs/2210.14129)
- [Published Version (Elsevier)](https://www.sciencedirect.com/science/article/abs/pii/S0045782522008064)
