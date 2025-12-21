# Adaptive hp-DFR: Goal-Oriented Deep Fourier Residual Methods

This repository contains the implementation and experiments for extending the Deep Fourier Residual (DFR) method with **adaptive hp-refinement** and **goal-oriented error estimation** for solving PDEs using neural networks.

**Full documentation**: [https://caverac.github.io/hp-dfr](https://caverac.github.io/hp-dfr)

## Table of Contents

- [Our Research](#our-research)
  - [Why Is This Novel?](#why-is-this-novel)
  - [Key Research Questions](#key-research-questions)
- [Background: The Deep Fourier Residual Method](#background-the-deep-fourier-residual-method)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Packages](#packages)
- [Contributing](#contributing)
- [License](#license)

## Our Research

We propose a novel extension to the Deep Fourier Residual method that combines three key innovations:

1. **Adaptive Fourier Mode Selection**: Instead of using a fixed truncation of Fourier modes, dynamically select modes based on their contribution to the residual norm, addressing the curse of dimensionality.

2. **Hierarchical Neural Network Architecture (hp-refinement)**: Use a multi-scale network with h-refinement (domain partitioning with local networks) and p-refinement (adaptive network depth/width).

3. **Goal-Oriented Error Estimation**: For quantities of interest (QoI), compute the dual-weighted residual to focus computational effort where it affects the output most.

### Why Is This Novel?

The original DFR method establishes that the H⁻¹ dual norm loss is equivalent to the H¹ error for well-posed problems. Our approach addresses key limitations:

- **Curse of dimensionality**: DFR requires O(Nᵈ) Fourier modes in d dimensions. Our sparse tensor methods reduce this to O(N(log N)^(d-1)).
- **Uniform refinement inefficiency**: Standard DFR uses the same number of modes everywhere. hp-adaptivity focuses resolution where needed.
- **Energy norm mismatch**: For certain PDEs (e.g., Helmholtz), H⁻¹ may not control the energy-norm error. Goal-oriented estimation directly targets quantities of interest.

This is the first integration of hp-adaptivity and goal-oriented error estimation with DFR-style dual norm losses for physics-informed learning.

### Key Research Questions

1. **Adaptive Fourier Mode Selection**: Can we maintain error-loss equivalence while using only O(N log N) modes instead of O(Nᵈ)? We investigate sparse tensor product Fourier spaces and prove error bounds for sparse mode approximation.

2. **hp-Adaptive Neural Network Architecture**: Can domain decomposition with local networks improve efficiency for problems with localized features? We implement h-adaptive DFR with automatic subdivision based on local residual indicators.

3. **Goal-Oriented DFR**: Can dual-weighted residuals focus the loss on quantities of interest? We implement simultaneous primal-adjoint neural network training and establish that the goal-oriented loss controls the error in the QoI.

## Background: The Deep Fourier Residual Method

This work builds upon the Deep Fourier Residual method introduced in:

> **A Deep Fourier Residual Method for solving PDEs using Neural Networks**
> Jamie M. Taylor, David Pardo, Ignacio Muga
> [arXiv:2210.14129](https://arxiv.org/abs/2210.14129) | [Published Version](https://www.sciencedirect.com/science/article/abs/pii/S0045782522008064)

The original DFR paper provides the theoretical foundation for using dual norm losses in physics-informed neural networks. Our work extends this framework with adaptive strategies to improve scalability and accuracy.

## Repository Structure

This monorepo contains four packages:

```
packages/
├── preprint/       # Paper draft (LaTeX)
├── experiments/    # Python code to reproduce results
├── docs/           # Documentation site (Docusaurus)
└── infra/          # AWS infrastructure (CDK TypeScript)
```

## Quick Start

### Prerequisites

- Node.js >= 22
- Yarn >= 4
- Python >= 3.10
- AWS CLI (for infrastructure deployment)

### Installation

```bash
# Enable Corepack for Yarn 4
corepack enable

# Install dependencies
yarn install

# Run experiments (see packages/experiments/README.md)
cd packages/experiments
uv sync
uv run python -m dfr_pinns

# Start documentation site
yarn docs:dev
```

### Upgrade Dependencies

```bash
uv lock --upgrade
uv sync
uv sync --extra pytorch
```

and 

```bash
yarn up "*"
```

## Packages

### Preprint (`packages/preprint`)

LaTeX source for the research paper presenting our adaptive hp-DFR method.

### Experiments (`packages/experiments`)

Python package implementing the adaptive hp-DFR methods with TensorFlow, JAX, and PyTorch backends. Includes benchmarks comparing against standard DFR and PINNs.

### Documentation (`packages/docs`)

Interactive documentation explaining the theory behind the methods, with tutorials and API reference.

```bash
yarn docs:dev    # Start dev server
yarn docs:build  # Build for production
```

### Infrastructure (`packages/infra`)

AWS CDK infrastructure for running experiments at scale using ECS/Fargate.

```bash
yarn infra:synth   # Synthesize CloudFormation
yarn infra:deploy  # Deploy to AWS
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
