# Goal-Oriented Deep Fourier Residual Methods

This repository contains the implementation and experiments for extending the Deep Fourier Residual (DFR) method with **goal-oriented error control** for solving PDEs using neural networks.

**Full documentation**: [https://caverac.github.io/hp-dfr](https://caverac.github.io/hp-dfr)

## Table of Contents

- [Our Research](#our-research)
  - [Key Idea](#key-idea)
- [Background: The Deep Fourier Residual Method](#background-the-deep-fourier-residual-method)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Packages](#packages)
- [Contributing](#contributing)
- [License](#license)

## Our Research

We extend the Deep Fourier Residual method with **goal-oriented error control**. Instead of minimizing the global H⁻¹ residual norm, we apply the dual-weighted residual (DWR) framework to the DFR loss: a primal network and an adjoint network are trained together, and the loss is the QoI-weighted residual functional `|<R(u), z>|`, evaluated with the same spectral machinery DFR uses for the H⁻¹ norm.

### Key Idea

The original DFR method establishes that the H⁻¹ dual-norm loss is equivalent to the H¹ error for well-posed problems. That controls the *global* error. For a quantity of interest (QoI) -- a point value, a subdomain average, a boundary flux -- this control is indirect. Goal-oriented DFR targets the QoI directly.

- **Contribution**: pairing DWR with the DFR H⁻¹ dual-norm loss specifically (distinct from existing goal-oriented PINN/Deep Ritz work).
- **Acceptance gate (theory)**: a goal-oriented analogue of the DFR error-loss equivalence, i.e., the QoI-weighted loss controls `|J(u) - J(u_h)|`.

> The project was originally scoped around three combined extensions (sparse Fourier modes, hp-adaptive domain decomposition, goal-oriented DFR). A June 2026 viability review narrowed it to goal-oriented DFR alone; the rationale (prior art, technical soundness, venue requirements) is recorded in `notebooks/notes/logs/20260629-idea-reframing.md`.

## Background: The Deep Fourier Residual Method

This work builds upon the Deep Fourier Residual method introduced in:

> **A Deep Fourier Residual Method for solving PDEs using Neural Networks**
> Jamie M. Taylor, David Pardo, Ignacio Muga
> [arXiv:2210.14129](https://arxiv.org/abs/2210.14129) | [Published Version](https://www.sciencedirect.com/science/article/abs/pii/S0045782522008064)

The original DFR paper provides the theoretical foundation for using dual norm losses in physics-informed neural networks. Our work extends this framework with adaptive strategies to improve scalability and accuracy.

### References

- **Reference Implementation**: [PINNS-and-DFR-examples](https://github.com/Mathmode/PINNS-and-DFR-examples) - Benchmarking platform by the MATHMODE group comparing PINNs and DFR implementations in TensorFlow, JAX, and PyTorch.

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
uv run hp-dfr --help
uv run hp-dfr research run --problem sine --qoi point

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

LaTeX source for the research paper presenting our goal-oriented DFR method.

### Experiments (`packages/experiments`)

Python package implementing goal-oriented DFR with TensorFlow and PyTorch backends, plus DFR and PINN baselines for comparison.

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
