# Goal-Oriented Deep Fourier Residual Methods

[![Documentation](https://img.shields.io/badge/docs-caverac.github.io%2Fhp--dfr-1f6feb)](https://caverac.github.io/hp-dfr/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](#license)

**Full documentation and results: [caverac.github.io/hp-dfr](https://caverac.github.io/hp-dfr/)**

A neural-network solver for partial differential equations that targets a specific
quantity of interest rather than the global error.

The Deep Fourier Residual (DFR) method solves a PDE by minimizing the $H^{-1}$ dual
norm of the weak residual -- a loss that is provably equivalent to the solution error.
This project extends DFR with **goal-oriented error control**: when the object of
interest is a single functional of the solution $J(u)$ -- a point value, a subdomain
average, a boundary flux -- rather than the solution everywhere, the loss is weighted
toward that functional using the dual-weighted residual (DWR) framework.

## The idea

A **primal** network $u_\theta$ approximates the solution and an **adjoint** network
$z_\phi$ approximates the sensitivity of $J$ to the residual. The primal network
minimizes the QoI-weighted residual

$$
\mathcal{L}_{\mathrm{QoI}}(u_\theta, z_\phi) = \bigl| \langle R(u_\theta),\, z_\phi \rangle \bigr|,
$$

while the adjoint network is trained on its own residual so that it approximates the
true adjoint. The central result is a goal-oriented counterpart of the DFR error-loss
equivalence, one-sided rather than an equivalence: under the standard well-posedness
(Banach-Necas-Babuska) hypotheses, with inf-sup constant $\gamma$,

$$
\bigl| J(u) - J(u_h) \bigr| \;\le\; \mathcal{L}_{\mathrm{QoI}}(u_h, z_h) \;+\; \frac{1}{\gamma}\, \lVert R(u_h) \rVert_{V^\ast}\, \lVert R^\ast(z_h) \rVert_{U^\ast},
$$

and every term on the right is computed during training. Measured across all 60
goal-oriented runs, the bound holds every time, at a median of ten times the true error.
The loss $\mathcal{L}_{\mathrm{QoI}}$ alone is **not** an error estimator, though: training
minimizes it directly, driving it some five orders of magnitude below the error that
remains, so the bound is carried by its remainder.

## Main result

Goal-orientation **reallocates accuracy from the global solution to the quantity of
interest**: at a matched training budget it reaches a QoI error about **15x smaller** than
plain DFR (Section 6.3) while its global energy error is about an **order of magnitude
larger**, on every seed. It therefore
helps wherever plain DFR leaves QoI-relevant residual unresolved, and hurts where it does
not. Sweeping the steepness `k` at fixed network and discretization shows both sides:

- **The problem is too hard for the network** (`k >= 8`): plain DFR's error grows **84x**
  from `k=2` to `k=16` while goal-orientation's grows **11x**, so goal-orientation ends up
  **6 to 15 times** more accurate.
- **The training budget is too small** (easy problem, larger network): plain DFR has not
  converged, and goal-orientation extracts more QoI accuracy from the same budget (**~15x**).
- **Neither** (easy problem, small network): goal-orientation is **2.4x worse**. The method
  can hurt.

Two honest caveats. Plain DFR is **not trained to convergence** at the shared budget (its
error is still falling at 10x the steps), so these are matched-_budget_, not matched-accuracy
comparisons, and the magnitudes on easy problems would shrink under a longer schedule; the
mechanism and the large-`k` advantage do not depend on it. And goal-oriented solutions are
accurate at the QoI but ~10x worse globally, so they are trustworthy at the quantity of
interest and nowhere else. The theory predicts none of this.

The figures, numbers, and reproduction commands are in the
[documentation](https://caverac.github.io/hp-dfr/docs/preprint/phase3-goal-oriented#results).

> The project was originally scoped around three combined extensions (sparse Fourier
> modes, hp-adaptive domain decomposition, and goal-oriented DFR). A June 2026 review
> narrowed it to goal-oriented DFR alone; the rationale is recorded in
> `notebooks/notes/logs/20260629-idea-reframing.md`.

## Quick start

**Prerequisites:** [uv](https://github.com/astral-sh/uv) (Python >= 3.11), and
Node.js >= 22 with Yarn >= 4 for the documentation site.

```bash
git clone https://github.com/caverac/hp-dfr.git
cd hp-dfr

# Python package (experiments). Install a backend extra; on Intel macOS use
# `uv pip install 'torch>=2.0.0,<2.2.0'` instead of the pytorch extra.
uv sync --extra pytorch

# Run a goal-oriented experiment: 1D sharp problem, point QoI at the 65% location.
uv run hp-dfr research run --problem arctan --qoi point --qoi-location 0.65

# See all commands (pinns / dfr / research / figures / backends).
uv run hp-dfr --help
```

Reproduce the paper figures (generate the sweep data, then render):

```bash
uv run hp-dfr data all   # both sweeps; about 20 minutes on a laptop CPU
uv run hp-dfr figures    # renders the three figures from the saved JSON
```

Run the documentation site locally:

```bash
yarn install
yarn docs:dev      # http://localhost:3000
```

## Repository structure

```
packages/
  preprint/       # Manuscript (LaTeX, single-file ms.tex)
  experiments/    # Python package: goal-oriented DFR + DFR/PINN baselines
  docs/           # Documentation site (Docusaurus)
```

## Development

Quality gates run in CI and via [pre-commit](https://pre-commit.com):

```bash
uv run pre-commit install      # black, isort, flake8, pylint, pydocstyle, mypy, prettier, eslint
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org); releases
and the version bump are automated by semantic-release on merge to `main`.

## Background

This work builds on the Deep Fourier Residual method:

> **A Deep Fourier Residual Method for solving PDEs using Neural Networks.**
> Jamie M. Taylor, David Pardo, Ignacio Muga.
> _Computer Methods in Applied Mechanics and Engineering_ **405** (2023) 115850.
> [arXiv:2210.14129](https://arxiv.org/abs/2210.14129) |
> [publisher](https://doi.org/10.1016/j.cma.2022.115850)

Reference implementations by the MATHMODE group:
[PINNS-and-DFR-examples](https://github.com/Mathmode/PINNS-and-DFR-examples).

## License

Released under the MIT License.
