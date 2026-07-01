---
sidebar_position: 1
---

# Introduction

The Deep Fourier Residual (DFR) method solves partial differential equations with
neural networks using a loss that is provably equivalent to the solution error.
This project extends DFR to _goal-oriented_ error control: when the object of
interest is a single functional of the solution rather than the solution
everywhere, the loss can be weighted toward that functional. This page states the
problem, the method, and the main empirical finding, and points to the pages that
develop each in detail.

## Solving PDEs with neural networks

A partial differential equation (PDE) relates an unknown field to its derivatives.
The model problem throughout these pages is Poisson's equation,

$$
-\,u''(x) = f(x),
$$

with $f$ a known source and $u$ the unknown solution. A neural-network solver
represents $u$ by a network $u_\theta$ with parameters $\theta$ and chooses $\theta$
to make $u_\theta$ satisfy the equation as closely as possible. What "as closely as
possible" means is fixed by the loss function, and the choice of loss determines
how faithfully the minimizer approximates the true solution.

## The DFR loss and its guarantee

The residual of a candidate solution is the amount by which it fails to satisfy the
equation. Minimizing the pointwise size of the residual is the basis of
physics-informed neural networks (PINNs). The size of the residual, however, is not
in general proportional to the size of the solution error: a small residual can
coexist with a solution that is still inaccurate.

DFR measures the residual in the $H^{-1}$ dual norm, which it evaluates with a
Fourier transform. Under the standard well-posedness conditions this norm is
equivalent to the energy-norm error, so the DFR loss and the error vanish together
and track each other during training. The [DFR theory page](/docs/theory/dfr)
derives the weak form, the dual norm, and the Fourier computation in full.

## Quantities of interest

The DFR guarantee concerns the global error. Applications frequently require instead
a single functional of the solution, a **quantity of interest** (QoI):

- a point value, $\;J(u) = u(x_0)$;
- an average over a region, $\;J(u) = |\omega|^{-1}\int_\omega u\,dx$;
- a boundary flux, $\;J(u) = \int_\Gamma \nabla u \cdot n \, ds$.

A small global error is sufficient but not necessary for an accurate QoI, and
driving the global error below a tolerance can be wasteful when only $J(u)$ matters.

## Goal-oriented DFR

The dual-weighted residual (DWR) method controls a single QoI in the finite-element
setting by introducing an adjoint problem whose solution measures the sensitivity of
$J$ to residual error in each region. Goal-oriented DFR applies this construction to
the DFR loss. Together with the primal network $u_\theta$ it trains an adjoint
network $z_\phi$, and it weights the primal loss toward the QoI through the residual
pairing

$$
\mathcal{L}_{\text{QoI}}(u_\theta, z_\phi) = \bigl|\langle R(u_\theta),\, z_\phi \rangle\bigr|.
$$

The [method and results page](/docs/preprint/phase3-goal-oriented) develops the
adjoint problem, the error representation, and the training objective, including the
one detail that makes the adjoint informative: it is trained on its own residual and
held fixed in the pairing, so it approximates the true adjoint rather than collapsing
to a value that makes the pairing vanish.

## Main finding

The method rests on two results, one theoretical and one empirical.

The theorem is a goal-oriented analogue of the DFR guarantee: the QoI-weighted loss,
together with a computable remainder, bounds the error in the quantity of interest,
and every term of the bound is available during training. The remainder is a
_product_ of the primal and adjoint residuals, which is the source of the empirical
behavior below.

The experiments locate where goal-orientation helps. On 1D Poisson problems, plain
DFR reaches its optimization floor (an error of order $10^{-5}$) at very small
networks; the solution is already accurate everywhere, the remainder is negligible,
and goal-orientation offers no advantage. On a 2D problem with a sharp feature, a
network of practical size cannot resolve the solution everywhere, the remainder is
no longer negligible, and goal-orientation reduces the QoI error by roughly three to
five times at matched degrees of freedom. The [results](/docs/preprint/phase3-goal-oriented#results)
present both cases, with the command that reproduces each figure.

## Scope

This project focuses on goal-oriented error control for DFR. Two adjacent directions
&mdash; sparse Fourier modes and hp-adaptive domain decomposition &mdash; are out of
scope; the [background and positioning page](/docs/preprint/literature-review)
explains why and places the work among related methods.

## Reading path

1. [Install the package](/docs/getting-started/installation) and
   [run an experiment](/docs/getting-started/quick-start).
2. [Theory: PINNs](/docs/theory/pinns) and [Theory: DFR](/docs/theory/dfr) &mdash;
   the two ingredients.
3. [Our research](/docs/preprint) &mdash; the goal-oriented method, the theorem, and
   the 1D and 2D results with reproducible figures.
4. [Background: the original DFR method](/docs/paper/summary) and the
   [API reference](/docs/api/pytorch).

## Reference

This work builds on the Deep Fourier Residual method:

> **A Deep Fourier Residual Method for solving PDEs using Neural Networks.**
> Jamie M. Taylor, David Pardo, Ignacio Muga.
> _Computer Methods in Applied Mechanics and Engineering_ **405** (2023) 115850.
> [arXiv:2210.14129](https://arxiv.org/abs/2210.14129)
