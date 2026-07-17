---
sidebar_position: 1
sidebar_label: Overview
---

# Our Research: Goal-Oriented DFR

This section presents goal-oriented error control for the Deep Fourier Residual
method: the construction, the guarantee it satisfies, and what the experiments
establish. It assumes the [Introduction](/docs/intro) and the
[DFR theory page](/docs/theory/dfr).

## Summary

The DFR loss minimizes the $H^{-1}$ norm of the weak residual, a quantity
equivalent to the global energy-norm error. When the object of interest is a single
functional of the solution &mdash; a **quantity of interest** (QoI) such as a point
value, an average, or a flux &mdash; goal-oriented DFR weights the loss toward that
functional. It trains a **primal** network $u_h$ together with an **adjoint** network
$z_h$ and minimizes a QoI-weighted residual. The construction comes with a theorem
bounding the QoI error, and the experiments show that it improves QoI accuracy in the
resolution-limited regime, and not otherwise.

## Construction

1. **Quantity of interest.** Fix a functional $J$ of the solution. The experiments
   use a mollified point value $J(u) = u(x_0)$.

2. **Adjoint problem.** For a linear $J$, the adjoint solution $z$ satisfies
   $a(v, z) = J(v)$ for all test functions $v$. It measures how strongly a residual
   error in each region affects $J$. It is approximated by a second network $z_h$.

3. **Goal-oriented loss.** The QoI error has an exact representation as the residual
   paired with the adjoint, which the primal network minimizes:

   $$
   \mathcal{L}_{\text{QoI}}(u_h, z_h) = \bigl|\langle R(u_h),\, z_h \rangle\bigr|.
   $$

The [Method and Results](/docs/preprint/phase3-goal-oriented) page develops the
adjoint problem, the error representation, the theorem, and the experiments.

## The guarantee

The central result is a goal-oriented counterpart of the DFR error&ndash;loss
equivalence. The QoI-weighted loss, plus a remainder built from the primal and
adjoint residuals, bounds the error in the quantity of interest, and every term of
the bound is computed during training. Unlike the DFR estimate the bound is
one-sided, and the loss alone is not an error estimator: training drives it far below
the error that remains, so the bound is carried by its remainder. That remainder is a
product of the two residuals; it is small only when both networks are accurate, which
means the bound is weakest in the very regime where goal-orientation proves useful. The precise statement is Theorem 4.3, on the
[Method and Results](/docs/preprint/phase3-goal-oriented#the-guarantee) page.

## What the experiments establish

The comparison against plain DFR is run at matched network sizes on Poisson problems
in one and two dimensions.

- **Resolution-limited: a large advantage.** As the solution sharpens at a fixed
  network, plain DFR's error grows 84 times while goal-orientation's grows 11.
  Goal-orientation ends up 6 to 15 times more accurate, winning 9 of 10 runs.

- **Stalled at the floor: also an advantage.** On the _easiest_ problem tested, plain
  DFR stops at an error near $10^{-5}$ that is flat across both problem difficulty and
  network capacity. Goal-orientation passes it by 15 times. This contradicts the
  natural expectation that there is nothing to gain once the global error is resolved.

- **Neither: a real loss.** On easy problems at small networks, goal-orientation is 2.4
  times _worse_, winning 2 of 10 runs. The method is not uniformly beneficial.

What unites the two wins is that plain DFR has stopped making progress, not that the
problem is hard. The theory predicts neither.

Both cases, with the figures and the exact commands that reproduce them, are on the
[Method and Results](/docs/preprint/phase3-goal-oriented#results) page.

## Scope

This project targets goal-oriented error control for DFR. Two adjacent directions are
out of scope, for reasons of novelty and existing coverage (see
[Background and Positioning](/docs/preprint/literature-review#how-this-project-is-scoped)):

- **Sparse Fourier modes.** The original DFR authors list better basis choices as
  future work; the hoped-for complexity savings rely on regularity a neural-network
  residual does not guarantee, and aggressive truncation weakens the two-sided
  error&ndash;loss equivalence.
- **hp-adaptive domain decomposition.** Much of this is provided by
  [Adaptive DFR](https://arxiv.org/abs/2401.04663) (overlapping subdomains, local
  DFR losses, Dorfler marking, refinement, with equivalence theory), leaving a
  narrow remaining gap.
- **Goal-oriented DFR.** Pairing the dual-weighted residual specifically with the
  DFR $H^{-1}$ loss is the least explored of these directions, and is the focus here.

## Status

| Component                         | Status                                                           |
| --------------------------------- | ---------------------------------------------------------------- |
| Goal-oriented model + QoI classes | Implemented (`models/goal_oriented_dfr.py`), unit-tested         |
| Dense DFR Fourier machinery       | Implemented and tested (`fourier/transforms.py`)                 |
| 1D and 2D Poisson problems        | Implemented (`problems/poisson_1d.py`, `problems/poisson_2d.py`) |
| QoI-error-control theorem         | Proved (Proposition 4.2 + Theorem 4.3 in the manuscript)         |
| 1D experiment (control)           | Done &mdash; figure `dfr-saturation-1d`                          |
| 2D experiment (advantage)         | Done &mdash; figure `dfr-vs-go-2d`                               |

## References

1. Taylor, J.M., Pardo, D., Muga, I. (2023). A Deep Fourier Residual Method for
   solving PDEs using Neural Networks. _CMAME_ **405**:115850.
   [arXiv:2210.14129](https://arxiv.org/abs/2210.14129)
2. Taylor, J.M., Bastidas, M., Calo, V.M., Pardo, D. (2024). Adaptive Deep Fourier
   Residual method via overlapping domain decomposition. _CMAME_.
   [arXiv:2401.04663](https://arxiv.org/abs/2401.04663)
3. Chakraborty, A., Wick, T., Rabczuk, T., Zhuang, X. (2025). Multigoal-oriented
   dual-weighted-residual error estimation using PINNs. _Machine Learning for
   Computational Science and Engineering_ **1**(1):13.
   [doi:10.1007/s44379-025-00012-4](https://doi.org/10.1007/s44379-025-00012-4)
4. Becker, R., Rannacher, R. (2001). An optimal control approach to a posteriori
   error estimation in finite element methods. _Acta Numerica_ **10**, 1&ndash;102.
