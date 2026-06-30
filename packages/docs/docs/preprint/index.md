---
sidebar_position: 1
sidebar_label: Overview
---

# Our Research: Goal-Oriented DFR

This project extends the Deep Fourier Residual (DFR) method with **goal-oriented
error control**: instead of minimizing the global $H^{-1}$ residual norm, we
weight the loss toward a specific **quantity of interest (QoI)** using a
dual-weighted residual (DWR) formulation.

## The idea

The original DFR method ([arXiv:2210.14129](https://arxiv.org/abs/2210.14129))
minimizes the $H^{-1}$ dual norm of the weak residual, which is equivalent to
the $H^1$ energy-norm error for well-posed problems. In many applications,
however, the goal is not the global error but a functional of the solution: a
point value $u(x_0)$, a subdomain average, or a boundary flux.

We train a **primal network** $u_h$ and an **adjoint network** $z_h$ together,
and minimize a QoI-weighted residual functional

$$
\mathcal{L}_{\text{QoI}}(u_h) = |\langle R(u_h), z_h \rangle|,
$$

so that error control focuses on the QoI. The adjoint network is what makes this
dual-norm-of-a-QoI-residual computable in the DFR setting.

## Why this scope (and not more)

This project was originally scoped around three combined extensions (sparse
Fourier modes, hp-adaptive domain decomposition, and goal-oriented DFR). A
viability review (see the session log `notebooks/notes/logs/20260629-idea-reframing.md`)
narrowed it to goal-oriented DFR alone:

- **Sparse Fourier modes** are explicitly signposted as future work by the DFR
  authors, the $O(N (\log N)^{d-1})$ speedup claim is not sound for a
  non-separable neural-network residual (the real cost is autodiff/quadrature,
  not the mode count), and sparse truncation breaks the two-sided error-loss
  equivalence. High risk, low novelty.
- **hp-adaptive domain decomposition** is largely pre-empted by
  [arXiv:2401.04663](https://arxiv.org/abs/2401.04663) (Adaptive DFR via
  overlapping domain decomposition), which already does local DFR losses, Dorfler
  marking, and residual-based refinement with equivalence theory.
- **Goal-oriented DFR** is the least pre-empted: while goal-oriented PINNs exist,
  pairing DWR specifically with the DFR $H^{-1}$ dual-norm loss appears
  unoccupied. This is the defensible novel core.

## The publishable unit

A single-contribution paper, **Goal-Oriented Deep Fourier Residual Methods**,
targeting CMAME / JCP (the DFR family's home). The acceptance gate in that venue
is a theorem, not just experiments: a goal-oriented analogue of DFR's error-loss
equivalence, stating that the QoI-weighted dual-norm loss controls
$|J(u) - J(u_h)|$.

See the [Goal-Oriented DFR plan](/docs/preprint/phase3-goal-oriented) for the
DWR framework, the theorem targets, and the experiment plan, and the
[Literature Review](/docs/preprint/literature-review) for prior art and
positioning.

## Status

| Component | Status |
|-----------|--------|
| Goal-oriented model + QoI classes | Implemented (`models/goal_oriented_dfr.py`); QoI classes unit-tested |
| Dense DFR Fourier machinery | Implemented and tested (`fourier/transforms.py`) |
| 1D Poisson problems | Implemented (`problems/poisson_1d.py`) |
| End-to-end goal-oriented training run | Validated in 1D (point QoI; L2 rel err ~1.9%, QoI err ~6e-3) |
| QoI-error-control theorem | **Proved** (Prop. 4.2 + Thm 4.3 in preprint, M1 done) |
| 2D problems, FEM-DWR baselines | Not started (M3) |

See the [Roadmap](/docs/preprint/phase3-goal-oriented#roadmap) for the sequenced
plan (M1-M4) and risks.

## References

1. Taylor, J.M., Pardo, D., Muga, I. (2023). A Deep Fourier Residual Method for
   solving PDEs using Neural Networks. CMAME 405:115850.
   [arXiv:2210.14129](https://arxiv.org/abs/2210.14129)
2. Taylor, J.M., Bastidas, M., Calo, V.M., Pardo, D. (2024). Adaptive Deep
   Fourier Residual method via overlapping domain decomposition. CMAME.
   [arXiv:2401.04663](https://arxiv.org/abs/2401.04663)
3. Chakraborty, A., Wick, T., Zhuang, X., Rabczuk, T. (2021/2025).
   Multigoal-oriented dual-weighted-residual error estimation using deep neural
   networks. [arXiv:2112.11360](https://arxiv.org/abs/2112.11360)
4. Becker, R., Rannacher, R. (2001). An optimal control approach to a posteriori
   error estimation in finite element methods. Acta Numerica, 10, 1-102.
