---
sidebar_position: 3
sidebar_label: 'Goal-Oriented DFR'
---

# Goal-Oriented DFR: Plan

This is the working plan for the single publishable contribution: goal-oriented
error control for the Deep Fourier Residual loss.

## Motivation

We are often interested not in the global $H^1$ error but in a specific
**quantity of interest (QoI)**:

- Point evaluation: $J(u) = u(x_0)$
- Subdomain average: $J(u) = \frac{1}{|\omega|} \int_\omega u \, dx$
- Boundary flux: $J(u) = \int_{\Gamma} \nabla u \cdot n \, ds$

Goal-oriented methods focus computational effort on reducing the error in $J(u)$
rather than the global error.

## Dual-Weighted Residual framework

### Adjoint problem

For a linear QoI $J$, the adjoint solution $z \in V$ solves

$$
a(v, z) = J(v) \quad \forall v \in V,
$$

where $a(\cdot, \cdot)$ is the bilinear form of the primal problem.

### Error representation

For the primal approximation $u_h$, the QoI error satisfies

$$
J(u) - J(u_h) = a(u - u_h, z) = \langle R(u_h), z \rangle,
$$

which motivates the **goal-oriented loss**

$$
\mathcal{L}_{\text{QoI}}(u_h) = |\langle R(u_h), z_h \rangle|,
$$

where $z_h$ is a neural-network approximation of the adjoint. The dual pairing
$\langle R(u_h), z_h \rangle$ is evaluated with the same DFR spectral machinery
(DST/DCT projection) used for the $H^{-1}$ norm.

## Theory (the acceptance gate) - PROVED

These are the analogue of DFR's error-loss equivalence and are what a CMAME/JCP
reviewer will require. **Both are now proved** in `packages/preprint/sections/theory.tex`
(Proposition 4.2 and Theorem 4.3) under the Banach-Necas-Babuska hypotheses
(boundedness constant $M$, inf-sup constant $\gamma$).

### Proposition 4.2: Exact error representation

For $u^*$ solving the primal problem and $z^*$ the adjoint problem, and any
$u_h, z_h$,

$$
J(u^*) - J(u_h) = R(u_h)(z^*) = R(u_h)(z_h) + R(u_h)(z^* - z_h).
$$

The first term is the computable goal-oriented loss $\mathcal{L}_{\text{QoI}}$;
the second is the adjoint-approximation error. This is **exact** (an equality).

### Theorem 4.3: QoI error control

$$
|J(u^*) - J(u_h)| \leq \mathcal{L}_{\text{QoI}}(u_h, z_h)
  + \frac{1}{\gamma}\, \|R(u_h)\|_{V^*}\, \|R^*(z_h)\|_{U^*}.
$$

The three terms on the right are **exactly the three terms of the training loss**
(goal-oriented loss + primal DFR loss + adjoint DFR loss), so the bound is fully
computable a posteriori. The remainder is the *product* of the primal and adjoint
residuals (sharper than the additive form first sketched), giving the
second-order DWR accuracy: at fewer DOF than global energy minimization the QoI
error is controlled to $O(\delta^2)$ in the residual (Remark 4.5). A corollary
(4.4) specializes to an adjoint trained to a tolerance $\varepsilon$.

## Implementation status

The model and QoI classes are implemented in
`packages/experiments/src/hp_dfr/models/goal_oriented_dfr.py`:

- `GoalOrientedDFRModel` - primal + adjoint networks, QoI-weighted dual-norm
  loss, built on the dense full-tensor DST machinery (`fourier/transforms.py`).
- `PointEvaluationQoI`, `AverageValueQoI`, `BoundaryFluxQoI` - QoI classes
  (unit-tested in `tests/test_goal_oriented.py`).

The QoI classes are tested, and the end-to-end `fit()` training loop is
**validated in 1D** (2026-06-30, PyTorch): on the `sine` Poisson problem with a
point QoI at $x = \pi/4$, training converges with the primal DFR loss dropping
from $7.06$ to $0.022$ (the weak residual $f + \Delta u$ driven to zero), global
$L^2$ relative error $\approx 1.9\%$, and QoI error $\approx 6\times 10^{-3}$.
Validation uncovered and fixed three sign/forcing bugs in the loss (the weak
residual was built as $f - \Delta u$, the goal-oriented pairing dropped the
forcing, and the $L^2$ pairing omitted the quadrature measure).

## Experiment plan

Start in 1D, then 2D. For each QoI, show goal-oriented DFR beats plain DFR on the
QoI error at matched degrees of freedom.

### Stage 1 (now): 1D point evaluation

Poisson 1D with $J(u) = u(x_0)$. Compare goal-oriented DFR against plain DFR:
QoI error and global $L^2$/$H^1$ error vs epochs at matched DOF.

| Method | DOF | QoI error | Global $H^1$ error |
|--------|-----|-----------|--------------------|
| Plain DFR | - | - | - |
| Goal-Oriented DFR | - | - | - |

### Stage 2: 2D

Add a 2D Poisson problem and an L-shaped domain; QoIs = point, subdomain
average, boundary flux. Baselines: plain DFR, PINN, and a FEM-DWR reference (FEM
as the ground-truth QoI).

## Roadmap

The path from the current state to a submittable paper, in order. Milestone 1 is
the bottleneck and gates acceptance.

### M1 - Theory (the acceptance gate) - DONE
The QoI-error-control theorem is proved: Proposition 4.2 (exact error
representation) and Theorem 4.3 (QoI error control) in `sections/theory.tex`,
under the BNB hypotheses, with the loss terms appearing exactly as the bound's
right-hand side. Remaining theory polish (optional for v1): nonlinear QoIs, and
making the spectral-truncation error explicit rather than inherited from DFR.

### M2 - 1D evidence: DEFINITIVE NEGATIVE (documented in the paper)
After fixing four bugs (goal-oriented loss sign; plain-DFR residual sign;
output-layer tanh capping amplitude; `DFRModel.predict` hardcoded domain) and
adding an LBFGS polish, **both** methods solve 1D Poisson (smooth and sharp). The
result: plain DFR reaches its accuracy floor (~1e-5 QoI) at the smallest networks
(tens of params), so there is **no resolution-limited regime in 1D** for
goal-orientation to exploit; goal-oriented is uniformly less accurate on the QoI.
Written up honestly in `sections/results.tex` (with three reasons: DFR
saturates; single-network DFR has no mesh to focus; the non-smooth goal loss is
harder to optimize) and Figure `dfr-saturation-1d`. The abstract, intro, and
conclusion were corrected (they had wrongly claimed GO beats DFR). Figures are
built via `hp-dfr figures` (see `hp_dfr/figures.py`, `_plotting.py`).

### M3 - 2D and the headline experiment
Add a 2D Poisson problem and an **L-shaped domain** (re-entrant corner). QoIs:
point, subdomain average, boundary flux. Baselines: plain DFR, a PINN, and a
**FEM-DWR reference** (FEM as ground-truth QoI). Headline: goal-oriented beats
plain DFR on the QoI at fixed DOF, most strongly near the singularity.

### M4 - Write-up and submit
Fill `experiments.tex` / `results.tex` with real numbers and figures, tighten the
positioning against prior art (Govoeyi-Richter, Chakraborty-Wick - extend, do not
re-claim), verify all citations. Target **CMAME** or **JCP**.

### Risks
- **Scoop:** the Taylor/Pardo group is active and may present hp/goal-oriented DFR
  at WCCM/ECCOMAS 2026 - move with urgency.
- **Loss tuning:** `primal_weight=0.1` is light and full-batch Adam is noisy; M2/M3
  will likely need weight tuning and possibly an LBFGS polish.
- **Scope creep:** sparse and hp stay cut unless a reviewer specifically asks.

## Task checklist

- [x] Validate the goal-oriented `fit()` end-to-end on 1D Poisson (point QoI).
- [x] (M1) Prove the error representation and QoI-error-control theorem (preprint `sections/theory.tex`).
- [x] (M2) 1D comparison done: definitive NEGATIVE (plain DFR saturates ~1e-5 at
  tiny DOF, smooth and sharp; GO uniformly worse). Documented in the paper with a
  figure. Four bugs fixed + LBFGS added along the way.
- [ ] (M3) Add a 2D Poisson problem and the L-shape; point/average/flux QoIs.
- [ ] (M3) Add the FEM-DWR reference baseline.
- [ ] (M4) Fill experiments/results sections; submit to CMAME/JCP.
