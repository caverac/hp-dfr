---
sidebar_position: 3
sidebar_label: "Method and Results"
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Goal-Oriented DFR: Method and Results

This page develops the method, states the error guarantee, and presents the 1D and
2D experiments. Each figure is accompanied by the command that reproduces it. It
assumes the [Overview](/docs/preprint) and the [DFR theory page](/docs/theory/dfr).

## The method

### Quantities of interest

A quantity of interest (QoI) is a functional of the solution. Three linear examples:

- **Point evaluation:** $J(u) = u(x_0)$
- **Subdomain average:** $J(u) = |\omega|^{-1} \displaystyle\int_\omega u \, dx$
- **Boundary flux:** $J(u) = \displaystyle\int_{\Gamma} \nabla u \cdot n \, ds$

Goal-oriented methods reduce the error in $J(u)$ rather than the error everywhere.

### The adjoint problem

Write the PDE in weak form as $a(u, v) = \ell(v)$ for all test functions $v$. For a
linear QoI $J$, the **adjoint solution** $z$ is defined by

$$
a(v, z) = J(v) \quad \text{for all } v.
$$

The adjoint quantifies the sensitivity of $J$ to residual error: where $z$ is large,
a residual contributes strongly to the QoI error.

### From error to loss

The QoI error satisfies the exact **error representation**

$$
J(u) - J(u_h) = a(u - u_h, z) = \langle R(u_h),\, z \rangle,
$$

where $R(u_h)$ is the residual of the primal approximation. Replacing the exact
adjoint by a network $z_h$ gives the computable **goal-oriented loss**

$$
\mathcal{L}_{\text{QoI}}(u_h, z_h) = \bigl|\langle R(u_h),\, z_h \rangle\bigr|.
$$

The pairing $\langle R(u_h), z_h \rangle = \int_\Omega (f + \Delta u_h)\, z_h\,dx$ is
an $L^2$ integral, evaluated on the same quadrature grid used to assemble the DFR
transforms.

### Training objective

Training minimizes the goal-oriented loss together with two DFR regularizers:

$$
\mathcal{L}(u_h, z_h) = \underbrace{|\langle R(u_h), z_h\rangle|}_{\text{goal}}
  + \alpha\,\underbrace{\|R^*(z_h)\|_{H^{-1}}^2}_{\text{adjoint DFR}}
  + \beta\,\underbrace{\|R(u_h)\|_{H^{-1}}^2}_{\text{primal DFR}}.
$$

One detail is essential. The goal term must update the **primal** network only: it
drives $u_h$ to reduce the residual where the adjoint is large. If the adjoint
parameters were also allowed to descend this term, the optimizer could drive the
pairing to zero by making $z_h$ orthogonal to the current residual, without $z_h$
solving the adjoint equation. The adjoint is therefore held fixed in the goal term
(a stop-gradient), so it is trained solely by its own residual
$\|R^*(z_h)\|_{H^{-1}}^2$ and approximates the true adjoint. Both networks enforce the
homogeneous Dirichlet conditions exactly through a cutoff construction, so no boundary
penalty is needed.

## The guarantee

The central result is a goal-oriented counterpart of the DFR error&ndash;loss
equivalence, under the standard well-posedness hypotheses (the
Banach&ndash;Necas&ndash;Babuska conditions: a boundedness constant $M$ and an
inf&ndash;sup constant $\gamma$). It is one-sided: a reliability bound, not an
equivalence. No lower bound holds, since an adjoint far from $z^*$ makes the loss
positive while the QoI error may vanish.

**Exact error representation (Proposition 4.2).** For any primal $u_h$ and adjoint
$z_h$,

$$
J(u) - J(u_h) = \underbrace{\langle R(u_h), z_h\rangle}_{\text{computable}}
  + \underbrace{\langle R(u_h), z - z_h\rangle}_{\text{adjoint error}}.
$$

The first term is the goal-oriented loss; the second involves the unknown exact
adjoint and is bounded next.

**QoI error control (Theorem 4.3).**

$$
\bigl| J(u) - J(u_h) \bigr|
  \;\le\;
  \mathcal{L}_{\text{QoI}}(u_h, z_h)
  \;+\;
  \frac{1}{\gamma}\,\|R(u_h)\|_{V^*}\,\|R^*(z_h)\|_{U^*}.
$$

The left-hand side is the QoI error. The right-hand side consists of the three
quantities minimized during training: the goal-oriented loss, the primal DFR
residual, and the adjoint DFR residual. The bound is therefore computable a
posteriori, without the true solution.

The remainder is a _product_ of the primal and adjoint residuals. If both are trained
to size $\delta$, the remainder is $O(\delta^2)$. Three qualifications matter, because
the natural reading of that sentence overreaches on all three.

First, the $O(\delta^2)$ statement presupposes $\delta$ is small, that is, that both
networks are well resolved. It says nothing about a resolution-limited regime, where
$\delta$ is not small and the bound is correspondingly weak. The bound is weakest
exactly where goal-orientation turns out to help.

Second, the classical second-order gain rests on Galerkin orthogonality, which has no
counterpart here. Minimizing the goal loss enforces a single scalar condition in its
place, which is enough for the error representation to reduce to the remainder, but is
a far weaker constraint and is satisfiable without $u_h$ being accurate.

Third, the goal loss is therefore **not** an error estimator for a network trained on
it. Training minimizes it directly, so it lands near $10^{-10}$ while the true QoI
error sits near $10^{-4}$. The useful computable object is the whole right-hand side of
the bound, which is carried by the remainder. Measured across all runs, that bound
holds every time, at a median of ten times the true error.

The next section shows both regimes.

## Results

The comparison against plain DFR is at matched primal-network degrees of freedom
(the same primal-network architecture for both), sweeping the network width and
repeating over three random seeds. The goal-oriented method also trains an adjoint
network of the same size, so it carries about twice the parameters and twice the
per-step cost at a given abscissa; the adjoint is auxiliary and discarded once the
primal is trained. The metric is the error in a mollified point QoI,
$|J_\sigma(u_h) - J_\sigma(u^*)|$; markers are medians over seeds and bands span the
seed min&ndash;max.

### One dimension: the baseline saturates

The 1D problems are a smooth solution ($u = \sin 2x$) and a peaked one (an $\arctan$
profile with its gradient concentrated near the center).

<figure class="scientific">
  <img src={useBaseUrl('/img/figures/dfr-saturation-1d.png')} alt="QoI error versus degrees of freedom in 1D for plain DFR and goal-oriented DFR" />
  <figcaption>QoI error versus primal-network degrees of freedom on the 1D Poisson
  problems (smooth, top; peaked, bottom). Plain DFR reaches its optimization floor
  (about $3\times 10^{-5}$) at the smallest networks, so goal-orientation offers no
  advantage &mdash; at either loss weight, including the one the 2D experiments use,
  which rules out the weight as the explanation for the 2D result. Markers are medians
  over random seeds; bands span the seed min&ndash;max.</figcaption>
</figure>

Reproduce with:

```bash
# 1. Generate the sweep data (requires a PyTorch backend; a few minutes).
#    On Intel macOS, install the backend once: uv pip install 'torch>=2.0.0,<2.2.0'
uv run hp-dfr data 1d

# 2. Build the figure from the saved data (fast).
uv run hp-dfr figures
```

The first command writes `assets/m2_1d_data.json`; the second reads it and writes
`assets/dfr-saturation-1d.png` and `.pdf`.

Plain DFR reaches its optimization floor (a QoI error of order $10^{-5}$) at the
smallest networks, a few tens of parameters. The solution is then accurate
everywhere, so the remainder in the bound is negligible and goal-orientation has no
error to reallocate; its two-network objective is somewhat harder to optimize, so it
is slightly less accurate at small and moderate sizes and reaches parity only at the
largest networks. One dimension is a control: it confirms the method does not report
an advantage where none is available.

### Two dimensions: the advantage

The 2D problem is a sharp separable bump on the unit square. Both methods use the
identical full tensor-product DST discretization, so the comparison isolates the
effect of goal-orientation. Unlike the 1D problems, a network of practical size does
not resolve this solution everywhere: plain DFR sits at a QoI error of $10^{-4}$ at
the smallest networks and only reaches $10^{-5}$ as the width grows.

<figure class="scientific">
  <img src={useBaseUrl('/img/figures/dfr-vs-go-2d.png')} alt="QoI error versus degrees of freedom in 2D for plain DFR and goal-oriented DFR" />
  <figcaption>QoI error versus primal-network degrees of freedom on the 2D arctan-bump
  Poisson problem. Goal-oriented DFR improves the median point-QoI error by 2.3 to 3.9
  times (3.1 times on a geometric mean) at matched primal-network degrees of freedom,
  winning 33 of 40 runs. Markers are medians over ten random seeds; bands span the seed
  min&ndash;max.</figcaption>
</figure>

Reproduce with:

```bash
# 1. Generate the 2D sweep data (requires a PyTorch backend; ~15 minutes).
uv run hp-dfr data 2d

# 2. Build the figure from the saved data (fast).
uv run hp-dfr figures
```

The first command writes `assets/m3_2d_data.json`; the second reads it and writes
`assets/dfr-vs-go-2d.png` and `.pdf`.

In this resolution-limited regime goal-orientation is the more accurate method at
every size, improving the median QoI error by 1.2 to 5.3 times (a geometric mean of
2.7) and winning on ten of the twelve configurations. Neither method is monotone in
the degrees of freedom, and the near-parity at 337 reflects the goal-oriented method
degrading rather than the baseline improving:

| Degrees of freedom |          Plain DFR |      Goal-oriented | Improvement |
| -----------------: | -----------------: | -----------------: | ----------: |
|                105 | $2.4\times10^{-4}$ | $4.5\times10^{-5}$ | $5.3\times$ |
|                337 | $1.6\times10^{-5}$ | $1.3\times10^{-5}$ | $1.2\times$ |
|                697 | $7.9\times10^{-6}$ | $2.8\times10^{-6}$ | $2.8\times$ |
|               1185 | $1.0\times10^{-5}$ | $3.2\times10^{-6}$ | $3.1\times$ |

The advantage is largest at the smallest network, where plain DFR is most starved of
resolution. Neither method is monotone in the degrees of freedom, and the near-parity at
337 is not the baseline doing unusually well &mdash; there plain DFR is in fact twice
worse than at 697. What narrows the gap is goal-orientation degrading. With three seeds
and a seed-to-seed spread reaching a factor of fifty at that size, we do not read the
337 column as evidence of a mechanism.

### Varying the difficulty directly

The sweeps above vary the network size at a fixed problem. They show _that_
goal-orientation helps in 2D and not in 1D, but not _why_: the two settings differ in more
than the difficulty of the problem. Sweeping the steepness `k` at a fixed network and a
fixed discretization isolates it.

The mode count is held at 24 throughout, chosen for the sharpest case rather than inherited
from the sweep above. This matters: the energy the sine basis fails to capture grows from
8e-8 at `k=8` to 1e-3 at `k=32`, so holding the count at 16 would make the test space, not
the network, the binding constraint, and the experiment would measure mode truncation while
appearing to measure resolution.

<figure class="scientific">
  <img src={useBaseUrl('/img/figures/dfr-vs-go-steepness.png')} alt="QoI error versus solution steepness for plain DFR and goal-oriented DFR at two network sizes" />
  <figcaption>QoI error versus steepness $k$ at fixed network size (105 DOF, top; 337,
  bottom) and fixed discretization, so $k$ is the only variable. Plain DFR sits at its
  optimization floor while the problem is easy, then degrades sharply (84x from $k=2$ to
  $k=16$ at 105 DOF); goal-oriented DFR degrades 11x, so the gap opens as the problem
  hardens and the curves cross near $k \approx 5$. Below the crossover goal-orientation is
  the worse method. Markers are medians over ten random seeds; bands span the seed
  min&ndash;max.</figcaption>
</figure>

Reproduce with `uv run hp-dfr data steepness` followed by `uv run hp-dfr figures`.

Three regimes appear:

- **Resolution-limited** (`k >= 8`): goal-orientation is 5.9x better at `k=8` and 11.1x at
  `k=16` (9/10 runs each). This is the regime the method is for, now isolated rather than
  inferred.
- **A real loss** (`k=4`, 105 DOF): goal-orientation is 2.4x _worse_, winning 2 of 10 runs.
  The problem is easy enough that there is no misallocated capacity to recover, and the
  second network is a pure cost.
- **Stalled at the floor** (`k=2`, 337 DOF): on the _easiest_ problem, goal-orientation is
  15.4x better, winning 9 of 10. This is not resolution limitation. Plain DFR's error there
  is flat to 1.3x across a fourfold change in difficulty, the signature of an optimization
  floor; goal-orientation reaches 15x below it.

What unites the two wins is that plain DFR has stopped making progress, for different
reasons: against the network's capacity in the first case, against its own accuracy floor
in the second. Where goal-orientation loses is where neither stall has set in. The
organizing variable is not how hard the problem is but whether the baseline is still
improving.

### Does the bound hold, and is the loss an estimator?

Every term of the bound is computed at each training step, so it can be checked rather
than assumed. Recording all three at the end of each of the 168 goal-oriented runs gives
two findings that pull in opposite directions.

<figure class="scientific">
  <img src={useBaseUrl('/img/figures/bound-verification.png')} alt="The QoI error bound and the goal term plotted against the true QoI error" />
  <figcaption>The bound and the goal term against the QoI error actually committed (1D,
  top; 2D, bottom); the dashed diagonal is equality. The bound (circles) holds on every
  one of the 168 runs across the whole study, exceeding the true error by a median factor
  of 15. The goal term alone (squares) lies about five orders of magnitude <em>below</em>
  the error it is supposed to estimate.</figcaption>
</figure>

Reproduce with `uv run hp-dfr data all` followed by `uv run hp-dfr figures`.

**The bound holds, and is usefully tight.** No violation on any run, despite the dual
norms being evaluated on truncated spectral sums that underestimate the true norms. It
exceeds the true error by a median factor of 9.8 (1D) and 17.7 (2D). Within an order of
magnitude, and never optimistic, is a useful a posteriori bound.

**The goal loss is not an error estimator.** Across the runs it has a median value of
$2\times 10^{-10}$ against a median QoI error near $10^{-5}$, and never once exceeds
the error. Training minimizes it directly, and being a single scalar condition it is
driven to zero without $u_h$ becoming accurate. Reporting it as an error estimate would
understate the error by five orders of magnitude; the quantity to use is the whole
right-hand side of the bound, which the remainder carries. This is a general hazard of
training on an a posteriori quantity: any residual functional that appears in the loss
is disqualified from also estimating what the loss failed to remove.

### Why the two regimes differ

Goal-orientation helps in two situations that look opposite and are not. In the first the
problem is too hard for the network, so the residual cannot be made uniformly small and the
goal-oriented objective spends a limited budget where the adjoint says it matters. In the
second the problem is easy and there is capacity to spare, but plain DFR stops improving
anyway, at a floor near $10^{-5}$ that is insensitive to the difficulty of the problem.

What the two share is that plain DFR's optimization has stalled: against the network's
capacity in the first case, against its own accuracy floor in the second. In both, the
residual left over is distributed without regard to the QoI, and reweighting it recovers
accuracy the baseline leaves on the table. Where goal-orientation loses is where neither
stall has set in: the baseline is still converging, nothing is misallocated, and the second
network is pure cost. The organizing variable is not how hard the problem is but whether the
baseline is still making progress, which is why "goal-orientation helps when the problem is
hard" is wrong at both ends of the steepness sweep.

The theorem predicts neither behavior. Its remainder is small only when both networks are
well resolved, which is what fails where the gains appear, so it accommodates the results
without explaining them. It has still less to say about the second regime, which is a claim
about where an optimizer stops, while the analysis is entirely approximation-theoretic and
contains no model of training at all.

The floor deserves its own note. Plain DFR's error is flat to 1.3x across a fourfold change
in problem difficulty, and to a factor of two across a twelvefold change in network size. A
limit insensitive to both is not an approximation limit; it is where the optimizer stops.
Diagnosing it is open, and it matters: a method whose accuracy is set by its optimizer
rather than its discretization is one whose error analysis describes something other than
what limits it in practice.

Two implementation points are necessary to observe this behavior: the adjoint must be
trained on its own residual and held fixed in the goal term, and in two dimensions the
$H^{-1}$ weights must use the true eigenvalues $\lambda_k = \sum_i (\pi k_i/L_i)^2$
rather than a separable product of one-dimensional weights.

## Using the code

The method is in `packages/experiments/src/hp_dfr/models/goal_oriented_dfr.py`:

- `GoalOrientedDFRModel` &mdash; primal and adjoint networks with the QoI-weighted
  dual-norm loss, on the dense full-tensor DST machinery in `fourier/transforms.py`.
  Supports 1D and 2D.
- `PointEvaluationQoI`, `AverageValueQoI`, `BoundaryFluxQoI` &mdash; the
  quantity-of-interest classes (unit-tested in `tests/test_goal_oriented.py`).

A command-line run:

```bash
# Goal-oriented DFR on the 1D sharp problem, point QoI at the 65% location.
uv run hp-dfr research run \
  --problem arctan --backend pytorch \
  --qoi point --qoi-location 0.65 \
  --hidden-layers "16,16" --n-modes 60 --epochs 2000
```

The [Quick Start](/docs/getting-started/quick-start) covers the Python API and the
remaining options.
