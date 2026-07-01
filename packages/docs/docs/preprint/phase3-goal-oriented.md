---
sidebar_position: 3
sidebar_label: "Method and Results"
---

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

The central result is a goal-oriented analogue of the DFR error&ndash;loss
equivalence, under the standard well-posedness hypotheses (the
Banach&ndash;Necas&ndash;Babuska conditions: a boundedness constant $M$ and an
inf&ndash;sup constant $\gamma$).

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
to size $\delta$, the goal loss estimates the QoI error to $O(\delta^2)$. This
second-order structure has a direct consequence for the experiments: when the primal
residual is already at its floor, the remainder is negligible and goal-orientation
cannot improve the QoI; when the primal residual cannot be made uniformly small, the
remainder is significant and goal-orientation reallocates the network's capacity
toward the QoI. The next section shows both regimes.

## Results

The comparison against plain DFR is at matched degrees of freedom (the same
primal-network architecture for both), sweeping the network width and repeating over
random seeds. The metric is the error in a mollified point QoI,
$|J_\sigma(u_h) - J_\sigma(u^*)|$; markers are medians over seeds and bands span the
seed min&ndash;max.

### One dimension: no advantage

The 1D problems are a smooth solution ($u = \sin 2x$) and a sharp one (an $\arctan$
profile with its gradient concentrated near the center).

![QoI error versus degrees of freedom in 1D, for plain DFR and goal-oriented DFR, on a smooth (top) and a sharp (bottom) problem.](/img/figures/dfr-saturation-1d.png)

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

![QoI error versus degrees of freedom in 2D, for plain DFR and goal-oriented DFR.](/img/figures/dfr-vs-go-2d.png)

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
almost every size, improving the median QoI error by roughly three to five times and
winning on ten of the twelve configurations:

| Degrees of freedom |          Plain DFR |      Goal-oriented | Improvement |
| -----------------: | -----------------: | -----------------: | ----------: |
|                105 | $2.4\times10^{-4}$ | $4.5\times10^{-5}$ | $5.3\times$ |
|                337 | $1.6\times10^{-5}$ | $1.3\times10^{-5}$ | $1.2\times$ |
|                697 | $7.9\times10^{-6}$ | $2.8\times10^{-6}$ | $2.8\times$ |
|               1185 | $1.0\times10^{-5}$ | $3.2\times10^{-6}$ | $3.1\times$ |

The advantage is largest at the smallest network, where plain DFR is most starved of
resolution, and persists at the larger sizes as the adjoint network becomes well
resolved. The two methods are comparable only at the intermediate size, where plain
DFR happens to resolve the bump particularly well.

### Why the two regimes differ

The contrast follows from the second-order remainder in Theorem 4.3. When plain DFR
already drives the primal residual to its floor (1D), the remainder is negligible and
there is no error to reallocate; the extra adjoint network only complicates the
optimization. When the primal residual cannot be made uniformly small at feasible
cost (2D), the goal-oriented objective spends the network's limited capacity where
the adjoint indicates it matters for the QoI, and the several-fold accuracy gain
follows. Goal-orientation is a tool for the resolution-limited regime.

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
