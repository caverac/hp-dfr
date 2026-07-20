"""Sweep-data generation for the goal-oriented DFR preprint figures.

Trains plain DFR and goal-oriented DFR at matched primal-network degrees of
freedom across a range of network widths and several random seeds, recording the
error in a point quantity of interest for each run. Goal-oriented runs also record
the three terms of the QoI error bound (:func:`_go_diagnostics`), so the bound can
be checked against the error it claims to control rather than only asserted.

Note that goal-oriented runs train a second network of the same size, so they carry
roughly twice the parameters and twice the per-step cost of the baseline at the
same recorded ``dof``, which counts the primal network alone.

The sweeps are:

- :func:`run_1d_sweep` -- two 1D Poisson problems (smooth ``sine`` and sharp
  ``arctan``); writes ``m2_1d_data.json``.
- :func:`run_2d_sweep` -- a 2D Poisson problem with a sharp separable bump; writes
  ``m3_2d_data.json``.
- :func:`run_2d_steepness_sweep` -- the same 2D bump swept over steepness at fixed
  networks; writes ``m4_2d_steepness.json``.
- :func:`run_2d_tradeoff_sweep` and :func:`run_2d_convergence_sweep` -- the two
  Section 6.4 diagnostics (what the method trades, and whether the baseline is
  converged); write ``floor_probe.json`` and ``convergence.json``.

All write to the shared ``assets`` directory (:data:`~hp_dfr._plotting.FIG_DIR`),
where :func:`hp_dfr.figures.make_all_figures` reads them to build the figures. The
sweeps are expensive; figure rendering is fast because it only reads the JSON.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

from hp_dfr._plotting import FIG_DIR
from hp_dfr.models import DFRModel, GoalOrientedDFRModel, PointEvaluationQoI
from hp_dfr.problems.poisson_1d import ArcTanProblem, PoissonProblem, SineProblem
from hp_dfr.problems.poisson_2d import ArcTanBump2D
from hp_dfr.types.common import BackendType
from hp_dfr.utils import console

ArrayF = npt.NDArray[np.float64]
_ShardItem = TypeVar("_ShardItem")

# --- 1D sweep configuration ------------------------------------------------
# Each problem maps to (instance, relative QoI location in [0, 1]).
_PROBLEMS_1D: dict[str, tuple[PoissonProblem, float]] = {
    "sine": (SineProblem(), 0.25),
    "arctan8": (ArcTanProblem(8.0), 0.65),
}
_ARCHS_1D: list[tuple[int, ...]] = [(4, 4), (8, 8), (16, 16), (32, 32)]
_SEEDS_1D = [0, 1, 2]
_MODES_1D, _NQ_1D, _ADAM_1D, _LR_1D, _LBFGS_1D = 60, 150, 2000, 2e-3, 400
_SIGMA_1D = 0.04
# The 1D control is run at both its own loss weight and the weight used by the 2D
# sweep (:data:`_PRIM_W_2D`). Running only the former would confound the 1D/2D
# comparison: a 1D null result at weight 1.0 cannot rule out that the 2D gain
# comes from the weight rather than from the dimension.
_PRIM_WS_1D: list[float] = [1.0, 5.0]
_OUT_1D = FIG_DIR / "m2_1d_data.json"

# --- 2D sweep configuration ------------------------------------------------
_STEEPNESS_2D = 8.0
_X0_2D = np.array([0.65, 0.65])
_SIGMA_2D = 0.07
_ARCHS_2D: list[tuple[int, ...]] = [(8, 8), (16, 16), (24, 24), (32, 32)]
_SEEDS_2D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_MODES_2D, _NQ_2D, _ADAM_2D, _LR_2D, _LBFGS_2D = 16, 32, 3000, 3e-3, 500
_ADJ_W_2D, _PRIM_W_2D = 1.0, 5.0  # goal-oriented loss weights (adjoint / primal)
_OUT_2D = FIG_DIR / "m3_2d_data.json"

# --- 2D steepness sweep configuration ---------------------------------------
# Tests the central claim directly: does the goal-oriented advantage grow as the
# problem becomes more resolution-limited? Steepness is the knob, at a fixed
# network, so it is the only variable.
#
# The mode count is deliberately *fixed* across the sweep and set high enough for
# the sharpest case. Holding it at the 16 of the main sweep would make the test
# space, not the network, the bottleneck at large k: the relative energy tail the
# sine basis fails to represent grows from 8e-8 at k=8 to 1e-3 at k=32, so the
# sweep would measure mode truncation instead. At 24 modes the tail stays below
# 7e-7 for every k below, and 48 quadrature points keep two points per mode.
_STEEPNESSES_2D: list[float] = [2.0, 4.0, 8.0, 16.0]
_ARCHS_STEEP_2D: list[tuple[int, ...]] = [(8, 8), (16, 16)]
_SEEDS_STEEP_2D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_MODES_STEEP_2D, _NQ_STEEP_2D = 24, 48
_OUT_STEEP_2D = FIG_DIR / "m4_2d_steepness.json"

# --- 2D trade-off and convergence configuration ----------------------------
# Two diagnostics behind Section 6.4 of the manuscript, both on the k=2,
# 337-parameter problem where the goal-oriented advantage is largest and least
# expected:
#
#   run_2d_tradeoff_sweep      what goal-orientation buys and sells: it records
#                              the QoI error, the global relative L2 error, and
#                              the dual-norm residual for plain DFR (at several
#                              training budgets) and for the goal-oriented method,
#                              so the reallocation (lower QoI error, higher global
#                              error) can be read off directly.
#   run_2d_convergence_sweep   whether plain DFR is converged at the shared
#                              budget: it trains far past it and records the
#                              residual along the way, showing the apparent
#                              accuracy floor is a truncated slow decay.
#
# The output file names match the exploratory artifacts these replace.
_TRADEOFF_ARCH: tuple[int, ...] = (16, 16)  # 337 primal parameters
_TRADEOFF_K = 2.0  # the easiest problem: the floor question, not resolution
_TRADEOFF_SEEDS = [0, 1, 2]
_TRADEOFF_LR = 3e-3
_TRADEOFF_MODES, _TRADEOFF_NQ = 24, 48  # the k=2 discretization of the steepness sweep
# Plain DFR at (adam, lbfgs): the shared budget, 3x, and 10x.
_TRADEOFF_BUDGETS: list[tuple[int, int]] = [(3000, 500), (10000, 2000), (30000, 5000)]
_OUT_TRADEOFF = FIG_DIR / "floor_probe.json"

_CONV_ARCH: tuple[int, ...] = (16, 16)
_CONV_SEEDS = [0, 1, 2]
_CONV_ADAM, _CONV_LBFGS, _CONV_LR = 50000, 500, 3e-3
# (name, steepness, modes, quadrature): the headline sweep's config and the
# steepness sweep's easy end.
_CONV_CONFIGS: list[tuple[str, float, int, int]] = [
    ("headline-k8-m16", 8.0, 16, 32),
    ("floor-k2-m24", 2.0, 24, 48),
]
_OUT_CONV = FIG_DIR / "convergence.json"


def _n_params(arch: tuple[int, ...], dim: int) -> int:
    """Trainable-parameter count of a fully connected ``dim -> arch -> 1`` net."""
    sizes = [dim, *arch, 1]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def _go_diagnostics(model: GoalOrientedDFRModel) -> dict[str, float]:
    """Extract the three terms of the QoI error bound from a trained model.

    The bound is ``|J(u*) - J(u_h)| <= L_QoI + ||R(u_h)|| ||R*(z_h)|| / gamma``,
    with ``gamma = 1`` for the symmetric coercive Poisson problem. Recording
    these alongside the true QoI error is what makes the bound checkable rather
    than merely stated: the estimator is computed every training step anyway.

    The history stores the *squared* dual norms, so the norms are their square
    roots. Values are taken from the last entry, which is written after the
    LBFGS polish.

    Parameters
    ----------
    model
        A trained goal-oriented model.

    Returns
    -------
    dict[str, float]
        The goal term, the two dual-norm residuals, and the resulting bound.
    """
    history = model.history
    goal_term = float(history["go_loss"][-1])
    res_primal = float(np.sqrt(history["primal_loss"][-1]))
    res_adjoint = float(np.sqrt(history["adjoint_loss"][-1]))
    return {
        "go_est": goal_term,
        "go_res_primal": res_primal,
        "go_res_adjoint": res_adjoint,
        "go_bound": goal_term + res_primal * res_adjoint,
    }


def run_1d_sweep(backend: BackendType = "pytorch") -> Path:
    """Run the 1D DFR-versus-goal-oriented sweep and write its JSON.

    Parameters
    ----------
    backend
        Deep-learning backend passed to both models.

    Returns
    -------
    Path
        Location of the written ``m2_1d_data.json`` file.
    """
    warnings.filterwarnings("ignore")
    records: list[dict[str, object]] = []
    for pname, (prob, x0_rel) in _PROBLEMS_1D.items():
        a, b = prob.domain
        x0 = a + x0_rel * (b - a)
        xg = np.linspace(a, b, 800)
        ue = prob.exact_solution(xg)
        qoi = PointEvaluationQoI(point=np.array([x0]), sigma=_SIGMA_1D)
        eta = qoi.adjoint_rhs(xg.reshape(-1, 1))
        je = float(np.trapz(eta * ue, xg))

        def qoi_err(up: ArrayF, xg: ArrayF = xg, eta: ArrayF = eta, je: float = je) -> float:
            return abs(float(np.trapz(eta * up, xg)) - je)

        for arch in _ARCHS_1D:
            dof = _n_params(arch, dim=1)
            for seed in _SEEDS_1D:
                # The baseline does not depend on the goal-oriented loss weight,
                # and is deterministic given the seed, so it is trained once and
                # reused across the weights compared below.
                dfr = DFRModel(hidden_layers=arch, n_fourier_modes=_MODES_1D, n_quadrature=_NQ_1D, backend=backend, seed=seed)
                dfr.build()
                dfr.fit(prob, epochs=_ADAM_1D, learning_rate=_LR_1D, verbose=False, lbfgs_iters=_LBFGS_1D)
                dfr_q = qoi_err(dfr.predict(xg))

                for primal_weight in _PRIM_WS_1D:
                    go = GoalOrientedDFRModel(
                        hidden_layers=arch,
                        dim=1,
                        qoi=qoi,
                        n_modes=_MODES_1D,
                        n_quadrature=_NQ_1D,
                        adjoint_weight=1.0,
                        primal_weight=primal_weight,
                        backend=backend,
                        seed=seed,
                    )
                    go.build()
                    go.fit(prob, epochs=_ADAM_1D, learning_rate=_LR_1D, verbose=False, lbfgs_iters=_LBFGS_1D)
                    go_q = qoi_err(go.predict(xg))

                    records.append(
                        {
                            "problem": pname,
                            "dof": dof,
                            "seed": seed,
                            "primal_weight": primal_weight,
                            "dfr_qoi": dfr_q,
                            "go_qoi": go_q,
                            **_go_diagnostics(go),
                        }
                    )
                    console.print(
                        f"{pname:8s} dof={dof:4d} seed={seed} pw={primal_weight:.1f} | DFR={dfr_q:.2e} GO={go_q:.2e}",
                        markup=False,
                    )

    _OUT_1D.write_text(json.dumps(records, indent=2), encoding="utf-8")
    console.print(f"wrote {_OUT_1D}", markup=False)
    return _OUT_1D


def _make_qoi_error_2d(qoi: PointEvaluationQoI, prob: ArcTanBump2D, n: int = 200) -> Callable[[ArrayF], float]:
    """Build a function returning the mollified point-QoI error on a fixed grid."""
    (a0, b0), (a1, b1) = prob.domain
    gx, gy = np.linspace(a0, b0, n), np.linspace(a1, b1, n)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    eta = qoi.adjoint_rhs(pts)
    je = float(np.sum(eta * prob.exact_solution(pts)) * cell)
    return lambda up: abs(float(np.sum(eta * up) * cell) - je)


def run_2d_sweep(backend: BackendType = "pytorch") -> Path:
    """Run the 2D DFR-versus-goal-oriented sweep and write its JSON.

    Parameters
    ----------
    backend
        Deep-learning backend passed to both models.

    Returns
    -------
    Path
        Location of the written ``m3_2d_data.json`` file.
    """
    warnings.filterwarnings("ignore")
    prob = ArcTanBump2D(steepness=_STEEPNESS_2D)
    qoi_err = _make_qoi_error_2d(PointEvaluationQoI(point=_X0_2D, sigma=_SIGMA_2D), prob)

    records: list[dict[str, object]] = []
    for arch in _ARCHS_2D:
        dof = _n_params(arch, dim=2)
        for seed in _SEEDS_2D:
            dfr = DFRModel(hidden_layers=arch, n_fourier_modes=_MODES_2D, n_quadrature=_NQ_2D, backend=backend, seed=seed, dim=2)
            dfr.build()
            dfr.fit(prob, epochs=_ADAM_2D, learning_rate=_LR_2D, verbose=False, lbfgs_iters=_LBFGS_2D)
            pts = np.column_stack([np.repeat(np.linspace(0, 1, 200), 200), np.tile(np.linspace(0, 1, 200), 200)])
            dfr_q = qoi_err(dfr.predict(pts))

            go = GoalOrientedDFRModel(
                hidden_layers=arch,
                dim=2,
                qoi=PointEvaluationQoI(point=_X0_2D, sigma=_SIGMA_2D),
                n_modes=_MODES_2D,
                n_quadrature=_NQ_2D,
                adjoint_weight=_ADJ_W_2D,
                primal_weight=_PRIM_W_2D,
                backend=backend,
                seed=seed,
            )
            go.build()
            go.fit(prob, epochs=_ADAM_2D, learning_rate=_LR_2D, verbose=False, lbfgs_iters=_LBFGS_2D)
            go_q = qoi_err(go.predict(pts))

            records.append(
                {
                    "problem": "arctanbump2d",
                    "dof": dof,
                    "seed": seed,
                    "primal_weight": _PRIM_W_2D,
                    "dfr_qoi": dfr_q,
                    "go_qoi": go_q,
                    **_go_diagnostics(go),
                }
            )
            console.print(f"dof={dof:4d} seed={seed} | DFR={dfr_q:.2e} GO={go_q:.2e}", markup=False)

    _OUT_2D.write_text(json.dumps(records, indent=2), encoding="utf-8")
    console.print(f"wrote {_OUT_2D}", markup=False)
    return _OUT_2D


def run_2d_steepness_sweep(backend: BackendType = "pytorch") -> Path:
    """Sweep the solution's steepness at fixed networks and write its JSON.

    The main 2D sweep varies the network size at one steepness, which shows *that*
    goal-orientation helps but not that it helps *because* the network is
    resolution-limited. This sweep varies the steepness instead, holding the
    network and the discretization fixed, so a rising advantage with ``k`` is
    direct evidence for that mechanism, and a flat one is evidence against it.

    Two network sizes are run rather than one: the smaller is the more
    resolution-limited and the one most favorable to goal-orientation, the larger
    is where the main sweep finds goal-orientation weakest. A trend present in
    both cannot be an artifact of the network size chosen.

    Parameters
    ----------
    backend
        Deep-learning backend passed to both models.

    Returns
    -------
    Path
        Location of the written ``m4_2d_steepness.json`` file.
    """
    warnings.filterwarnings("ignore")
    records: list[dict[str, object]] = []
    pts = np.column_stack([np.repeat(np.linspace(0, 1, 200), 200), np.tile(np.linspace(0, 1, 200), 200)])

    for steepness in _STEEPNESSES_2D:
        prob = ArcTanBump2D(steepness=steepness)
        qoi_err = _make_qoi_error_2d(PointEvaluationQoI(point=_X0_2D, sigma=_SIGMA_2D), prob)

        for arch in _ARCHS_STEEP_2D:
            dof = _n_params(arch, dim=2)
            for seed in _SEEDS_STEEP_2D:
                dfr = DFRModel(
                    hidden_layers=arch,
                    n_fourier_modes=_MODES_STEEP_2D,
                    n_quadrature=_NQ_STEEP_2D,
                    backend=backend,
                    seed=seed,
                    dim=2,
                )
                dfr.build()
                dfr.fit(prob, epochs=_ADAM_2D, learning_rate=_LR_2D, verbose=False, lbfgs_iters=_LBFGS_2D)
                dfr_q = qoi_err(dfr.predict(pts))

                go = GoalOrientedDFRModel(
                    hidden_layers=arch,
                    dim=2,
                    qoi=PointEvaluationQoI(point=_X0_2D, sigma=_SIGMA_2D),
                    n_modes=_MODES_STEEP_2D,
                    n_quadrature=_NQ_STEEP_2D,
                    adjoint_weight=_ADJ_W_2D,
                    primal_weight=_PRIM_W_2D,
                    backend=backend,
                    seed=seed,
                )
                go.build()
                go.fit(prob, epochs=_ADAM_2D, learning_rate=_LR_2D, verbose=False, lbfgs_iters=_LBFGS_2D)
                go_q = qoi_err(go.predict(pts))

                records.append(
                    {
                        "problem": "arctanbump2d",
                        "steepness": steepness,
                        "dof": dof,
                        "seed": seed,
                        "primal_weight": _PRIM_W_2D,
                        "dfr_qoi": dfr_q,
                        "go_qoi": go_q,
                        **_go_diagnostics(go),
                    }
                )
                console.print(
                    f"k={steepness:4.1f} dof={dof:4d} seed={seed} | DFR={dfr_q:.2e} GO={go_q:.2e}",
                    markup=False,
                )
                _OUT_STEEP_2D.write_text(json.dumps(records, indent=2), encoding="utf-8")

    console.print(f"wrote {_OUT_STEEP_2D}", markup=False)
    return _OUT_STEEP_2D


def _rel_l2_2d(pred: ArrayF, exact: ArrayF) -> float:
    """Global relative L2 error, to separate the QoI from the whole solution."""
    return float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))


def _full_grid_2d(n: int = 200) -> ArrayF:
    """Build the evaluation grid on the unit square used by the sweep metrics."""
    axis = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.repeat(axis, n), np.tile(axis, n)])


def run_2d_tradeoff_sweep(backend: BackendType = "pytorch") -> Path:
    """Measure what goal-orientation trades, and against what baseline.

    On the easiest 2D problem, records the QoI error, the global relative L2
    error, and the dual-norm residual for plain DFR at several training budgets
    and for the goal-oriented method at the shared budget. The goal-oriented run
    reaches a smaller QoI error while its global error is larger, which is the
    reallocation the method is built to produce; the multiple DFR budgets show
    the baseline is still improving past the shared budget rather than sitting at
    a floor.

    Parameters
    ----------
    backend
        Deep-learning backend passed to both models.

    Returns
    -------
    Path
        Location of the written ``floor_probe.json`` file.
    """
    warnings.filterwarnings("ignore")
    prob = ArcTanBump2D(steepness=_TRADEOFF_K)
    qoi = PointEvaluationQoI(point=_X0_2D, sigma=_SIGMA_2D)
    qoi_err = _make_qoi_error_2d(qoi, prob)
    pts = _full_grid_2d()
    exact = prob.exact_solution(pts)

    records: list[dict[str, object]] = []
    for seed in _TRADEOFF_SEEDS:
        for adam, lbfgs in _TRADEOFF_BUDGETS:
            dfr = DFRModel(
                hidden_layers=_TRADEOFF_ARCH,
                n_fourier_modes=_TRADEOFF_MODES,
                n_quadrature=_TRADEOFF_NQ,
                backend=backend,
                seed=seed,
                dim=2,
            )
            dfr.build()
            dfr.fit(prob, epochs=adam, learning_rate=_TRADEOFF_LR, verbose=False, lbfgs_iters=lbfgs)
            pred = dfr.predict(pts)
            records.append(
                {
                    "method": "dfr",
                    "seed": seed,
                    "adam": adam,
                    "lbfgs": lbfgs,
                    "qoi": qoi_err(pred),
                    # The nD DFR loss is ||R||^2 in the dual norm.
                    "res_dual": float(np.sqrt(dfr.history["loss"][-1])),
                    "rel_l2": _rel_l2_2d(pred, exact),
                }
            )
            r = records[-1]
            console.print(
                f"DFR seed={seed} adam={adam:5d} lbfgs={lbfgs:4d} | "
                f"QoI={r['qoi']:.2e} ||R||={r['res_dual']:.2e} relL2={r['rel_l2']:.2e}",
                markup=False,
            )
            _OUT_TRADEOFF.write_text(json.dumps(records, indent=2), encoding="utf-8")

        go = GoalOrientedDFRModel(
            hidden_layers=_TRADEOFF_ARCH,
            dim=2,
            qoi=PointEvaluationQoI(point=_X0_2D, sigma=_SIGMA_2D),
            n_modes=_TRADEOFF_MODES,
            n_quadrature=_TRADEOFF_NQ,
            adjoint_weight=_ADJ_W_2D,
            primal_weight=_PRIM_W_2D,
            backend=backend,
            seed=seed,
        )
        go.build()
        go.fit(prob, epochs=_TRADEOFF_BUDGETS[0][0], learning_rate=_TRADEOFF_LR, verbose=False, lbfgs_iters=_TRADEOFF_BUDGETS[0][1])
        pred = go.predict(pts)
        records.append(
            {
                "method": "go",
                "seed": seed,
                "adam": _TRADEOFF_BUDGETS[0][0],
                "lbfgs": _TRADEOFF_BUDGETS[0][1],
                "qoi": qoi_err(pred),
                "res_dual": float(np.sqrt(go.history["primal_loss"][-1])),
                "rel_l2": _rel_l2_2d(pred, exact),
            }
        )
        r = records[-1]
        console.print(
            f"GO  seed={seed} adam={r['adam']:5d} lbfgs={r['lbfgs']:4d} | "
            f"QoI={r['qoi']:.2e} ||R||={r['res_dual']:.2e} relL2={r['rel_l2']:.2e}",
            markup=False,
        )
        _OUT_TRADEOFF.write_text(json.dumps(records, indent=2), encoding="utf-8")

    console.print(f"wrote {_OUT_TRADEOFF}", markup=False)
    return _OUT_TRADEOFF


def run_2d_convergence_sweep(backend: BackendType = "pytorch") -> Path:
    """Trace plain DFR's residual far past the shared training budget.

    The DFR loss is the squared dual-norm residual and the model records it every
    epoch, so a single long run yields the whole convergence curve. Run well past
    the budget used by the sweeps, the residual keeps falling with no plateau,
    which shows the apparent accuracy floor of the earlier figures is a fixed
    budget truncating a slow decay rather than a limit of the method.

    Parameters
    ----------
    backend
        Deep-learning backend passed to the model.

    Returns
    -------
    Path
        Location of the written ``convergence.json`` file.
    """
    warnings.filterwarnings("ignore")
    records: list[dict[str, object]] = []
    for name, steepness, modes, nq in _CONV_CONFIGS:
        prob = ArcTanBump2D(steepness=steepness)
        for seed in _CONV_SEEDS:
            model = DFRModel(
                hidden_layers=_CONV_ARCH,
                n_fourier_modes=modes,
                n_quadrature=nq,
                backend=backend,
                seed=seed,
                dim=2,
            )
            model.build()
            model.fit(prob, epochs=_CONV_ADAM, learning_rate=_CONV_LR, verbose=False, lbfgs_iters=_CONV_LBFGS)

            # history["loss"] is ||R||^2 per Adam epoch; the last entry is written
            # after the LBFGS polish, so keep the two apart.
            loss = np.asarray(model.history["loss"], dtype=np.float64)
            adam_curve = loss[:_CONV_ADAM]
            # Downsample the curve for storage; keep its shape, not every step.
            idx = np.unique(np.geomspace(1, _CONV_ADAM, 60).astype(int)) - 1
            records.append(
                {
                    "config": name,
                    "k": steepness,
                    "seed": seed,
                    "adam_steps": idx.tolist(),
                    "adam_res": np.sqrt(adam_curve[idx]).tolist(),
                    "res_at_3000": float(np.sqrt(adam_curve[2999])),
                    "res_at_end_adam": float(np.sqrt(adam_curve[-1])),
                    "res_post_lbfgs": float(np.sqrt(loss[-1])),
                }
            )
            r = records[-1]
            console.print(
                f"{name:16s} seed={seed} | ||R|| 3k={r['res_at_3000']:.2e} "
                f"50k={r['res_at_end_adam']:.2e} post-LBFGS={r['res_post_lbfgs']:.2e}",
                markup=False,
            )
            _OUT_CONV.write_text(json.dumps(records, indent=2), encoding="utf-8")

    console.print(f"wrote {_OUT_CONV}", markup=False)
    return _OUT_CONV


# --- matched-cost sweep configuration --------------------------------------
# Goal-orientation trains a second network, so a run costs roughly 2.5 times a
# plain-DFR run (Section 6.4 of the manuscript). A matched-*budget* comparison
# therefore under-resources plain DFR. The matched-*cost* comparison gives plain
# DFR _DFR_STEP_MULT times the Adam and LBFGS budget so the two methods get
# comparable wall-clock. The multiplier is a conservative upper estimate of the
# measured cost ratio (~1.9 to 2.5), so if goal-orientation still wins, the
# baseline was if anything over-resourced. It is designed to be run in shards, one
# Fargate task per shard; the shards are merged with :func:`merge_matched_cost_shards`.
_DFR_STEP_MULT = 2.5

# A single (steepness, network architecture, seed) unit of work.
MatchedCostItem = tuple[float, tuple[int, ...], int]


class _MatchedCostSpec(NamedTuple):
    """A matched-cost sweep: its flat work list and the discretization it uses."""

    items: list[MatchedCostItem]
    modes: int
    nq: int


# "steepness" varies the steepness at two fixed networks (the mechanism figure);
# "2d" varies the network size at a fixed steepness (the headline table).
_MATCHED_COST_SWEEPS: dict[str, _MatchedCostSpec] = {
    "steepness": _MatchedCostSpec(
        items=[(k, arch, seed) for k in _STEEPNESSES_2D for arch in _ARCHS_STEEP_2D for seed in _SEEDS_STEEP_2D],
        modes=_MODES_STEEP_2D,
        nq=_NQ_STEEP_2D,
    ),
    "2d": _MatchedCostSpec(
        items=[(_STEEPNESS_2D, arch, seed) for arch in _ARCHS_2D for seed in _SEEDS_2D],
        modes=_MODES_2D,
        nq=_NQ_2D,
    ),
}


def _shard_items(items: list[_ShardItem], shard: int, num_shards: int) -> list[_ShardItem]:
    """Return the shard-th slice of ``items`` under round-robin partitioning.

    Round-robin (``index % num_shards == shard``) rather than contiguous blocks so
    that each shard sees a mix of cheap and expensive runs, keeping shard runtimes
    balanced when cost varies with the work item (larger networks cost more).

    Parameters
    ----------
    items
        The full work list.
    shard
        Zero-based shard index, in ``[0, num_shards)``.
    num_shards
        Total number of shards.

    Returns
    -------
    list[object]
        The work items assigned to this shard.

    Raises
    ------
    ValueError
        If ``num_shards`` is not positive or ``shard`` is out of range.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")
    return [it for i, it in enumerate(items) if i % num_shards == shard]


def _matched_cost_path(which: str, shard: int, num_shards: int) -> Path:
    """Output path for a matched-cost run, per shard or merged (``num_shards == 1``)."""
    if num_shards == 1:
        return FIG_DIR / f"matched_cost_{which}.json"
    return FIG_DIR / f"matched_cost_{which}.shard{shard}of{num_shards}.json"


def run_2d_matched_cost_sweep(
    which: str = "steepness",
    shard: int = 0,
    num_shards: int = 1,
    backend: BackendType = "pytorch",
) -> Path:
    """Repeat a 2D sweep giving plain DFR matched wall-clock rather than matched steps.

    Trains plain DFR at :data:`_DFR_STEP_MULT` times the budget of goal-oriented
    DFR, so the comparison holds computational cost fixed instead of step count.
    The work list is partitioned into ``num_shards`` shards (:func:`_shard_items`)
    and only ``shard`` is run, so the sweep can be spread over parallel Fargate
    tasks; :func:`merge_matched_cost_shards` recombines them.

    Parameters
    ----------
    which
        Which sweep to repeat: ``"steepness"`` or ``"2d"`` (see
        :data:`_MATCHED_COST_SWEEPS`).
    shard
        Zero-based shard index to run.
    num_shards
        Total number of shards the work is partitioned into.
    backend
        Deep-learning backend passed to both models.

    Returns
    -------
    Path
        Location of the written shard (or merged) JSON file.

    Raises
    ------
    KeyError
        If ``which`` is not a known sweep.
    """
    if which not in _MATCHED_COST_SWEEPS:
        raise KeyError(f"unknown matched-cost sweep {which!r}; choose from {sorted(_MATCHED_COST_SWEEPS)}")
    warnings.filterwarnings("ignore")
    spec = _MATCHED_COST_SWEEPS[which]
    modes, nq = spec.modes, spec.nq
    items = _shard_items(spec.items, shard, num_shards)
    dfr_adam, dfr_lbfgs = int(_ADAM_2D * _DFR_STEP_MULT), int(_LBFGS_2D * _DFR_STEP_MULT)
    pts = _full_grid_2d()

    records: list[dict[str, object]] = []
    out = _matched_cost_path(which, shard, num_shards)
    for steepness, arch, seed in items:
        prob = ArcTanBump2D(steepness=steepness)
        qoi_err = _make_qoi_error_2d(PointEvaluationQoI(point=_X0_2D, sigma=_SIGMA_2D), prob)
        dof = _n_params(arch, dim=2)

        dfr = DFRModel(hidden_layers=arch, n_fourier_modes=modes, n_quadrature=nq, backend=backend, seed=seed, dim=2)
        dfr.build()
        dfr.fit(prob, epochs=dfr_adam, learning_rate=_LR_2D, verbose=False, lbfgs_iters=dfr_lbfgs)
        dfr_q = qoi_err(dfr.predict(pts))

        go = GoalOrientedDFRModel(
            hidden_layers=arch,
            dim=2,
            qoi=PointEvaluationQoI(point=_X0_2D, sigma=_SIGMA_2D),
            n_modes=modes,
            n_quadrature=nq,
            adjoint_weight=_ADJ_W_2D,
            primal_weight=_PRIM_W_2D,
            backend=backend,
            seed=seed,
        )
        go.build()
        go.fit(prob, epochs=_ADAM_2D, learning_rate=_LR_2D, verbose=False, lbfgs_iters=_LBFGS_2D)
        go_q = qoi_err(go.predict(pts))

        records.append(
            {
                "problem": "arctanbump2d",
                "which": which,
                "steepness": steepness,
                "dof": dof,
                "seed": seed,
                "primal_weight": _PRIM_W_2D,
                "matched_cost": True,
                "dfr_adam": dfr_adam,
                "dfr_lbfgs": dfr_lbfgs,
                "dfr_qoi": dfr_q,
                "go_qoi": go_q,
                **_go_diagnostics(go),
            }
        )
        console.print(
            f"[{which}] k={steepness:4.1f} dof={dof:4d} seed={seed} shard={shard}/{num_shards} | DFR={dfr_q:.2e} GO={go_q:.2e}",
            markup=False,
        )
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")

    console.print(f"wrote {out}", markup=False)
    return out


def merge_matched_cost_shards(which: str, num_shards: int) -> Path:
    """Combine matched-cost shard files into the canonical ``matched_cost_<which>.json``.

    Parameters
    ----------
    which
        Which sweep was run: ``"steepness"`` or ``"2d"``.
    num_shards
        Number of shards to combine.

    Returns
    -------
    Path
        Location of the merged JSON file.

    Raises
    ------
    FileNotFoundError
        If any shard file is missing.
    ValueError
        If the shards contain duplicate (steepness, dof, seed) work items.
    """
    records: list[dict[str, object]] = []
    for shard in range(num_shards):
        path = _matched_cost_path(which, shard, num_shards)
        if not path.exists():
            raise FileNotFoundError(f"missing shard {path}; run all {num_shards} shards first")
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    keys = [(r["steepness"], r["dof"], r["seed"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (steepness, dof, seed) across shards; check the shard/num_shards used")
    records.sort(key=lambda r: (r["steepness"], r["dof"], r["seed"]))
    merged = FIG_DIR / f"matched_cost_{which}.json"
    merged.write_text(json.dumps(records, indent=2), encoding="utf-8")
    console.print(f"wrote {merged} ({len(records)} runs from {num_shards} shards)", markup=False)
    return merged


__all__ = [
    "merge_matched_cost_shards",
    "run_1d_sweep",
    "run_2d_convergence_sweep",
    "run_2d_matched_cost_sweep",
    "run_2d_steepness_sweep",
    "run_2d_sweep",
    "run_2d_tradeoff_sweep",
]
