"""Sweep-data generation for the goal-oriented DFR preprint figures.

Trains plain DFR and goal-oriented DFR at matched degrees of freedom across a
range of network widths and several random seeds, recording the error in a point
quantity of interest for each run. Two sweeps are provided:

- :func:`run_1d_sweep` -- two 1D Poisson problems (smooth ``sine`` and sharp
  ``arctan``); writes ``m2_1d_data.json``.
- :func:`run_2d_sweep` -- a 2D Poisson problem with a sharp separable bump; writes
  ``m3_2d_data.json``.

Both write to the shared ``assets`` directory (:data:`~hp_dfr._plotting.FIG_DIR`),
where :func:`hp_dfr.figures.make_all_figures` reads them to build the figures. The
sweeps are expensive; figure rendering is fast because it only reads the JSON.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from hp_dfr._plotting import FIG_DIR
from hp_dfr.models import DFRModel, GoalOrientedDFRModel, PointEvaluationQoI
from hp_dfr.problems.poisson_1d import ArcTanProblem, PoissonProblem, SineProblem
from hp_dfr.problems.poisson_2d import ArcTanBump2D
from hp_dfr.types.common import BackendType
from hp_dfr.utils import console

ArrayF = npt.NDArray[np.float64]

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
_OUT_1D = FIG_DIR / "m2_1d_data.json"

# --- 2D sweep configuration ------------------------------------------------
_STEEPNESS_2D = 8.0
_X0_2D = np.array([0.65, 0.65])
_SIGMA_2D = 0.07
_ARCHS_2D: list[tuple[int, ...]] = [(8, 8), (16, 16), (24, 24), (32, 32)]
_SEEDS_2D = [0, 1, 2]
_MODES_2D, _NQ_2D, _ADAM_2D, _LR_2D, _LBFGS_2D = 16, 32, 3000, 3e-3, 500
_ADJ_W_2D, _PRIM_W_2D = 1.0, 5.0  # goal-oriented loss weights (adjoint / primal)
_OUT_2D = FIG_DIR / "m3_2d_data.json"


def _n_params(arch: tuple[int, ...], dim: int) -> int:
    """Trainable-parameter count of a fully connected ``dim -> arch -> 1`` net."""
    sizes = [dim, *arch, 1]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


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
                dfr = DFRModel(hidden_layers=arch, n_fourier_modes=_MODES_1D, n_quadrature=_NQ_1D, backend=backend, seed=seed)
                dfr.build()
                dfr.fit(prob, epochs=_ADAM_1D, learning_rate=_LR_1D, verbose=False, lbfgs_iters=_LBFGS_1D)
                dfr_q = qoi_err(dfr.predict(xg))

                go = GoalOrientedDFRModel(
                    hidden_layers=arch,
                    dim=1,
                    qoi=qoi,
                    n_modes=_MODES_1D,
                    n_quadrature=_NQ_1D,
                    adjoint_weight=1.0,
                    primal_weight=1.0,
                    backend=backend,
                    seed=seed,
                )
                go.build()
                go.fit(prob, epochs=_ADAM_1D, learning_rate=_LR_1D, verbose=False, lbfgs_iters=_LBFGS_1D)
                go_q = qoi_err(go.predict(xg))

                records.append({"problem": pname, "dof": dof, "seed": seed, "dfr_qoi": dfr_q, "go_qoi": go_q})
                console.print(f"{pname:8s} dof={dof:4d} seed={seed} | DFR={dfr_q:.2e} GO={go_q:.2e}", markup=False)

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

            records.append({"problem": "arctanbump2d", "dof": dof, "seed": seed, "dfr_qoi": dfr_q, "go_qoi": go_q})
            console.print(f"dof={dof:4d} seed={seed} | DFR={dfr_q:.2e} GO={go_q:.2e}", markup=False)

    _OUT_2D.write_text(json.dumps(records, indent=2), encoding="utf-8")
    console.print(f"wrote {_OUT_2D}", markup=False)
    return _OUT_2D


__all__ = ["run_1d_sweep", "run_2d_sweep"]
