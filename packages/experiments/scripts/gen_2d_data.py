"""Generate the 2D sweep data: plain DFR vs goal-oriented DFR at matched DOF.

Trains both methods on a 2D Poisson problem with a sharp separable bump across a
range of network widths and several seeds, recording the error in a point quantity
of interest. Both methods share the identical full tensor-product DST
discretization, so the comparison isolates the effect of goal-orientation. The
output JSON is read by ``hp_dfr.figures`` to build the ``dfr-vs-go-2d`` figure.

Run with a PyTorch backend installed::

    uv pip install 'torch>=2.0.0,<2.2.0'   # once, if needed (Intel macOS pin)
    KERAS_BACKEND=torch uv run python packages/experiments/scripts/gen_2d_data.py
"""

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore")

from hp_dfr.models import DFRModel, GoalOrientedDFRModel, PointEvaluationQoI  # noqa: E402
from hp_dfr.problems.poisson_2d import ArcTanBump2D  # noqa: E402

STEEPNESS = 8.0
X0 = np.array([0.65, 0.65])
SIGMA = 0.07
ARCHS = [[8, 8], [16, 16], [24, 24], [32, 32]]
SEEDS = [0, 1, 2]
MODES, NQ, ADAM, LR, LBFGS = 16, 32, 3000, 3e-3, 500
ADJ_W, PRIM_W = 1.0, 5.0  # goal-oriented loss weights (adjoint / primal)

OUT = Path(__file__).resolve().parents[3] / "assets" / "m3_2d_data.json"


def n_params(arch: list[int], dim: int = 2) -> int:
    """Trainable-parameter count of a fully connected ``dim -> arch -> 1`` net."""
    sizes = [dim, *arch, 1]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def make_qoi_error(qoi: PointEvaluationQoI, prob: ArcTanBump2D, n: int = 200) -> Callable[[np.ndarray], float]:
    """Build a function returning the mollified point-QoI error on a fixed grid."""
    (a0, b0), (a1, b1) = prob.domain
    gx, gy = np.linspace(a0, b0, n), np.linspace(a1, b1, n)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    eta = qoi.adjoint_rhs(pts)
    je = float(np.sum(eta * prob.exact_solution(pts)) * cell)
    return lambda up: abs(float(np.sum(eta * up) * cell) - je)


def main() -> None:
    """Run the sweep and write the JSON consumed by the figure builder."""
    prob = ArcTanBump2D(steepness=STEEPNESS)
    qoi_err = make_qoi_error(PointEvaluationQoI(point=X0, sigma=SIGMA), prob)

    records = []
    for arch in ARCHS:
        dof = n_params(arch)
        for seed in SEEDS:
            dfr = DFRModel(hidden_layers=arch, n_fourier_modes=MODES, n_quadrature=NQ, backend="pytorch", seed=seed, dim=2)
            dfr.build()
            dfr.fit(prob, epochs=ADAM, learning_rate=LR, verbose=False, lbfgs_iters=LBFGS)
            pts = np.column_stack([np.repeat(np.linspace(0, 1, 200), 200), np.tile(np.linspace(0, 1, 200), 200)])
            dfr_q = qoi_err(dfr.predict(pts))

            go = GoalOrientedDFRModel(
                hidden_layers=arch, dim=2, qoi=PointEvaluationQoI(point=X0, sigma=SIGMA),
                n_modes=MODES, n_quadrature=NQ, adjoint_weight=ADJ_W, primal_weight=PRIM_W, backend="pytorch", seed=seed,
            )
            go.build()
            go.fit(prob, epochs=ADAM, learning_rate=LR, verbose=False, lbfgs_iters=LBFGS)
            go_q = qoi_err(go.predict(pts))

            records.append({"problem": "arctanbump2d", "dof": dof, "seed": seed, "dfr_qoi": dfr_q, "go_qoi": go_q})
            print(f"dof={dof:4d} seed={seed} | DFR={dfr_q:.2e} GO={go_q:.2e}", flush=True)

    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
