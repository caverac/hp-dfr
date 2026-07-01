"""Generate the 1D sweep data: plain DFR vs goal-oriented DFR at matched DOF.

Trains both methods on two 1D Poisson problems (a smooth ``sine`` and a sharp
``arctan``) across a range of network widths and several random seeds, and records
the error in a point quantity of interest for each run. The output JSON is read by
``hp_dfr.figures`` to build the ``dfr-saturation-1d`` figure.

Run with a PyTorch backend installed::

    uv pip install 'torch>=2.0.0,<2.2.0'   # once, if needed (Intel macOS pin)
    KERAS_BACKEND=torch uv run python packages/experiments/scripts/gen_1d_data.py
"""

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from hp_dfr.models import DFRModel, GoalOrientedDFRModel, PointEvaluationQoI  # noqa: E402
from hp_dfr.problems.poisson_1d import ArcTanProblem, SineProblem  # noqa: E402

# Each problem maps to (instance, relative QoI location in [0, 1]).
PROBLEMS = {"sine": (SineProblem(), 0.25), "arctan8": (ArcTanProblem(8.0), 0.65)}
ARCHS = [[4, 4], [8, 8], [16, 16], [32, 32]]
SEEDS = [0, 1, 2]
MODES, NQ, ADAM, LR, LBFGS = 60, 150, 2000, 2e-3, 400

OUT = Path(__file__).resolve().parents[3] / "assets" / "m2_1d_data.json"


def n_params(arch: list[int], dim: int = 1) -> int:
    """Trainable-parameter count of a fully connected ``dim -> arch -> 1`` net."""
    sizes = [dim, *arch, 1]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def main() -> None:
    """Run the sweep and write the JSON consumed by the figure builder."""
    records = []
    for pname, (prob, x0_rel) in PROBLEMS.items():
        a, b = prob.domain
        x0 = a + x0_rel * (b - a)
        xg = np.linspace(a, b, 800)
        ue = prob.exact_solution(xg)
        qoi = PointEvaluationQoI(point=np.array([x0]), sigma=0.04)
        eta = qoi.adjoint_rhs(xg.reshape(-1, 1))
        je = float(np.trapz(eta * ue, xg))

        def qoi_err(up: np.ndarray, eta: np.ndarray = eta, je: float = je) -> float:
            return abs(float(np.trapz(eta * up, xg)) - je)

        for arch in ARCHS:
            dof = n_params(arch)
            for seed in SEEDS:
                dfr = DFRModel(hidden_layers=arch, n_fourier_modes=MODES, n_quadrature=NQ, backend="pytorch", seed=seed)
                dfr.build()
                dfr.fit(prob, epochs=ADAM, learning_rate=LR, verbose=False, lbfgs_iters=LBFGS)
                dfr_q = qoi_err(dfr.predict(xg))

                go = GoalOrientedDFRModel(
                    hidden_layers=arch, dim=1, qoi=qoi, n_modes=MODES, n_quadrature=NQ,
                    adjoint_weight=1.0, primal_weight=1.0, backend="pytorch", seed=seed,
                )
                go.build()
                go.fit(prob, epochs=ADAM, learning_rate=LR, verbose=False, lbfgs_iters=LBFGS)
                go_q = qoi_err(go.predict(xg))

                records.append({"problem": pname, "dof": dof, "seed": seed, "dfr_qoi": dfr_q, "go_qoi": go_q})
                print(f"{pname:8s} dof={dof:4d} seed={seed} | DFR={dfr_q:.2e} GO={go_q:.2e}", flush=True)

    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
