"""Research CLI subcommands for Goal-Oriented DFR.

Goal-Oriented DFR applies dual-weighted residual (DWR) error control to the DFR
H^{-1} dual-norm loss: a primal and an adjoint network are trained together so
that the loss targets a specific quantity of interest (QoI) rather than the
global energy error.
"""

import click
import numpy as np
from matplotlib import pyplot as plt

from hp_dfr.models import (
    AverageValueQoI,
    GoalOrientedDFRModel,
    PointEvaluationQoI,
    QuantityOfInterest,
)
from hp_dfr.models.base import Problem
from hp_dfr.problems import poisson_1d

from .common import (
    PROBLEM_DESCRIPTIONS,
    common_options,
    console,
    list_problems_table,
    network_options,
    parse_hidden_layers,
    problem_option,
)


@click.group()
def research() -> None:
    """Goal-Oriented DFR experiments.

    Train primal and adjoint networks with a QoI-weighted dual-norm loss so that
    error control focuses on a quantity of interest (point value, subdomain
    average) instead of the global energy norm.
    """


@research.command("run")
@problem_option
@common_options
@network_options(default_layers="32,32")
@click.option(
    "--qoi",
    type=click.Choice(["point", "average"]),
    default="point",
    help="Quantity of interest (1D): point evaluation or subdomain average",
)
@click.option(
    "--qoi-location",
    default=0.5,
    help="Relative location (in [0, 1]) for the point-evaluation QoI",
)
@click.option("--n-modes", default=20, help="Number of Fourier modes per dimension")
@click.option("--n-quadrature", default=64, help="Quadrature points per dimension")
def run(
    problem: str,
    backend: str,
    epochs: int,
    lr: float,
    hidden_layers: str,
    qoi: str,
    qoi_location: float,
    n_modes: int,
    n_quadrature: int,
    seed: int,
    output: str | None,
    plot: bool,
) -> None:
    """Run Goal-Oriented DFR on a 1D Poisson problem.

    Trains primal and adjoint networks with the QoI-weighted dual-norm loss and
    reports the QoI error and the global L2 error.
    """
    console.print(f"[bold blue]Goal-Oriented DFR on '{problem}' problem[/bold blue]")
    console.print(f"Backend: {backend}, QoI: {qoi}, modes/dim: {n_modes}")

    layers = parse_hidden_layers(hidden_layers)
    prob = poisson_1d.get_problem(problem)

    # Absolute QoI location within the problem domain.
    a, b = prob.domain
    abs_location = a + qoi_location * (b - a)

    qoi_obj: QuantityOfInterest
    if qoi == "point":
        qoi_obj = PointEvaluationQoI(point=np.array([abs_location]))
        console.print(f"Point QoI at x = {abs_location:.3f}")
    else:
        qoi_obj = AverageValueQoI()
        console.print("Average-value QoI over the full domain")

    model = GoalOrientedDFRModel(
        hidden_layers=tuple(layers),
        dim=1,
        qoi=qoi_obj,
        n_quadrature=n_quadrature,
        n_modes=n_modes,
        backend=backend,
        seed=seed,
    )

    model.build()
    history = model.fit(prob, epochs=epochs, learning_rate=lr, verbose=True)

    x_test = np.linspace(a, b, 200)
    u_pred = model.predict(x_test)

    if prob.exact_solution is not None:
        u_exact = prob.exact_solution(x_test)
        l2_error = float(np.sqrt(np.mean((u_pred - u_exact) ** 2)))

        if qoi == "point":
            qoi_pred = float(model.predict(np.array([abs_location]))[0])
            qoi_exact = float(prob.exact_solution(np.array([abs_location]))[0])
            qoi_error = abs(qoi_pred - qoi_exact)
            console.print(f"\n[bold green]QoI error (point at x={abs_location:.3f}): " f"{qoi_error:.6e}[/bold green]")

        console.print(f"[bold green]Global L2 error: {l2_error:.6e}[/bold green]")

    if plot:
        _plot_goal_oriented_results(x_test, u_pred, prob, history, abs_location, output)


@research.command()
def problems() -> None:
    """List available 1D problems for Goal-Oriented DFR experiments."""
    console.print("[bold]Problems available for Goal-Oriented DFR:[/bold]\n")
    list_problems_table(PROBLEM_DESCRIPTIONS)


def _plot_goal_oriented_results(
    x_test: "np.ndarray",
    u_pred: "np.ndarray",
    prob: Problem,
    history: dict,
    qoi_location: float,
    output: str | None,
) -> None:
    """Plot the predicted solution (with QoI marker) and the training loss."""
    try:
        _fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(x_test, u_pred, "b-", label="Predicted", linewidth=2)
        if prob.exact_solution is not None:
            u_exact = prob.exact_solution(x_test)
            axes[0].plot(x_test, u_exact, "r--", label="Exact", linewidth=2)
            axes[0].axvline(x=qoi_location, color="green", linestyle="-", alpha=0.7, label="QoI")

        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(x)")
        axes[0].set_title("Goal-Oriented DFR Solution")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.7)

        axes[1].semilogy(history["loss"], linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss (DWR)")
        axes[1].set_title("Training Loss")
        axes[1].grid(True, linestyle=":", alpha=0.7)

        plt.tight_layout()

        if output:
            plt.savefig(output, dpi=150)
            console.print(f"[green]Saved plot to {output}[/green]")
        else:
            plt.show()

    except ImportError:
        console.print("[yellow]matplotlib not available, skipping plots[/yellow]")
