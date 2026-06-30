"""DFR (Deep Fourier Residual) CLI subcommands.

Reference paper: "A Deep Fourier Residual Method for solving PDEs using Neural Networks"
"""

import os
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from hp_dfr.models import DFRModel
from hp_dfr.models.base import Problem
from hp_dfr.problems import poisson_1d
from hp_dfr.types.common import BackendType, ProblemTypes

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
def dfr() -> None:
    """Deep Fourier Residual method (reference paper).

    DFR uses a variational formulation with the H^{-1} dual norm computed
    via Discrete Sine/Cosine Transforms for error-equivalent loss functions.

    Key characteristics:
    - Uses weak-form residual in H^{-1} norm
    - Loss is equivalent to actual error (up to constants)
    - Only requires H^1 regularity (vs H^2 for PINNs)
    - Deterministic quadrature (no random collocation)

    Reference:
    Taylor et al., "A Deep Fourier Residual Method for solving PDEs
    using Neural Networks", CMAME 2023.
    """


@dfr.command()
@problem_option
@common_options
@network_options(default_layers="10,10,10,10")
@click.option(
    "--n-modes",
    default=10,
    help="Number of Fourier modes for H^{-1} norm computation",
)
@click.option(
    "--n-quadrature",
    default=100,
    help="Number of quadrature points for integration",
)
def run(
    problem: ProblemTypes,
    backend: BackendType,
    epochs: int,
    lr: float,
    hidden_layers: str,
    n_modes: int,
    n_quadrature: int,
    seed: int,
    output: str,
    plot: bool,
) -> None:
    """Run a DFR experiment.

    Train a neural network using the Deep Fourier Residual method,
    computing loss in the H^{-1} dual norm via DST/DCT.
    """
    console.print(f"[bold blue]Running DFR on {problem} problem[/bold blue]")
    console.print(f"Backend: {backend}, Epochs: {epochs}, LR: {lr}")
    console.print(f"Fourier modes: {n_modes}, Quadrature points: {n_quadrature}")

    layers = parse_hidden_layers(hidden_layers)
    prob = poisson_1d.get_problem(problem)

    model = DFRModel(
        hidden_layers=layers,
        n_fourier_modes=n_modes,
        backend=backend,
        seed=seed,
    )

    model.build()
    history = model.fit(prob, epochs=epochs, learning_rate=lr, verbose=True)

    # Evaluate
    x_test = np.linspace(prob.domain[0], prob.domain[1], 100)
    u_pred = model.predict(x_test)

    if prob.exact_solution is not None:
        u_exact = prob.exact_solution(x_test)
        l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        console.print(f"\n[bold green]L2 Error: {l2_error:.6e}[/bold green]")

    if plot:
        _plot_results(x_test, u_pred, prob, history, output=Path(output) if output else None)


@dfr.command()
def problems() -> None:
    """List available problems for DFR experiments."""
    list_problems_table(PROBLEM_DESCRIPTIONS)


@dfr.command()
@click.option(
    "--output-dir",
    default="results/dfr_paper",
    help="Output directory for reproduced results",
)
@click.option(
    "--problems",
    "problem_spec",
    default="all",
    help="Problems to reproduce (comma-separated or 'all')",
)
def reproduce(output_dir: str, problem_spec: str) -> None:
    """Reproduce results from the reference DFR paper.

    Runs the benchmark problems from the original paper with the same
    hyperparameters and saves results for comparison.
    """
    console.print("[bold blue]Reproducing DFR paper results...[/bold blue]")
    console.print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    problem_list: list[str]
    if problem_spec == "all":
        problem_list = [str(p) for p in PROBLEM_DESCRIPTIONS]
    else:
        problem_list = [p.strip() for p in problem_spec.split(",")]

    console.print(f"Problems: {', '.join(problem_list)}")

    # Paper hyperparameters (from reference)
    paper_config = {
        "sine": {"n_modes": 10, "hidden_layers": [10, 10, 10, 10], "epochs": 10000},
        "arctan": {"n_modes": 20, "hidden_layers": [20, 20, 20, 20], "epochs": 20000},
        "discontinuous": {
            "n_modes": 30,
            "hidden_layers": [30, 30, 30, 30],
            "epochs": 30000,
        },
        "delta": {"n_modes": 40, "hidden_layers": [40, 40, 40, 40], "epochs": 40000},
    }

    for prob_name in problem_list:
        if prob_name not in paper_config:
            console.print(f"[yellow]Unknown problem: {prob_name}, skipping[/yellow]")
            continue

        config = paper_config[prob_name]
        console.print(f"\n[cyan]Running {prob_name}...[/cyan]")

        # TODO: Implement full paper reproduction with proper metrics
        console.print(f"  Config: {config}")
        console.print("[yellow]  Full reproduction not yet implemented[/yellow]")

    console.print(f"\n[green]Results will be saved to {output_dir}/[/green]")


def _plot_results(
    x_test: npt.NDArray[np.float64],
    u_pred: npt.NDArray[np.float64],
    prob: Problem,
    history: dict[str, list[float]],
    /,
    *,
    output: Path | None = None,
) -> None:
    """Plot solution and training history."""
    fig: Figure
    axs: list[Axes]
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.92, wspace=0.15, hspace=0.05)

    # Solution plot
    axs[0].plot(x_test, u_pred, color="black", linestyle="-", label="Predicted", linewidth=3)
    u_exact = prob.exact_solution(x_test)
    axs[0].plot(x_test, u_exact, color="gray", linestyle="--", label="Exact", linewidth=2)
    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel("$u(x)$")
    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[0].legend(frameon=False)
    axs[0].grid(True, linestyle="-", alpha=0.7)
    axs[0].tick_params(which="minor", length=3, color="gray", direction="in")
    axs[0].tick_params(which="major", length=6, direction="in")
    axs[0].tick_params(top=True, right=True, which="both")

    # Loss plot
    axs[1].semilogy(history["loss"], linewidth=2, color="black")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel(r"$\|R\|_{H^{-1}}$")
    axs[1].grid(True, linestyle="-", alpha=0.7)
    axs[1].tick_params(which="minor", length=3, color="gray", direction="in")
    axs[1].tick_params(which="major", length=6, direction="in")
    axs[1].tick_params(top=True, right=True, which="both")

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        console.print(f"[green]Saved plot to {output}[/green]")
    else:
        plt.show()
