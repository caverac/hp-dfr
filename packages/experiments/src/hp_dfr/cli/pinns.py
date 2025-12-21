# pyright: reportUnknownMemberType=false
"""PINNs (Physics-Informed Neural Networks) CLI subcommands.

The pyright directive above suppresses "Type of X is partially unknown" warnings
caused by incomplete type stubs in matplotlib (specifically plt.subplots).
"""

from pathlib import Path
from typing import Any, Literal
import click
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from hp_dfr.models import PINNsModel
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
def pinns() -> None:
    """Physics-Informed Neural Networks experiments.

    PINNs solve PDEs by minimizing the strong-form residual at collocation
    points, with boundary conditions enforced via penalty terms.

    Key characteristics:
    - Uses L2 norm of PDE residual as loss
    - Requires tuning of BC penalty weight
    - Works well for smooth solutions
    """


@pinns.command()
@problem_option
@common_options
@network_options(default_layers="64,64,64")
@click.option(
    "--n-collocation",
    default=1000,
    help="Number of collocation points for residual evaluation",
)
@click.option(
    "--bc-weight",
    default=100.0,
    help="Boundary condition penalty weight (lambda)",
)
def run(  # pylint: disable=too-many-positional-arguments
    problem: str,
    backend: Literal["tensorflow", "jax", "pytorch"],
    epochs: int,
    lr: float,
    hidden_layers: str,
    n_collocation: int,
    bc_weight: float,
    seed: int,
    output: str,
    plot: bool,
) -> None:
    """Run a PINNs experiment.

    Train a physics-informed neural network to solve the specified problem
    using the strong-form residual minimization approach.

    Parameters
    ----------
    problem : str
        Problem type (e.g. sine), see 'pinns problems' for options
    backend : str
        Deep learning backend to use (tensorflow, pytorch)
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    hidden_layers : str
        Comma-separated list of hidden layer sizes
    n_collocation : int
        Number of collocation points for residual evaluation
    bc_weight : float
        Boundary condition penalty weight (lambda)
    seed : int
        Random seed for reproducibility
    output : str
        Path to save output plots (if empty, plots are shown interactively)
    plot : bool
        Whether to generate plots of the solution and training history
    """

    console.print(f"[bold blue]Running PINNs on {problem} problem[/bold blue]")
    console.print(f"Backend: {backend}, Epochs: {epochs}, LR: {lr}")
    console.print(f"Collocation points: {n_collocation}, BC weight: {bc_weight}")

    layers = parse_hidden_layers(hidden_layers)
    prob = poisson_1d.get_problem(problem)

    model = PINNsModel(
        hidden_layers=layers,
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


@pinns.command()
def problems() -> None:
    """List available problems for PINNs experiments."""
    list_problems_table(PROBLEM_DESCRIPTIONS)


def _plot_results(
    x_test: npt.NDArray[np.float64],
    u_pred: npt.NDArray[np.float64],
    prob: poisson_1d.Poisson1D[Any],
    history: dict[str, list[float]],
    /,
    *,
    output: Path | None = None,
) -> None:
    """Plot solution and training history."""
    try:
        fig: Figure
        axs: list[Axes]
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))

        fig.subplots_adjust(
            left=0.1,
            right=0.98,
            bottom=0.1,
            top=0.92,
            wspace=0.15,
            hspace=0.05,
        )

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
        axs[1].set_ylabel("Loss")
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

    except ImportError:
        console.print("[yellow]matplotlib not available, skipping plots[/yellow]")
