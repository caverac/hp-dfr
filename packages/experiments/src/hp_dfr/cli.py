"""Command-line interface for hp-DFR experiments.

This module re-exports the main CLI from hp_dfr.cli for backwards compatibility.
The CLI has been restructured into nested subcommands:

    hp-dfr pinns run ...       # PINNs experiments
    hp-dfr dfr run ...         # DFR reference paper
    hp-dfr research run ...    # Our hp-DFR research
    hp-dfr backends            # List available backends

Legacy commands are preserved for backwards compatibility but may be removed
in a future version.
"""

import click
import numpy as np
from rich.console import Console

from hp_dfr.cli.common import (
    PROBLEM_DESCRIPTIONS,
    list_backends_table,
    list_problems_table,
)


console = Console()


# Create a new main group that combines old and new commands
@click.group()
@click.version_option()
def main():
    """hp-DFR: Adaptive hp-refinement Deep Fourier Residual Methods.

    A toolkit for solving PDEs using neural networks with multiple methods:

    \b
    NEW STRUCTURE (recommended):
      hp-dfr pinns run ...       PINNs experiments
      hp-dfr dfr run ...         DFR reference paper
      hp-dfr research run ...    Our hp-DFR research
      hp-dfr backends            List available backends

    \b
    LEGACY COMMANDS (deprecated):
      hp-dfr run --method ...    Use 'hp-dfr <method> run' instead
      hp-dfr list-problems       Use 'hp-dfr <method> problems' instead
      hp-dfr list-backends       Use 'hp-dfr backends' instead
    """
    pass


# Import and register new subcommand groups
from hp_dfr.cli.dfr import dfr
from hp_dfr.cli.pinns import pinns
from hp_dfr.cli.research import research

main.add_command(pinns)
main.add_command(dfr)
main.add_command(research)


@main.command()
def backends():
    """List available deep learning backends and their status."""
    list_backends_table()


# Legacy commands for backwards compatibility
@main.command("run")
@click.option(
    "--method",
    type=click.Choice(["pinns", "dfr"]),
    default="dfr",
    help="Method to use (pinns or dfr)",
)
@click.option(
    "--problem",
    type=click.Choice(["sine", "arctan", "discontinuous", "delta"]),
    default="sine",
    help="Problem to solve",
)
@click.option(
    "--backend",
    type=click.Choice(["tensorflow", "jax", "pytorch"]),
    default="tensorflow",
    help="Deep learning backend",
)
@click.option("--epochs", default=1000, help="Number of training epochs")
@click.option("--lr", default=1e-3, help="Learning rate")
@click.option(
    "--hidden-layers", default="10,10,10,10", help="Hidden layer sizes (comma-separated)"
)
@click.option("--n-modes", default=10, help="Number of Fourier modes (DFR only)")
@click.option("--seed", default=1234, help="Random seed")
@click.option("--output", default=None, help="Output file for results")
@click.option("--plot/--no-plot", default=True, help="Show plots")
def run_legacy(
    method, problem, backend, epochs, lr, hidden_layers, n_modes, seed, output, plot
):
    """[DEPRECATED] Run a single experiment.

    This command is deprecated. Please use:

    \b
      hp-dfr pinns run ...    for PINNs experiments
      hp-dfr dfr run ...      for DFR experiments
    """
    console.print(
        "[yellow]Warning: 'hp-dfr run' is deprecated. "
        f"Use 'hp-dfr {method} run' instead.[/yellow]\n"
    )

    from hp_dfr.models import DFRModel, PINNsModel
    from hp_dfr.problems import poisson_1d

    console.print(f"[bold blue]Running {method.upper()} on {problem} problem[/bold blue]")
    console.print(f"Backend: {backend}, Epochs: {epochs}, LR: {lr}")

    layers = [int(x) for x in hidden_layers.split(",")]
    prob = poisson_1d.get_problem(problem)

    if method == "pinns":
        model = PINNsModel(
            hidden_layers=layers,
            backend=backend,
            seed=seed,
        )
    else:
        model = DFRModel(
            hidden_layers=layers,
            n_fourier_modes=n_modes,
            backend=backend,
            seed=seed,
        )

    model.build()
    history = model.fit(prob, epochs=epochs, learning_rate=lr, verbose=True)

    x_test = np.linspace(prob.domain[0], prob.domain[1], 100)
    u_pred = model.predict(x_test)

    if prob.exact_solution is not None:
        u_exact = prob.exact_solution(x_test)
        l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        console.print(f"\n[bold green]L2 Error: {l2_error:.6e}[/bold green]")

    if plot:
        _plot_results(x_test, u_pred, prob, history, method.upper(), output)


@main.command("list-problems")
def list_problems_legacy():
    """[DEPRECATED] List available problems.

    This command is deprecated. Please use:

    \b
      hp-dfr pinns problems
      hp-dfr dfr problems
      hp-dfr research problems
    """
    console.print(
        "[yellow]Warning: 'hp-dfr list-problems' is deprecated. "
        "Use 'hp-dfr <method> problems' instead.[/yellow]\n"
    )
    list_problems_table(PROBLEM_DESCRIPTIONS)


@main.command("list-backends")
def list_backends_legacy():
    """[DEPRECATED] List available backends.

    This command is deprecated. Please use 'hp-dfr backends' instead.
    """
    console.print(
        "[yellow]Warning: 'hp-dfr list-backends' is deprecated. "
        "Use 'hp-dfr backends' instead.[/yellow]\n"
    )
    list_backends_table()


@main.command("reproduce-paper")
@click.option("--output-dir", default="results", help="Output directory for results")
def reproduce_paper_legacy(output_dir):
    """[DEPRECATED] Reproduce all results from the paper.

    This command is deprecated. Please use 'hp-dfr dfr reproduce' instead.
    """
    console.print(
        "[yellow]Warning: 'hp-dfr reproduce-paper' is deprecated. "
        "Use 'hp-dfr dfr reproduce' instead.[/yellow]\n"
    )

    # Delegate to the new command
    from hp_dfr.cli.dfr import reproduce

    ctx = click.Context(reproduce)
    ctx.invoke(reproduce, output_dir=output_dir)


def _plot_results(x_test, u_pred, prob, history, method_name, output):
    """Plot solution and training history."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(x_test, u_pred, "b-", label="Predicted")
        if prob.exact_solution is not None:
            u_exact = prob.exact_solution(x_test)
            axes[0].plot(x_test, u_exact, "r--", label="Exact")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(x)")
        axes[0].set_title(f"{method_name} Solution")
        axes[0].legend()
        axes[0].grid(True, linestyle=":")

        axes[1].semilogy(history["loss"])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss")
        axes[1].grid(True, linestyle=":")

        plt.tight_layout()

        if output:
            plt.savefig(output)
            console.print(f"[green]Saved plot to {output}[/green]")
        else:
            plt.show()

    except ImportError:
        console.print("[yellow]matplotlib not available, skipping plots[/yellow]")


if __name__ == "__main__":
    main()
