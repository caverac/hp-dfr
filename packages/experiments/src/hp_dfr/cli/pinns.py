"""PINNs (Physics-Informed Neural Networks) CLI subcommands."""

import click
import numpy as np

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
def pinns():
    """Physics-Informed Neural Networks experiments.

    PINNs solve PDEs by minimizing the strong-form residual at collocation
    points, with boundary conditions enforced via penalty terms.

    \b
    Key characteristics:
    - Uses L2 norm of PDE residual as loss
    - Requires tuning of BC penalty weight
    - Works well for smooth solutions
    """
    pass


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
def run(
    problem,
    backend,
    epochs,
    lr,
    hidden_layers,
    n_collocation,
    bc_weight,
    seed,
    output,
    plot,
):
    """Run a PINNs experiment.

    Train a physics-informed neural network to solve the specified problem
    using the strong-form residual minimization approach.
    """
    from hp_dfr.models import PINNsModel
    from hp_dfr.problems import poisson_1d

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
        _plot_results(x_test, u_pred, prob, history, "PINNs", output)


@pinns.command()
def problems():
    """List available problems for PINNs experiments."""
    list_problems_table(PROBLEM_DESCRIPTIONS)


def _plot_results(x_test, u_pred, prob, history, method_name, output):
    """Plot solution and training history."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Solution plot
        axes[0].plot(x_test, u_pred, "b-", label="Predicted", linewidth=2)
        if prob.exact_solution is not None:
            u_exact = prob.exact_solution(x_test)
            axes[0].plot(x_test, u_exact, "r--", label="Exact", linewidth=2)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(x)")
        axes[0].set_title(f"{method_name} Solution")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.7)

        # Loss plot
        axes[1].semilogy(history["loss"], linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
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
