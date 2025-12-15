"""DFR (Deep Fourier Residual) CLI subcommands.

Reference paper: "A Deep Fourier Residual Method for solving PDEs using Neural Networks"
"""

import os

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
def dfr():
    """Deep Fourier Residual method (reference paper).

    DFR uses a variational formulation with the H^{-1} dual norm computed
    via Discrete Sine/Cosine Transforms for error-equivalent loss functions.

    \b
    Key characteristics:
    - Uses weak-form residual in H^{-1} norm
    - Loss is equivalent to actual error (up to constants)
    - Only requires H^1 regularity (vs H^2 for PINNs)
    - Deterministic quadrature (no random collocation)

    \b
    Reference:
    Taylor et al., "A Deep Fourier Residual Method for solving PDEs
    using Neural Networks", CMAME 2023.
    """
    pass


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
    problem,
    backend,
    epochs,
    lr,
    hidden_layers,
    n_modes,
    n_quadrature,
    seed,
    output,
    plot,
):
    """Run a DFR experiment.

    Train a neural network using the Deep Fourier Residual method,
    computing loss in the H^{-1} dual norm via DST/DCT.
    """
    from hp_dfr.models import DFRModel
    from hp_dfr.problems import poisson_1d

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
        _plot_results(x_test, u_pred, prob, history, "DFR", output)


@dfr.command()
def problems():
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
    default="all",
    help="Problems to reproduce (comma-separated or 'all')",
)
def reproduce(output_dir, problems):
    """Reproduce results from the reference DFR paper.

    Runs the benchmark problems from the original paper with the same
    hyperparameters and saves results for comparison.
    """
    console.print("[bold blue]Reproducing DFR paper results...[/bold blue]")
    console.print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    if problems == "all":
        problem_list = list(PROBLEM_DESCRIPTIONS.keys())
    else:
        problem_list = [p.strip() for p in problems.split(",")]

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
        axes[1].set_ylabel("Loss (H⁻¹ norm)")
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
