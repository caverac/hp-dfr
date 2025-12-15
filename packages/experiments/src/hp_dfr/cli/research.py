"""Research CLI subcommands for hp-DFR (our new research).

This module contains commands for our adaptive hp-refinement DFR methods:
- hp-DFR with domain decomposition and adaptive refinement
- Sparse DFR for high-dimensional problems
- Goal-oriented DFR with dual-weighted residual
"""

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
def research():
    """Adaptive hp-DFR methods (our research).

    Our extensions to the DFR method featuring:

    \b
    - Adaptive hp-refinement: Domain decomposition with local networks
    - Sparse Fourier methods: O(N(log N)^{d-1}) complexity for high dimensions
    - Goal-oriented error: Dual-weighted residual for quantities of interest

    These methods address limitations of standard DFR in challenging scenarios.
    """
    pass


@research.command()
@problem_option
@common_options
@network_options(default_layers="32,32")
@click.option("--dim", default=1, help="Problem dimension")
@click.option(
    "--initial-divisions",
    default=2,
    help="Initial number of domain divisions per dimension",
)
@click.option(
    "--interface-penalty",
    default=10.0,
    help="Penalty weight for interface coupling",
)
@click.option(
    "--adapt-every",
    default=100,
    help="Epochs between adaptive refinement checks",
)
@click.option(
    "--max-subdomains",
    default=16,
    help="Maximum number of subdomains",
)
@click.option(
    "--refinement-threshold",
    default=0.1,
    help="Error threshold for triggering refinement",
)
def run(
    problem,
    backend,
    epochs,
    lr,
    hidden_layers,
    dim,
    initial_divisions,
    interface_penalty,
    adapt_every,
    max_subdomains,
    refinement_threshold,
    seed,
    output,
    plot,
):
    """Run an hp-DFR experiment with adaptive refinement.

    Uses domain decomposition with local neural networks on each subdomain.
    The mesh is adaptively refined based on local error indicators.
    """
    from hp_dfr.models import HPDFRModel
    from hp_dfr.problems import poisson_1d

    console.print(f"[bold blue]Running hp-DFR on {problem} problem[/bold blue]")
    console.print(f"Backend: {backend}, Epochs: {epochs}, LR: {lr}")
    console.print(f"Initial divisions: {initial_divisions}, Max subdomains: {max_subdomains}")
    console.print(f"Interface penalty: {interface_penalty}, Adapt every: {adapt_every} epochs")

    layers = parse_hidden_layers(hidden_layers)
    prob = poisson_1d.get_problem(problem)

    model = HPDFRModel(
        hidden_layers=layers,
        dim=dim,
        initial_divisions=initial_divisions,
        interface_penalty=interface_penalty,
        backend=backend,
        seed=seed,
    )

    model.build()

    # Training with periodic adaptation
    total_epochs = 0
    while total_epochs < epochs:
        batch_epochs = min(adapt_every, epochs - total_epochs)
        history = model.fit(prob, epochs=batch_epochs, learning_rate=lr, verbose=True)
        total_epochs += batch_epochs

        # Check for adaptation
        if total_epochs < epochs and model.n_subdomains < max_subdomains:
            refined = model.adapt(prob, threshold=refinement_threshold)
            if refined:
                console.print(
                    f"[yellow]Refined mesh at epoch {total_epochs}, "
                    f"now {model.n_subdomains} subdomains[/yellow]"
                )

    # Evaluate
    x_test = np.linspace(prob.domain[0], prob.domain[1], 100)
    u_pred = model.predict(x_test)

    if prob.exact_solution is not None:
        u_exact = prob.exact_solution(x_test)
        l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        console.print(f"\n[bold green]L2 Error: {l2_error:.6e}[/bold green]")
        console.print(f"[bold green]Final subdomains: {model.n_subdomains}[/bold green]")

    if plot:
        _plot_hp_results(x_test, u_pred, prob, model, history, output)


@research.command("run-sparse")
@problem_option
@common_options
@network_options(default_layers="64,64,64")
@click.option("--dim", default=2, help="Problem dimension")
@click.option(
    "--max-level",
    default=16,
    help="Maximum Fourier level for hyperbolic cross",
)
@click.option(
    "--use-sparse/--full-tensor",
    default=True,
    help="Use sparse (hyperbolic cross) vs full tensor indices",
)
def run_sparse(
    problem,
    backend,
    epochs,
    lr,
    hidden_layers,
    dim,
    max_level,
    use_sparse,
    seed,
    output,
    plot,
):
    """Run SparseDFR for high-dimensional problems.

    Uses hyperbolic cross index sets to reduce complexity from O(N^d)
    to O(N(log N)^{d-1}), enabling DFR in higher dimensions.
    """
    from hp_dfr.models import SparseDFRModel
    from hp_dfr.problems import poisson_1d

    console.print(f"[bold blue]Running Sparse DFR on {problem} problem[/bold blue]")
    console.print(f"Backend: {backend}, Dimension: {dim}")
    console.print(f"Max Fourier level: {max_level}, Sparse: {use_sparse}")

    layers = parse_hidden_layers(hidden_layers)

    # For now, only 1D problems available
    if dim > 1:
        console.print("[yellow]Note: Only 1D problems currently implemented[/yellow]")
        console.print("[yellow]Running with dim=1 for demonstration[/yellow]")
        dim = 1

    prob = poisson_1d.get_problem(problem)

    model = SparseDFRModel(
        hidden_layers=layers,
        dim=dim,
        max_level=max_level,
        use_sparse=use_sparse,
        backend=backend,
        seed=seed,
    )

    model.build()
    history = model.fit(prob, epochs=epochs, learning_rate=lr, verbose=True)

    # Report compression ratio
    n_sparse = model.n_modes_sparse
    n_full = model.n_modes_full
    compression = n_full / n_sparse if n_sparse > 0 else 1.0
    console.print(f"\n[cyan]Fourier modes: {n_sparse} sparse vs {n_full} full ({compression:.1f}x compression)[/cyan]")

    # Evaluate
    x_test = np.linspace(prob.domain[0], prob.domain[1], 100)
    u_pred = model.predict(x_test)

    if prob.exact_solution is not None:
        u_exact = prob.exact_solution(x_test)
        l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        console.print(f"[bold green]L2 Error: {l2_error:.6e}[/bold green]")

    if plot:
        _plot_results(x_test, u_pred, prob, history, "Sparse DFR", output)


@research.command("run-goal-oriented")
@problem_option
@common_options
@network_options(default_layers="32,32,32")
@click.option(
    "--qoi",
    type=click.Choice(["point", "average", "flux"]),
    default="point",
    help="Quantity of interest type",
)
@click.option(
    "--qoi-location",
    default=0.5,
    help="Location for point evaluation QoI (relative to domain)",
)
@click.option(
    "--n-modes",
    default=20,
    help="Number of Fourier modes",
)
def run_goal_oriented(
    problem,
    backend,
    epochs,
    lr,
    hidden_layers,
    qoi,
    qoi_location,
    n_modes,
    seed,
    output,
    plot,
):
    """Run goal-oriented DFR with dual-weighted residual.

    Focuses error control on a specific quantity of interest (QoI)
    using adjoint-based error estimation.
    """
    from hp_dfr.models import GoalOrientedDFRModel
    from hp_dfr.problems import poisson_1d

    console.print(f"[bold blue]Running Goal-Oriented DFR on {problem} problem[/bold blue]")
    console.print(f"Backend: {backend}, QoI: {qoi}")
    console.print(f"QoI location: {qoi_location}, Fourier modes: {n_modes}")

    layers = parse_hidden_layers(hidden_layers)
    prob = poisson_1d.get_problem(problem)

    # Compute absolute QoI location
    domain_length = prob.domain[1] - prob.domain[0]
    abs_location = prob.domain[0] + qoi_location * domain_length

    model = GoalOrientedDFRModel(
        hidden_layers=layers,
        n_fourier_modes=n_modes,
        qoi_type=qoi,
        qoi_params={"location": abs_location},
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

        # Compute QoI error
        if qoi == "point":
            qoi_pred = model.evaluate_qoi(u_pred, x_test)
            qoi_exact = prob.exact_solution(abs_location)
            qoi_error = abs(qoi_pred - qoi_exact)
            console.print(f"\n[bold green]QoI Error (point at x={abs_location:.2f}): {qoi_error:.6e}[/bold green]")

        console.print(f"[bold green]Global L2 Error: {l2_error:.6e}[/bold green]")

    if plot:
        _plot_goal_oriented_results(x_test, u_pred, prob, model, history, abs_location, output)


@research.command()
def problems():
    """List available problems for hp-DFR research experiments."""
    console.print("[bold]Problems available for hp-DFR research:[/bold]\n")
    list_problems_table(PROBLEM_DESCRIPTIONS)

    console.print("\n[dim]Note: hp-DFR methods are particularly effective for:[/dim]")
    console.print("[dim]  - arctan: Large gradients benefit from adaptive refinement[/dim]")
    console.print("[dim]  - discontinuous: Domain decomposition handles jumps well[/dim]")
    console.print("[dim]  - delta: Local refinement near singularity[/dim]")


def _plot_results(x_test, u_pred, prob, history, method_name, output):
    """Plot solution and training history."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(x_test, u_pred, "b-", label="Predicted", linewidth=2)
        if prob.exact_solution is not None:
            u_exact = prob.exact_solution(x_test)
            axes[0].plot(x_test, u_exact, "r--", label="Exact", linewidth=2)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(x)")
        axes[0].set_title(f"{method_name} Solution")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.7)

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


def _plot_hp_results(x_test, u_pred, prob, model, history, output):
    """Plot hp-DFR results with subdomain visualization."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Solution plot
        axes[0].plot(x_test, u_pred, "b-", label="Predicted", linewidth=2)
        if prob.exact_solution is not None:
            u_exact = prob.exact_solution(x_test)
            axes[0].plot(x_test, u_exact, "r--", label="Exact", linewidth=2)

        # Mark subdomain boundaries
        for boundary in model.subdomain_boundaries:
            axes[0].axvline(x=boundary, color="gray", linestyle=":", alpha=0.5)

        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(x)")
        axes[0].set_title("hp-DFR Solution")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.7)

        # Loss plot
        axes[1].semilogy(history["loss"], linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss")
        axes[1].grid(True, linestyle=":", alpha=0.7)

        # Error distribution per subdomain
        subdomain_errors = model.get_subdomain_errors(prob)
        axes[2].bar(range(len(subdomain_errors)), subdomain_errors, color="steelblue")
        axes[2].set_xlabel("Subdomain")
        axes[2].set_ylabel("Local Error")
        axes[2].set_title("Error Distribution")
        axes[2].grid(True, linestyle=":", alpha=0.7, axis="y")

        plt.tight_layout()

        if output:
            plt.savefig(output, dpi=150)
            console.print(f"[green]Saved plot to {output}[/green]")
        else:
            plt.show()

    except ImportError:
        console.print("[yellow]matplotlib not available, skipping plots[/yellow]")


def _plot_goal_oriented_results(x_test, u_pred, prob, model, history, qoi_location, output):
    """Plot goal-oriented DFR results with QoI visualization."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Solution with QoI marker
        axes[0].plot(x_test, u_pred, "b-", label="Predicted", linewidth=2)
        if prob.exact_solution is not None:
            u_exact = prob.exact_solution(x_test)
            axes[0].plot(x_test, u_exact, "r--", label="Exact", linewidth=2)
            axes[0].axvline(x=qoi_location, color="green", linestyle="-", alpha=0.7, label="QoI location")
            axes[0].scatter([qoi_location], [prob.exact_solution(qoi_location)], color="green", s=100, zorder=5)

        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(x)")
        axes[0].set_title("Goal-Oriented DFR Solution")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.7)

        # Loss plot
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
