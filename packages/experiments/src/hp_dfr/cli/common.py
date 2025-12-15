"""Shared CLI options and utilities."""

from functools import wraps
from typing import Callable, List

import click
from rich.console import Console
from rich.table import Table


console = Console()


def common_options(func: Callable) -> Callable:
    """Shared options for all run commands."""

    @click.option(
        "--backend",
        type=click.Choice(["tensorflow", "jax", "pytorch"]),
        default="tensorflow",
        help="Deep learning backend",
    )
    @click.option("--epochs", default=1000, help="Number of training epochs")
    @click.option("--lr", default=1e-3, help="Learning rate")
    @click.option("--seed", default=1234, help="Random seed")
    @click.option("--output", default=None, help="Output file for results")
    @click.option("--plot/--no-plot", default=True, help="Show plots")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def network_options(default_layers: str = "64,64,64") -> Callable:
    """Network architecture options with configurable default."""

    def decorator(func: Callable) -> Callable:
        @click.option(
            "--hidden-layers",
            default=default_layers,
            help="Hidden layer sizes (comma-separated)",
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def problem_option(func: Callable) -> Callable:
    """Standard problem selection option."""

    @click.option(
        "--problem",
        type=click.Choice(["sine", "arctan", "discontinuous", "delta"]),
        default="sine",
        help="Problem to solve",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def parse_hidden_layers(hidden_layers: str) -> List[int]:
    """Parse comma-separated layer sizes into list of integers."""
    return [int(x.strip()) for x in hidden_layers.split(",")]


def list_backends_table() -> None:
    """Display available backends and their status."""
    table = Table(title="Backend Status")
    table.add_column("Backend", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version", style="yellow")

    backends = [
        ("tensorflow", "tensorflow"),
        ("jax", "jax"),
        ("pytorch", "torch"),
    ]

    for name, module in backends:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            table.add_row(name, "[green]Available[/green]", version)
        except ImportError:
            table.add_row(name, "[red]Not installed[/red]", "-")

    console.print(table)


def list_problems_table(problems: dict) -> None:
    """Display available problems in a table."""
    table = Table(title="Available Problems")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")

    for name, desc in problems.items():
        table.add_row(name, desc)

    console.print(table)


# Standard problem descriptions
PROBLEM_DESCRIPTIONS = {
    "sine": "Smooth sine solution (Model Problem 1)",
    "arctan": "Large gradients (Model Problem 2)",
    "discontinuous": "Discontinuous coefficients (Model Problem 3)",
    "delta": "Point source (Model Problem 4)",
}
