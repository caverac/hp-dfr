"""Shared CLI options and utilities."""

from functools import wraps
from typing import Any, Callable, TypeVar

import click
from rich.console import Console
from rich.table import Table

console = Console()

# TypeVar for Click command decorators that preserves function signature
F = TypeVar("F", bound=Callable[..., Any])


def common_options(func: F) -> F:
    """Shared options for all run commands.

    Parameters
    ----------
    func
        The Click command function to decorate.

    Returns
    -------
    F
        The decorated function with common options added.
    """

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
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def network_options(default_layers: str = "64,64,64") -> Callable[[F], F]:
    """Network architecture options with configurable default.

    Parameters
    ----------
    default_layers
        Default hidden layer sizes as comma-separated string.

    Returns
    -------
    Callable[[F], F]
        A decorator that adds the --hidden-layers option.
    """

    def decorator(func: F) -> F:
        @click.option(
            "--hidden-layers",
            default=default_layers,
            help="Hidden layer sizes (comma-separated)",
        )
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def problem_option(func: F) -> F:
    """Standard problem selection option.

    Parameters
    ----------
    func
        The Click command function to decorate.

    Returns
    -------
    F
        The decorated function with --problem option added.
    """

    @click.option(
        "--problem",
        type=click.Choice(["sine", "arctan", "discontinuous", "delta"]),
        default="sine",
        help="Problem to solve",
    )
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def parse_hidden_layers(hidden_layers: str) -> tuple[int, ...]:
    """Parse comma-separated layer sizes into list of integers.

    Parameters
    ----------
    hidden_layers
        Comma-separated string of layer sizes (e.g., "64,64,64").

    Returns
    -------
    list[int]
        List of integer layer sizes.
    """
    return tuple(int(x.strip()) for x in hidden_layers.split(","))


def list_backends_table() -> None:
    """Display available backends and their status."""
    table = Table(title="Backend Status")
    table.add_column("Backend", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version", style="yellow")

    backends: list[tuple[str, str]] = [
        ("tensorflow", "tensorflow"),
        ("jax", "jax"),
        ("pytorch", "torch"),
    ]

    for name, module in backends:
        try:
            mod = __import__(module)
            version: str = getattr(mod, "__version__", "unknown")
            table.add_row(name, "[green]Available[/green]", version)
        except ImportError:
            table.add_row(name, "[red]Not installed[/red]", "-")

    console.print(table)


def list_problems_table(problems: dict[str, str]) -> None:
    """Display available problems in a table.

    Parameters
    ----------
    problems
        Dictionary mapping problem names to descriptions.
    """
    table = Table(title="Available Problems")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")

    for name, desc in problems.items():
        table.add_row(name, desc)

    console.print(table)


# Standard problem descriptions
PROBLEM_DESCRIPTIONS: dict[str, str] = {
    "sine": "Smooth sine solution (Model Problem 1)",
    "arctan": "Large gradients (Model Problem 2)",
    "discontinuous": "Discontinuous coefficients (Model Problem 3)",
    "delta": "Point source (Model Problem 4)",
}
