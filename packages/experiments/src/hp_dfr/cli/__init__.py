"""Command-line interface for hp-DFR experiments.

This module provides a structured CLI with subcommands for different experiment types:
- pinns: Physics-Informed Neural Networks experiments
- dfr: Deep Fourier Residual method (reference paper reproduction)
- research: Adaptive hp-DFR methods (our new research)

Legacy commands are preserved for backwards compatibility but are deprecated.
"""

import click

from .common import list_backends_table
from .dfr import dfr
from .pinns import pinns
from .research import research


@click.group()
@click.version_option()
def main() -> None:
    """hp-DFR: Adaptive hp-refinement Deep Fourier Residual Methods.

    A toolkit for solving PDEs using neural networks with multiple methods:

    \b
    hp-dfr pinns run ...       PINNs experiments
    hp-dfr dfr run ...         DFR reference paper
    hp-dfr research run ...    Our hp-DFR research
    hp-dfr backends            List available backends
    """


# Register subcommand groups
main.add_command(pinns)
main.add_command(dfr)
main.add_command(research)


@main.command()
def backends() -> None:
    """List available deep learning backends and their status."""
    list_backends_table()


# Export main for entry point
__all__ = ["main"]


if __name__ == "__main__":
    main()
