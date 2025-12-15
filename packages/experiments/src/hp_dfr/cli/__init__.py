"""Command-line interface for hp-DFR experiments.

This module provides a structured CLI with subcommands for different experiment types:
- pinns: Physics-Informed Neural Networks experiments
- dfr: Deep Fourier Residual method (reference paper reproduction)
- research: Adaptive hp-DFR methods (our new research)
"""

import click

from .common import list_backends_table
from .dfr import dfr
from .pinns import pinns
from .research import research


@click.group()
@click.version_option()
def main():
    """hp-DFR: Adaptive hp-refinement Deep Fourier Residual Methods.

    A toolkit for solving PDEs using neural networks with multiple methods:

    \b
    - PINNs: Physics-Informed Neural Networks (collocation-based)
    - DFR: Deep Fourier Residual method (variational, H^{-1} norm loss)
    - hp-DFR: Our research - adaptive refinement with goal-oriented error

    Use 'hp-dfr <command> --help' for more information on each command.
    """
    pass


# Register subcommand groups
main.add_command(pinns)
main.add_command(dfr)
main.add_command(research)


@main.command()
def backends():
    """List available deep learning backends and their status."""
    list_backends_table()


# Export main for entry point
__all__ = ["main"]
