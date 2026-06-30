"""Command-line interface for the DFR experiments.

Subcommands:
- pinns: Physics-Informed Neural Networks baseline.
- dfr: Deep Fourier Residual method (reference paper reproduction).
- research: Goal-Oriented DFR (this project's contribution).
"""

import click

from hp_dfr.figures import make_all_figures

from .common import list_backends_table
from .dfr import dfr
from .pinns import pinns
from .research import research


@click.group()
@click.version_option()
def main() -> None:
    r"""Deep Fourier Residual experiments: baselines and Goal-Oriented DFR.

    \b
    hp-dfr pinns run ...       PINNs baseline
    hp-dfr dfr run ...         DFR reference paper
    hp-dfr research run ...    Goal-Oriented DFR
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


@main.command()
def figures() -> None:
    """Build the preprint figures from saved experiment data."""
    make_all_figures()


# Export main for entry point
__all__ = ["main"]


if __name__ == "__main__":
    main()
