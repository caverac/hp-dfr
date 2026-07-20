"""Sweep-data generation CLI subcommands.

Regenerate the training-sweep JSON that :func:`hp_dfr.figures.make_all_figures`
reads to build the preprint figures. The sweeps are expensive (many widths x
seeds x epochs); the ``figures`` command is fast because it only reads the JSON.
"""

import click

from hp_dfr.data import (
    merge_matched_cost_shards,
    run_1d_sweep,
    run_2d_convergence_sweep,
    run_2d_matched_cost_sweep,
    run_2d_steepness_sweep,
    run_2d_sweep,
    run_2d_tradeoff_sweep,
)
from hp_dfr.types.common import BackendType

from .common import console

_backend_option = click.option(
    "--backend",
    type=click.Choice(["tensorflow", "jax", "pytorch"]),
    default="pytorch",
    help="Deep learning backend",
)


@click.group()
def data() -> None:
    r"""Generate the preprint sweep data consumed by ``hp-dfr figures``.

    \b
    hp-dfr data 1d             1D DFR vs goal-oriented sweep (m2_1d_data.json)
    hp-dfr data 2d             2D DFR vs goal-oriented sweep (m3_2d_data.json)
    hp-dfr data steepness      2D steepness sweep (m4_2d_steepness.json)
    hp-dfr data tradeoff       2D reallocation + convergence diagnostics (Section 6.4)
    hp-dfr data matched-cost   2D sweep giving plain DFR matched wall-clock (shardable)
    hp-dfr data merge-shards   Combine matched-cost shard files
    hp-dfr data all            The three preprint sweeps
    """


@data.command(name="1d")
@_backend_option
def data_1d(backend: BackendType) -> None:
    """Generate the 1D sweep data into the assets directory."""
    console.print("[bold blue]Generating 1D sweep data[/bold blue]")
    run_1d_sweep(backend=backend)


@data.command(name="2d")
@_backend_option
def data_2d(backend: BackendType) -> None:
    """Generate the 2D sweep data into the assets directory."""
    console.print("[bold blue]Generating 2D sweep data[/bold blue]")
    run_2d_sweep(backend=backend)


@data.command(name="steepness")
@_backend_option
def data_steepness(backend: BackendType) -> None:
    """Generate the 2D steepness sweep data into the assets directory."""
    console.print("[bold blue]Generating 2D steepness sweep data[/bold blue]")
    run_2d_steepness_sweep(backend=backend)


@data.command(name="tradeoff")
@_backend_option
def data_tradeoff(backend: BackendType) -> None:
    """Generate the Section 6.4 reallocation and convergence diagnostics.

    Writes ``floor_probe.json`` (what goal-orientation trades) and
    ``convergence.json`` (whether plain DFR is converged at the shared budget)
    into the assets directory. This is a long run: plain DFR is trained to many
    times the sweep budget to trace its convergence.
    """
    console.print("[bold blue]Generating 2D reallocation data[/bold blue]")
    run_2d_tradeoff_sweep(backend=backend)
    console.print("[bold blue]Generating 2D convergence data[/bold blue]")
    run_2d_convergence_sweep(backend=backend)


@data.command(name="matched-cost")
@click.option("--which", type=click.Choice(["steepness", "2d"]), default="steepness", help="Which sweep to repeat")
@click.option("--shard", type=int, default=0, help="Zero-based shard index to run")
@click.option("--num-shards", type=int, default=1, help="Total number of shards")
@_backend_option
def data_matched_cost(which: str, shard: int, num_shards: int, backend: BackendType) -> None:
    """Run one shard of the matched-cost sweep (plain DFR gets matched wall-clock)."""
    console.print(f"[bold blue]Matched-cost {which} sweep, shard {shard}/{num_shards}[/bold blue]")
    run_2d_matched_cost_sweep(which=which, shard=shard, num_shards=num_shards, backend=backend)


@data.command(name="merge-shards")
@click.option("--which", type=click.Choice(["steepness", "2d"]), default="steepness", help="Which sweep to merge")
@click.option("--num-shards", type=int, required=True, help="Number of shards to combine")
def data_merge_shards(which: str, num_shards: int) -> None:
    """Combine matched-cost shard files into the canonical JSON."""
    console.print(f"[bold blue]Merging {num_shards} {which} shards[/bold blue]")
    merge_matched_cost_shards(which=which, num_shards=num_shards)


@data.command(name="all")
@_backend_option
def data_all(backend: BackendType) -> None:
    """Generate the 1D, 2D and 2D-steepness sweep data."""
    console.print("[bold blue]Generating 1D sweep data[/bold blue]")
    run_1d_sweep(backend=backend)
    console.print("[bold blue]Generating 2D sweep data[/bold blue]")
    run_2d_sweep(backend=backend)
    console.print("[bold blue]Generating 2D steepness sweep data[/bold blue]")
    run_2d_steepness_sweep(backend=backend)


__all__ = ["data"]
