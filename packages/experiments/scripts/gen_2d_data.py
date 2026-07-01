"""Generate the 2D sweep data: plain DFR vs goal-oriented DFR at matched DOF.

Thin wrapper around :func:`hp_dfr.data.run_2d_sweep`, kept for backward
compatibility. Prefer the CLI::

    uv run hp-dfr data 2d

Direct invocation still works::

    KERAS_BACKEND=torch uv run python packages/experiments/scripts/gen_2d_data.py
"""

from hp_dfr.data import run_2d_sweep

if __name__ == "__main__":
    run_2d_sweep(backend="pytorch")
