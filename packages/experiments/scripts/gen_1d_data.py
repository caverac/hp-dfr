"""Generate the 1D sweep data: plain DFR vs goal-oriented DFR at matched DOF.

Thin wrapper around :func:`hp_dfr.data.run_1d_sweep`, kept for backward
compatibility. Prefer the CLI::

    uv run hp-dfr data 1d

Direct invocation still works::

    KERAS_BACKEND=torch uv run python packages/experiments/scripts/gen_1d_data.py
"""

from hp_dfr.data import run_1d_sweep

if __name__ == "__main__":
    run_1d_sweep(backend="pytorch")
