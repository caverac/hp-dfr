"""Publication figures for the goal-oriented DFR preprint.

Each function builds one figure with matplotlib's object-oriented API and returns
it; the :func:`~hp_dfr._plotting.figure` decorator applies the house style and
writes ``.png`` (documentation) and ``.pdf`` (paper) to the preprint figures
directory. Expensive training data is generated separately and read from JSON, so
regenerating figures is fast and reproducible. :func:`make_all_figures` builds the
full set.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import cast

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from hp_dfr._plotting import FIG_DIR, figure

Record = dict[str, object]
MedianBand = tuple[
    npt.NDArray[np.int_],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]


def _load(name: str) -> list[Record]:
    path = FIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing figure data {path}; generate it before building figures")
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, list):
        raise ValueError(f"expected a JSON array in {path}")
    records: list[Record] = loaded
    return records


def _median_band(records: list[Record], problem: str, key: str) -> MedianBand:
    """Return (dofs, median, lo, hi) over seeds for one method/problem."""
    by_dof: dict[int, list[float]] = defaultdict(list)
    for r in records:
        if r["problem"] == problem:
            by_dof[cast(int, r["dof"])].append(cast(float, r[key]))
    dofs = sorted(by_dof)
    med = np.array([np.median(by_dof[d]) for d in dofs])
    lo = np.array([np.min(by_dof[d]) for d in dofs])
    hi = np.array([np.max(by_dof[d]) for d in dofs])
    return np.array(dofs), med, lo, hi


@figure("dfr-saturation-1d")
def figure_dfr_saturation_1d() -> Figure:
    r"""Plot the QoI error versus degrees of freedom for both methods.

    On both a smooth (sine) and a sharp (arctan) 1D Poisson problem, plain DFR
    reaches its accuracy floor (~1e-5) already at the smallest network sizes, so
    there is no resolution-limited regime for goal-orientation to exploit; the
    goal-oriented variant is uniformly less accurate on the quantity of interest.
    Markers are medians over seeds; bands span the seed min--max.
    """
    records = _load("m2_1d_data.json")
    panels = [("sine", "smooth (sine)"), ("arctan8", r"sharp (arctan, $k=8$)")]
    styles = [
        ("dfr_qoi", "-", "o", "DFR"),
        ("go_qoi", "--", "s", "Goal-Oriented DFR"),
    ]

    fig = Figure(figsize=(5.4, 6.2))
    top = fig.add_subplot(2, 1, 1)
    bot = fig.add_subplot(2, 1, 2, sharex=top)
    axlist = [top, bot]
    for i, (prob, panel_label) in enumerate(panels):
        ax = axlist[i]
        for key, ls, marker, label in styles:
            dofs, med, lo, hi = _median_band(records, prob, key)
            ax.fill_between(dofs, lo, hi, color="0.6", alpha=0.25, linewidth=0)
            ax.plot(
                dofs, med, color="black", linestyle=ls, marker=marker, linewidth=1.6, markersize=6, markerfacecolor="white", label=label
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$|J_\sigma(u_h) - J_\sigma(u^*)|$")
        # Panel identity as an in-axes annotation (not a title).
        ax.text(0.97, 0.93, panel_label, transform=ax.transAxes, ha="right", va="top")
    bot.set_xlabel(r"$N_{\mathrm{dof}}$")
    top.tick_params(labelbottom=False)  # x-axis shared with the bottom panel
    top.legend(loc="lower left")
    return fig


def make_all_figures() -> None:
    """Build and write every preprint figure."""
    figure_dfr_saturation_1d()
