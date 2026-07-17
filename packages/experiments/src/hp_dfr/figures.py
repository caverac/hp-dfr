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


def _median_band(records: list[Record], problem: str, key: str, primal_weight: float | None = None) -> MedianBand:
    """Return (dofs, median, lo, hi) over seeds for one method/problem.

    ``primal_weight`` selects one goal-oriented loss weight. The 1D sweep records
    several, so omitting it would pool distinct configurations into one band.
    """
    by_dof: dict[int, list[float]] = defaultdict(list)
    for r in records:
        if r["problem"] != problem:
            continue
        if primal_weight is not None and r.get("primal_weight") != primal_weight:
            continue
        by_dof[cast(int, r["dof"])].append(cast(float, r[key]))
    if not by_dof:
        raise ValueError(f"no records for problem={problem!r} primal_weight={primal_weight!r}")
    dofs = sorted(by_dof)
    med = np.array([np.median(by_dof[d]) for d in dofs])
    lo = np.array([np.min(by_dof[d]) for d in dofs])
    hi = np.array([np.max(by_dof[d]) for d in dofs])
    return np.array(dofs), med, lo, hi


@figure("dfr-saturation-1d")
def figure_dfr_saturation_1d() -> Figure:
    r"""Plot the QoI error versus degrees of freedom for both methods.

    On both a smooth (sine) and a peaked (arctan) 1D Poisson problem, plain DFR
    reaches its accuracy floor (~3e-5) already at the smallest network sizes and
    does not improve on it thereafter. The goal-oriented variant is less accurate
    at small and moderate sizes, at either loss weight, so the weight the 2D sweep
    uses does not by itself manufacture an advantage.

    At the largest networks the heavier weight overtakes the baseline on both
    problems, passing a floor the baseline cannot. The steepness sweep shows this
    is a reproducible effect, not seed noise.

    Markers are medians over seeds; bands span the seed min--max.
    """
    records = _load("m2_1d_data.json")
    panels = [("sine", "smooth (sine)"), ("arctan8", r"peaked (arctan, $k=8$)")]
    styles = [
        ("dfr_qoi", 1.0, "-", "o", "DFR"),
        ("go_qoi", 1.0, "--", "s", r"GO-DFR ($\alpha=\beta=1$)"),
        ("go_qoi", 5.0, "-.", "^", r"GO-DFR ($\alpha=\beta=5$)"),
    ]

    fig = Figure(figsize=(5.4, 6.2))
    top = fig.add_subplot(2, 1, 1)
    bot = fig.add_subplot(2, 1, 2, sharex=top)
    axlist = [top, bot]
    for i, (prob, panel_label) in enumerate(panels):
        ax = axlist[i]
        for key, weight, ls, marker, label in styles:
            dofs, med, lo, hi = _median_band(records, prob, key, primal_weight=weight)
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


@figure("dfr-vs-go-2d")
def figure_dfr_vs_go_2d() -> Figure:
    r"""Plot the QoI error versus degrees of freedom for both methods in 2D.

    On the 2D Poisson problem with an arctan bump, goal-orientation is the more
    accurate method at all but one of the swept sizes, winning ten of the twelve
    runs and improving the median QoI error by 1.2x to 5.3x (2.7x on a geometric
    mean) at matched primal-network degrees of freedom. Markers are medians over
    seeds; bands span the seed min--max.
    """
    records = _load("m3_2d_data.json")
    styles = [
        ("dfr_qoi", "-", "o", "DFR"),
        ("go_qoi", "--", "s", "Goal-Oriented DFR"),
    ]

    fig = Figure(figsize=(5.4, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    for key, ls, marker, label in styles:
        dofs, med, lo, hi = _median_band(records, "arctanbump2d", key)
        ax.fill_between(dofs, lo, hi, color="0.6", alpha=0.25, linewidth=0)
        ax.plot(dofs, med, color="black", linestyle=ls, marker=marker, linewidth=1.6, markersize=6, markerfacecolor="white", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\mathrm{dof}}$")
    ax.set_ylabel(r"$|J_\sigma(u_h) - J_\sigma(u^*)|$")
    ax.legend(loc="best")
    return fig


@figure("bound-verification")
def figure_bound_verification() -> Figure:
    r"""Plot the QoI error bound and the goal term against the true QoI error.

    Each marker is one goal-oriented run. The dashed diagonal is equality, so a
    valid bound lies above it. The bound holds on every run at roughly ten times
    the true error, while the goal term alone lies about five orders of magnitude
    below it: training minimizes the goal term directly, so it cannot also report
    the error that remains.
    """
    panels = [
        ("m2_1d_data.json", "1D"),
        ("m3_2d_data.json", "2D"),
    ]
    series = [
        ("go_bound", "o", r"$\mathcal{L}_{\mathrm{QoI}} + \|R(u_h)\|_{V^*}\|R^*(z_h)\|_{U^*}$"),
        ("go_est", "s", r"$\mathcal{L}_{\mathrm{QoI}}$"),
    ]

    fig = Figure(figsize=(5.4, 6.2))
    top = fig.add_subplot(2, 1, 1)
    bot = fig.add_subplot(2, 1, 2, sharex=top)
    for ax, (name, panel_label) in zip([top, bot], panels):
        records = _load(name)
        true_err = np.array([cast(float, r["go_qoi"]) for r in records])
        for key, marker, label in series:
            ax.plot(
                true_err,
                np.array([cast(float, r[key]) for r in records]),
                linestyle="none",
                marker=marker,
                color="black",
                markersize=5,
                markerfacecolor="white",
                label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\mathcal{L}_{\mathrm{QoI}}$, bound")
        ax.text(0.03, 0.93, panel_label, transform=ax.transAxes, ha="left", va="top")
    bot.set_xlabel(r"$|J_\sigma(u_h) - J_\sigma(u^*)|$")
    top.tick_params(labelbottom=False)  # x-axis shared with the bottom panel
    top.legend(loc="center right")

    # Equality, drawn last across the settled limits: a valid bound lies above it.
    for ax in (top, bot):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.5", linewidth=1.0, zorder=0)
        ax.set_xlim(lo, hi)
    return fig


def _steepness_band(records: list[Record], dof: int, key: str) -> MedianBand:
    """Return (steepnesses, median, lo, hi) over seeds for one method/network size."""
    by_k: dict[float, list[float]] = defaultdict(list)
    for r in records:
        if r["dof"] == dof:
            by_k[cast(float, r["steepness"])].append(cast(float, r[key]))
    ks = sorted(by_k)
    med = np.array([np.median(by_k[k]) for k in ks])
    lo = np.array([np.min(by_k[k]) for k in ks])
    hi = np.array([np.max(by_k[k]) for k in ks])
    return np.array(ks), med, lo, hi


@figure("dfr-vs-go-steepness")
def figure_dfr_vs_go_steepness() -> Figure:
    r"""Plot the QoI error against the solution's steepness at fixed network size.

    Steepness is the knob that makes the problem resolution-limited, at a fixed
    network and a fixed discretization, so it isolates the mechanism the method is
    supposed to exploit. Plain DFR sits at its optimization floor while the problem
    is easy and then degrades sharply; the goal-oriented method degrades far more
    slowly, so the gap opens as the problem hardens.
    """
    records = _load("m4_2d_steepness.json")
    styles = [
        ("dfr_qoi", "-", "o", "DFR"),
        ("go_qoi", "--", "s", "Goal-Oriented DFR"),
    ]
    dofs = sorted({cast(int, r["dof"]) for r in records})

    fig = Figure(figsize=(5.4, 6.2))
    top = fig.add_subplot(2, 1, 1)
    bot = fig.add_subplot(2, 1, 2, sharex=top)
    for ax, dof in zip([top, bot], dofs):
        for key, ls, marker, label in styles:
            ks, med, lo, hi = _steepness_band(records, dof, key)
            ax.fill_between(ks, lo, hi, color="0.6", alpha=0.25, linewidth=0)
            ax.plot(ks, med, color="black", linestyle=ls, marker=marker, linewidth=1.6, markersize=6, markerfacecolor="white", label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$|J_\sigma(u_h) - J_\sigma(u^*)|$")
        ax.text(0.03, 0.93, rf"$N_{{\mathrm{{dof}}}} = {dof}$", transform=ax.transAxes, ha="left", va="top")
    bot.set_xlabel(r"$k$")
    top.tick_params(labelbottom=False)  # x-axis shared with the bottom panel
    top.legend(loc="upper left", bbox_to_anchor=(0.03, 0.86))
    return fig


def make_all_figures() -> None:
    """Build and write every preprint figure."""
    figure_dfr_saturation_1d()
    figure_dfr_vs_go_2d()
    figure_bound_verification()
    figure_dfr_vs_go_steepness()
