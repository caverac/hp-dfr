"""Shared matplotlib style and the :func:`figure` decorator.

A figure-building function returns a :class:`~matplotlib.figure.Figure`; the
:func:`figure` decorator applies the shared house style, then writes the figure to
the preprint ``figures`` directory as **both** a ``.png`` (documentation) and a
``.pdf`` (paper). The render is compared pixel-for-pixel against the existing
``.png`` and rewritten only when the content changed, so regenerating an unchanged
figure produces no spurious diff. Mirrors the separatrix-sentinel convention.
"""

from __future__ import annotations

import functools
import io
import os
from pathlib import Path
from typing import Callable, ParamSpec

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from hp_dfr.utils import console


def _default_fig_dir() -> Path:
    """Locate the project-root ``assets`` directory in a source checkout.

    In the repository ``_plotting.py`` sits four levels below the root
    (``packages/experiments/src/hp_dfr``), so the assets directory is
    ``parents[4] / "assets"``. When the package is installed at a different depth
    (for example inside a container), set :envvar:`HP_DFR_ASSETS_DIR` instead.
    """
    return Path(__file__).resolve().parents[4] / "assets"


#: Figures (and their source data) live in the ``assets`` directory. Overridable
#: via :envvar:`HP_DFR_ASSETS_DIR` for installs whose layout is not the repo's.
FIG_DIR: Path = Path(os.environ["HP_DFR_ASSETS_DIR"]) if os.environ.get("HP_DFR_ASSETS_DIR") else _default_fig_dir()

# Shared serif/STIX figure style.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

FigureParams = ParamSpec("FigureParams")

#: Resolution (DPI) used when rasterizing both the PNG and the PDF.
SAVEFIG_DPI = 300


def _write_figure(fig: Figure, target: Path | io.BytesIO, fmt: str) -> None:
    """Write *fig* to *target* in *fmt* using the shared savefig settings."""
    fig.savefig(target, format=fmt, dpi=SAVEFIG_DPI, facecolor="white", edgecolor="none")


def configure_axes(axes: Axes) -> None:
    """Apply the shared inward-tick style to *axes* on all four sides."""
    if axes.get_xscale() == "linear":
        axes.xaxis.set_minor_locator(AutoMinorLocator())
    if axes.get_yscale() == "linear":
        axes.yaxis.set_minor_locator(AutoMinorLocator())
    axes.tick_params(which="minor", length=3, color="gray", direction="in")
    axes.tick_params(which="major", length=6, direction="in")
    axes.tick_params(top=True, right=True, which="both")


def _apply_house_style(fig: Figure) -> None:
    """Style every axes of *fig*, make legends frameless, and reflow the layout."""
    for axes in fig.axes:
        configure_axes(axes)
        legend = axes.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
    fig.tight_layout()


def save_figure_set_if_changed(fig: Figure, png_path: Path, pdf_path: Path) -> bool:
    """Write *fig* as PNG and PDF, but only if its rendered content changed."""
    buf = io.BytesIO()
    _write_figure(fig, buf, "png")
    buf.seek(0)
    new_img = plt.imread(buf)

    if png_path.exists() and pdf_path.exists():
        existing = plt.imread(png_path)
        if new_img.shape == existing.shape and bool(np.array_equal(new_img, existing)):
            return False

    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(buf.getvalue())
    _write_figure(fig, pdf_path, "pdf")
    return True


def figure(name: str) -> Callable[[Callable[FigureParams, Figure]], Callable[FigureParams, Figure]]:
    """Save the returned :class:`Figure` to the preprint figures dir if it changed.

    The decorated function must return a :class:`~matplotlib.figure.Figure`. The
    decorator applies the house style, writes ``<name>.png`` and ``<name>.pdf``,
    reports the outcome, and closes the figure.
    """

    def decorator(
        func: Callable[FigureParams, Figure],
    ) -> Callable[FigureParams, Figure]:
        @functools.wraps(func)
        def wrapper(*args: FigureParams.args, **kwargs: FigureParams.kwargs) -> Figure:
            fig = func(*args, **kwargs)
            _apply_house_style(fig)
            png_path = FIG_DIR / f"{name}.png"
            pdf_path = FIG_DIR / f"{name}.pdf"
            if save_figure_set_if_changed(fig, png_path, pdf_path):
                console.print(f"  saved {png_path.name} and {pdf_path.name}")
            else:
                console.print(f"  unchanged {name}", markup=False)
            plt.close(fig)
            return fig

        return wrapper

    return decorator
