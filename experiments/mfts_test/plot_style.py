"""Publication plotting helpers shared by the MFTS experiments.

The dimensions follow Elsevier's nominal 90 mm single-column and 190 mm
double-column artwork widths.  Figures are authored at their final print size
so typography and line weights do not depend on downstream LaTeX scaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

MM_PER_INCH = 25.4
SINGLE_COLUMN_WIDTH = 90.0 / MM_PER_INCH
DOUBLE_COLUMN_WIDTH = 190.0 / MM_PER_INCH

PALETTE = {
    "warm_red": "#E64B35",
    "orange": "#E69F00",
    "teal_green": "#009E73",
    "dark_teal": "#00897B",
    "blue": "#0072B2",
    "cyan_blue": "#56B4E9",
    "muted_blue": "#80B1D3",
    "purple": "#8C79B8",
    "salmon": "#F08070",
    "olive": "#8DA35B",
    "neutral_gray": "#5F5F5F",
    "light_gray": "#BDBDBD",
    "black": "#222222",
}

# Keep method semantics stable across comparison, sweep, MLE, and Pareto plots.
METHOD_COLORS = {
    "WSC-KF": PALETTE["warm_red"],
    "WSC-KF-AD": PALETTE["warm_red"],
    "WSC-KF-FD": PALETTE["salmon"],
    "WSC-KF-product": PALETTE["warm_red"],
    "WSC-KF-LV": PALETTE["salmon"],
    "WSC-KF-recombined": PALETTE["salmon"],
    "CD-EKF": PALETTE["blue"],
    "EKF": PALETTE["blue"],
    "CD-UKF": PALETTE["teal_green"],
    "UKF": PALETTE["teal_green"],
    "CD-UKF-fine": PALETTE["teal_green"],
    "CD-UKF-coarse": PALETTE["olive"],
    "CD-CKF": PALETTE["cyan_blue"],
    "CKF": PALETTE["cyan_blue"],
    "CD-EnKF": PALETTE["orange"],
    "EnKF": PALETTE["orange"],
    "PF": PALETTE["purple"],
}

METHOD_MARKERS = {
    "WSC-KF": "o",
    "WSC-KF-AD": "o",
    "WSC-KF-FD": "o",
    "WSC-KF-product": "*",
    "WSC-KF-LV": "P",
    "WSC-KF-recombined": "P",
    "CD-EKF": "s",
    "EKF": "s",
    "CD-UKF": "D",
    "UKF": "D",
    "CD-UKF-fine": "D",
    "CD-UKF-coarse": "d",
    "CD-CKF": "X",
    "CKF": "X",
    "CD-EnKF": "^",
    "EnKF": "^",
    "PF": "v",
}

mpl.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 450,
        "savefig.bbox": None,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 3.0,
        "axes.labelsize": 8.0,
        "axes.labelpad": 2.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "legend.title_fontsize": 7.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.45,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.minor.size": 1.7,
        "ytick.minor.size": 1.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "xtick.color": PALETTE["black"],
        "ytick.color": PALETTE["black"],
        "axes.edgecolor": PALETTE["black"],
        "axes.labelcolor": PALETTE["black"],
        "text.color": PALETTE["black"],
        "legend.frameon": False,
        "legend.handlelength": 1.7,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 0.9,
    }
)


def style_curve_axis(
    ax: Axes,
    *,
    grid_axis: Literal["x", "y", "both"] | None = None,
) -> None:
    """Style a one-dimensional curve axis with open top/right spines."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(PALETTE["black"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(direction="out")
    ax.set_axisbelow(True)
    if grid_axis is not None:
        ax.grid(
            True,
            axis=grid_axis,
            which="major",
            color=PALETTE["light_gray"],
            linewidth=0.45,
            alpha=0.45,
        )


def style_box_axis(
    ax: Axes,
    *,
    grid_axis: Literal["x", "y", "both"] | None = None,
) -> None:
    """Style a bar/image axis with a thin black frame."""

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(PALETTE["black"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(direction="out")
    ax.set_axisbelow(True)
    if grid_axis is not None:
        ax.grid(
            True,
            axis=grid_axis,
            which="major",
            color=PALETTE["light_gray"],
            linewidth=0.45,
            alpha=0.45,
        )


def save_publication_figure(
    fig: Figure,
    output_path: str | Path,
    *,
    dpi: int = 450,
) -> tuple[Path, Path]:
    """Write a vector manuscript figure and a high-resolution PNG preview.

    ``output_path`` may have any suffix; both ``.pdf`` and ``.png`` are always
    produced beside it.  The returned tuple is ``(png_path, pdf_path)``.
    """

    stem = Path(output_path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    common = {
        "facecolor": "white",
        "edgecolor": "none",
    }
    fig.savefig(pdf_path, **common)
    fig.savefig(png_path, dpi=dpi, **common)
    return png_path, pdf_path
