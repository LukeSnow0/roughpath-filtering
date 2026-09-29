"""Manuscript-wide plotting style for the experiments.

Follows the user's scientific plot style: white background, thin black open
axes with outward ticks, no gridlines, explicitly assigned moderately saturated
colors (the main method warm red), legends outside the data, and typography
that stays legible at print size instead of shrinking to fit.

Figures are authored at their final print size (Elsevier's nominal 90 mm
single-column and 190 mm double-column widths), so typography and line weights
do not depend on downstream LaTeX scaling.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

MM_PER_INCH = 25.4
SINGLE_COLUMN_WIDTH = 90.0 / MM_PER_INCH
DOUBLE_COLUMN_WIDTH = 190.0 / MM_PER_INCH
# Measured from the two-column A4 manuscript: \columnwidth and \textwidth. Author
# figures at these widths and include them with width=\columnwidth (or
# \textwidth) so fonts print at their nominal size.
PAPER_COLUMN_WIDTH = 84.4 / MM_PER_INCH
PAPER_TEXT_WIDTH = 177.1 / MM_PER_INCH

PALETTE = {
    "warm_red": "#E64B35",
    "orange": "#E69F00",
    "teal_green": "#009E73",
    "dark_teal": "#00897B",
    "blue": "#0072B2",
    "cyan_blue": "#56B4E9",
    "muted_blue": "#80B1D3",
    "purple": "#8C79B8",
    "lavender": "#BEBADA",
    "salmon": "#F08070",
    "olive": "#8DA35B",
    "light_green": "#7CC943",
    "coral": "#F04E3E",
    "neutral_gray": "#5F5F5F",
    "light_gray": "#BDBDBD",
    "black": "#222222",
}

# Keep method semantics stable across comparison, sweep, MLE, and Pareto plots.
# WSC-KF (the paper's method) is always warm red; its lower-degree or
# alternative variants use salmon.
METHOD_COLORS = {
    "WSC-KF": PALETTE["warm_red"],
    "WSC-KF-AD": PALETTE["warm_red"],
    "WSC-KF-FD": PALETTE["salmon"],
    "WSC-KF-product": PALETTE["warm_red"],
    "WSC-KF-LV": PALETTE["salmon"],
    "WSC-KF-recombined": PALETTE["salmon"],
    "WSC-KF-m5": PALETTE["warm_red"],
    "WSC-KF-m3": PALETTE["salmon"],
    "CD-EKF": PALETTE["blue"],
    "EKF": PALETTE["blue"],
    "CD-UKF": PALETTE["teal_green"],
    "UKF": PALETTE["teal_green"],
    "CD-UKF-fine": PALETTE["teal_green"],
    "CD-UKF-coarse": PALETTE["olive"],
    "SE-UKF": PALETTE["dark_teal"],
    "CD-CKF": PALETTE["cyan_blue"],
    "CKF": PALETTE["cyan_blue"],
    "TME": PALETTE["olive"],
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
    "WSC-KF-m5": "s",
    "WSC-KF-m3": "D",
    "CD-EKF": "s",
    "EKF": "s",
    "CD-UKF": "D",
    "UKF": "D",
    "CD-UKF-fine": "D",
    "CD-UKF-coarse": "d",
    "SE-UKF": "h",
    "CD-CKF": "X",
    "CKF": "X",
    "TME": "P",
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
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.titlepad": 4.0,
        "axes.labelsize": 9.0,
        "axes.labelpad": 3.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.0,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "patch.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.color": PALETTE["black"],
        "ytick.color": PALETTE["black"],
        "axes.edgecolor": PALETTE["black"],
        "axes.labelcolor": PALETTE["black"],
        "text.color": PALETTE["black"],
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
    }
)


def style_curve_axis(ax: Axes) -> None:
    """Style a one-dimensional curve axis: open top/right spines, no gridlines."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(PALETTE["black"])
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(direction="out")
    ax.set_axisbelow(True)


def style_box_axis(ax: Axes) -> None:
    """Style a bar/image axis with a thin black frame, no gridlines."""

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(PALETTE["black"])
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(direction="out")
    ax.set_axisbelow(True)


def add_subfigure_label(
    fig: Figure, ax: Axes, label: str, *, y_offset: float = 0.055
) -> None:
    """Place an ``(a)``-style label centered below ``ax`` (in figure coordinates)."""

    bbox = ax.get_position()
    fig.text(
        (bbox.x0 + bbox.x1) / 2,
        bbox.y0 - y_offset,
        label,
        ha="center",
        va="top",
        fontsize=10,
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
