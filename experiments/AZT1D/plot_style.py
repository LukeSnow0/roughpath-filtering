"""AZT1D aliases for the manuscript-wide experiment plotting style."""

from experiments.mfts_test.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    MM_PER_INCH,
    PALETTE,
    SINGLE_COLUMN_WIDTH,
    add_subfigure_label,
    save_publication_figure,
    style_box_axis,
    style_curve_axis,
)
from experiments.mfts_test.plot_style import METHOD_COLORS as _METHOD_COLORS
from experiments.mfts_test.plot_style import METHOD_MARKERS as _METHOD_MARKERS

METHOD_COLORS = {**_METHOD_COLORS, "CGM": PALETTE["neutral_gray"]}
METHOD_MARKERS = {**_METHOD_MARKERS, "CGM": "."}

__all__ = [
    "DOUBLE_COLUMN_WIDTH",
    "METHOD_COLORS",
    "METHOD_MARKERS",
    "MM_PER_INCH",
    "PALETTE",
    "SINGLE_COLUMN_WIDTH",
    "add_subfigure_label",
    "save_publication_figure",
    "style_box_axis",
    "style_curve_axis",
]
