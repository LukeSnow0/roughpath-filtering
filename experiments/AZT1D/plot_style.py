"""AZT1D aliases for the manuscript-wide experiment plotting style."""

from experiments.mfts_test.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    MM_PER_INCH,
    PALETTE,
    SINGLE_COLUMN_WIDTH,
    save_publication_figure,
    style_box_axis,
    style_curve_axis,
)

METHOD_COLORS = {
    "WSC-KF": PALETTE["warm_red"],
    "WSC-KF-product": PALETTE["warm_red"],
    "EKF": PALETTE["blue"],
    "CD-EKF": PALETTE["blue"],
    "UKF": PALETTE["teal_green"],
    "CD-UKF": PALETTE["teal_green"],
    "CKF": PALETTE["cyan_blue"],
    "CD-CKF": PALETTE["cyan_blue"],
    "EnKF": PALETTE["orange"],
    "CD-EnKF": PALETTE["orange"],
    "PF": PALETTE["purple"],
    "CGM": PALETTE["neutral_gray"],
}

METHOD_MARKERS = {
    "WSC-KF": "o",
    "WSC-KF-product": "*",
    "EKF": "s",
    "CD-EKF": "s",
    "UKF": "D",
    "CD-UKF": "D",
    "CKF": "X",
    "CD-CKF": "X",
    "EnKF": "^",
    "CD-EnKF": "^",
    "PF": "v",
    "CGM": ".",
}

__all__ = [
    "DOUBLE_COLUMN_WIDTH",
    "METHOD_COLORS",
    "METHOD_MARKERS",
    "MM_PER_INCH",
    "PALETTE",
    "SINGLE_COLUMN_WIDTH",
    "save_publication_figure",
    "style_box_axis",
    "style_curve_axis",
]
