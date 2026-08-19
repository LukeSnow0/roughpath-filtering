"""Algorithm-in-one-picture: Wiener Space Cubature Kalman Filter pipeline.

Four-step horizontal diagram for talks:
  1. Gaussian posterior
  2. Spatial cubature nodes
  3. Wiener-cubature ODE predict
  4. Measurement update

Outputs under ``output/``:
  algorithm_one_picture.png
  algorithm_one_picture.pdf
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")

# Ink / paper / accent — presentation-friendly, not default purple-on-white
INK = "#1a2332"
MUTED = "#5a6577"
PAPER = "#ffffff"
PANEL = "#ffffff"
ACCENT = "#c45c26"  # observation / update
PATH = "#2a6f7f"  # Wiener paths / predict
NODE = "#3d5a80"  # spatial nodes
GAUSS = "#4a5568"


def _panel(ax, x0, y0, w, h):
    box = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=1.2,
        edgecolor="#d4cfc6",
        facecolor=PANEL,
        transform=ax.transAxes,
        clip_on=False,
        zorder=1,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, x0, x1, y=0.52):
    arr = FancyArrowPatch(
        (x0, y),
        (x1, y),
        transform=ax.transAxes,
        arrowstyle="-|>,head_length=4,head_width=3",
        mutation_scale=12,
        linewidth=1.6,
        color=INK,
        clip_on=False,
        zorder=5,
    )
    ax.add_patch(arr)


def _step_label(ax, x, y, n, title):
    ax.text(
        x,
        y,
        f"{n}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=ACCENT,
        fontfamily="sans-serif",
        zorder=3,
    )
    ax.text(
        x + 0.018,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="semibold",
        color=INK,
        fontfamily="sans-serif",
        zorder=3,
    )


def _draw_gaussian(ax, cx, cy, sx=0.055, sy=0.09):
    """Bell curve + mean marker in axes fraction coords."""
    t = np.linspace(-2.6, 2.6, 120)
    pdf = np.exp(-0.5 * t**2)
    xs = cx + sx * t / 2.6
    ys = cy - 0.02 + sy * pdf
    ax.fill_between(
        xs,
        cy - 0.02,
        ys,
        transform=ax.transAxes,
        color=GAUSS,
        alpha=0.18,
        zorder=2,
        linewidth=0,
    )
    ax.plot(xs, ys, transform=ax.transAxes, color=GAUSS, lw=2.0, zorder=3)
    ax.plot(
        [cx, cx],
        [cy - 0.02, cy - 0.02 + sy],
        transform=ax.transAxes,
        color=ACCENT,
        lw=1.4,
        ls="--",
        zorder=3,
    )
    ax.plot(cx, cy - 0.02, "o", transform=ax.transAxes, color=ACCENT, ms=5, zorder=4)
    ax.text(
        cx,
        cy - 0.055,
        r"$\mathcal{N}(\hat{x}_k,\,P_k)$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
        zorder=3,
    )


def _draw_spatial_nodes(ax, cx, cy):
    """2D ellipse with ±σ cubature nodes."""
    th = np.linspace(0, 2 * np.pi, 200)
    a, b = 0.055, 0.048
    ex = cx + a * np.cos(th)
    ey = cy + 0.01 + b * np.sin(th)
    ax.fill(ex, ey, transform=ax.transAxes, color=NODE, alpha=0.10, zorder=2, lw=0)
    ax.plot(ex, ey, transform=ax.transAxes, color=NODE, lw=1.4, alpha=0.55, zorder=2)

    # Four axis-aligned cubature nodes (2n for n=2)
    pts = [
        (cx + a * 0.92, cy + 0.01),
        (cx - a * 0.92, cy + 0.01),
        (cx, cy + 0.01 + b * 0.92),
        (cx, cy + 0.01 - b * 0.92),
    ]
    for x, y in pts:
        ax.plot(x, y, "o", transform=ax.transAxes, color=NODE, ms=7, zorder=4)
        ax.plot(x, y, "o", transform=ax.transAxes, color="white", ms=2.5, zorder=5)
    ax.plot(cx, cy + 0.01, "+", transform=ax.transAxes, color=ACCENT, ms=8, mew=1.5, zorder=4)
    ax.text(
        cx,
        cy - 0.055,
        r"$\{\zeta_i\}_{i=1}^{2n}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
        zorder=3,
    )


def _draw_wiener_predict(ax, cx, cy):
    """Fan of controlled ODE trajectories from a node."""
    t = np.linspace(0, 1, 80)
    base_x0, base_y0 = cx - 0.048, cy - 0.01
    colors = ["#2a6f7f", "#3d8a9a", "#1f5a66", "#5aa0ad"]
    for i, col in enumerate(colors):
        # piecewise-linear-ish cubature-looking paths in (time, state) view
        kink = 0.35 + 0.12 * i
        wiggle = 0.018 * np.sin(2 * np.pi * (t + 0.15 * i)) + 0.012 * (i - 1.5) * t
        slope = np.where(t < kink, 0.9 + 0.15 * i, -0.35 + 0.1 * i)
        integ = np.cumsum(slope) * (t[1] - t[0])
        integ = integ - integ[0]
        xs = base_x0 + 0.11 * t
        ys = base_y0 + 0.07 * integ / (np.max(np.abs(integ)) + 1e-9) + wiggle
        ax.plot(xs, ys, transform=ax.transAxes, color=col, lw=1.5, alpha=0.9, zorder=3)
        ax.plot(xs[-1], ys[-1], "o", transform=ax.transAxes, color=col, ms=4.5, zorder=4)

    ax.plot(base_x0, base_y0, "o", transform=ax.transAxes, color=NODE, ms=7, zorder=5)
    ax.text(
        cx,
        cy - 0.055,
        r"ODE along $\{w_j,\lambda_j\}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=MUTED,
        zorder=3,
    )


def _draw_update(ax, cx, cy):
    """Prior cloud → posterior after y_k."""
    # prior (lighter, wider)
    th = np.linspace(0, 2 * np.pi, 200)
    a1, b1 = 0.048, 0.038
    ax.fill(
        cx - 0.012 + a1 * np.cos(th),
        cy + 0.015 + b1 * np.sin(th),
        transform=ax.transAxes,
        color=GAUSS,
        alpha=0.12,
        zorder=2,
        lw=0,
    )
    ax.plot(
        cx - 0.012 + a1 * np.cos(th),
        cy + 0.015 + b1 * np.sin(th),
        transform=ax.transAxes,
        color=GAUSS,
        lw=1.2,
        alpha=0.5,
        ls="--",
        zorder=2,
    )
    # posterior (tighter)
    a2, b2 = 0.028, 0.024
    ax.fill(
        cx + 0.02 + a2 * np.cos(th),
        cy + 0.015 + b2 * np.sin(th),
        transform=ax.transAxes,
        color=ACCENT,
        alpha=0.22,
        zorder=3,
        lw=0,
    )
    ax.plot(
        cx + 0.02 + a2 * np.cos(th),
        cy + 0.015 + b2 * np.sin(th),
        transform=ax.transAxes,
        color=ACCENT,
        lw=1.8,
        zorder=3,
    )
    ax.plot(cx + 0.02, cy + 0.015, "o", transform=ax.transAxes, color=ACCENT, ms=6, zorder=4)
    # observation tick
    ax.plot(
        [cx + 0.055, cx + 0.055],
        [cy - 0.02, cy + 0.055],
        transform=ax.transAxes,
        color=INK,
        lw=1.6,
        zorder=4,
    )
    ax.plot(
        cx + 0.055,
        cy + 0.055,
        "v",
        transform=ax.transAxes,
        color=INK,
        ms=7,
        zorder=5,
    )
    ax.text(
        cx + 0.055,
        cy + 0.072,
        r"$y_{k}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK,
        fontweight="semibold",
        zorder=5,
    )
    ax.text(
        cx,
        cy - 0.055,
        r"Kalman update $\to \mathcal{N}(\hat{x}_{k+1},P_{k+1})$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
        zorder=3,
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 3.6), dpi=200)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.94,
        "Wiener Space Cubature Kalman Filter — one cycle",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=INK,
        fontfamily="sans-serif",
    )
    ax.text(
        0.5,
        0.875,
        "Same outer loop as a sigma-point Kalman filter — the predict step transports uncertainty through the SDE with Wiener cubature",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color=MUTED,
        fontfamily="sans-serif",
    )

    # Panel geometry (axes fraction)
    panels = [
        (0.02, 0.10, 0.20, 0.68),
        (0.27, 0.10, 0.20, 0.68),
        (0.52, 0.10, 0.20, 0.68),
        (0.77, 0.10, 0.20, 0.68),
    ]
    titles = [
        "Represent",
        "Spatial cubature",
        "Wiener predict",
        "Update",
    ]
    drawers = [_draw_gaussian, _draw_spatial_nodes, _draw_wiener_predict, _draw_update]

    for i, ((x0, y0, w, h), title, draw) in enumerate(
        zip(panels, titles, drawers), start=1
    ):
        _panel(ax, x0, y0, w, h)
        _step_label(ax, x0 + 0.012, y0 + h - 0.055, i, title)
        cx = x0 + w / 2
        cy = y0 + 0.34
        draw(ax, cx, cy)

    # Arrows between panels
    for i in range(3):
        x0 = panels[i][0] + panels[i][2] + 0.008
        x1 = panels[i + 1][0] - 0.008
        _arrow(ax, x0, x1, y=0.42)

    # Footnote
    ax.text(
        0.5,
        0.035,
        r"No Jacobians  ·  no Brownian sampling  ·  deterministic ODEs along cubature paths $\{w_j\}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=MUTED,
        style="italic",
    )

    fig.tight_layout(pad=0.3)
    png = os.path.join(OUTPUT_DIR, "algorithm_one_picture.png")
    pdf = os.path.join(OUTPUT_DIR, "algorithm_one_picture.pdf")
    fig.savefig(png, dpi=450, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(pdf, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
