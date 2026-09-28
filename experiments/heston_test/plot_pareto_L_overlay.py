"""Overlay Heston Pareto for L-scale ∈ {2,3,4}: mean RMSE with ±kσ bands.

Reads raw CSVs from ``noise_hi_L2``, ``noise_hi`` (L=3), ``noise_hi_L4``.
WSC-KF is split by Lyons degree: m=3 (P=4,16,64) vs m=5 (P=9,81).

Run:
  uv run --extra experiments python experiments/heston_test/plot_pareto_L_overlay.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from collections.abc import Callable

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
OUTPUT_DIR = os.path.join(HERE, "output")
from mfts_test.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    PALETTE,
    save_publication_figure,
    style_curve_axis,
)

# Shade is mean ± BAND_SCALE * σ (mean sits in the middle).
BAND_SCALE = 1.0
SHADE_ALPHA = 0.14
SINGLE_X_HALF = 1.04

# (display name, color, marker, annotate?, row predicate)
PlotSeries = tuple[str, str, str, bool, Callable[[dict], bool]]
PLOT_SERIES: list[PlotSeries] = [
    (
        "WSC-KF m=3",
        METHOD_COLORS["WSC-KF"],
        "s",
        True,
        lambda r: r["family"] == "WSC-KF-product" and "-m3-" in r["label"],
    ),
    (
        "WSC-KF m=5",
        PALETTE["salmon"],
        "D",
        True,
        lambda r: r["family"] == "WSC-KF-product" and "-m5-" in r["label"],
    ),
    ("CD-EnKF", METHOD_COLORS["CD-EnKF"], "^", True, lambda r: r["family"] == "CD-EnKF"),
    ("PF", METHOD_COLORS["PF"], "v", True, lambda r: r["family"] == "PF"),
    ("CD-EKF", METHOD_COLORS["CD-EKF"], "s", False, lambda r: r["family"] == "CD-EKF"),
    ("CD-UKF (coarse)", METHOD_COLORS["CD-UKF"], "D", False, lambda r: r["family"] == "CD-UKF"),
    ("CD-CKF", METHOD_COLORS["CD-CKF"], "X", False, lambda r: r["family"] == "CD-CKF"),
]

_LABEL_OFFSET = {
    "CD-EnKF": (0, -10),
    "PF": (0, 7),
    "WSC-KF m=3": (0, 8),
    "WSC-KF m=5": (0, -9),
}

SERIES = [
    (2, "noise_hi_L2"),
    (3, "noise_hi"),
    (4, "noise_hi_L4"),
]


def _load_raw_summary(tag: str) -> tuple[list[dict], float, float]:
    """Aggregate raw seed rows → mean / std RMSE and mean wall time."""
    path = os.path.join(OUTPUT_DIR, f"pareto_budget_vs_rmse_{tag}_raw.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    buckets: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    delta_t = horizon = None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            delta_t = float(r["delta_t"])
            horizon = float(r["horizon"])
            key = (r["family"], r["label"], int(float(r["budget"])))
            buckets[key].append((float(r["rmse"]), float(r["time_s"])))

    rows: list[dict] = []
    for (family, label, budget), pairs in buckets.items():
        rmses = np.asarray([p[0] for p in pairs], dtype=np.float64)
        times = np.asarray([p[1] for p in pairs], dtype=np.float64)
        mean = float(np.mean(rmses))
        std = float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0
        half = BAND_SCALE * std
        rows.append(
            {
                "family": family,
                "label": label,
                "budget": budget,
                "n_seeds": len(pairs),
                "rmse_mean": mean,
                "rmse_ylo": mean - half,
                "rmse_yhi": mean + half,
                "time_s_mean": float(np.mean(times)),
                "delta_t": float(delta_t),
                "horizon": float(horizon),
            }
        )
    return rows, float(delta_t), float(horizon)


def _annotate(ax, disp: str, color: str, pts, xs, ys):
    base_dx, base_dy = _LABEL_OFFSET.get(disp, (0, 6))
    for i, (pt, x, y) in enumerate(zip(pts, xs, ys)):
        dy = base_dy if (i % 2 == 0) else (base_dy + (4 if base_dy > 0 else -4))
        ax.annotate(
            str(int(pt["budget"])),
            (x, y),
            textcoords="offset points",
            xytext=(base_dx, dy),
            fontsize=6,
            color=color,
            ha="center",
            va="bottom" if dy >= 0 else "top",
            alpha=0.9,
        )


def _draw_panel(ax, summary: list[dict], *, show_legend: bool):
    all_xs: list[float] = []
    all_ys: list[float] = []
    for disp, color, marker, do_ann, pred in PLOT_SERIES:
        pts = [r for r in summary if pred(r)]
        if not pts:
            continue
        pts = sorted(pts, key=lambda r: r["budget"])
        xs = np.asarray([1e3 * r["time_s_mean"] for r in pts])
        ys = np.asarray([r["rmse_mean"] for r in pts])
        ylo = np.asarray([r["rmse_ylo"] for r in pts])
        yhi = np.asarray([r["rmse_yhi"] for r in pts])
        all_xs.extend(xs.tolist())
        all_ys.extend(ylo.tolist())
        all_ys.extend(yhi.tolist())
        if len(pts) == 1:
            x0 = float(xs[0])
            ax.fill_between(
                [x0 / SINGLE_X_HALF, x0 * SINGLE_X_HALF],
                [ylo[0]] * 2,
                [yhi[0]] * 2,
                color=color,
                alpha=SHADE_ALPHA,
                lw=0,
                zorder=1,
            )
            ax.plot(xs, ys, marker, color=color, markersize=6, label=disp, zorder=3)
        else:
            ax.fill_between(xs, ylo, yhi, color=color, alpha=SHADE_ALPHA, lw=0, zorder=1)
            ax.plot(
                xs, ys, f"-{marker}", color=color, lw=1.4, markersize=5, label=disp, zorder=3
            )
        if do_ann:
            _annotate(ax, disp, color, pts, xs, ys)
    if show_legend:
        ax.legend(fontsize=7, loc="best")
    return all_xs, all_ys


def main():
    panels = []
    for L, tag in SERIES:
        summary, delta_t, horizon = _load_raw_summary(tag)
        panels.append((L, tag, summary, delta_t, horizon))

    fig, axes = plt.subplots(
        1, 3, figsize=(DOUBLE_COLUMN_WIDTH, 2.55), sharex=True, sharey=True
    )
    xlim_xs: list[float] = []
    ylim_ys: list[float] = []
    for ax, (L, tag, summary, _, _) in zip(axes, panels):
        xs, ys = _draw_panel(ax, summary, show_legend=False)
        xlim_xs.extend(xs)
        ylim_ys.extend(ys)
        n_seeds = summary[0]["n_seeds"] if summary else 0
        ax.set_title(rf"$L$-scale = {L} ({n_seeds} seeds)")
        ax.set_xscale("log")
        style_curve_axis(ax)
        ax.set_xlabel("Wall-clock time [ms]")
    axes[0].set_ylabel("Fine-grid RMSE")

    if xlim_xs:
        for ax in axes:
            ax.set_xlim(min(xlim_xs) / 1.25, max(xlim_xs) * 1.35)
    if ylim_ys:
        y0, y1 = min(ylim_ys), max(ylim_ys)
        pad = 0.05 * (y1 - y0 + 1e-12)
        for ax in axes:
            ax.set_ylim(max(0.0, y0 - pad), y1 + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.30, top=0.90, wspace=0.16)
    out = os.path.join(OUTPUT_DIR, "pareto_budget_vs_rmse_noise_hi_L234.png")
    out_png, out_pdf = save_publication_figure(fig, out)
    plt.close()
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
