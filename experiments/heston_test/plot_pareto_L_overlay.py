"""Overlay Heston Pareto for L-scale ∈ {2,3,4}: median RMSE with [q10, q90] bands.

Reads raw CSVs from ``noise_hi_L2``, ``noise_hi`` (L=3), ``noise_hi_L4``.
WS-CKF is split by Lyons degree: m=3 (P=4,16,64) vs m=5 (P=9,81).

Run:
  uv run --extra experiments python experiments/heston_test/plot_pareto_L_overlay.py
"""

from __future__ import annotations

import csv
import os
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
OUTPUT_DIR = os.path.join(HERE, "output")

Q_LO, Q_HI = 10.0, 90.0

# (display name, color, marker, annotate?, row predicate)
PlotSeries = tuple[str, str, str, bool, Callable[[dict], bool]]
PLOT_SERIES: list[PlotSeries] = [
    (
        "WS-CKF m=3",
        "#6A4C93",
        "s",
        True,
        lambda r: r["family"] == "CRF-product" and "-m3-" in r["label"],
    ),
    (
        "WS-CKF m=5",
        "#C77DFF",
        "D",
        True,
        lambda r: r["family"] == "CRF-product" and "-m5-" in r["label"],
    ),
    ("CD-EnKF", "#F6AE2D", "o", True, lambda r: r["family"] == "CD-EnKF"),
    ("PF", "#F24236", "o", True, lambda r: r["family"] == "PF"),
    ("CD-EKF", "#2E86AB", "s", False, lambda r: r["family"] == "CD-EKF"),
    ("CD-UKF (coarse)", "#3DA35D", "s", False, lambda r: r["family"] == "CD-UKF"),
    ("CD-CKF", "#7DCEA0", "s", False, lambda r: r["family"] == "CD-CKF"),
]

_LABEL_OFFSET = {
    "CD-EnKF": (0, -10),
    "PF": (0, 7),
    "WS-CKF m=3": (0, 8),
    "WS-CKF m=5": (0, -9),
}

SERIES = [
    (2, "noise_hi_L2"),
    (3, "noise_hi"),
    (4, "noise_hi_L4"),
]


def _load_raw_summary(tag: str) -> tuple[list[dict], float, float]:
    """Aggregate raw seed rows → median / q10 / q90 RMSE and median wall time."""
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
        rows.append(
            {
                "family": family,
                "label": label,
                "budget": budget,
                "n_seeds": len(pairs),
                "rmse_med": float(np.median(rmses)),
                "rmse_qlo": float(np.percentile(rmses, Q_LO)),
                "rmse_qhi": float(np.percentile(rmses, Q_HI)),
                "time_s_med": float(np.median(times)),
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
        xs = np.asarray([1e3 * r["time_s_med"] for r in pts])
        ys = np.asarray([r["rmse_med"] for r in pts])
        ylo = np.asarray([r["rmse_qlo"] for r in pts])
        yhi = np.asarray([r["rmse_qhi"] for r in pts])
        all_xs.extend(xs.tolist())
        all_ys.extend(ylo.tolist())
        all_ys.extend(yhi.tolist())
        if len(pts) == 1:
            x0 = float(xs[0])
            ax.fill_between(
                [x0 / 1.04, x0 * 1.04],
                [ylo[0]] * 2,
                [yhi[0]] * 2,
                color=color,
                alpha=0.22,
                lw=0,
            )
            ax.plot(xs, ys, marker, color=color, markersize=6, label=disp)
        else:
            ax.fill_between(xs, ylo, yhi, color=color, alpha=0.22, lw=0)
            ax.plot(xs, ys, f"-{marker}", color=color, lw=1.4, markersize=5, label=disp)
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

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, sharey=True)
    xlim_xs: list[float] = []
    ylim_ys: list[float] = []
    for ax, (L, tag, summary, _, _) in zip(axes, panels):
        xs, ys = _draw_panel(ax, summary, show_legend=(L == 2))
        xlim_xs.extend(xs)
        ylim_ys.extend(ys)
        n_seeds = summary[0]["n_seeds"] if summary else 0
        ax.set_title(rf"$L$-scale={L}  ({n_seeds} seeds)", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("Wall-clock time [ms]")
    axes[0].set_ylabel("Fine-grid RMSE (overall)")

    if xlim_xs:
        for ax in axes:
            ax.set_xlim(min(xlim_xs) / 1.25, max(xlim_xs) * 1.35)
    if ylim_ys:
        y0, y1 = min(ylim_ys), max(ylim_ys)
        pad = 0.05 * (y1 - y0 + 1e-12)
        for ax in axes:
            ax.set_ylim(max(0.0, y0 - pad), y1 + pad)

    delta_t, horizon = panels[0][3], panels[0][4]
    fig.suptitle(
        rf"Heston Pareto at $\Delta t={delta_t:g}$, $\xi=1$, $\rho=-0.7$ "
        rf"(median, shade $[{Q_LO:g}\%,{Q_HI:g}\%]$; horizon={horizon:g}; "
        r"WS-CKF $m{=}3$: $P\in\{4,16,64\}$, $m{=}5$: $P\in\{9,81\}$)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "pareto_budget_vs_rmse_noise_hi_L234.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
