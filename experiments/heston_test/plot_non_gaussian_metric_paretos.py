"""Plot per-metric Pareto fronts from non_gaussian_flow_eval.csv.

Compact family legend + on-plot config labels (k=, n=, x=).
"""

from __future__ import annotations

import csv
import os

import matplotlib
import numpy as np

os.environ["MPLCONFIGDIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".mplcache"
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output", "non_gaussian")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "non_gaussian_flow_eval.csv")

# One color per filter family; config encoded by on-plot labels.
FAMILY_STYLE = {
    "WSC-KF": "#7B2CBF",
    "EKF": "#FF7F50",
    "UKF": "#E9C46A",
    "CKF": "#2E86AB",
    "EnKF": "#9B2226",
    "PF": "#2A9D8F",
    "KLV-PF": "#F77F00",
}

# (method_key, family, short on-plot tag, refinement matcher)
SERIES = [
    ("WSC-KF-m5-k1-cloud", "WSC-KF", "k=1", 1),
    ("WSC-KF-m5-k2-cloud", "WSC-KF", "k=2", 2),
    ("WSC-KF-m5-k3-cloud", "WSC-KF", "k=3", 3),
    ("CD-EKF-gauss", "EKF", "x8", 8),
    ("CD-UKF-gauss", "UKF", "x8", 8),
    ("CD-CKF-gauss", "CKF", "x1", 1),
    ("CD-CKF-gauss", "CKF", "x2", 2),
    ("CD-CKF-gauss", "CKF", "x4", 4),
    ("CD-CKF-gauss", "CKF", "x8", 8),
    ("CD-EnKF-forecast", "EnKF", "n=500", 500),
    ("CD-EnKF-forecast", "EnKF", "n=1000", 1000),
    ("CD-EnKF-forecast", "EnKF", "n=2000", 2000),
    ("CD-EnKF-forecast", "EnKF", "n=5000", 5000),
    ("Bootstrap-PF-forecast", "PF", "n=500", 500),
    ("Bootstrap-PF-forecast", "PF", "n=1000", 1000),
    ("Bootstrap-PF-forecast", "PF", "n=2000", 2000),
    ("Bootstrap-PF-forecast", "PF", "n=5000", 5000),
    ("KLV-PF-forecast", "KLV-PF", "n=500", 500),
    ("KLV-PF-forecast", "KLV-PF", "n=1000", 1000),
    ("KLV-PF-forecast", "KLV-PF", "n=2000", 2000),
    ("KLV-PF-forecast", "KLV-PF", "n=5000", 5000),
]

TRACKS = {
    "WSC-KF": ["k=1", "k=2", "k=3"],
    "CKF": ["x1", "x2", "x4", "x8"],
    "EnKF": ["n=500", "n=1000", "n=2000", "n=5000"],
    "PF": ["n=500", "n=1000", "n=2000", "n=5000"],
    "KLV-PF": ["n=500", "n=1000", "n=2000", "n=5000"],
}


def _metric_pretty(metric: str) -> str:
    return {
        "v_skew": r"$\mathrm{skew}(v_{t+\Delta})$",
        "v_kurt": r"$\mathrm{kurt}(v_{t+\Delta})$",
        "E_dX_dv": r"$E[\Delta X\,\Delta v]$",
        "E_dX2_dv": r"$E[(\Delta X)^2\Delta v]$",
    }[metric]


def _family_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=FAMILY_STYLE[fam],
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=fam,
            markersize=9,
            linestyle="None",
        )
        for fam in ("WSC-KF", "EKF", "UKF", "CKF", "EnKF", "PF", "KLV-PF")
    ]


def _aggregate_point(rows, *, regime, method_key, metric, refinement):
    subset = [
        r
        for r in rows
        if r["regime"] == regime
        and r["method"] == method_key
        and r["metric"] == metric
        and int(float(r["refinement"])) == int(refinement)
    ]
    if not subset:
        return None
    by_seed_err, by_seed_t = [], []
    for sid in sorted({int(r["seed"]) for r in subset}):
        ss = [r for r in subset if int(r["seed"]) == sid]
        err = float(np.mean([float(r["abs_error"]) for r in ss]))
        ref = max(1e-15, float(np.mean([abs(float(r["reference"])) for r in ss])))
        by_seed_err.append(err / ref)
        by_seed_t.append(1e3 * float(np.mean([float(r["time_s"]) for r in ss])))
    return {
        "x": float(np.mean(by_seed_t)),
        "y": float(np.mean(by_seed_err)),
        "xstd": float(np.std(by_seed_t, ddof=1)) if len(by_seed_t) > 1 else 0.0,
        "ystd": float(np.std(by_seed_err, ddof=1)) if len(by_seed_err) > 1 else 0.0,
        "n_seeds": len(by_seed_t),
    }


def _draw_metric(ax, rows, metric: str, regime: str, *, annotate: bool):
    title_fs, label_fs, tick_fs, ann_fs = 18, 15, 13, 9
    points: dict[tuple[str, str], dict] = {}

    for method_key, family, tag, refinement in SERIES:
        pt = _aggregate_point(
            rows,
            regime=regime,
            method_key=method_key,
            metric=metric,
            refinement=refinement,
        )
        if pt is None:
            continue
        color = FAMILY_STYLE[family]
        points[(family, tag)] = {**pt, "color": color, "tag": tag, "family": family}
        ax.errorbar(
            pt["x"],
            pt["y"],
            xerr=pt["xstd"],
            yerr=pt["ystd"],
            fmt="none",
            ecolor=color,
            alpha=0.45,
            capsize=2,
            linewidth=1.0,
            zorder=1,
        )
        ax.scatter(
            pt["x"],
            pt["y"],
            s=70,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )

    # Connect refinement tracks
    for family, tags in TRACKS.items():
        seq = [points[(family, t)] for t in tags if (family, t) in points]
        if len(seq) < 2:
            continue
        ax.plot(
            [p["x"] for p in seq],
            [p["y"] for p in seq],
            "-",
            color=seq[0]["color"],
            lw=1.6,
            alpha=0.85,
            zorder=2,
        )

    if annotate:
        # Stagger offsets by family to reduce collisions
        offsets = {
            "WSC-KF": (4, 7),
            "EKF": (4, -10),
            "UKF": (4, 8),
            "CKF": (4, -9),
            "EnKF": (4, 7),
            "PF": (4, -10),
            "KLV-PF": (4, 8),
        }
        for (family, tag), p in points.items():
            dx, dy = offsets.get(family, (4, 4))
            # Short tags on plot: k=1, x8, n=500 → 1 / x8 / 500
            short = tag
            if tag.startswith("k="):
                short = tag.split("=", 1)[1]
            elif tag.startswith("n="):
                short = tag.split("=", 1)[1]
            ax.annotate(
                short,
                (p["x"], p["y"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=ann_fs,
                color=p["color"],
                ha="left",
                va="center",
                zorder=4,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("One-step forecast wall-clock [ms]", fontsize=label_fs)
    ax.set_ylabel(f"Relative error  {_metric_pretty(metric)}", fontsize=label_fs)
    ax.set_title(_metric_pretty(metric), fontsize=title_fs, pad=8)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.grid(True, which="both", alpha=0.3)
    return points


def _save_mild_panel(rows, out: str, n_seeds: int):
    metrics = ["v_skew", "v_kurt", "E_dX_dv", "E_dX2_dv"]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 10.0))
    axs = axs.ravel()
    for ax, metric in zip(axs, metrics):
        _draw_metric(ax, rows, metric, "mild", annotate=True)

    fig.legend(
        handles=_family_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=7,
        fontsize=13,
        frameon=True,
        title="Filter family  (labels: WSC k · CKF x · particles n)",
        title_fontsize=13,
        columnspacing=1.4,
        handletextpad=0.45,
    )
    fig.suptitle(
        rf"Heston one-step forecast Pareto (mild; $n_{{\mathrm{{seeds}}}}={n_seeds}$)",
        fontsize=19,
        y=0.998,
    )
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.96))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out}")


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    mild_seeds = sorted({int(r["seed"]) for r in rows if r["regime"] == "mild"})
    n_seeds = len(mild_seeds)
    print(f"Loaded {len(rows)} rows; mild seeds={mild_seeds} (n={n_seeds})")

    _save_mild_panel(
        rows,
        os.path.join(OUTPUT_DIR, "non_gaussian_flow_eval_pareto_panel_mild.png"),
        n_seeds,
    )

    # Keep combined + per-metric plots with the same style (mild+stress markers optional)
    # For now only mild panel was requested; still refresh stress+mild panel compactly.
    metrics = ["v_skew", "v_kurt", "E_dX_dv", "E_dX2_dv"]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 10.0))
    for ax, metric in zip(axs.ravel(), metrics):
        _draw_metric(ax, rows, metric, "mild", annotate=True)
        # overlay stress with squares, no extra labels
        for method_key, family, tag, refinement in SERIES:
            pt = _aggregate_point(
                rows,
                regime="stress",
                method_key=method_key,
                metric=metric,
                refinement=refinement,
            )
            if pt is None:
                continue
            ax.scatter(
                pt["x"],
                pt["y"],
                s=55,
                marker="s",
                color=FAMILY_STYLE[family],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.55,
                zorder=2,
            )
    handles = _family_legend_handles() + [
        Line2D([0], [0], marker="o", color="0.3", linestyle="None", label="mild", markersize=8),
        Line2D([0], [0], marker="s", color="0.3", linestyle="None", label="stress", markersize=8),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=11,
        frameon=True,
        title="Filter family  |  ○ mild   □ stress",
        title_fontsize=12,
    )
    fig.suptitle("Heston one-step forecast Pareto (mild + stress)", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.97))
    out = os.path.join(OUTPUT_DIR, "non_gaussian_flow_eval_pareto_panel.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out}")


if __name__ == "__main__":
    main()
