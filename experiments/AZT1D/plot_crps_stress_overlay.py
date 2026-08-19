"""Overlay cohort CRPS-Pareto summaries across σ_G stress scales.

Reads ``cohort_calibrated_summary_{tag}.csv`` files and writes a single
CRPS-vs-wall-clock figure with filter colors and scale linestyles in the legend.
No budget-number annotations (avoids the WSC path-count label).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    save_publication_figure,
    style_curve_axis,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")


def _load_summary(path: str) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["budget"] = int(r["budget"])
        for k in (
            "crps_mean",
            "crps_std",
            "time_s_mean",
            "time_s_std",
            "nll_mean",
            "nll_std",
        ):
            if k in r and r[k] != "":
                r[k] = float(r[k])
    return rows


def main():
    ap = argparse.ArgumentParser(description="Overlay CRPS Pareto across σ_G scales")
    ap.add_argument(
        "--tags",
        type=str,
        required=True,
        help="comma-separated OUT_TAG suffixes after cohort_calibrated_summary_ "
        "(e.g. dt20_stress_sg1,dt20_stress_sg2,dt20_stress_sg3)",
    )
    ap.add_argument(
        "--scales",
        type=str,
        default="",
        help="comma-separated scale labels matching --tags (default: parse sgN from tag)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=os.path.join(OUTPUT_DIR, "cohort_calibrated_crps_dt20_stress.png"),
    )
    ap.add_argument("--dt-assim", type=float, default=20.0, dest="dt_assim")
    ap.add_argument("--err-scale", type=float, default=1.0, dest="err_scale")
    args = ap.parse_args()

    tags = [t.strip().lstrip("_") for t in args.tags.split(",") if t.strip()]
    if args.scales.strip():
        scales = [float(x) for x in args.scales.split(",") if x.strip()]
    else:
        scales = []
        for t in tags:
            # expect ..._sg{N}
            if "_sg" in t:
                scales.append(float(t.rsplit("_sg", 1)[-1]))
            else:
                raise SystemExit(f"cannot parse scale from tag {t!r}; pass --scales")
    if len(scales) != len(tags):
        raise SystemExit("--tags and --scales length mismatch")

    families = ["WSC-KF", "EKF", "UKF", "CKF", "EnKF", "PF"]
    colors = {name: METHOD_COLORS[name] for name in families}
    # Distinct linestyles / markers per scale (readable in B&W too)
    scale_style = {
        1.0: {"ls": "-", "marker": "o", "alpha": 1.00},
        2.0: {"ls": "--", "marker": "s", "alpha": 0.92},
        3.0: {"ls": ":", "marker": "^", "alpha": 0.88},
    }
    bundles = []
    for tag, sc in zip(tags, scales):
        path = os.path.join(OUTPUT_DIR, f"cohort_calibrated_summary_{tag}.csv")
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}")
        bundles.append((sc, _load_summary(path)))

    n_subj = int(bundles[0][1][0].get("n_subjects", 0)) if bundles[0][1] else 0
    err_scale = float(args.err_scale)

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 2.65))
    present_fams: list[str] = []
    for sc, summary in bundles:
        sty = scale_style.get(
            float(sc),
            {"ls": "-.", "marker": "D", "alpha": 0.85},
        )
        for fam in families:
            pts = [s for s in summary if s["filter"] == fam]
            if not pts:
                continue
            if fam not in present_fams:
                present_fams.append(fam)
            pts = sorted(pts, key=lambda s: s["time_s_mean"])
            xs = np.asarray([1e3 * s["time_s_mean"] for s in pts])
            ys = np.asarray([s["crps_mean"] for s in pts])
            yerr = err_scale * np.asarray([s["crps_std"] for s in pts])
            xerr = err_scale * np.asarray([1e3 * s["time_s_std"] for s in pts])
            col = colors.get(fam, "0.3")
            if len(pts) > 1:
                ax.plot(
                    xs, ys, sty["ls"], color=col, linewidth=1.15,
                    alpha=sty["alpha"], zorder=2,
                )
            ax.errorbar(
                xs, ys, xerr=xerr, yerr=yerr,
                fmt=sty["marker"], color=col, markerfacecolor="white",
                markeredgewidth=0.8, markersize=4.2, capsize=2.0,
                elinewidth=0.7, linestyle="none",
                alpha=sty["alpha"], zorder=3,
            )

    # Dual legend: filter color + σ_G scale style
    fam_handles = [
        Line2D(
            [0], [0], color=colors[f], marker="o", linestyle="none",
            markerfacecolor="white", markersize=4.2, label=f,
        )
        for f in present_fams
    ]
    scale_handles = []
    for sc, _ in bundles:
        sty = scale_style.get(
            float(sc),
            {"ls": "-.", "marker": "D", "alpha": 0.85},
        )
        scale_handles.append(
            Line2D(
                [0], [0], color="0.25", marker=sty["marker"], linestyle=sty["ls"],
                linewidth=1.15, markerfacecolor="white", markersize=4.2,
                label=rf"$\sigma_G\times{sc:g}$",
            )
        )
    leg1 = fig.legend(
        handles=fam_handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
        ncol=max(1, len(fam_handles)), title="Filter",
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=scale_handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=max(1, len(scale_handles)), title=r"Process-noise scale",
    )

    ax.set_xlabel("Mean wall time [ms]")
    ax.set_ylabel("Held-out CRPS [mg/dL]")
    ax.set_xscale("log")
    style_curve_axis(ax, grid_axis="both")
    ax.text(
        0.02,
        0.98,
        rf"$N={n_subj}$; $\Delta t={args.dt_assim:g}$ min; bars: $\pm {err_scale:g}$ SD",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.36, top=0.97)
    out_png, out_pdf = save_publication_figure(fig, args.out)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
