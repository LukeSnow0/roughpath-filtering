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
    ap.add_argument("--err-scale", type=float, default=0.25, dest="err_scale")
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

    colors = {
        "WSC-KF": "#6A4C93",
        "EKF": "#2E86AB",
        "UKF": "#E8B923",
        "CKF": "#3DA35D",
        "EnKF": "#D97706",
        "PF": "#F24236",
    }
    # Distinct linestyles / markers per scale (readable in B&W too)
    scale_style = {
        1.0: {"ls": "-", "marker": "o", "alpha": 1.00},
        2.0: {"ls": "--", "marker": "s", "alpha": 0.92},
        3.0: {"ls": ":", "marker": "^", "alpha": 0.88},
    }
    families = ["WSC-KF", "EKF", "UKF", "CKF", "EnKF", "PF"]

    bundles = []
    for tag, sc in zip(tags, scales):
        path = os.path.join(OUTPUT_DIR, f"cohort_calibrated_summary_{tag}.csv")
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}")
        bundles.append((sc, _load_summary(path)))

    n_subj = int(bundles[0][1][0].get("n_subjects", 0)) if bundles[0][1] else 0
    err_scale = float(args.err_scale)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
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
                    xs, ys, sty["ls"], color=col, linewidth=1.35,
                    alpha=sty["alpha"], zorder=2,
                )
            ax.errorbar(
                xs, ys, xerr=xerr, yerr=yerr,
                fmt=sty["marker"], color=col, markersize=7, capsize=2.5,
                elinewidth=1.0, linestyle="none",
                alpha=sty["alpha"], zorder=3,
            )

    # Dual legend: filter color + σ_G scale style
    fam_handles = [
        Line2D(
            [0], [0], color=colors[f], marker="o", linestyle="none",
            markersize=7, label=f,
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
                linewidth=1.35, markersize=7,
                label=rf"$\sigma_G\times{sc:g}$",
            )
        )
    leg1 = ax.legend(
        handles=fam_handles, loc="upper left", frameon=False, fontsize=8, title="filter",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=scale_handles, loc="upper right", frameon=False, fontsize=8, title="noise",
    )

    ax.set_xlabel(rf"Wall-clock [ms] (mean $\pm$ {err_scale:g}\,std over subjects)")
    ax.set_ylabel("Held-out CRPS")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    scale_txt = ", ".join(f"$\\times{sc:g}$" for sc, _ in bundles)
    fig.suptitle(
        f"AZT1D cohort CRPS Pareto (PF=mixture; $\\Delta t$={args.dt_assim:g} min)\n"
        f"N={n_subj} subjects; EnKF $\\hat\\theta_i$; "
        rf"$\sigma_G$ stress {scale_txt}; bars = $\pm${err_scale:g} std",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
