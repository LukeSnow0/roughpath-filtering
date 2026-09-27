"""Rebuild numerics_section figures from archived CSVs.

  .venv/bin/python experiments/numerics_section/assemble_figures.py
"""

from __future__ import annotations

import csv
import os
import sys

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"),
)

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for path in (EXPERIMENTS, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from mfts_test.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    save_publication_figure,
    style_curve_axis,
)

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

WSC3, WSC5 = "#9B5DE5", "#7B2CBF"
TME_C = "#CC79A7"
ENKF_C = METHOD_COLORS["CD-EnKF"]
PF_C = "#C1121F"
CKF_C = METHOD_COLORS["CD-CKF"]
EKF_C = METHOD_COLORS["CD-EKF"]
SEUKF_C = "#00897B"

MOMENT_Y_LO = 1e-8
MOMENT_Y_HI = 0.45


def _load(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if k in ("family", "label", "regime", "rmse_comp"):
                continue
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                pass
        out.append(d)
    return out


def _find(rows, family, label):
    for r in rows:
        if r["family"] == family and r["label"] == label:
            return r
    return None


def _sem(std, n):
    return float(std) / np.sqrt(max(float(n), 1.0))


def _fit_slope(x, y, xmin, xmax):
    m = (x >= xmin) & (x <= xmax) & np.isfinite(y) & (y > 0)
    if int(np.sum(m)) < 2:
        return np.nan
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def _point(ax, x, y, *, color, marker, ms=6.0, z=5):
    ax.plot(
        x,
        y,
        linestyle="none",
        marker=marker,
        color=color,
        markerfacecolor="white",
        markersize=ms,
        markeredgewidth=1.0,
        zorder=z,
    )


def _ann(ax, x, y, text, *, color, xytext=(5, 5), ha="left", va="bottom", fs=6.2):
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=xytext,
        fontsize=fs,
        color=color,
        ha=ha,
        va=va,
        zorder=6,
    )


def plot_levy_weak_error(rows: list[dict]) -> None:
    """Local weak error of E[cos(ω x²)].

    CD-CKF cubatures the state, then integrates with L at the mean. From a
    Dirac law that is one node at the origin, so x²≡0. WSC m=3 omits Lévy
    area and predicts the same value. Neither is a one-step IT-1.5 of the
    driving Brownian motion; m=5 is, of the mixed integrals.
    """
    rows = [r for r in rows if r["delta"] <= 0.55]
    taus = np.array([r["delta"] for r in rows])
    fig, ax = plt.subplots(figsize=(0.52 * DOUBLE_COLUMN_WIDTH, 2.12))
    series = [
        ("ckf_err", CKF_C, "X-", r"CD-CKF"),
        ("wsckf_err_m3", WSC3, "D--", r"WSC-KF $m$=3"),
        ("wsckf_err_m5", WSC5, "s-", r"WSC-KF $m$=5"),
    ]
    plotted = {}
    for key, color, fmt, lab in series:
        ys = np.array([max(float(r[key]), 1e-16) for r in rows])
        slope = _fit_slope(taus, ys, 0.10, 0.35)
        label = rf"{lab} (fit={slope:.2f})"
        ax.loglog(
            taus,
            ys,
            fmt,
            color=color,
            label=label,
            markerfacecolor="white",
            markersize=5.6,
            lw=1.45,
        )
        plotted[key] = ys

    mid = min(4, len(taus) - 1)
    for slope, key, lab, ls in (
        (2.0, "ckf_err", r"$\propto\tau^{2}$", "--"),
        (3.0, "wsckf_err_m5", r"$\propto\tau^{3}$", ":"),
    ):
        ys = plotted[key]
        x0, y0 = float(taus[mid]), max(float(ys[mid]), 1e-16)
        xx = np.array([taus[0], taus[-1]])
        ax.loglog(xx, y0 * (xx / x0) ** slope, ls, color="0.45", lw=0.95, label=lab, zorder=1)

    ax.set_xlabel(r"Forecast interval $\tau$")
    ax.set_ylabel(
        r"$|\widehat{\mathbb{E}}[\cos(\omega x^{2})]"
        r"-[\cosh(\omega\tau)]^{-1/2}|$"
    )
    style_curve_axis(ax, grid_axis="both")
    ax.tick_params(labelsize=7.6)
    ax.xaxis.label.set_size(8.4)
    ax.yaxis.label.set_size(8.4)
    ax.legend(fontsize=6.8, loc="lower right", frameon=False, borderaxespad=0.15)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.20, top=0.97)
    save_publication_figure(fig, os.path.join(FIG, "fig_levy_area_weak_error.png"))
    save_publication_figure(fig, os.path.join(FIG, "fig_levy_area_weak_error_local.png"))
    plt.close(fig)


def plot_weak_error(rows: list[dict]) -> None:
    """Heston covariance weak error: CD-CKF vs WSC m=3 and m=5."""
    deltas = np.array([r["delta"] for r in rows])
    fig, ax = plt.subplots(figsize=(0.52 * DOUBLE_COLUMN_WIDTH, 2.72))
    series = [
        ("wsckf_cov_err_m3", WSC3, "D-", r"WSC-KF $m$=3"),
        ("wsckf_cov_err_m5", WSC5, "s-", r"WSC-KF $m$=5"),
        ("ckf_cov_err", CKF_C, "s-", "CD-CKF"),
    ]
    plotted = {}
    for key, color, fmt, lab in series:
        ys = np.array([max(float(r[key]), 1e-16) for r in rows])
        slope = _fit_slope(deltas, ys, 0.15, 0.50)
        label = rf"{lab} (fit={slope:.2f})"
        ax.loglog(
            deltas,
            ys,
            fmt,
            color=color,
            label=label,
            markerfacecolor="white",
            markersize=5.4,
            lw=1.45,
        )
        plotted[key] = ys

    mid = min(6, len(deltas) - 1)
    for slope, ys, lab, ls in (
        (2.0, plotted["ckf_cov_err"], r"$\propto\Delta^{2}$", "--"),
        (3.0, plotted["wsckf_cov_err_m5"], r"$\propto\Delta^{3}$", ":"),
    ):
        x0, y0 = float(deltas[mid]), max(float(ys[mid]), 1e-16)
        xx = np.array([deltas[0], deltas[-1]])
        ax.loglog(xx, y0 * (xx / x0) ** slope, ls, color="0.45", lw=0.95, label=lab, zorder=1)

    ax.set_xlabel(r"Forecast interval $\Delta$")
    ax.set_ylabel(r"$\|P_{\Delta}-\widehat{P}_{\Delta}\|_{\mathrm{F}}$")
    style_curve_axis(ax, grid_axis="both")
    ax.legend(fontsize=6.2, loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.96)
    save_publication_figure(fig, os.path.join(FIG, "fig_heston_weak_error.png"))
    plt.close(fig)


def plot_levy_moment(ax, rows: list[dict], *, compact: bool = False) -> None:
    """One-step |E[(x²)²]−Δ²/2| by method. Not a cost Pareto: TME and m=5 are both exact."""
    specs = [
        ("CD-EKF", "EKF", "EKF" if not compact else "EKF", EKF_C, "s", False),
        ("WSC-KF-m3", "m3-k1", r"$m$=3", WSC3, "D", False),
        ("CD-EnKF", "EnKF-25", "EnKF\n25" if compact else "EnKF 25", ENKF_C, "o", False),
        ("CD-EnKF", "EnKF-2000", "EnKF\n2000" if compact else "EnKF 2000", ENKF_C, "o", False),
        ("TME", "TME-2", "TME-2" if not compact else "TME", TME_C, "P", True),
        ("WSC-KF-m5", "m5-k1", r"$m$=5", WSC5, "s", True),
    ]
    xs, labels = [], []
    pos = 0.0
    gap_done = False
    for fam, lab, disp, color, marker, exact in specs:
        r = _find(rows, fam, lab)
        if r is None:
            continue
        if exact and not gap_done:
            pos += 0.55
            gap_done = True
        x = float(pos)
        y = max(float(r["abs_err"]), MOMENT_Y_LO)
        nseed = max(float(r.get("n_seeds", 1) or 1), 1.0)
        yerr = _sem(r.get("abs_err_std", 0.0) or 0.0, nseed)
        yerr = min(yerr, 0.85 * y) if yerr > 1e-18 else 0.0
        ax.errorbar(
            [x],
            [y],
            yerr=(yerr if yerr > 1e-18 else None),
            fmt=marker,
            linestyle="none",
            color=color,
            markerfacecolor=color if exact else "white",
            markeredgecolor=color,
            markersize=7.2 if exact else 6.2,
            markeredgewidth=1.05,
            ecolor=color,
            elinewidth=0.7,
            capsize=1.6,
            zorder=5,
        )
        if exact:
            _ann(ax, x, y, "exact", color=color, xytext=(0, 9), ha="center", va="bottom", fs=6.0)
        xs.append(x)
        labels.append(disp)
        pos += 1.0

    if len(xs) >= 6:
        ax.axvline(0.5 * (xs[3] + xs[4]), color="0.82", lw=0.7, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5.8 if compact else 6.4)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.set_yscale("log")
    ax.set_ylim(3e-9, MOMENT_Y_HI)
    ax.set_xlabel("")
    ax.set_ylabel(r"$|\widehat{\mathbb{E}}[(x^{2})^{2}]-\Delta^{2}/2|$")
    style_curve_axis(ax, grid_axis="y")
    ax.tick_params(axis="x", length=0)


CHAR_Y_LO = 4e-3
CHAR_Y_HI = 0.55


def _one_step_err_polyline(ax, rows, specs, *, color, marker, ls="-", mfc="white", lw=1.25, z=4):
    pts = []
    for fam, lab in specs:
        r = _find(rows, fam, lab)
        if r is not None:
            pts.append(r)
    if len(pts) < 2:
        return
    pts = sorted(pts, key=lambda r: r["time_s"])
    xs = np.array([1e3 * r["time_s"] for r in pts])
    ys = np.array([max(float(r["abs_err"]), CHAR_Y_LO) for r in pts])
    nseed = np.array([max(float(r.get("n_seeds", 1) or 1), 1.0) for r in pts])
    yerr = np.array([_sem(r.get("abs_err_std", 0.0) or 0.0, n) for r, n in zip(pts, nseed)])
    yerr = np.minimum(yerr, 0.85 * ys)
    ax.errorbar(
        xs,
        ys,
        yerr=np.where(yerr > 1e-18, yerr, np.nan),
        fmt=marker,
        linestyle=ls,
        color=color,
        markerfacecolor=mfc,
        markeredgecolor=color,
        markersize=5.6,
        markeredgewidth=1.0,
        ecolor=color,
        elinewidth=0.7,
        capsize=1.6,
        lw=lw,
        zorder=z,
    )


def plot_levy_char(ax, rows: list[dict]) -> None:
    """One-step |E[cos(ω x²)] − [cosh(ωΔ)]^{-1/2}|: non-polynomial test of P_Δ."""
    _one_step_err_polyline(
        ax, rows,
        [("CD-EnKF", "EnKF-25"), ("CD-EnKF", "EnKF-50"), ("CD-EnKF", "EnKF-100"),
         ("CD-EnKF", "EnKF-500"), ("CD-EnKF", "EnKF-2000")],
        color=ENKF_C, marker="o", ls="--", lw=1.15, z=3,
    )
    _one_step_err_polyline(
        ax, rows,
        [("WSC-KF-m3", "m3-k2"), ("WSC-KF-m3", "m3-k3")],
        color=WSC3, marker="D", mfc=WSC3, lw=1.7, z=6,
    )
    _one_step_err_polyline(
        ax, rows,
        [("WSC-KF-m5", "m5-k1"), ("WSC-KF-m5", "m5-k2")],
        color=WSC5, marker="s", mfc=WSC5, lw=1.45, z=5,
    )

    labels = [
        ("CD-EKF", "EKF", "EKF", EKF_C, True, (0, 7), "center", "bottom"),
        ("WSC-KF-m3", "m3-k1", r"$m$=3, $N_\tau$=1", WSC3, True, (5, 6), "left", "bottom"),
        ("TME", "TME-2", "TME-2", TME_C, True, (-6, 7), "right", "bottom"),
        ("WSC-KF-m5", "m5-k1", r"$m$=5, $N_\tau$=1", WSC5, False, (6, -10), "left", "top"),
        ("WSC-KF-m3", "m3-k3", r"$m$=3, $N_\tau$=3", WSC3, False, (-6, -10), "right", "top"),
        ("WSC-KF-m5", "m5-k2", r"$m$=5, $N_\tau$=2", WSC5, False, (5, 6), "left", "bottom"),
        ("CD-EnKF", "EnKF-25", "25", ENKF_C, False, (6, 5), "left", "bottom"),
        ("CD-EnKF", "EnKF-2000", "2000", ENKF_C, False, (0, -11), "center", "top"),
    ]
    marker_of = {"CD-EKF": "s", "WSC-KF-m3": "D", "TME": "P", "WSC-KF-m5": "s", "CD-EnKF": "o"}
    for fam, lab, disp, color, draw, off, ha, va in labels:
        r = _find(rows, fam, lab)
        if r is None:
            continue
        x = 1e3 * r["time_s"]
        y = max(float(r["abs_err"]), CHAR_Y_LO)
        if draw:
            _point(ax, x, y, color=color, marker=marker_of[fam], ms=6.0)
        _ann(ax, x, y, disp, color=color, xytext=off, ha=ha, va=va)

    ax.set_xlabel("One-step forecast wall-clock [ms]")
    ax.set_ylabel(r"$|\widehat{\mathbb{E}}[\cos(\omega x^{2})]-[\cosh(\omega\Delta)]^{-1/2}|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.006, 55)
    ax.set_ylim(CHAR_Y_LO, CHAR_Y_HI)
    style_curve_axis(ax, grid_axis="both")


def _filter_polyline(
    ax,
    rows,
    specs,
    *,
    color,
    marker,
    ls="-",
    mfc="white",
    mec=None,
    lw=1.2,
    ms=5.6,
    z=4,
    mew=1.0,
    alpha=1.0,
):
    pts = []
    for fam, lab in specs:
        r = _find(rows, fam, lab)
        if r is not None:
            pts.append(r)
    if len(pts) < 2:
        return
    pts = sorted(pts, key=lambda r: r["time_s_mean"])
    xs = np.array([1e3 * r["time_s_mean"] for r in pts])
    ys = np.array([r["pred_nll_per_obs_mean"] for r in pts])
    yerr = np.array([_sem(r["pred_nll_per_obs_std"], r["n_seeds"]) for r in pts])
    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        fmt=marker,
        linestyle=ls,
        color=color,
        markerfacecolor=mfc,
        markeredgecolor=color if mec is None else mec,
        markersize=ms,
        markeredgewidth=mew,
        ecolor=color,
        elinewidth=0.7,
        capsize=1.6,
        lw=lw,
        zorder=z,
        alpha=alpha,
    )


# (family, label, annotation, color, marker, xytext, ha, va) for single filters.
LEVY_FILTER_SINGLES = [
    ("CD-EKF", "EKF", None, EKF_C, "s", (0, 8), "center", "bottom"),
    ("CD-CKF", "CKF", "EKF, CKF", CKF_C, "X", (8, 0), "left", "center"),
    ("SE-UKF", "N8", "SE-UKF", SEUKF_C, "X", (6, 6), "left", "bottom"),
    ("TME", "TME-2", "TME-2", TME_C, "P", (0, 8), "center", "bottom"),
    ("WSC-KF-m3", "m3-k1", r"$m$=3, $N_\tau$=1", WSC3, "D", (0, 8), "center", "bottom"),
    ("WSC-KF-m3", "m3-k3", r"$m$=3, $N_\tau$=3", WSC3, "D", (-6, -11), "right", "top"),
    ("WSC-KF-m5", "m5-k2", r"$m$=5, $N_\tau$=2", WSC5, "s", (5, 6), "left", "bottom"),
    ("CD-EnKF", "EnKF-5", "5", ENKF_C, "o", (-7, 8), "right", "bottom"),
    ("CD-EnKF", "EnKF-100", "100", ENKF_C, "o", (4, 6), "left", "bottom"),
    ("PF", "PF-5", "5", PF_C, "v", (6, -10), "left", "top"),
    ("PF", "PF-500", "500", PF_C, "v", (5, -11), "left", "top"),
]


def plot_filter_pareto(ax, rows: list[dict], *, singles, xlim, ylim, yscale="log") -> None:
    """Sequential predictive NLL vs wall-clock. Type-II singles are drawn at their true NLL."""
    _filter_polyline(
        ax, rows,
        [
            ("CD-EnKF", "EnKF-5"),
            ("CD-EnKF", "EnKF-10"),
            ("CD-EnKF", "EnKF-25"),
            ("CD-EnKF", "EnKF-50"),
            ("CD-EnKF", "EnKF-100"),
            ("CD-EnKF", "EnKF-500"),
        ],
        color=ENKF_C, marker="o", ls="--", mfc="white", lw=1.05, ms=5.2, z=3,
    )
    _filter_polyline(
        ax, rows,
        [
            ("PF", "PF-5"),
            ("PF", "PF-10"),
            ("PF", "PF-25"),
            ("PF", "PF-50"),
            ("PF", "PF-100"),
            ("PF", "PF-500"),
        ],
        color=PF_C, marker="v", ls="--", mfc="white", lw=1.05, ms=5.2, z=3,
    )
    _filter_polyline(
        ax, rows,
        [("WSC-KF-m5", "m5-k1"), ("WSC-KF-m5", "m5-k2")],
        color=WSC5, marker="s", ls="-", mfc=WSC5, mec="white",
        lw=1.45, ms=6.2, z=5, mew=0.7,
    )
    _filter_polyline(
        ax, rows,
        [("WSC-KF-m3", "m3-k1"), ("WSC-KF-m3", "m3-k2"), ("WSC-KF-m3", "m3-k3")],
        color=WSC3, marker="D", ls="-", mfc=WSC3, mec="white",
        lw=1.9, ms=6.8, z=6, mew=0.7,
    )

    drawn = {("WSC-KF-m3", "m3-k1"), ("WSC-KF-m3", "m3-k3"),
             ("WSC-KF-m5", "m5-k1"), ("WSC-KF-m5", "m5-k2")}
    for fam, lab, disp, color, marker, xytext, ha, va in singles:
        r = _find(rows, fam, lab)
        if r is None:
            continue
        x = 1e3 * r["time_s_mean"]
        y = r["pred_nll_per_obs_mean"]
        if (fam, lab) not in drawn:
            _point(ax, x, y, color=color, marker=marker, ms=6.0)
        if disp is not None:
            _ann(ax, x, y, disp, color=color, xytext=xytext, ha=ha, va=va)

    ax.set_xlabel("Wall-clock / track [ms]")
    ax.set_ylabel("Predictive NLL [nats / obs.]")
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    style_curve_axis(ax, grid_axis="both")


def plot_levy_filter(ax, rows: list[dict]) -> None:
    """Sequential cosine NLL on the Lévy-area SDE."""
    plot_filter_pareto(
        ax, rows, singles=LEVY_FILTER_SINGLES, xlim=(0.025, 36), ylim=(0.75, 12.0)
    )


def _levy_legend_filter():
    return [
        Line2D(
            [0], [0], color=WSC3, marker="D", markerfacecolor=WSC3,
            markeredgecolor="white", markeredgewidth=0.7, lw=1.8,
            label=r"WSC-KF $m$=3",
        ),
        Line2D(
            [0], [0], color=WSC5, marker="s", markerfacecolor=WSC5,
            markeredgecolor="white", markeredgewidth=0.7, lw=1.45,
            label=r"WSC-KF $m$=5",
        ),
        Line2D([0], [0], color=TME_C, marker="P", markerfacecolor="white", lw=0, label="TME-2"),
        Line2D([0], [0], color=EKF_C, marker="s", markerfacecolor="white", lw=0, label="CD-EKF"),
        Line2D([0], [0], color=CKF_C, marker="X", markerfacecolor="white", lw=0, label="CD-CKF"),
        Line2D([0], [0], color=SEUKF_C, marker="X", markerfacecolor="white", lw=0, label="SE-UKF"),
        Line2D(
            [0], [0], color=ENKF_C, marker="o", markerfacecolor="white",
            lw=1.15, ls="--", label="CD-EnKF",
        ),
        Line2D(
            [0], [0], color=PF_C, marker="v", markerfacecolor="white",
            lw=1.15, ls="--", label="PF",
        ),
    ]


def _levy_legend():
    return [
        Line2D(
            [0], [0], color=WSC3, marker="D", markerfacecolor=WSC3,
            markeredgecolor="white", markeredgewidth=0.7, lw=1.8,
            label=r"WSC-KF $m$=3",
        ),
        Line2D(
            [0], [0], color=WSC5, marker="s", markerfacecolor=WSC5,
            markeredgecolor="white", markeredgewidth=0.7, lw=1.45,
            label=r"WSC-KF $m$=5",
        ),
        Line2D([0], [0], color=TME_C, marker="P", markerfacecolor="white", lw=0, label="TME-2"),
        Line2D([0], [0], color=EKF_C, marker="s", markerfacecolor="white", lw=0, label="CD-EKF"),
        Line2D([0], [0], color=CKF_C, marker="X", markerfacecolor="white", lw=0, label="CD-CKF"),
        Line2D([0], [0], color=SEUKF_C, marker="X", markerfacecolor="white", lw=0, label="SE-UKF"),
        Line2D(
            [0], [0], color=ENKF_C, marker="o", markerfacecolor="white",
            lw=1.15, ls="--", label="CD-EnKF",
        ),
        Line2D(
            [0], [0], color=PF_C, marker="v", markerfacecolor="white",
            lw=1.15, ls="--", label="PF",
        ),
    ]


def plot_levy_area() -> None:
    filt = _load(os.path.join(DATA, "levy_area_filter_pareto.csv"))
    mom = _load(os.path.join(DATA, "levy_area_moment_pareto.csv"))

    fig, ax = plt.subplots(figsize=(0.62 * DOUBLE_COLUMN_WIDTH, 3.05))
    plot_levy_filter(ax, filt)
    fig.legend(
        handles=_levy_legend_filter(),
        loc="lower center",
        bbox_to_anchor=(0.56, -0.02),
        ncol=3,
        fontsize=5.8,
        frameon=False,
        handletextpad=0.28,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.24, top=0.97)
    save_publication_figure(fig, os.path.join(FIG, "fig_levy_area_split.png"))
    save_publication_figure(fig, os.path.join(FIG, "fig_levy_area_filter.png"))
    plt.close(fig)

    fig1, ax1 = plt.subplots(figsize=(0.62 * DOUBLE_COLUMN_WIDTH, 2.55))
    plot_levy_moment(ax1, mom)
    fig1.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.96)
    save_publication_figure(fig1, os.path.join(FIG, "fig_levy_area_moment.png"))
    plt.close(fig1)

    char = _load(os.path.join(DATA, "levy_area_char_pareto.csv"))
    fig3, ax3 = plt.subplots(figsize=(0.62 * DOUBLE_COLUMN_WIDTH, 2.72))
    plot_levy_char(ax3, char)
    ax3.legend(
        handles=[h for h in _levy_legend() if h.get_label() in (
            r"WSC-KF $m$=3", r"WSC-KF $m$=5", "TME-2", "CD-EKF", "CD-EnKF",
        )],
        fontsize=6.0,
        loc="upper right",
        frameon=False,
    )
    fig3.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.96)
    save_publication_figure(fig3, os.path.join(FIG, "fig_levy_area_char.png"))
    plt.close(fig3)


SIN_FIT_WINDOW = (0.02, 0.2)

SIN_FILTER_SINGLES = [
    ("CD-EKF", "EKF", "EKF", EKF_C, "s", (6, 0), "left", "center"),
    ("CD-CKF", "CKF", "CKF", CKF_C, "X", (6, 0), "left", "center"),
    ("SE-UKF", "N8", "SE-UKF", SEUKF_C, "X", (-5, 5), "right", "bottom"),
    ("TME", "TME-2", "TME-2", TME_C, "P", (0, 8), "center", "bottom"),
    ("WSC-KF-m3", "m3-k1", r"$m$=3, $N_\tau$=1", WSC3, "D", (0, -9), "center", "top"),
    ("WSC-KF-m3", "m3-k3", r"$m$=3, $N_\tau$=3", WSC3, "D", (-2, 8), "left", "bottom"),
    ("WSC-KF-m5", "m5-k1", r"$m$=5, $N_\tau$=1", WSC5, "s", (-6, -6), "right", "top"),
    ("WSC-KF-m5", "m5-k2", r"$m$=5, $N_\tau$=2", WSC5, "s", (0, 7), "center", "bottom"),
    ("CD-EnKF", "EnKF-5", "5", ENKF_C, "o", (-7, 4), "right", "bottom"),
    ("CD-EnKF", "EnKF-500", "500", ENKF_C, "o", (4, 5), "left", "bottom"),
    ("PF", "PF-5", "5", PF_C, "v", (6, -2), "left", "top"),
    ("PF", "PF-500", "500", PF_C, "v", (6, -2), "left", "top"),
]


def plot_sin_weak_error(rows: list[dict]) -> None:
    """One-step weak error on dx² = sin(x¹) dW² from (1, 0), for two functionals.

    From a point mass CD-EKF and CD-CKF coincide (one node, L at the mean). The
    stochastic Taylor expansion does not terminate here, so m=3 and m=5 show
    their generic orders 2 and 3.
    """
    taus = np.array([r["delta"] for r in rows])
    series = [
        (("ckf_err", "ckf_err2"), CKF_C, "X-", "CD-EKF / CD-CKF"),
        (("tme2_err", "tme2_err2"), TME_C, "P-.", "TME-2"),
        (("wsckf_err_m3", "wsckf_err2_m3"), WSC3, "D--", r"WSC-KF $m$=3"),
        (("wsckf_err_m5", "wsckf_err2_m5"), WSC5, "s-", r"WSC-KF $m$=5"),
    ]
    panels = (
        (r"$\mathbb{E}[\cos(\omega x^{2}_{\tau})]$", "(a)"),
        (r"$\mathbb{E}[(x^{2}_{\tau})^{2}]$", "(b)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.55))
    handles = []
    for p, (ax, (title, tag)) in enumerate(zip(axes, panels)):
        for keys, color, fmt, _lab in series:
            ys = np.array([max(float(r[keys[p]]), 1e-16) for r in rows])
            # CD-CKF is drawn wider so it stays visible where WSC m=3 coincides with it.
            lw = 2.4 if keys[0] == "ckf_err" else 1.35
            (line,) = ax.loglog(
                taus, ys, fmt, color=color, markerfacecolor="white", markersize=4.6, lw=lw
            )
            if p == 0:
                handles.append(line)
        x_ref = np.array([taus[0], taus[-1]])
        for slope, keys, ls in ((2.0, series[0][0], "--"), (3.0, series[3][0], ":")):
            y0 = 0.35 * float(rows[0][keys[p]])
            ax.loglog(x_ref, y0 * (x_ref / taus[0]) ** slope, ls, color="0.45", lw=0.95, zorder=1)
        ax.set_title(title, fontweight="normal")
        ax.set_xlabel(r"Forecast interval $\tau$")
        ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
        ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.text(0.5, -0.30, tag, transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
        style_curve_axis(ax, grid_axis="both")
    axes[0].set_ylabel("One-step weak error")

    labels = []
    for keys, _color, _fmt, lab in series:
        fits = [
            _fit_slope(taus, np.array([float(r[k]) for r in rows]), *SIN_FIT_WINDOW)
            for k in keys
        ]
        labels.append(f"{lab} (slopes {fits[0]:.2f} / {fits[1]:.2f})")
    ref = [
        Line2D([0], [0], color="0.45", lw=0.95, ls="--", label=r"$\propto\tau^{2}$"),
        Line2D([0], [0], color="0.45", lw=0.95, ls=":", label=r"$\propto\tau^{3}$"),
    ]
    fig.legend(
        handles=handles + ref,
        labels=labels + [h.get_label() for h in ref],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=6.8,
        frameon=False,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.74, wspace=0.22)
    save_publication_figure(fig, os.path.join(FIG, "fig_sin_area_weak_error.png"))
    plt.close(fig)


def plot_sin_filter(rows: list[dict]) -> None:
    """Sequential sine-observation NLL on dx² = sin(x¹) dW²."""
    fig, ax = plt.subplots(figsize=(0.62 * DOUBLE_COLUMN_WIDTH, 3.05))
    plot_filter_pareto(
        ax, rows, singles=SIN_FILTER_SINGLES, xlim=(0.01, 60), ylim=(0.9, 2.15), yscale="linear"
    )
    fig.legend(
        handles=_levy_legend_filter(),
        loc="lower center",
        bbox_to_anchor=(0.56, -0.02),
        ncol=3,
        fontsize=5.8,
        frameon=False,
        handletextpad=0.28,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.24, top=0.97)
    save_publication_figure(fig, os.path.join(FIG, "fig_sin_area_filter.png"))
    plt.close(fig)


def _sci(x: float) -> str:
    """``3.8\\times10^{-2}`` for LaTeX tables."""
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def write_sin_one_step_table(moment: list[dict], char: list[dict]) -> None:
    """LaTeX table of one-step errors at τ = 0.5 on the sin(x¹) system (Table 1 analogue)."""
    entries = [
        ("CD-EKF", "EKF", "CD-EKF"),
        ("CD-CKF", "CKF", "CD-CKF"),
        ("SE-UKF", "N8", "SE-UKF ($N=8$)"),
        ("TME", "TME-2", "TME-2"),
        ("TME", "TME-3", "TME-3"),
        ("WSC-KF-m3", "m3-k1", r"WSC-KF $m=3$, $N_\tau=1$"),
        ("WSC-KF-m3", "m3-k2", r"WSC-KF $m=3$, $N_\tau=2$"),
        ("WSC-KF-m3", "m3-k3", r"WSC-KF $m=3$, $N_\tau=3$"),
        ("WSC-KF-m5", "m5-k1", r"WSC-KF $m=5$, $N_\tau=1$"),
        ("WSC-KF-m5", "m5-k2", r"WSC-KF $m=5$, $N_\tau=2$"),
        ("CD-EnKF", "EnKF-25", "CD-EnKF ($n=25$)"),
        ("CD-EnKF", "EnKF-100", "CD-EnKF ($n=100$)"),
        ("CD-EnKF", "EnKF-2000", r"CD-EnKF ($n=2\,000$)"),
    ]

    def cell(r):
        if r["n_seeds"] > 1:
            return rf"{_sci(r['abs_err'])} $\pm$ {_sci(_sem(r['abs_err_std'], r['n_seeds']))}"
        return _sci(r["abs_err"])

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Paths & $|\Delta\mathbb{E}[(x^2)^2]|$ & $|\Delta\mathbb{E}[\cos(\omega x^2)]|$ \\",
        r"\midrule",
    ]
    for fam, lab, name in entries:
        m, c = _find(moment, fam, lab), _find(char, fam, lab)
        paths = f"{int(m['budget'])}" if fam.startswith(("WSC", "CD-EnKF")) else "--"
        lines.append(rf"{name} & {paths} & {cell(m)} & {cell(c)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(os.path.join(FIG, "tab_sin_area_one_step.tex"), "w") as f:
        f.write("\n".join(lines))


def plot_sin_area() -> None:
    plot_sin_weak_error(_load(os.path.join(DATA, "sin_area_weak_error.csv")))
    plot_sin_filter(_load(os.path.join(DATA, "sin_area_filter_pareto.csv")))
    write_sin_one_step_table(
        _load(os.path.join(DATA, "sin_area_moment_pareto.csv")),
        _load(os.path.join(DATA, "sin_area_char_pareto.csv")),
    )


def main() -> None:
    levy_weak = _load(os.path.join(DATA, "levy_area_weak_error.csv"))
    plot_levy_weak_error(levy_weak)
    heston_weak = _load(os.path.join(DATA, "heston_weak_error.csv"))
    plot_weak_error(heston_weak)
    plot_levy_area()
    plot_sin_area()
    print(f"Wrote figures in {FIG}")


if __name__ == "__main__":
    main()
