"""Rebuild numerics_section figures from archived CSVs.

  .venv/bin/python experiments/numerics_section/assemble_figures.py

Styling comes from ``mfts_test/plot_style.py`` (colors, markers, typography);
this module only lays figures out. Every figure is authored at the manuscript's
single-column width (``PAPER_COLUMN_WIDTH``); include it with
``width=\\columnwidth`` so text prints at its nominal size.
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
from matplotlib.ticker import FixedLocator, NullFormatter

from mfts_test.plot_style import (
    METHOD_COLORS,
    METHOD_MARKERS,
    PAPER_COLUMN_WIDTH,
    PAPER_TEXT_WIDTH,
    PALETTE,
    save_publication_figure,
    style_curve_axis,
)

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

WSC3 = METHOD_COLORS["WSC-KF-m3"]
WSC5 = METHOD_COLORS["WSC-KF-m5"]
TME_C = METHOD_COLORS["TME"]
ENKF_C = METHOD_COLORS["CD-EnKF"]
PF_C = METHOD_COLORS["PF"]
CKF_C = METHOD_COLORS["CD-CKF"]
EKF_C = METHOD_COLORS["CD-EKF"]
SEUKF_C = METHOD_COLORS["SE-UKF"]
REF_C = PALETTE["neutral_gray"]
MARKER = {fam: METHOD_MARKERS[fam] for fam in (
    "WSC-KF-m3", "WSC-KF-m5", "TME", "CD-EKF", "CD-CKF", "SE-UKF", "CD-EnKF", "PF",
)}

ANN_FS = 7.5  # direct labels next to points
CURVE_LW = 1.8
MARKER_SIZE = 5.0

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


def _decimal_ticks(axis, ticks, labels=None) -> None:
    """Plain-number major ticks on a log axis (e.g. 0.1, 0.2, 0.5), unlabeled minors."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_ticklabels(labels if labels is not None else [f"{t:g}" for t in ticks])
    axis.set_minor_formatter(NullFormatter())


def _point(ax, x, y, *, color, marker, ms=MARKER_SIZE + 0.5, z=5):
    ax.plot(
        x,
        y,
        linestyle="none",
        marker=marker,
        color=color,
        markerfacecolor="white",
        markersize=ms,
        markeredgewidth=1.2,
        zorder=z,
    )


def _ann(ax, x, y, text, *, color, xytext=(5, 5), ha="left", va="bottom", fs=ANN_FS):
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


# ---------------------------------------------------------------------------
# One-step weak-error rate plots
# ---------------------------------------------------------------------------


def _rate_panel(ax, xs, curves, refs, *, fit_window, xticks, markevery=1,
                markersize=MARKER_SIZE):
    """Log-log error curves with dark-gray reference slopes.

    ``curves``: ``(ys, color, fmt, lw[, markerfacecolor])``; returns the fitted
    slope of each.
    ``refs``: ``(slope, anchor_ys, linestyle)``; each reference line passes half a
    decade-factor below ``anchor_ys`` at the start of the fit window.
    """
    slopes = []
    for ys, color, fmt, lw, *mfc in curves:
        ys = np.maximum(np.asarray(ys, dtype=float), 1e-16)
        ax.loglog(xs, ys, fmt, color=color, markerfacecolor=mfc[0] if mfc else "white",
                  markevery=markevery,
                  markersize=markersize, markeredgewidth=1.1, lw=lw)
        slopes.append(_fit_slope(xs, ys, *fit_window))
    i0 = int(np.argmin(np.abs(np.log(xs / fit_window[0]))))
    x_ref = np.array([xs[0], xs[-1]])
    for slope, anchor, ls in refs:
        y0 = 0.5 * float(anchor[i0])
        ax.loglog(x_ref, y0 * (x_ref / xs[i0]) ** slope, ls, color=REF_C, lw=1.2, zorder=1)
    _decimal_ticks(ax.xaxis, xticks)
    style_curve_axis(ax)
    return slopes


def _rate_legend_handles(styles, labels):
    handles = [
        Line2D([0], [0], color=c, marker=fmt[0], ls=fmt[1:], lw=lw, markersize=MARKER_SIZE,
               markerfacecolor="white", markeredgewidth=1.1, label=lab)
        for (c, fmt, lw), lab in zip(styles, labels)
    ]
    return handles + [
        Line2D([0], [0], color=REF_C, lw=1.2, ls="--", label=r"$\propto\tau^{2}$"),
        Line2D([0], [0], color=REF_C, lw=1.2, ls=":", label=r"$\propto\tau^{3}$"),
    ]


def plot_levy_weak_error(rows: list[dict]) -> None:
    """Local weak error of E[cos(ω x²)] on the Lévy-area SDE.

    CD-CKF cubatures the state, then integrates with L at the mean. From a
    Dirac law that is one node at the origin, so x²≡0. WSC m=3 omits Lévy
    area and predicts the same value, so it lies on top of CD-CKF.
    """
    rows = [r for r in rows if r["delta"] <= 0.55]
    taus = np.array([r["delta"] for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])
    styles = [(CKF_C, "X-", 3.0), (WSC3, "D--", CURVE_LW), (WSC5, "s-", CURVE_LW)]
    fig, ax = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 2.45))
    slopes = _rate_panel(
        ax, taus,
        [(col(k), *st) for k, st in zip(("ckf_err", "wsckf_err_m3", "wsckf_err_m5"), styles)],
        [(2.0, col("ckf_err"), "--"), (3.0, col("wsckf_err_m5"), ":")],
        fit_window=(0.10, 0.35), xticks=[0.1, 0.2, 0.3, 0.5],
    )
    ax.set_xlabel(r"Forecast interval $\tau$")
    ax.set_ylabel(r"Error in $\mathbb{E}[\cos(\omega x^{(2)}_{\tau})]$")
    names = ["CD-CKF", r"WSC-KF $m$=3", r"WSC-KF $m$=5"]
    fig.legend(
        handles=_rate_legend_handles(
            styles, [f"{n} (fit {s:.2f})" for n, s in zip(names, slopes)]
        ),
        loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.19, top=0.70)
    save_publication_figure(fig, os.path.join(FIG, "fig_levy_area_weak_error.png"))
    save_publication_figure(fig, os.path.join(FIG, "fig_levy_area_weak_error_local.png"))
    plt.close(fig)


def plot_weak_error(rows: list[dict]) -> None:
    """Heston covariance weak error: CD-CKF vs WSC m=3 and m=5."""
    deltas = np.array([r["delta"] for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])
    styles = [(CKF_C, "X-", CURVE_LW), (WSC3, "D--", CURVE_LW), (WSC5, "s-", CURVE_LW)]
    keys = ("ckf_cov_err", "wsckf_cov_err_m3", "wsckf_cov_err_m5")
    fig, ax = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 2.45))
    slopes = _rate_panel(
        ax, deltas,
        [(col(k), *st) for k, st in zip(keys, styles)],
        [(2.0, col("ckf_cov_err"), "--"), (3.0, col("wsckf_cov_err_m5"), ":")],
        fit_window=(0.15, 0.50), xticks=[0.2, 0.3, 0.5],
    )
    ax.set_xlabel(r"Forecast interval $\Delta$")
    ax.set_ylabel(r"$\|P_{\Delta}-\widehat{P}_{\Delta}\|_{\mathrm{F}}$")
    names = ["CD-CKF", r"WSC-KF $m$=3", r"WSC-KF $m$=5"]
    handles = _rate_legend_handles(
        styles, [f"{n} (fit {s:.2f})" for n, s in zip(names, slopes)]
    )
    for h, lab in zip(handles[-2:], (r"$\propto\Delta^{2}$", r"$\propto\Delta^{3}$")):
        h.set_label(lab)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2,
               columnspacing=0.8)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.19, top=0.70)
    save_publication_figure(fig, os.path.join(FIG, "fig_heston_weak_error.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# One-step moment / characteristic-function comparisons
# ---------------------------------------------------------------------------


def plot_levy_moment(ax, rows: list[dict]) -> None:
    """One-step |E[(x²)²]−Δ²/2| by method. Not a cost Pareto: TME and m=5 are both exact."""
    specs = [
        ("CD-EKF", "EKF", "EKF", EKF_C, False),
        ("WSC-KF-m3", "m3-k1", r"$m$=3", WSC3, False),
        ("CD-EnKF", "EnKF-25", "EnKF\n25", ENKF_C, False),
        ("CD-EnKF", "EnKF-2000", "EnKF\n2000", ENKF_C, False),
        ("TME", "TME-2", "TME-2", TME_C, True),
        ("WSC-KF-m5", "m5-k1", r"$m$=5", WSC5, True),
    ]
    xs, labels = [], []
    pos = 0.0
    gap_done = False
    for fam, lab, disp, color, exact in specs:
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
            fmt=MARKER[fam],
            linestyle="none",
            color=color,
            markerfacecolor=color if exact else "white",
            markeredgecolor=color,
            markersize=7.0,
            markeredgewidth=1.2,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.0,
            zorder=5,
        )
        if exact:
            _ann(ax, x, y, "exact", color=color, xytext=(0, 9), ha="center", va="bottom")
        xs.append(x)
        labels.append(disp)
        pos += 1.0

    if len(xs) >= 6:
        ax.axvline(0.5 * (xs[3] + xs[4]), color=PALETTE["light_gray"], lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.set_yscale("log")
    ax.set_ylim(3e-9, MOMENT_Y_HI)
    ax.set_ylabel(r"Error in $\mathbb{E}[(x^{(2)}_{\Delta})^{2}]$")
    style_curve_axis(ax)
    ax.tick_params(axis="x", length=0)


CHAR_Y_LO = 4e-3
CHAR_Y_HI = 0.55


def _one_step_err_polyline(ax, rows, specs, *, color, marker, ls="-", mfc="white",
                           lw=CURVE_LW, z=4):
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
        markersize=MARKER_SIZE + 0.5,
        markeredgewidth=1.1,
        ecolor=color,
        elinewidth=0.9,
        capsize=2.0,
        lw=lw,
        zorder=z,
    )


def plot_levy_char(ax, rows: list[dict]) -> None:
    """One-step |E[cos(ω x²)] − [cosh(ωΔ)]^{-1/2}|: non-polynomial test of P_Δ."""
    _one_step_err_polyline(
        ax, rows,
        [("CD-EnKF", "EnKF-25"), ("CD-EnKF", "EnKF-50"), ("CD-EnKF", "EnKF-100"),
         ("CD-EnKF", "EnKF-500"), ("CD-EnKF", "EnKF-2000")],
        color=ENKF_C, marker=MARKER["CD-EnKF"], ls="--", lw=1.4, z=3,
    )
    _one_step_err_polyline(
        ax, rows,
        [("WSC-KF-m3", "m3-k2"), ("WSC-KF-m3", "m3-k3")],
        color=WSC3, marker=MARKER["WSC-KF-m3"], mfc=WSC3, z=6,
    )
    _one_step_err_polyline(
        ax, rows,
        [("WSC-KF-m5", "m5-k1"), ("WSC-KF-m5", "m5-k2")],
        color=WSC5, marker=MARKER["WSC-KF-m5"], mfc=WSC5, z=5,
    )

    labels = [
        ("CD-EKF", "EKF", "EKF", EKF_C, True, (0, 7), "center", "bottom"),
        ("TME", "TME-2", "TME-2", TME_C, True, (-6, 7), "right", "bottom"),
        ("WSC-KF-m3", "m3-k1", r"$N_\tau$=1", WSC3, True, (0, 7), "center", "bottom"),
        ("WSC-KF-m3", "m3-k2", "2", WSC3, False, (-7, 0), "right", "center"),
        ("WSC-KF-m3", "m3-k3", "3", WSC3, False, (0, -9), "center", "top"),
        ("WSC-KF-m5", "m5-k1", r"$N_\tau$=1", WSC5, False, (-8, 0), "right", "center"),
        ("WSC-KF-m5", "m5-k2", "2", WSC5, False, (0, 7), "center", "bottom"),
        ("CD-EnKF", "EnKF-25", r"$n$=25", REF_C, False, (8, -3), "left", "top"),
        ("CD-EnKF", "EnKF-2000", r"$n$=2000", REF_C, False, (0, -10), "center", "top"),
    ]
    for fam, lab, disp, color, draw, off, ha, va in labels:
        r = _find(rows, fam, lab)
        if r is None:
            continue
        x = 1e3 * r["time_s"]
        y = max(float(r["abs_err"]), CHAR_Y_LO)
        if draw:
            _point(ax, x, y, color=color, marker=MARKER[fam])
        _ann(ax, x, y, disp, color=color, xytext=off, ha=ha, va=va)

    ax.set_xlabel("One-step forecast wall-clock [ms]")
    ax.set_ylabel(r"Error in $\mathbb{E}[\cos(\omega x^{(2)}_{\Delta})]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.006, 55)
    ax.set_ylim(CHAR_Y_LO, CHAR_Y_HI)
    _decimal_ticks(ax.xaxis, [0.01, 0.1, 1, 10])
    _decimal_ticks(ax.yaxis, [0.01, 0.1])
    style_curve_axis(ax)


# ---------------------------------------------------------------------------
# Sequential filtering: predictive NLL vs wall-clock
# ---------------------------------------------------------------------------


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
    lw=CURVE_LW,
    ms=MARKER_SIZE + 0.5,
    z=4,
    mew=1.1,
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
        elinewidth=0.9,
        capsize=2.0,
        lw=lw,
        zorder=z,
        alpha=alpha,
    )


# Single Gaussian filters: (family, label, annotation, xytext, ha, va); color
# and marker come from the family.
LEVY_FILTER_SINGLES = [
    ("CD-EKF", "EKF", None, (0, 8), "center", "bottom"),
    ("CD-CKF", "CKF", "EKF, CKF", (8, 0), "left", "center"),
    ("SE-UKF", "N8", "SE-UKF", (6, 6), "left", "bottom"),
    ("TME", "TME-2", "TME-2", (0, 8), "center", "bottom"),
]

# Points on the WSC-KF and ensemble curves: (family, label, text, xytext, ha, va,
# color). WSC-KF points carry their segment count; ensemble sizes are gray.
LEVY_FILTER_LABELS = [
    ("WSC-KF-m3", "m3-k1", r"$N_\tau$=1", (0, 8), "center", "bottom", WSC3),
    ("WSC-KF-m3", "m3-k2", "2", (-6, -6), "right", "top", WSC3),
    ("WSC-KF-m3", "m3-k3", "3", (0, -9), "center", "top", WSC3),
    ("WSC-KF-m5", "m5-k1", r"$N_\tau$=1", (-8, 0), "right", "center", WSC5),
    ("WSC-KF-m5", "m5-k2", "2", (0, 8), "center", "bottom", WSC5),
    ("CD-EnKF", "EnKF-5", r"$n$=5", (8, 0), "left", "center", REF_C),
    ("PF", "PF-500", r"$n$=500", (7, 2), "left", "center", REF_C),
]

FAMILY_COLOR = {
    "WSC-KF-m3": WSC3, "WSC-KF-m5": WSC5, "TME": TME_C, "CD-EKF": EKF_C,
    "CD-CKF": CKF_C, "SE-UKF": SEUKF_C, "CD-EnKF": ENKF_C, "PF": PF_C,
}


def plot_filter_pareto(
    ax, rows: list[dict], *, singles, labels, xlim, ylim, yscale="log"
) -> None:
    """Sequential predictive NLL vs wall-clock. Type-II singles are drawn at their true NLL."""
    ens = [5, 10, 25, 50, 100, 500]
    _filter_polyline(ax, rows, [("CD-EnKF", f"EnKF-{n}") for n in ens],
                     color=ENKF_C, marker=MARKER["CD-EnKF"], ls="--", lw=1.4, z=3)
    _filter_polyline(ax, rows, [("PF", f"PF-{n}") for n in ens],
                     color=PF_C, marker=MARKER["PF"], ls="--", lw=1.4, z=3)
    _filter_polyline(ax, rows, [("WSC-KF-m5", "m5-k1"), ("WSC-KF-m5", "m5-k2")],
                     color=WSC5, marker=MARKER["WSC-KF-m5"], mfc=WSC5, mec="white",
                     ms=6.6, z=5, mew=0.8)
    _filter_polyline(ax, rows, [("WSC-KF-m3", f"m3-k{k}") for k in (1, 2, 3)],
                     color=WSC3, marker=MARKER["WSC-KF-m3"], mfc=WSC3, mec="white",
                     ms=6.6, z=6, mew=0.8)

    drawn = {("WSC-KF-m3", "m3-k1"), ("WSC-KF-m3", "m3-k3"),
             ("WSC-KF-m5", "m5-k1"), ("WSC-KF-m5", "m5-k2")}
    for fam, lab, disp, xytext, ha, va in singles:
        r = _find(rows, fam, lab)
        if r is None:
            continue
        x = 1e3 * r["time_s_mean"]
        y = r["pred_nll_per_obs_mean"]
        if (fam, lab) not in drawn:
            _point(ax, x, y, color=FAMILY_COLOR[fam], marker=MARKER[fam])
        if disp is not None:
            _ann(ax, x, y, disp, color=FAMILY_COLOR[fam], xytext=xytext, ha=ha, va=va)
    for fam, lab, text, xytext, ha, va, color in labels:
        r = _find(rows, fam, lab)
        if r is not None:
            _ann(ax, 1e3 * r["time_s_mean"], r["pred_nll_per_obs_mean"], text,
                 color=color, xytext=xytext, ha=ha, va=va)

    ax.set_xlabel("Wall-clock per track [ms]")
    ax.set_ylabel("Predictive NLL [nats / obs.]")
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _decimal_ticks(ax.xaxis, [t for t in (0.01, 0.1, 1, 10, 100) if xlim[0] <= t <= xlim[1]])
    style_curve_axis(ax)


def plot_levy_filter(ax, rows: list[dict]) -> None:
    """Sequential cosine NLL on the Lévy-area SDE."""
    plot_filter_pareto(
        ax, rows, singles=LEVY_FILTER_SINGLES, labels=LEVY_FILTER_LABELS,
        xlim=(0.025, 36), ylim=(0.75, 12.0),
    )
    _decimal_ticks(ax.yaxis, [1, 2, 5, 10])


def _method_legend():
    point = dict(markerfacecolor="white", markeredgewidth=1.2, markersize=6.0, lw=0)
    line = dict(markerfacecolor="white", markeredgewidth=1.1, markersize=5.5)
    solid = dict(markeredgecolor="white", markeredgewidth=0.8, markersize=6.6, lw=CURVE_LW)
    return [
        Line2D([0], [0], color=WSC3, marker=MARKER["WSC-KF-m3"], markerfacecolor=WSC3,
               label=r"WSC-KF $m$=3", **solid),
        Line2D([0], [0], color=WSC5, marker=MARKER["WSC-KF-m5"], markerfacecolor=WSC5,
               label=r"WSC-KF $m$=5", **solid),
        Line2D([0], [0], color=ENKF_C, marker=MARKER["CD-EnKF"], ls="--", lw=1.4,
               label="CD-EnKF", **line),
        Line2D([0], [0], color=PF_C, marker=MARKER["PF"], ls="--", lw=1.4, label="PF", **line),
        Line2D([0], [0], color=TME_C, marker=MARKER["TME"], label="TME-2", **point),
        Line2D([0], [0], color=EKF_C, marker=MARKER["CD-EKF"], label="CD-EKF", **point),
        Line2D([0], [0], color=CKF_C, marker=MARKER["CD-CKF"], label="CD-CKF", **point),
        Line2D([0], [0], color=SEUKF_C, marker=MARKER["SE-UKF"], label="SE-UKF", **point),
    ]


# Compact legend spacing for figures where vertical space is tight.
TIGHT_LEGEND = dict(handlelength=1.5, handletextpad=0.35, labelspacing=0.25,
                    columnspacing=0.7, borderaxespad=0.1)


def _filter_figure(plot_fn, rows, outnames) -> None:
    fig, ax = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 2.9))
    plot_fn(ax, rows)
    fig.legend(handles=_method_legend(), loc="lower center", bbox_to_anchor=(0.53, 0.0),
               ncol=3, **TIGHT_LEGEND)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.315, top=0.985)
    for name in outnames:
        save_publication_figure(fig, os.path.join(FIG, name))
    plt.close(fig)


def plot_levy_area() -> None:
    filt = _load(os.path.join(DATA, "levy_area_filter_pareto.csv"))
    _filter_figure(plot_levy_filter, filt,
                   ("fig_levy_area_split.png", "fig_levy_area_filter.png"))

    mom = _load(os.path.join(DATA, "levy_area_moment_pareto.csv"))
    fig1, ax1 = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 2.2))
    plot_levy_moment(ax1, mom)
    fig1.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.96)
    save_publication_figure(fig1, os.path.join(FIG, "fig_levy_area_moment.png"))
    plt.close(fig1)

    char = _load(os.path.join(DATA, "levy_area_char_pareto.csv"))
    fig3, ax3 = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 2.95))
    plot_levy_char(ax3, char)
    fig3.legend(
        handles=[h for h in _method_legend() if h.get_label() in (
            r"WSC-KF $m$=3", r"WSC-KF $m$=5", "CD-EnKF", "TME-2", "CD-EKF",
        )],
        loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=3, columnspacing=0.8,
    )
    fig3.subplots_adjust(left=0.18, right=0.97, bottom=0.31, top=0.97)
    save_publication_figure(fig3, os.path.join(FIG, "fig_levy_area_char.png"))
    plt.close(fig3)


# ---------------------------------------------------------------------------
# sin(x¹) area SDE
# ---------------------------------------------------------------------------

SIN_FIT_WINDOW = (0.02, 0.2)

SIN_FILTER_SINGLES = [
    ("CD-EKF", "EKF", "EKF", (7, 0), "left", "center"),
    ("CD-CKF", "CKF", "CKF", (7, 0), "left", "center"),
    ("SE-UKF", "N8", "SE-UKF", (-6, 5), "right", "bottom"),
    ("TME", "TME-2", "TME-2", (0, 8), "center", "bottom"),
]

SIN_FILTER_LABELS = [
    ("WSC-KF-m3", "m3-k1", r"$N_\tau$=1", (0, -10), "center", "top", WSC3),
    ("WSC-KF-m3", "m3-k2", "2", (0, -9), "center", "top", WSC3),
    ("WSC-KF-m3", "m3-k3", "3", (0, 8), "center", "bottom", WSC3),
    ("WSC-KF-m5", "m5-k1", r"$N_\tau$=1", (-8, 0), "right", "center", WSC5),
    ("WSC-KF-m5", "m5-k2", "2", (0, 8), "center", "bottom", WSC5),
    ("PF", "PF-5", r"$n$=5", (-8, 0), "right", "center", REF_C),
    ("CD-EnKF", "EnKF-500", r"$n$=500", (7, -4), "left", "center", REF_C),
]


def _sin_weak_error_panels(axes, rows: list[dict], *, filled_wsc: bool = False):
    """Draw the two sin weak-error panels on ``axes``; return (series, slopes).

    From a point mass CD-EKF and CD-CKF coincide (one node, L at the mean); in
    (b) WSC m=3 coincides with them too. The stochastic Taylor expansion does
    not terminate here, so m=3 and m=5 show their generic orders 2 and 3.
    ``filled_wsc`` draws WSC-KF with solid lines and filled markers, matching
    the filter panels in the combined figure.
    """
    taus = np.array([r["delta"] for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])
    wsc3 = (WSC3, "D-", CURVE_LW, WSC3) if filled_wsc else (WSC3, "D--", CURVE_LW)
    wsc5 = (WSC5, "s-", CURVE_LW, WSC5) if filled_wsc else (WSC5, "s-", CURVE_LW)
    series = [
        (("ckf_err", "ckf_err2"), (CKF_C, "X-", 3.0), "CD-EKF, CD-CKF"),
        (("tme2_err", "tme2_err2"), (TME_C, "P-.", CURVE_LW), "TME-2"),
        (("wsckf_err_m3", "wsckf_err2_m3"), wsc3, r"WSC-KF $m$=3"),
        (("wsckf_err_m5", "wsckf_err2_m5"), wsc5, r"WSC-KF $m$=5"),
    ]
    titles = (
        r"(a) $\mathbb{E}[\cos(\omega x^{(2)}_{\tau})]$",
        r"(b) $\mathbb{E}[(x^{(2)}_{\tau})^{2}]$",
    )
    slopes = []
    for p, (ax, title) in enumerate(zip(axes, titles)):
        slopes.append(_rate_panel(
            ax, taus,
            [(col(keys[p]), *style) for keys, style, _ in series],
            [(2.0, col(series[0][0][p]), "--"), (3.0, col(series[3][0][p]), ":")],
            fit_window=SIN_FIT_WINDOW, xticks=[0.02, 0.1, 0.5], markevery=2, markersize=4.2,
        ))
        ax.set_title(title, fontweight="normal", fontsize=9)
        ax.set_xlabel(r"Interval $\tau$")
    axes[0].set_ylabel("One-step weak error")
    return series, slopes


def plot_sin_weak_error(rows: list[dict]) -> None:
    """One-step weak error on dx² = sin(x¹) dW² from (1, 0), for two functionals."""
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_COLUMN_WIDTH, 2.6), sharey=True)
    series, slopes = _sin_weak_error_panels(axes, rows)
    for (_, _, lab), sa, sb in zip(series, slopes[0], slopes[1]):
        print(f"  sin weak error, {lab}: fitted slopes {sa:.2f} (a) / {sb:.2f} (b)")
    fig.legend(handles=_rate_legend_handles([s[1][:3] for s in series], [s[2] for s in series]),
               loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, columnspacing=0.8)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.17, top=0.68, wspace=0.08)
    save_publication_figure(fig, os.path.join(FIG, "fig_sin_area_weak_error.png"))
    plt.close(fig)


def plot_sin_filter(ax, rows: list[dict]) -> None:
    """Sequential sine-observation NLL on dx² = sin(x¹) dW²."""
    plot_filter_pareto(
        ax, rows, singles=SIN_FILTER_SINGLES, labels=SIN_FILTER_LABELS,
        xlim=(0.01, 100), ylim=(0.9, 2.15), yscale="linear",
    )


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
        r"Method & Paths & $|\Delta\mathbb{E}[(x^{(2)})^2]|$ & $|\Delta\mathbb{E}[\cos(\omega x^{(2)})]|$ \\",
        r"\midrule",
    ]
    for fam, lab, name in entries:
        m, c = _find(moment, fam, lab), _find(char, fam, lab)
        paths = f"{int(m['budget'])}" if fam.startswith(("WSC", "CD-EnKF")) else "--"
        lines.append(rf"{name} & {paths} & {cell(m)} & {cell(c)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(os.path.join(FIG, "tab_sin_area_one_step.tex"), "w") as f:
        f.write("\n".join(lines))


SIN_K100_FILTER_SINGLES = [
    ("CD-EKF", "EKF", "EKF", (7, 0), "left", "center"),
    ("CD-CKF", "CKF", "CKF", (7, 0), "left", "center"),
    ("SE-UKF", "N8", "SE-UKF", (0, 8), "center", "bottom"),
    ("TME", "TME-2", "TME-2", (0, 8), "center", "bottom"),
]

SIN_K100_FILTER_LABELS = [
    ("WSC-KF-m3", "m3-k1", r"$N_\tau$=1", (0, -10), "center", "top", WSC3),
    ("WSC-KF-m3", "m3-k2", "2", (0, -9), "center", "top", WSC3),
    ("WSC-KF-m3", "m3-k3", "3", (8.5, 1.5), "left", "center", WSC3),
    ("WSC-KF-m5", "m5-k1", r"$N_\tau$=1", (-4, 7), "right", "bottom", WSC5),
    ("WSC-KF-m5", "m5-k2", "2", (0, 4), "center", "bottom", WSC5),
    ("PF", "PF-5", r"$n$=5", (8, 0), "left", "center", REF_C),
    ("PF", "PF-500", r"$n$=500", (8, -3), "left", "center", REF_C),
]


def plot_sin_filter_k100(ax, rows: list[dict]) -> None:
    """Sequential sine-observation NLL on dx² = sin(x¹) dW² with K = 100 observations."""
    plot_filter_pareto(
        ax, rows, singles=SIN_K100_FILTER_SINGLES, labels=SIN_K100_FILTER_LABELS,
        xlim=(0.08, 1300), ylim=(0.85, 2.35), yscale="linear",
    )


def plot_sin_weak_error_and_filter(weak_rows: list[dict], filt_rows: list[dict]) -> None:
    """Full width: (a, b) one-step weak error and (c) K = 100 filtering, one shared legend."""
    fig = plt.figure(figsize=(PAPER_TEXT_WIDTH, 2.5))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.32, 1.9], wspace=0.08)
    ax_a = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1], sharey=ax_a)
    ax_b.tick_params(labelleft=False)
    ax_c = fig.add_subplot(grid[3])
    _sin_weak_error_panels((ax_a, ax_b), weak_rows, filled_wsc=True)
    plot_sin_filter_k100(ax_c, filt_rows)
    ax_c.set_title(r"(c) Filtering, $K$ = 100", fontweight="normal", fontsize=9)
    # Log NLL axis gives the crowded region near 1 more room than a linear one.
    ax_c.set_yscale("log")
    ax_c.set_ylim(0.86, 2.35)
    _decimal_ticks(ax_c.yaxis, [1, 1.5, 2])
    handles = _method_legend() + [
        Line2D([0], [0], color=REF_C, lw=1.2, ls="--", label=r"$\propto\tau^{2}$"),
        Line2D([0], [0], color=REF_C, lw=1.2, ls=":", label=r"$\propto\tau^{3}$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               **TIGHT_LEGEND)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.3, top=0.915)
    save_publication_figure(fig, os.path.join(FIG, "fig_sin_area_weak_error_filter.png"))
    plt.close(fig)


def plot_sin_area() -> None:
    plot_sin_weak_error(_load(os.path.join(DATA, "sin_area_weak_error.csv")))
    _filter_figure(plot_sin_filter, _load(os.path.join(DATA, "sin_area_filter_pareto.csv")),
                   ("fig_sin_area_filter.png",))
    _filter_figure(plot_sin_filter_k100,
                   _load(os.path.join(DATA, "sin_area_filter_pareto_K100.csv")),
                   ("fig_sin_area_filter_K100.png",))
    plot_sin_weak_error_and_filter(
        _load(os.path.join(DATA, "sin_area_weak_error.csv")),
        _load(os.path.join(DATA, "sin_area_filter_pareto_K100.csv")),
    )
    write_sin_one_step_table(
        _load(os.path.join(DATA, "sin_area_moment_pareto.csv")),
        _load(os.path.join(DATA, "sin_area_char_pareto.csv")),
    )


def main() -> None:
    plot_levy_weak_error(_load(os.path.join(DATA, "levy_area_weak_error.csv")))
    plot_weak_error(_load(os.path.join(DATA, "heston_weak_error.csv")))
    plot_levy_area()
    plot_sin_area()
    print(f"Wrote figures in {FIG}")


if __name__ == "__main__":
    main()
