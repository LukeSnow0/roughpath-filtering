"""Full sequential Heston filter: NLL and RMSE vs wall-clock Pareto over N obs steps.

Simulated twin with noisy log-price observations. Each method runs the full
assimilation track (N sequential predict–update steps) and is scored by filter
log-likelihood and RMSE of posterior means vs latent truth at obs times.

Run:
  uv run --extra experiments python experiments/heston_test/filter_nll_pareto.py
  uv run --extra experiments python experiments/heston_test/filter_nll_pareto.py \\
      --seeds 100,101,102,103,104 --n-obs 32 --regime stress

Outputs under ``experiments/heston_test/output/filter_nll/``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, replace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax import jit

jax.config.update("jax_enable_x64", True)

from cd_dynamax_mirror.filters import UKFParams
from heston_model import (
    HestonParams,
    init_estimate,
    make_heston_dynamics,
    rmse,
    rmse_components,
    simulate_heston,
)
from mfts_test.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    PALETTE,
    save_publication_figure,
    style_curve_axis,
)
from predictive_nll_stepwise import (
    run_ekf_predictive_nll,
    run_enkf_predictive_nll,
    run_pf_predictive_nll,
    run_ukf_predictive_nll,
    run_wsc_predictive_nll,
)
from klv_particle_filter import run_klv_predictive_nll
from wsckf.library import lv_path_count

OUTPUT_DIR = os.path.join(HERE, "output", "filter_nll")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPATIAL_NODES = 4
REGIMES = {
    "mild": HestonParams(xi=0.3, rho=-0.4, meas_std=0.02),
    "stress": HestonParams(xi=1.0, rho=-0.9, meas_std=0.02),
}
CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)


def _parse_ints(s: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


@dataclass(frozen=True)
class WSCKFConfig:
    family: str
    label: str
    lyons_degree: int
    k: int
    budget: int


def default_wsc_configs() -> list[WSCKFConfig]:
    configs: list[WSCKFConfig] = []
    for m in (3, 5):
        for k in (1, 2, 3):
            wiener = lv_path_count(2, k, lyons_degree=m)
            budget = SPATIAL_NODES * wiener
            configs.append(
                WSCKFConfig(
                    family=f"WSC-KF-m{m}",
                    label=f"m{m}-k{k}",
                    lyons_degree=m,
                    k=k,
                    budget=budget,
                )
            )
    return configs


def _mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    xs = np.asarray([r[key] for r in rows], dtype=np.float64)
    if xs.size == 0:
        return float("nan"), float("nan")
    if xs.size == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1))


def _simulate_track(p: HestonParams, *, n_obs: int, delta: float, seed: int):
    horizon = n_obs * delta
    n_fine = max(n_obs * 8, int(round(horizon / (delta / 4))))
    obs_stride = max(1, n_fine // n_obs)
    true_fine, ys_np, _ = simulate_heston(
        horizon=horizon,
        n_fine=n_fine,
        p=p,
        seed=seed,
        obs_stride=obs_stride,
    )
    true_obs = np.asarray(true_fine, dtype=np.float64)[obs_stride::obs_stride][:n_obs]
    return true_fine, true_obs, jnp.asarray(ys_np)


def _score_rmse(est: np.ndarray, truth: np.ndarray, *, component: str) -> float:
    if component == "all":
        return rmse(est, truth)
    idx = 0 if component == "X" else 1
    return float(rmse_components(est, truth)[idx])


def _stepwise_call(fn, repeats: int = 5):
    out = fn()
    jax.block_until_ready(out)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - t0)
    pred_nll, filt_nll, means = out
    return float(pred_nll), float(filt_nll), np.asarray(means), float(np.median(samples))


def _annotate_budget(ax, pts, xs, ys, *, fam: str):
    """Budget labels — endpoints only (all points if ≤3) to avoid clutter."""
    if not (fam.startswith("WSC-KF") or fam in ("CD-EnKF", "PF", "KLV-PF")):
        return
    color = COLORS.get(fam, "#333333")
    n = len(pts)
    if n == 0:
        return
    if n <= 3:
        idxs = list(range(n))
    else:
        idxs = [0, n - 1]
    for i in idxs:
        pt, x, y = pts[i], xs[i], ys[i]
        if fam == "WSC-KF-m3":
            xytext = (0, 7)
        elif fam == "WSC-KF-m5":
            xytext = (4, -11)
        elif fam == "CD-EnKF":
            xytext = (0, 9) if i == 0 else (4, 9)
        elif fam == "PF":
            xytext = (0, -11) if i == 0 else (4, -11)
        elif fam == "KLV-PF":
            xytext = (0, 10)
        else:
            xytext = (0, 6)
        ax.annotate(
            str(int(pt["budget"])),
            (x, y),
            textcoords="offset points",
            xytext=xytext,
            fontsize=5.8,
            color=color,
            ha="center",
            va="center",
        )


def _plot_metric_pareto(
    summary: list[dict],
    *,
    n_obs: int,
    delta: float,
    regime: str,
    n_seeds: int,
    y_mean_key: str,
    y_std_key: str,
    ylabel: str,
    outfile: str,
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
    clip_band: float | None = None,
    band_scale: float = 1.0,
    band_alpha: float = 0.14,
    offscale_families: tuple[str, ...] = (),
    diff_scale: float = 1.0,
    park_outliers: bool = False,
    shade_regions: list[tuple[float | None, float | None, float | None, float | None, str]] | None = None,
    figsize: tuple[float, float] | None = None,
    x_pad: tuple[float, float] = (1.25, 1.35),
):
    w, h = figsize if figsize is not None else (DOUBLE_COLUMN_WIDTH, 2.85)
    fig, ax = plt.subplots(figsize=(w, h))
    groups = [
        ("WSC-KF m=3", "WSC-KF-m3", lambda r: r["family"] == "WSC-KF-m3"),
        ("WSC-KF m=5", "WSC-KF-m5", lambda r: r["family"] == "WSC-KF-m5"),
        ("CD-EnKF", "CD-EnKF", lambda r: r["family"] == "CD-EnKF"),
        ("PF", "PF", lambda r: r["family"] == "PF"),
        ("KLV-PF", "KLV-PF", lambda r: r["family"] == "KLV-PF"),
        ("CD-EKF", "CD-EKF", lambda r: r["family"] == "CD-EKF"),
        ("CD-UKF", "CD-UKF", lambda r: r["family"] == "CD-UKF"),
        ("CD-CKF", "CD-CKF", lambda r: r["family"] == "CD-CKF"),
    ]
    markers = {
        "WSC-KF m=3": "s",
        "WSC-KF m=5": "D",
        "CD-EKF": "s",
        "CD-UKF": "s",
        "CD-CKF": "s",
        "KLV-PF": "^",
    }
    offscale = set(offscale_families)
    all_xs: list[float] = []
    xlim_xs: list[float] = []  # in-range only — avoid stretching log-x to slow outliers
    # (disp, color_key, x_ms, y_true, budget_or_none)
    offscale_pts: list[tuple[str, str, float, float, int | None]] = []
    y_hi_cut = float(ylim[1]) if (ylim is not None and park_outliers) else None
    # In-range series endpoints for dashed connectors from parked low-n points.
    series_first_inrange: dict[str, tuple[float, float]] = {}

    for disp, color_key, pred in groups:
        pts = sorted([r for r in summary if pred(r)], key=lambda r: r["budget"])
        if not pts:
            continue
        xs = np.asarray([1e3 * r["time_s_mean"] for r in pts])
        ys = np.asarray([r[y_mean_key] for r in pts])
        ystd_raw = np.asarray([r[y_std_key] for r in pts], dtype=np.float64)
        xstd = np.asarray(
            [1e3 * float(r.get("time_s_std", 0.0)) for r in pts], dtype=np.float64
        )
        ystd = ystd_raw * float(band_scale)
        if clip_band is not None:
            ystd = np.minimum(ystd, float(clip_band))
        all_xs.extend(xs.tolist())
        color = COLORS[color_key]
        marker = markers.get(disp, "o")

        if color_key in offscale:
            for x, y, pt in zip(xs, ys, pts):
                offscale_pts.append(
                    (disp, color_key, float(x), float(y), int(pt["budget"]))
                )
            # Legend handle only (actual markers drawn at top of frame later).
            ax.plot(
                [],
                [],
                marker,
                color=color,
                markerfacecolor="white",
                markersize=4.5,
                label=disp,
            )
            continue

        # Optionally park individual high outliers (keeps low-NLL cluster readable).
        if y_hi_cut is not None:
            keep = ys <= y_hi_cut
            for x, y, pt in zip(xs[~keep], ys[~keep], [p for p, k in zip(pts, keep) if not k]):
                offscale_pts.append(
                    (disp, color_key, float(x), float(y), int(pt["budget"]))
                )
            if not np.any(keep):
                ax.plot(
                    [],
                    [],
                    marker,
                    color=color,
                    markerfacecolor="white",
                    markersize=4.5,
                    label=disp,
                )
                continue
            pts = [p for p, k in zip(pts, keep) if k]
            xs, ys, ystd, ystd_raw, xstd = (
                xs[keep],
                ys[keep],
                ystd[keep],
                ystd_raw[keep],
                xstd[keep],
            )
            series_first_inrange[color_key] = (float(xs[0]), float(ys[0]))

        xlim_xs.extend(float(x) for x in xs)

        if len(pts) == 1:
            x0 = float(xs[0])
            ax.fill_between(
                [x0 / 1.04, x0 * 1.04],
                [ys[0] - ystd[0]] * 2,
                [ys[0] + ystd[0]] * 2,
                color=color,
                alpha=band_alpha,
                lw=0,
            )
            ax.errorbar(
                xs,
                ys,
                xerr=xstd * float(band_scale),
                yerr=ystd,
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                alpha=0.75,
                zorder=3,
            )
            ax.plot(xs, ys, marker, color=color, markerfacecolor="white", markersize=4.5, label=disp)
        else:
            ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=band_alpha, lw=0)
            ax.errorbar(
                xs,
                ys,
                xerr=xstd * float(band_scale),
                yerr=ystd,
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                alpha=0.75,
                zorder=3,
            )
            ax.plot(
                xs,
                ys,
                f"-{marker}",
                color=color,
                markerfacecolor="white",
                lw=1.25,
                markersize=4.2,
                label=disp,
            )
        _annotate_budget(ax, pts, xs, ys, fam=color_key)

    ax.set_xlabel("Wall-clock time [ms]")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")
    xs_for_lim = xlim_xs if xlim_xs else all_xs
    if xs_for_lim:
        pad_lo, pad_hi = x_pad
        ax.set_xlim(min(xs_for_lim) / pad_lo, max(xs_for_lim) * pad_hi)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if shade_regions:
        xl, xr = ax.get_xlim()
        yb, yt = ax.get_ylim()
        for x_lo_s, x_hi_s, y_lo_s, y_hi_s, color_key in shade_regions:
            color = COLORS.get(color_key, color_key)
            x0 = float(xl if x_lo_s is None else x_lo_s)
            x1 = float(xr if x_hi_s is None else x_hi_s)
            y0 = float(yb if y_lo_s is None else y_lo_s)
            y1 = float(yt if y_hi_s is None else y_hi_s)
            x0, x1 = max(min(x0, x1), xl), min(max(x0, x1), xr)
            y0, y1 = max(min(y0, y1), yb), min(max(y0, y1), yt)
            if x1 <= x0 or y1 <= y0:
                continue
            ax.fill_between(
                [x0, x1],
                [y0, y0],
                [y1, y1],
                color=color,
                alpha=0.12,
                lw=0,
                zorder=0,
                label="_nolegend_",
            )
            ax.plot([x1, x1], [y0, y1], color=color, lw=0.8, alpha=0.45, ls="--", zorder=0)
            ax.plot([x0, x1], [y0, y0], color=color, lw=0.8, alpha=0.45, ls="--", zorder=0)
            ax.plot([x0, x1], [y1, y1], color=color, lw=0.8, alpha=0.45, ls="--", zorder=0)
            ax.plot([x0, x0], [y0, y1], color=color, lw=0.8, alpha=0.45, ls="--", zorder=0)

    # Indicate off-scale methods at the top of the zoomed y-range.
    if offscale_pts and ylim is not None:
        y_lo, y_hi = float(ylim[0]), float(ylim[1])
        y_mark = y_hi - 0.04 * (y_hi - y_lo)
        xl, xr = ax.get_xlim()

        def _place_x(x_ms: float, occupied: list[float]) -> float:
            x_draw = float(np.clip(x_ms, xl * 1.02, xr / 1.02))
            for _ in range(12):
                if all(abs(np.log10(x_draw) - np.log10(p)) >= 0.12 for p in occupied):
                    break
                x_draw *= 1.28
                if x_draw >= xr:
                    x_draw = max(xl * 1.05, xr / (1.2 ** (len(occupied) + 1)))
            return min(x_draw, xr)

        # One compact label per family (avoid a forest of ↑ texts).
        by_fam: dict[str, list[tuple[str, str, float, float, int | None]]] = {}
        for pt in offscale_pts:
            by_fam.setdefault(pt[1], []).append(pt)

        fam_order = ("CD-UKF", "CD-CKF", "CD-EnKF", "PF", "KLV-PF", "WSC-KF-m3", "WSC-KF-m5")
        occupied_x: list[float] = []
        for fi, fam in enumerate(fam_order):
            pts_f = by_fam.get(fam)
            if not pts_f:
                continue
            color = COLORS[fam]
            marker = markers.get(
                next(d for d, c, _ in groups if c == fam),
                "o",
            )
            # Representative x: geometric mean of member times (clipped).
            x_rep = float(np.exp(np.mean(np.log([max(p[2], 1e-9) for p in pts_f]))))
            x_draw = _place_x(x_rep, occupied_x)
            occupied_x.append(x_draw)
            ax.plot(
                x_draw,
                y_mark,
                marker,
                color=color,
                markerfacecolor="white",
                markersize=5.0,
                zorder=6,
                clip_on=False,
            )
            budgets = sorted({p[4] for p in pts_f if p[4] is not None})
            y_true = float(np.median([p[3] for p in pts_f]))
            if fam in ("CD-EnKF", "PF", "KLV-PF") and budgets:
                if len(budgets) == 1:
                    txt = f"{budgets[0]}↑"
                else:
                    txt = f"{budgets[0]}–{budgets[-1]}↑"
            else:
                txt = rf"{y_true:.1f}$\uparrow$"
            dy = 7 + 9 * (fi % 2)
            ax.annotate(
                txt,
                (x_draw, y_mark),
                textcoords="offset points",
                xytext=(0, dy),
                fontsize=5.6,
                color=color,
                ha="center",
                va="bottom",
                clip_on=False,
            )

        # One dashed link per EnKF/PF family from parked cluster into the front.
        for fam in ("CD-EnKF", "PF"):
            if fam not in series_first_inrange or fam not in by_fam:
                continue
            x_ms = float(np.median([p[2] for p in by_fam[fam]]))
            color = COLORS[fam]
            x1, y1 = series_first_inrange[fam]
            ax.plot(
                [min(x_ms, xr), x1],
                [y_mark, y1],
                "--",
                color=color,
                lw=0.9,
                alpha=0.45,
                zorder=1,
            )

    style_curve_axis(ax)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.32, top=0.94)
    out = os.path.join(OUTPUT_DIR, outfile)
    save_publication_figure(fig, out)
    plt.close(fig)
    print(f"Wrote {out}")


def _pred_nll_focus_ylim(summary: list[dict]) -> tuple[float, float, tuple[str, ...]]:
    """Zoom onto the low-NLL cluster; return ylim + families entirely above it."""
    vals = [float(r["pred_nll_per_obs_mean"]) for r in summary]
    best = min(vals)
    y_lo = best - 0.08
    y_hi = -0.5
    off: list[str] = []
    for fam in ("CD-UKF", "CD-CKF", "KLV-PF"):
        fam_vals = [float(r["pred_nll_per_obs_mean"]) for r in summary if r["family"] == fam]
        if fam_vals and min(fam_vals) > y_hi:
            off.append(fam)
    return y_lo, y_hi, tuple(off)


def _plot_all_paretos(
    summary: list[dict],
    *,
    n_obs: int,
    delta: float,
    regime: str,
    n_seeds: int,
    rmse_comp: str,
    diff_scale: float = 1.0,
):
    _plot_metric_pareto(
        summary,
        n_obs=n_obs,
        delta=delta,
        regime=regime,
        n_seeds=n_seeds,
        y_mean_key="nll_per_obs_mean",
        y_std_key="nll_per_obs_std",
        ylabel="NLL [nats / observation]",
        outfile="filter_nll_pareto.png",
        band_scale=0.35,
        clip_band=0.04,
        band_alpha=0.10,
        diff_scale=diff_scale,
    )
    # Zoom onto competitive (low-NLL) cluster; park high outliers at the top edge.
    # (Log y-scale is not usable: predictive NLL is typically negative.)
    pred_lo, pred_hi, off_fams = _pred_nll_focus_ylim(summary)
    _plot_metric_pareto(
        summary,
        n_obs=n_obs,
        delta=delta,
        regime=regime,
        n_seeds=n_seeds,
        y_mean_key="pred_nll_per_obs_mean",
        y_std_key="pred_nll_per_obs_std",
        ylabel="Predictive NLL [nats / observation]",
        outfile="filter_pred_nll_pareto.png",
        ylim=(pred_lo, pred_hi),
        band_scale=0.55,
        clip_band=0.06,
        band_alpha=0.10,
        offscale_families=off_fams,
        park_outliers=True,
        diff_scale=diff_scale,
        # Three regions for the Pareto story (15-seed means):
        #   left  — Gaussian filters (cheap, higher / parked NLL)
        #   mid   — efficient WSC-KF front (beats Gaussians, cheaper than large EnKF/PF)
        #   right — EnKF (≥100) / large PF (accurate, expensive)
        shade_regions=[
            (None, 7.0, -1.42, None, "CD-EKF"),
            (18.0, 1.1e2, None, -1.48, "WSC-KF-m3"),
            (1.3e2, None, None, -1.55, "PF"),
        ],
        figsize=(0.78 * DOUBLE_COLUMN_WIDTH, 2.85),
        x_pad=(1.15, 1.25),
    )
    rmse_labels = {
        "all": "RMSE (overall state)",
        "X": r"RMSE ($X=\log S$)",
        "v": r"RMSE ($v$)",
    }
    _plot_metric_pareto(
        summary,
        n_obs=n_obs,
        delta=delta,
        regime=regime,
        n_seeds=n_seeds,
        y_mean_key="rmse_mean",
        y_std_key="rmse_std",
        ylabel=rmse_labels[rmse_comp],
        outfile="filter_rmse_pareto.png",
        yscale="log" if rmse_comp == "v" else "linear",
        band_scale=0.35,
        clip_band=0.01,
        band_alpha=0.10,
        diff_scale=diff_scale,
    )


COLORS = {
    "WSC-KF-m3": "#7B2CBF",
    "WSC-KF-m5": "#9B5DE5",
    "CD-EnKF": METHOD_COLORS["CD-EnKF"],
    "PF": "#C1121F",
    "KLV-PF": "#F77F00",
    "CD-EKF": METHOD_COLORS["CD-EKF"],
    "CD-UKF": METHOD_COLORS["CD-UKF-coarse"],
    "CD-CKF": METHOD_COLORS["CD-CKF"],
}


def _summarize(raw_rows: list[dict], args) -> list[dict]:
    summary: list[dict] = []
    keys = sorted({(r["family"], r["label"], int(r["budget"])) for r in raw_rows})
    for fam, lab, bud in keys:
        group = [
            r
            for r in raw_rows
            if r["family"] == fam and r["label"] == lab and int(r["budget"]) == bud
        ]
        nll_m, nll_s = _mean_std(group, "nll_per_obs")
        pred_m, pred_s = _mean_std(group, "pred_nll_per_obs")
        rmse_m, rmse_s = _mean_std(group, "rmse")
        t_m, t_s = _mean_std(group, "time_s")
        summary.append(
            {
                "regime": args.regime if hasattr(args, "regime") else group[0]["regime"],
                "n_obs": int(group[0]["n_obs"]),
                "delta": float(group[0]["delta"]),
                "rmse_comp": group[0].get("rmse_comp", getattr(args, "rmse_comp", "v")),
                "family": fam,
                "label": lab,
                "budget": bud,
                "n_seeds": len(group),
                "nll_per_obs_mean": nll_m,
                "nll_per_obs_std": nll_s,
                "pred_nll_per_obs_mean": pred_m,
                "pred_nll_per_obs_std": pred_s,
                "rmse_mean": rmse_m,
                "rmse_std": rmse_s,
                "time_s_mean": t_m,
                "time_s_std": t_s,
            }
        )
    return summary


def run_benchmark(args) -> tuple[list[dict], list[dict]]:
    p = replace(REGIMES[args.regime], diff_scale=float(args.diff_scale))
    seeds = _parse_ints(args.seeds)
    enkf_list = _parse_ints(args.enkf)
    pf_list = _parse_ints(args.pf)
    klv_list = _parse_ints(args.klv) if str(args.klv).strip() else []
    wsc_configs = default_wsc_configs()
    if args.wsc_max_budget > 0:
        wsc_configs = [c for c in wsc_configs if c.budget <= args.wsc_max_budget]

    dt = float(args.delta)
    dt0 = dt / float(args.inner_div)
    n_obs = int(args.n_obs)
    raw_rows: list[dict] = []

    print(
        f"Heston filter NLL Pareto: regime={args.regime}, diff_scale={p.diff_scale:g}, "
        f"N={n_obs}, Δ={dt:g}, "
        f"seeds={seeds}, WSC={'skip' if args.skip_wsc or args.klv_only else [c.label for c in wsc_configs]}, "
        f"CD={'skip' if args.skip_cd or args.klv_only else 'EKF/UKF/CKF'}, "
        f"EnKF={enkf_list or 'skip'}, PF={pf_list or 'skip'}, KLV={klv_list or 'skip'}"
    )

    run_wsc = not args.klv_only and not args.skip_wsc
    run_cd = not args.klv_only and not args.skip_cd
    run_enkf = not args.klv_only and bool(enkf_list) and not args.skip_enkf
    run_pf = not args.klv_only and bool(pf_list) and not args.skip_pf

    n_seeds = len(seeds)
    n_jobs_per_seed = (
        (len(wsc_configs) if run_wsc else 0)
        + (3 if run_cd else 0)
        + (len(enkf_list) if run_enkf else 0)
        + (len(pf_list) if run_pf else 0)
        + len(klv_list)
    )
    print(
        f"[progress] {n_seeds} seeds × ~{n_jobs_per_seed} configs/seed "
        f"(~{n_seeds * n_jobs_per_seed} filter runs)",
        flush=True,
    )
    t_run0 = time.perf_counter()
    seed_durations: list[float] = []

    for seed_i, seed in enumerate(seeds, start=1):
        t_seed0 = time.perf_counter()
        print(
            f"\n=== seed {seed}  [{seed_i}/{n_seeds}] ===",
            flush=True,
        )
        true_fine, true_obs, ys = _simulate_track(p, n_obs=n_obs, delta=dt, seed=seed)
        m0_np, P0_np = init_estimate(p, true_state=true_fine[0], cov_scale=1.0)
        m0 = jnp.asarray(m0_np, dtype=jnp.float64)
        P0 = jnp.asarray(P0_np, dtype=jnp.float64)
        chol_P0 = jnp.linalg.cholesky(P0)
        drift, L_fn, Qc_fn, h_obs, R = make_heston_dynamics(p)
        chol_R = jnp.linalg.cholesky(R)
        job_i = 0

        def _tick(stage: str):
            nonlocal job_i
            job_i += 1
            elapsed = time.perf_counter() - t_run0
            done_seeds = seed_i - 1
            if seed_durations:
                avg = float(np.mean(seed_durations))
                eta = avg * (n_seeds - done_seeds) - (time.perf_counter() - t_seed0)
                eta = max(0.0, eta)
                eta_s = f"ETA~{eta / 60.0:.1f}m"
            else:
                eta_s = "ETA~…"
            print(
                f"  [progress {seed_i}/{n_seeds} job {job_i}/{n_jobs_per_seed}] "
                f"{stage}  elapsed={elapsed / 60.0:.1f}m  {eta_s}",
                flush=True,
            )

        def record(family, label, budget, pred_nll, filt_nll, means, wall_s):
            est = np.asarray(means, dtype=np.float64)
            err = _score_rmse(est, true_obs, component=args.rmse_comp)
            raw_rows.append(
                {
                    "seed": seed,
                    "regime": args.regime,
                    "n_obs": n_obs,
                    "delta": dt,
                    "rmse_comp": args.rmse_comp,
                    "family": family,
                    "label": label,
                    "budget": budget,
                    "nll_per_obs": filt_nll,
                    "pred_nll_per_obs": pred_nll,
                    "rmse": err,
                    "time_s": wall_s,
                }
            )
            print(
                f"  {family:12s} {label:12s} budget={budget:4d}  "
                f"NLL/obs={filt_nll:8.4f}  predNLL={pred_nll:8.4f}  "
                f"RMSE={err:8.5f}  time={1e3 * wall_s:8.1f} ms",
                flush=True,
            )

        if run_wsc:
            for cfg in wsc_configs:
                _tick(f"WSC {cfg.label}")
                wsc_jit = jit(
                    lambda ly=cfg.lyons_degree, kk=cfg.k: run_wsc_predictive_nll(
                        drift=drift,
                        L_fn=L_fn,
                        h=h_obs,
                        R=R,
                        chol_R=chol_R,
                        ys=ys,
                        m0=m0,
                        chol_P0=chol_P0,
                        dt=dt,
                        dt0=dt0,
                        lyons_degree=ly,
                        k=kk,
                    )
                )
                pred_nll, filt_nll, means, wall = _stepwise_call(
                    wsc_jit, repeats=args.n_timing
                )
                record(cfg.family, cfg.label, cfg.budget, pred_nll, filt_nll, means, wall)

        if run_cd:
            _tick("CD-EKF")
            ekf_jit = jit(
                lambda: run_ekf_predictive_nll(
                    drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
                    ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0,
                )
            )
            pred_nll, filt_nll, means, wall = _stepwise_call(ekf_jit, repeats=args.n_timing)
            record("CD-EKF", "EKF", 0, pred_nll, filt_nll, means, wall)

            _tick("CD-UKF")
            ukf_jit = jit(
                lambda: run_ukf_predictive_nll(
                    drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
                    ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0, ukf_params=UKFParams(),
                )
            )
            pred_nll, filt_nll, means, wall = _stepwise_call(ukf_jit, repeats=args.n_timing)
            record("CD-UKF", "UKF", 0, pred_nll, filt_nll, means, wall)

            _tick("CD-CKF")
            ckf_jit = jit(
                lambda: run_ukf_predictive_nll(
                    drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
                    ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0, ukf_params=CKF_PARAMS,
                )
            )
            pred_nll, filt_nll, means, wall = _stepwise_call(ckf_jit, repeats=args.n_timing)
            record("CD-CKF", "CKF", 0, pred_nll, filt_nll, means, wall)

        key = jr.key(seed)
        if run_enkf:
            for n_part in enkf_list:
                _tick(f"EnKF n={n_part}")
                key, sub = jr.split(key)
                enkf_jit = jit(
                    lambda k, n=n_part: run_enkf_predictive_nll(
                        drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
                        ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0, key=k, n_particles=n,
                    )
                )
                pred_nll, filt_nll, means, wall = _stepwise_call(
                    lambda k=sub: enkf_jit(k), repeats=args.n_timing
                )
                record("CD-EnKF", f"EnKF-{n_part}", n_part, pred_nll, filt_nll, means, wall)

        if run_pf:
            for n_part in pf_list:
                _tick(f"PF n={n_part}")
                key, sub = jr.split(key)
                pf_jit = jit(
                    lambda k, n=n_part: run_pf_predictive_nll(
                        drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
                        ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0, key=k, n_particles=n,
                    )
                )
                pred_nll, filt_nll, means, wall = _stepwise_call(
                    lambda k=sub: pf_jit(k), repeats=args.n_timing
                )
                record("PF", f"PF-{n_part}", n_part, pred_nll, filt_nll, means, wall)

        for n_part in klv_list:
            _tick(f"KLV-PF n={n_part}")
            pred_nll, filt_nll, means, wall = run_klv_predictive_nll(
                p=p,
                h=h_obs,
                R=R,
                ys=ys,
                m0=m0_np,
                P0=P0_np,
                dt=dt,
                n_particles=n_part,
                seed=seed + 17,
                n_macro=int(args.klv_macro),
                lyons_degree=int(args.klv_lyons),
                cubature_order=int(args.klv_k),
            )
            record("KLV-PF", f"KLV-{n_part}", n_part, pred_nll, filt_nll, means, wall)

        seed_dt = time.perf_counter() - t_seed0
        seed_durations.append(seed_dt)
        avg = float(np.mean(seed_durations))
        left = n_seeds - seed_i
        print(
            f"[progress] seed {seed} done in {seed_dt / 60.0:.1f}m  "
            f"({seed_i}/{n_seeds}, {100.0 * seed_i / n_seeds:.0f}%)  "
            f"avg/seed={avg / 60.0:.1f}m  ETA~{avg * left / 60.0:.1f}m",
            flush=True,
        )

    return raw_rows, _summarize(raw_rows, args)


def _coerce_raw_row(r: dict) -> dict:
    out = dict(r)
    out["seed"] = int(out["seed"])
    out["n_obs"] = int(float(out["n_obs"]))
    out["delta"] = float(out["delta"])
    out["budget"] = int(float(out["budget"]))
    for k in ("nll_per_obs", "pred_nll_per_obs", "rmse", "time_s"):
        out[k] = float(out[k])
    return out


def _merge_raw(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Replace rows with the same (seed, family, label, budget)."""
    def key(r):
        return (int(r["seed"]), r["family"], r["label"], int(r["budget"]))

    merged = {key(r): _coerce_raw_row(r) for r in existing}
    for r in new_rows:
        merged[key(r)] = r
    return list(merged.values())


def _write_csv(raw_rows: list[dict], summary: list[dict]) -> None:
    raw_path = os.path.join(OUTPUT_DIR, "filter_nll_pareto_raw.csv")
    csv_path = os.path.join(OUTPUT_DIR, "filter_nll_pareto.csv")
    if raw_rows:
        fields = sorted({k for r in raw_rows for k in r})
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in raw_rows:
                w.writerow(r)
    if summary:
        fields = sorted({k for r in summary for k in r})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sorted(summary, key=lambda x: (x["family"], x["budget"], x["label"])):
                w.writerow(r)
    print(f"Wrote {raw_path}")
    print(f"Wrote {csv_path}")


def _read_summary_csv() -> list[dict]:
    path = os.path.join(OUTPUT_DIR, "filter_nll_pareto.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("n_obs", "budget", "n_seeds"):
            if k in r:
                r[k] = int(float(r[k]))
        for k in (
            "delta",
            "nll_per_obs_mean",
            "nll_per_obs_std",
            "pred_nll_per_obs_mean",
            "pred_nll_per_obs_std",
            "rmse_mean",
            "rmse_std",
            "time_s_mean",
            "time_s_std",
        ):
            if k in r:
                r[k] = float(r[k])
    return rows


def _read_raw_csv() -> list[dict]:
    path = os.path.join(OUTPUT_DIR, "filter_nll_pareto_raw.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Heston full-filter NLL/RMSE vs time Pareto")
    parser.add_argument("--seeds", type=str, default="100,101,102,103,104")
    parser.add_argument("--n-obs", type=int, default=32, dest="n_obs")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--inner-div", type=int, default=256, dest="inner_div")
    parser.add_argument("--regime", choices=tuple(REGIMES), default="stress")
    parser.add_argument(
        "--diff-scale",
        type=float,
        default=1.0,
        dest="diff_scale",
        help="Multiplies diffusion L (process-noise power); default 1",
    )
    parser.add_argument("--enkf", type=str, default="500,1000,2000,5000")
    parser.add_argument("--pf", type=str, default="500,1000,2000,5000")
    parser.add_argument(
        "--klv",
        type=str,
        default="500,1000,2000",
        help="Comma-separated KLV-PF particle counts (empty = skip)",
    )
    parser.add_argument(
        "--klv-macro",
        type=int,
        default=8,
        dest="klv_macro",
        help="KLV cubature macro-steps over each Δ",
    )
    parser.add_argument(
        "--klv-lyons",
        type=int,
        default=5,
        dest="klv_lyons",
        help="Lyons–Victoir degree m for KLV local cubature",
    )
    parser.add_argument(
        "--klv-k",
        type=int,
        default=1,
        dest="klv_k",
        help="Cubature order k for KLV local paths",
    )
    parser.add_argument(
        "--klv-only",
        action="store_true",
        dest="klv_only",
        help="Only run KLV-PF (implies --append)",
    )
    parser.add_argument("--skip-wsc", action="store_true", dest="skip_wsc")
    parser.add_argument("--skip-cd", action="store_true", dest="skip_cd", help="Skip EKF/UKF/CKF")
    parser.add_argument("--skip-enkf", action="store_true", dest="skip_enkf")
    parser.add_argument("--skip-pf", action="store_true", dest="skip_pf")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge into existing raw CSV (replace matching seed/family/label)",
    )
    parser.add_argument(
        "--rmse-comp",
        type=str,
        default="v",
        choices=("all", "X", "v"),
        dest="rmse_comp",
        help="State component for RMSE Pareto (default: latent variance v)",
    )
    parser.add_argument(
        "--wsc-max-budget",
        type=int,
        default=0,
        dest="wsc_max_budget",
        help="Skip WSC configs with node×path budget above this (0 = no cap)",
    )
    parser.add_argument("--n-timing", type=int, default=5, dest="n_timing")
    parser.add_argument("--plot-only", action="store_true", help="Replot from summary CSV")
    args = parser.parse_args()
    if args.klv_only:
        args.append = True

    if args.plot_only:
        summary = _read_summary_csv()
        if not summary:
            raise SystemExit("No summary CSV found; run benchmark first.")
        n_seeds = max(r["n_seeds"] for r in summary)
        rmse_comp = summary[0].get("rmse_comp", args.rmse_comp)
        _plot_all_paretos(
            summary,
            n_obs=summary[0]["n_obs"],
            delta=summary[0]["delta"],
            regime=summary[0]["regime"],
            n_seeds=n_seeds,
            rmse_comp=rmse_comp,
            diff_scale=float(args.diff_scale),
        )
        return

    raw_rows, summary = run_benchmark(args)
    if args.append:
        existing = _read_raw_csv()
        if existing:
            raw_rows = _merge_raw(existing, raw_rows)
            summary = _summarize(raw_rows, args)
            print(f"Append-merged into existing raw (now {len(raw_rows)} rows)")
    _write_csv(raw_rows, summary)
    n_seeds = max(r["n_seeds"] for r in summary)
    _plot_all_paretos(
        summary,
        n_obs=args.n_obs,
        delta=args.delta,
        regime=args.regime,
        n_seeds=n_seeds,
        rmse_comp=args.rmse_comp,
        diff_scale=float(args.diff_scale),
    )


if __name__ == "__main__":
    main()
