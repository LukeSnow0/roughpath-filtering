"""Lyons–Victoir explicit paths on multiplicative-noise GBM (point-mass predict).

Isolates the Wiener-cubature rate on a model where path shape / Lévy area matter
(state-dependent diffusion), with an **exact** moment reference — no MC noise.

* 1D GBM: ``dx = μ x dt + σ x dW``
* 2D independent GBM: diagonal ``μ, σ`` (noise fields commute; drift×diffusion
  brackets still make the √22 time-warping visible)

Compares explicit LV degrees ``m ∈ {3, 5}`` (``multidim_method="lv"`` in
``wsckf.library``; 1D uses the classical formulae). Theory: weak error
``O(δ^{(m+1)/2})``.

Run:
  uv run --extra experiments python experiments/multiplicative_noise_test/weak_error_lv_paths.py
  uv run --extra experiments python experiments/multiplicative_noise_test/weak_error_lv_paths.py --quick
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (EXPERIMENTS_DIR, ROOT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from wsckf.library import default_num_paths, precompute_cubature_paths, predict
from wsckf.library.conversions import ito_to_stratonovich_dynamics
from mfts_test.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    PALETTE,
    save_publication_figure,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MU = 0.5
SIGMA = 0.6
X0 = 1.0
MU2 = np.array([0.5, -0.3])
SIGMA2 = np.array([0.6, 0.4])
X02 = np.array([1.0, 1.0])
INNER_STEPS = 64


def exact_mean(delta, x0=X0, mu=MU):
    return x0 * np.exp(mu * delta)


def exact_var(delta, x0=X0, mu=MU, sigma=SIGMA):
    return x0**2 * np.exp(2.0 * mu * delta) * (np.exp(sigma**2 * delta) - 1.0)


def exact_mean_2d(delta):
    return X02 * np.exp(MU2 * delta)


def exact_cov_2d(delta):
    # Independent coordinates → diagonal covariance.
    v = X02**2 * np.exp(2.0 * MU2 * delta) * (np.exp(SIGMA2**2 * delta) - 1.0)
    return np.diag(v)


def make_gbm_1d(mu=MU, sigma=SIGMA):
    def ito_drift(t, x):
        del t
        return mu * x

    def diffusion(t, x):
        del t
        return (sigma * x).reshape(1, 1)

    return ito_drift, diffusion


def make_gbm_2d():
    mu = jnp.asarray(MU2)
    sig = jnp.asarray(SIGMA2)

    def ito_drift(t, x):
        del t
        return mu * x

    def diffusion(t, x):
        del t
        return jnp.diag(sig * x)

    return ito_drift, diffusion


ITO_1D, DIFF_1D = make_gbm_1d()
STRAT_DRIFT_1D, STRAT_DIFF_1D = ito_to_stratonovich_dynamics(ITO_1D, DIFF_1D)
ITO_2D, DIFF_2D = make_gbm_2d()
STRAT_DRIFT_2D, STRAT_DIFF_2D = ito_to_stratonovich_dynamics(ITO_2D, DIFF_2D)


def lv_predict(delta, noise_dim, lyons_degree, mean0, strat_drift, strat_diff):
    ts, paths, weights = precompute_cubature_paths(
        horizon=float(delta),
        noise_dim=noise_dim,
        integration_dt=float(delta) / INNER_STEPS,
        cubature_order=1,
        multidim_method="lv",
        lyons_degree=lyons_degree,
    )
    mean = jnp.asarray(mean0, dtype=jnp.float64)
    chol = jnp.zeros((mean.shape[0], mean.shape[0]), dtype=jnp.float64)
    m_pred, chol_pred, _, _ = predict(
        mean,
        chol,
        jnp.asarray(ts),
        jnp.asarray(paths),
        jnp.asarray(weights),
        strat_drift,
        strat_diff,
    )
    P = chol_pred @ chol_pred.T
    return np.asarray(m_pred), np.asarray(P), int(paths.shape[0])


def fit_slope(deltas, errs, dmax=0.4, floor=1e-13):
    d = np.asarray(deltas, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    mask = (d <= dmax) & (e > floor)
    if mask.sum() < 2:
        mask = e > floor
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(d[mask]), np.log(e[mask]), 1)[0])


def guide(
    ax,
    deltas,
    anchor_err,
    anchor_d,
    slope,
    label,
    *,
    color,
    linestyle,
):
    d = np.asarray(deltas)
    c = anchor_err / (anchor_d**slope)
    return ax.loglog(
        d,
        c * d**slope,
        color=color,
        linestyle=linestyle,
        alpha=0.72,
        lw=1.0,
        label=label,
    )[0]


def parse_args():
    p = argparse.ArgumentParser(description="LV path weak-error scaling on GBM")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        deltas = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    else:
        deltas = np.logspace(np.log10(0.03), np.log10(1.0), 12)

    degrees = (3, 5)
    floor = 1e-15

    for d in (1, 2):
        counts = ", ".join(f"m={m}: {default_num_paths(d, 1, 'lv', lyons_degree=m)}" for m in degrees)
        print(f"LV paths (d={d}): {counts}")
    print(f"GBM multiplicative | deltas={len(deltas)} | INNER_STEPS={INNER_STEPS}")
    print("-" * 72)

    # ---- 1D ----
    err_mean_1d = {m: [] for m in degrees}
    err_var_1d = {m: [] for m in degrees}
    n_paths_1d = {}
    for d in deltas:
        tm, tv = exact_mean(d), exact_var(d)
        for m in degrees:
            mp, Pp, npth = lv_predict(
                d, 1, m, [X0], STRAT_DRIFT_1D, STRAT_DIFF_1D
            )
            n_paths_1d[m] = npth
            err_mean_1d[m].append(abs(float(mp[0]) - tm))
            err_var_1d[m].append(abs(float(Pp[0, 0]) - tv))
    for m in degrees:
        err_mean_1d[m] = np.asarray(err_mean_1d[m])
        err_var_1d[m] = np.asarray(err_var_1d[m])

    # ---- 2D ----
    err_mean_2d = {m: [] for m in degrees}
    err_cov_2d = {m: [] for m in degrees}
    n_paths_2d = {}
    for d in deltas:
        tm = exact_mean_2d(d)
        tP = exact_cov_2d(d)
        for m in degrees:
            mp, Pp, npth = lv_predict(
                d, 2, m, X02, STRAT_DRIFT_2D, STRAT_DIFF_2D
            )
            n_paths_2d[m] = npth
            err_mean_2d[m].append(float(np.linalg.norm(mp - tm)))
            err_cov_2d[m].append(float(np.linalg.norm(Pp - tP, ord="fro")))
    for m in degrees:
        err_mean_2d[m] = np.asarray(err_mean_2d[m])
        err_cov_2d[m] = np.asarray(err_cov_2d[m])

    print("1D GBM mean / var weak-error slopes (theory (m+1)/2):")
    for m in degrees:
        print(
            f"  m={m}  P={n_paths_1d[m]:2d}  "
            f"mean={fit_slope(deltas, err_mean_1d[m]):.2f}  "
            f"var={fit_slope(deltas, err_var_1d[m]):.2f}  "
            f"(theory {0.5 * (m + 1):.1f})"
        )
    print("2D GBM mean / cov weak-error slopes:")
    for m in degrees:
        print(
            f"  m={m}  P={n_paths_2d[m]:2d}  "
            f"mean={fit_slope(deltas, err_mean_2d[m]):.2f}  "
            f"cov={fit_slope(deltas, err_cov_2d[m]):.2f}  "
            f"(theory {0.5 * (m + 1):.1f})"
        )

    # ----- CSV -----
    csv_path = os.path.join(OUTPUT_DIR, "weak_error_lv_paths.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["delta"]
        for m in degrees:
            header += [
                f"err_mean_1d_m{m}",
                f"err_var_1d_m{m}",
                f"err_mean_2d_m{m}",
                f"err_cov_2d_m{m}",
            ]
        w.writerow(header)
        for i, d in enumerate(deltas):
            row = [float(d)]
            for m in degrees:
                row += [
                    err_mean_1d[m][i],
                    err_var_1d[m][i],
                    err_mean_2d[m][i],
                    err_cov_2d[m][i],
                ]
            w.writerow(row)

    slope_csv = os.path.join(OUTPUT_DIR, "weak_error_lv_paths_slopes.csv")
    with open(slope_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "setting",
                "lyons_degree",
                "n_paths",
                "theory_slope",
                "fitted_mean",
                "fitted_second",
            ]
        )
        for m in degrees:
            w.writerow(
                [
                    "1d_gbm",
                    m,
                    n_paths_1d[m],
                    0.5 * (m + 1),
                    fit_slope(deltas, err_mean_1d[m]),
                    fit_slope(deltas, err_var_1d[m]),
                ]
            )
            w.writerow(
                [
                    "2d_gbm",
                    m,
                    n_paths_2d[m],
                    0.5 * (m + 1),
                    fit_slope(deltas, err_mean_2d[m]),
                    fit_slope(deltas, err_cov_2d[m]),
                ]
            )

    # ----- plot -----
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 4.60),
        sharex=True,
    )
    colors = {3: PALETTE["warm_red"], 5: PALETTE["teal_green"]}
    linestyles = {3: "-", 5: "--"}
    markers = {3: "o", 5: "s"}

    panels = [
        (axes[0, 0], err_mean_1d, "(a) 1D mean", r"$|\hat\mu-\mu|$"),
        (axes[0, 1], err_var_1d, "(b) 1D variance", r"$|\hat v-v|$"),
        (axes[1, 0], err_mean_2d, "(c) 2D mean", r"$\|\hat\mu-\mu\|_2$"),
        (axes[1, 1], err_cov_2d, "(d) 2D covariance", r"$\|\hat P-P\|_F$"),
    ]
    for ax, series, title, ylab in panels:
        fitted_slopes = {}
        for m in degrees:
            ys = np.maximum(series[m], floor)
            fitted = fit_slope(deltas, series[m])
            fitted_slopes[m] = fitted
            ax.loglog(
                deltas,
                ys,
                color=colors[m],
                linestyle=linestyles[m],
                marker=markers[m],
                markerfacecolor="white",
                markeredgecolor=colors[m],
                markeredgewidth=0.8,
                lw=1.45,
                ms=3.6,
                label=rf"LV $m={m}$",
            )
        # guide lines anchored at largest-δ m=3 point
        anchor_y = max(series[3][-1], floor)
        anchor_x = float(deltas[-1])
        guide(
            ax,
            deltas,
            anchor_y,
            anchor_x,
            2.0,
            r"$O(\delta^2)$",
            color=PALETTE["neutral_gray"],
            linestyle=":",
        )
        guide(
            ax,
            deltas,
            anchor_y,
            anchor_x,
            3.0,
            r"$O(\delta^3)$",
            color=PALETTE["light_gray"],
            linestyle=(0, (4, 2)),
        )
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.text(
            0.025,
            0.955,
            rf"fitted: $p_3={fitted_slopes[3]:.2f}$, $p_5={fitted_slopes[5]:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            color=PALETTE["black"],
        )
        style_curve_axis(ax, grid_axis="both")

    axes[1, 0].set_xlabel(r"Assimilation interval $\delta$")
    axes[1, 1].set_xlabel(r"Assimilation interval $\delta$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        columnspacing=1.35,
        handletextpad=0.45,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.205,
        top=0.955,
        wspace=0.27,
        hspace=0.34,
    )
    png_path = os.path.join(OUTPUT_DIR, "weak_error_lv_paths.png")
    png_path, pdf_path = save_publication_figure(fig, png_path)
    plt.close()

    print("-" * 72)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {slope_csv}")


if __name__ == "__main__":
    main()
