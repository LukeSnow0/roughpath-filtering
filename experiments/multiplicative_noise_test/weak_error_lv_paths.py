"""Lyons–Victoir explicit paths on multiplicative-noise GBM (point-mass predict).

Isolates the Wiener-cubature rate on a model where path shape / Lévy area matter
(state-dependent diffusion), with an **exact** moment reference — no MC noise.

* 1D GBM: ``dx = μ x dt + σ x dW``
* 2D independent GBM: diagonal ``μ, σ`` (noise fields commute; drift×diffusion
  brackets still make the √22 time-warping visible)

Compares explicit LV degrees ``m ∈ {3, 5}`` from
``experiments/multiplicative_noise_test/lyons_victoir_paths.py`` (does not modify
the repo product/recombined builders). Theory: weak error ``O(δ^{(m+1)/2})``.

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

from crf.library import predict
from crf.library.conversions import ito_to_stratonovich_dynamics
from lyons_victoir_paths import describe, precompute_lv_paths

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
    ts, paths, weights = precompute_lv_paths(
        horizon=float(delta),
        noise_dim=noise_dim,
        integration_dt=float(delta) / INNER_STEPS,
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


def guide(ax, deltas, anchor_err, anchor_d, slope, label, color="k"):
    d = np.asarray(deltas)
    c = anchor_err / (anchor_d**slope)
    ax.loglog(d, c * d**slope, "--", color=color, alpha=0.55, lw=1.2, label=label)


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

    print(describe(1))
    print(describe(2))
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
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), sharex=True)
    colors = {3: "#6A4C93", 5: "#9B5DE5"}

    panels = [
        (axes[0, 0], err_mean_1d, "1D GBM: mean error", r"$|\hat\mu - \mu|$"),
        (axes[0, 1], err_var_1d, "1D GBM: variance error", r"$|\hat v - v|$"),
        (axes[1, 0], err_mean_2d, "2D GBM: mean error", r"$\|\hat\mu - \mu\|_2$"),
        (axes[1, 1], err_cov_2d, "2D GBM: cov error", r"$\|\hat P - P\|_F$"),
    ]
    for ax, series, title, ylab in panels:
        for m in degrees:
            ys = np.maximum(series[m], floor)
            fitted = fit_slope(deltas, series[m])
            theor = 0.5 * (m + 1)
            ax.loglog(
                deltas,
                ys,
                "o-",
                color=colors[m],
                lw=1.8,
                ms=5,
                label=f"LV m={m}  (fit={fitted:.2f}, theory={theor:.0f})",
            )
        # guide lines anchored at largest-δ m=3 point
        anchor_y = max(series[3][-1], floor)
        anchor_x = float(deltas[-1])
        guide(ax, deltas, anchor_y, anchor_x, 2.0, r"$\propto\delta^{2}$", color="0.45")
        guide(ax, deltas, anchor_y, anchor_x, 3.0, r"$\propto\delta^{3}$", color="0.55")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    axes[1, 0].set_xlabel(r"assimilation interval $\delta$")
    axes[1, 1].set_xlabel(r"assimilation interval $\delta$")
    fig.suptitle(
        "Explicit LV multi-D paths on multiplicative GBM "
        r"($P_0=0$; exact moments)",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "weak_error_lv_paths.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("-" * 72)
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {slope_csv}")


if __name__ == "__main__":
    main()
