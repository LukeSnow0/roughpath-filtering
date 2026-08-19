"""Temporal weak-error δ-scaling of WSC-KF vs CD filters on Heston ``(S, v)``.

Textbook Itô Heston (not log-price)::

    dS = μ S dt + S √v dW¹
    dv = κ(θ − v) dt + ξ √v (ρ dW¹ + √(1−ρ²) dW²)

``E[LLᵀ]_{SS} = E[S² v]`` does not close at second order, so CD freeze-at-mean
moment ODEs are *not* the true covariance equations (unlike ``(log S, v)``).

Protocol: one predict step from a point mass ``P0 = 0``. Mean errors use the
closed affine mean (:func:`heston_model.exact_heston_sv_mean`).

Diffusion uses the globally ``C^∞`` map ``φ(v)=(v²+ε²)^{1/4}`` in place of
``√v`` (``ε=θ``) so every derivative of the noise fields is bounded. Drift is
unchanged, so the exact mean is the same as textbook Heston. This tests whether
the leftover ``δ²`` m=5 rate was due to CIR ``√v`` regularity.

Cubature paths are the explicit multi-D Lyons–Victoir formulae
(``multidim_method="lv"``). Degrees ``m=3,5`` only.

Run:
  uv run --extra experiments python experiments/heston_test/weak_error_delta_scaling.py
"""

from __future__ import annotations

import csv
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
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
sys.path.insert(0, HERE)
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, ROOT_DIR)

from cd_dynamax_mirror.filters import UKFParams, _ekf_predict, _ukf_predict, _ukf_weights
from heston_model import (
    HestonParams,
    exact_heston_sv_mean,
    make_heston_sv_dynamics,
)
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feller-friendly defaults (2 κ θ = 0.16 > ξ² = 0.09). State is [S, v].
PARAMS = HestonParams()
V0 = float(PARAMS.theta if PARAMS.v0 is None else PARAMS.v0)
S0 = float(np.exp(PARAMS.x0))  # log-price x0=0 → S0=1
X0 = np.array([S0, V0], dtype=np.float64)

INNER_STEPS = 128  # RK4 steps per interval → dt0 = δ/128
# C^∞ stand-in for √v: φ(v)=(v²+ε²)^{1/4}. Drift (hence the exact mean) unchanged.
SMOOTH_VOL_EPS = float(PARAMS.theta)
ITO_DRIFT, DIFFUSION, QC_FN, _, _ = make_heston_sv_dynamics(
    PARAMS, smooth_vol_eps=SMOOTH_VOL_EPS
)
STRAT_DRIFT, STRAT_DIFF = ito_to_stratonovich_dynamics(ITO_DRIFT, DIFFUSION)

CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)


def wsckf_predict(delta: float, lyons_degree: int, k: int = 1):
    """Point-mass predict with library multi-D LV paths (incl. Lévy area)."""
    integration_dt = delta / INNER_STEPS
    ts, path_values, path_weights = precompute_cubature_paths(
        horizon=float(delta),
        noise_dim=2,
        integration_dt=integration_dt,
        cubature_order=int(k),
        multidim_method="lv",
        lyons_degree=lyons_degree,
    )
    mean = jnp.asarray(X0)
    chol_cov = jnp.zeros((2, 2))
    pred_mean, pred_chol, _, _ = predict(
        mean,
        chol_cov,
        jnp.asarray(ts),
        jnp.asarray(path_values),
        jnp.asarray(path_weights),
        STRAT_DRIFT,
        STRAT_DIFF,
    )
    P = pred_chol @ pred_chol.T
    return np.asarray(pred_mean), np.asarray(P)


def cd_predict(delta: float, which: str):
    """CD-EKF / UKF / CKF predict from ``P0 = 0`` (moment ODEs, frozen L)."""
    m = jnp.asarray(X0)
    P = jnp.zeros((2, 2))
    n_inner = INNER_STEPS
    dt0 = jnp.asarray(delta / n_inner, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(delta, dtype=jnp.float64)
    if which == "ekf":
        m_pred, P_pred = _ekf_predict(
            m,
            P,
            ITO_DRIFT,
            DIFFUSION,
            QC_FN,
            t0,
            t1,
            dt0,
            state_order="first",
            solver="rk4",
            n_steps=n_inner,
        )
    else:
        n = 2
        p = CKF_PARAMS if which == "ckf" else UKFParams()
        lamb = p.alpha**2 * (n + p.kappa) - n
        w_mean, w_cov, W_matrix = _ukf_weights(n, p.alpha, p.beta, lamb)
        m_pred, P_pred = _ukf_predict(
            m,
            P,
            ITO_DRIFT,
            DIFFUSION,
            QC_FN,
            t0,
            t1,
            dt0,
            n,
            lamb,
            w_mean,
            w_cov,
            W_matrix,
            state_order="first",
            solver="rk4",
            n_steps=n_inner,
        )
    return np.asarray(m_pred), np.asarray(P_pred)


def fit_slope(deltas, errs, dmax=0.25, floor=1e-15):
    d = np.asarray(deltas)
    e = np.asarray(errs)
    mask = (d <= dmax) & (e > floor)
    if mask.sum() < 2:
        mask = e > floor
    if mask.sum() < 2:
        return float("nan")
    # Guard against all-zero / underflow floors.
    if np.all(e[mask] <= floor * 10):
        return float("nan")
    return float(np.polyfit(np.log(d[mask]), np.log(e[mask]), 1)[0])


def guide(ax, deltas, anchor_err, anchor_d, slope, label, color="k", ls="--"):
    d = np.asarray(deltas)
    c = anchor_err / (anchor_d**slope)
    (line,) = ax.loglog(
        d, c * d**slope, ls, color=color, alpha=0.75, lw=1.6, label=label
    )
    return line


def _plot_scaling(
    deltas,
    series: dict[str, tuple[np.ndarray, str, str]],
    *,
    ylabel: str,
    png_path: str,
    guides: list[tuple[float, np.ndarray, str, str]],
    ncol: int = 3,
):
    """``series``: name -> (errs, color, marker_style)."""
    AXIS_FS, LEGEND_FS, TICK_FS = 13, 11, 11
    rng_floor = 1e-18
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    handles = []
    for name, (errs, color, fmt) in series.items():
        slope = fit_slope(deltas, errs)
        label = name if not np.isfinite(slope) else rf"{name} (fit={slope:.2f})"
        (ln,) = ax.loglog(
            deltas, np.maximum(errs, rng_floor), fmt, color=color, label=label
        )
        handles.append(ln)
    guide_handles = []
    for slope, anchor_errs, lab, ls in guides:
        mid = min(8, len(deltas) - 1)
        guide_handles.append(
            guide(
                ax,
                deltas,
                max(float(anchor_errs[mid]), rng_floor),
                float(deltas[mid]),
                slope,
                lab,
                color="0.35",
                ls=ls,
            )
        )
    ax.set_xlabel(r"assimilation interval $\delta$", fontsize=AXIS_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.set_xlim(deltas[0], deltas[-1])
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(
        handles=[*handles, *guide_handles],
        fontsize=LEGEND_FS,
        ncol=ncol,
        columnspacing=1.0,
        handletextpad=0.4,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # Small-δ asymptotic (LV rates) down through 10^{-3}; large-δ for prefactors.
    deltas = np.logspace(np.log10(1e-3), np.log10(1.5), 16)
    wsckf_degrees = (3, 5)
    wsckf_colors = {3: "#6A4C93", 5: "#9B5DE5"}

    print(
        f"Heston (S,v) C^∞-vol weak-error δ-scaling  "
        f"φ(v)=(v²+ε²)^{{1/4}} ε={SMOOTH_VOL_EPS:g}  "
        f"κ={PARAMS.kappa:g} θ={PARAMS.theta:g} ξ={PARAMS.xi:g} ρ={PARAMS.rho:g}  "
        f"x0={X0.tolist()}  (P0=0, mean=exact)\n"
    )

    ws_mean_err: dict[int, np.ndarray] = {}
    for m in wsckf_degrees:
        em = []
        for d in deltas:
            m_ref = exact_heston_sv_mean(float(d), X0, PARAMS)
            m_hat, _ = wsckf_predict(float(d), lyons_degree=m, k=1)
            em.append(float(np.linalg.norm(m_hat - m_ref)))
        ws_mean_err[m] = np.array(em)

    ekf_mean, ukf_mean, ckf_mean = [], [], []
    for d in deltas:
        m_ref = exact_heston_sv_mean(float(d), X0, PARAMS)
        for which, em_list in (
            ("ekf", ekf_mean),
            ("ukf", ukf_mean),
            ("ckf", ckf_mean),
        ):
            m_hat, _ = cd_predict(float(d), which)
            em_list.append(float(np.linalg.norm(m_hat - m_ref)))
    ekf_mean, ukf_mean, ckf_mean = map(np.array, (ekf_mean, ukf_mean, ckf_mean))

    print("Mean weak-error slope vs delta:")
    for m in wsckf_degrees:
        print(
            f"    WSC-KF m={m}: slope = {fit_slope(deltas, ws_mean_err[m]):.2f}"
            f"   (theory {(m + 1) / 2:.1f})"
        )
    print(f"    CD-EKF  : slope = {fit_slope(deltas, ekf_mean):.2f}")
    print(f"    CD-UKF  : slope = {fit_slope(deltas, ukf_mean):.2f}")
    print(f"    CD-CKF  : slope = {fit_slope(deltas, ckf_mean):.2f}")

    csv_path = os.path.join(OUTPUT_DIR, "weak_error_delta_scaling_sv_smooth.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["delta"]
            + [f"wsckf_mean_err_m{m}" for m in wsckf_degrees]
            + ["ekf_mean_err", "ukf_mean_err", "ckf_mean_err"]
        )
        for i, d in enumerate(deltas):
            w.writerow(
                [d]
                + [ws_mean_err[m][i] for m in wsckf_degrees]
                + [ekf_mean[i], ukf_mean[i], ckf_mean[i]]
            )

    png_mean = os.path.join(OUTPUT_DIR, "weak_error_delta_scaling_sv_mean.png")
    _plot_scaling(
        deltas,
        {
            rf"WSC-KF $m$=3": (ws_mean_err[3], wsckf_colors[3], "o-"),
            rf"WSC-KF $m$=5": (ws_mean_err[5], wsckf_colors[5], "o-"),
            "CD-EKF": (ekf_mean, "C3", "s-"),
            "CD-UKF": (ukf_mean, "C0", "^-"),
            "CD-CKF": (ckf_mean, "#2E86AB", "D-"),
        },
        ylabel=r"$\|\hat m - m\|_2$",
        png_path=png_mean,
        guides=[
            (2.0, ws_mean_err[3], r"$\propto\delta^{2}$", "--"),
            (3.0, ws_mean_err[5], r"$\propto\delta^{3}$", ":"),
        ],
        ncol=2,
    )

    print(f"\nSaved plot -> {png_mean}")
    print(f"Saved data -> {csv_path}")


if __name__ == "__main__":
    main()
