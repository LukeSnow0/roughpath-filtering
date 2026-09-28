"""Temporal weak-error delta-scaling of the WSC-KF: the Lyons-Victoir rate O(delta^{(m+1)/2}).

The Lyons-Victoir cubature-on-Wiener-space theorem bounds the one-interval weak
(semigroup) error of a degree-``m`` formula by

    | E[f(x_delta) | x0] - sum_j w_j f(Phi_delta(x0, omega_j)) |  <=  C * delta^{(m+1)/2}.

We measure this exponent empirically. To isolate the *temporal* Wiener-cubature
error from the *spatial* (Gaussian-closure) error, we start each predict from a
**point mass** ``P0 = 0``: the WSC-KF's spatial cubature nodes then collapse onto the
mean and contribute no error, so the predicted-moment error is purely the
semigroup approximation the theorem bounds.

Model (scalar GBM, Ito): dx = mu x dt + sigma x dW, x(0) = x0 (deterministic).
Closed-form conditional moments give an exact reference:

    E[x_delta | x0]   = x0 * exp(mu * delta)
    Var[x_delta | x0] = x0^2 * exp(2 mu delta) * (exp(sigma^2 delta) - 1)

Two panels:
  (1) WSC-KF mean weak-error vs delta for cubature_order k in {1,3,5,7} (1D product
      Lyons paths). All collapse to the SAME slope -> the rate is set by the
      cubature *degree*, not the macro-segment count k. The observed slope ~3
      implies degree m=5 (the 1D base rule is 3-point Gauss-Hermite, exact to
      degree 5 -- not degree 3 as the docstring says).
  (2) **2D independent GBM**: WSC-KF (Lyons degrees m=3,5,7) vs CD-EKF/UKF/CKF
      covariance weak-error vs delta. WSC-KF slopes track (m+1)/2; CD moment-ODE
      filters (diffusion frozen at the mean) stay near the Gaussian-closure rate.
      m=3,5 use product Lyons–Victoir paths; m=7 uses the product of 1D
      4-point Gauss–Hermite straight lines (16 paths) — enough for independent
      coordinates, whose law depends only on the marginals of W.

Run:
    uv run --extra experiments python experiments/multiplicative_noise_test/weak_error_delta_scaling.py
"""

from __future__ import annotations

import csv
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENTS_DIR, ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, ROOT_DIR)

from cd_dynamax_mirror.filters import UKFParams, _ekf_predict, _ukf_predict, _ukf_weights

from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict
from mfts_test.plot_style import (
    METHOD_COLORS,
    METHOD_MARKERS,
    PALETTE,
    SINGLE_COLUMN_WIDTH,
    save_publication_figure,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MU = 0.5
SIGMA = 0.6
X0 = 1.0
INNER_STEPS = 64  # fixed RK4 inner steps per interval -> dt0 = delta/64, so the
#                   discretization error stays O(delta^5), well below the terms measured.


# ---------------------------------------------------------------------------
# Exact conditional moments (point initial condition)
# ---------------------------------------------------------------------------
def exact_mean(delta, x0=X0, mu=MU):
    return x0 * np.exp(mu * delta)


def exact_var(delta, x0=X0, mu=MU, sigma=SIGMA):
    return x0**2 * np.exp(2.0 * mu * delta) * (np.exp(sigma**2 * delta) - 1.0)


# ---------------------------------------------------------------------------
# 1D GBM dynamics
# ---------------------------------------------------------------------------
def make_gbm_1d(mu=MU, sigma=SIGMA):
    def ito_drift(t, x):
        del t
        return mu * x

    def diffusion(t, x):
        del t
        return (sigma * x).reshape(1, 1)

    def qc_fn(t, x):
        del t, x
        return jnp.eye(1)

    return ito_drift, diffusion, qc_fn


ITO_DRIFT_1D, DIFFUSION_1D, QC_1D = make_gbm_1d()
STRAT_DRIFT_1D, STRAT_DIFF_1D = ito_to_stratonovich_dynamics(ITO_DRIFT_1D, DIFFUSION_1D)


def wsckf_predict_1d(
    delta,
    k,
    multidim_paths="auto",
    recombination_degree=3,
    num_paths=None,
    lyons_degree=5,
):
    """Point-mass WSC-KF predict on 1D GBM.

    ``lyons_degree`` ∈ {3, 5} uses the repo product Lyons–Victoir paths.
    ``lyons_degree=7`` is 1D-only: 4-point Gauss–Hermite terminals with
    straight-line paths. For scalar GBM the flow depends only on ``W_δ``, so
    this matches the Litterer degree-7 *Gaussian* projection (theory slope 4).
    (Gyurkó–Lyons treat degrees 9 and 11 in 1D; full Litterer path geometry is
    unnecessary for this observable.)
    """
    integration_dt = delta / INNER_STEPS
    if int(lyons_degree) == 7:
        if int(k) != 1:
            raise ValueError("lyons_degree=7 helper is implemented for k=1 only")
        ts, path_values, path_weights = _gauss_hermite_straight_paths(
            horizon=float(delta),
            n_nodes=4,
            integration_dt=integration_dt,
            noise_dim=1,
        )
    else:
        ts, path_values, path_weights = precompute_cubature_paths(
            horizon=delta,
            noise_dim=1,
            integration_dt=integration_dt,
            cubature_order=k,
            multidim_method=multidim_paths,
            recombination_degree=recombination_degree,
            num_paths=num_paths,
            lyons_degree=lyons_degree,
        )
    mean = jnp.array([X0])
    chol_cov = jnp.zeros((1, 1))  # point mass: pure temporal (semigroup) error
    pred_mean, pred_chol, _, _ = predict(
        mean, chol_cov, ts, path_values, path_weights, STRAT_DRIFT_1D, STRAT_DIFF_1D
    )
    return float(pred_mean[0]), float((pred_chol @ pred_chol.T)[0, 0])


def _gauss_hermite_straight_paths(
    horizon: float,
    n_nodes: int,
    integration_dt: float,
    noise_dim: int = 1,
):
    """Straight-line cubature paths to (product) probabilists' Gauss–Hermite nodes."""
    from itertools import product as iterproduct

    from numpy.polynomial.hermite_e import hermegauss

    nodes_1d, w_raw = hermegauss(int(n_nodes))
    w_1d = np.asarray(w_raw / np.sqrt(2.0 * np.pi), dtype=np.float64)
    w_1d = w_1d / w_1d.sum()
    d = int(noise_dim)
    if d == 1:
        nodes = nodes_1d.reshape(-1, 1)
        weights = w_1d.copy()
    else:
        nodes = np.array(list(iterproduct(nodes_1d, repeat=d)), dtype=np.float64)
        weights = np.array(
            [float(np.prod([w_1d[j] for j in js])) for js in iterproduct(range(n_nodes), repeat=d)],
            dtype=np.float64,
        )
        weights = weights / weights.sum()

    n_steps = max(2, int(np.ceil(horizon / integration_dt)))
    ts = np.linspace(0.0, horizon, n_steps + 1)
    t_unit = ts / horizon
    # ω(t) = √T · z · (t/T)
    paths = np.stack(
        [np.sqrt(horizon) * z[None, :] * t_unit[:, None] for z in nodes], axis=0
    )
    return (
        jnp.asarray(ts),
        jnp.asarray(paths),
        jnp.asarray(weights),
    )


# CD-CKF ≡ spherical-radial cubature (Arasaratnam–Haykin): UKF with λ = 0.
CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)


def cd_predict_1d(delta, which):
    """Continuous-discrete Gaussian predict from a point mass ``P0 = 0``.

    ``which`` ∈ {``"ekf"``, ``"ukf"``, ``"ckf"``}.  Diffusion is frozen at the
    mean in all three moment-ODE filters (``L(t, m)``), so on this linear-drift
    GBM they share the same Gaussian-closure variance rate.
    """
    m = jnp.array([X0])
    P = jnp.zeros((1, 1))
    n_inner = INNER_STEPS
    dt0 = delta / n_inner
    if which == "ekf":
        m_pred, P_pred = _ekf_predict(
            m, P, ITO_DRIFT_1D, DIFFUSION_1D, QC_1D, 0.0, delta, dt0,
            state_order="first", solver="rk4", n_steps=n_inner,
        )
    else:
        n = 1
        p = CKF_PARAMS if which == "ckf" else UKFParams()
        lamb = p.alpha**2 * (n + p.kappa) - n
        w_mean, w_cov, W_matrix = _ukf_weights(n, p.alpha, p.beta, lamb)
        m_pred, P_pred = _ukf_predict(
            m, P, ITO_DRIFT_1D, DIFFUSION_1D, QC_1D, 0.0, delta, dt0,
            n, lamb, w_mean, w_cov, W_matrix,
            state_order="first", solver="rk4", n_steps=n_inner,
        )
    return float(m_pred[0]), float(P_pred[0, 0])


# ---------------------------------------------------------------------------
# 2D independent GBM
# ---------------------------------------------------------------------------
MU2 = np.array([0.5, -0.3])
SIGMA2 = np.array([0.6, 0.4])
X02 = np.array([1.0, 1.0])


def exact_mean_2d(delta):
    return X02 * np.exp(MU2 * delta)


def exact_cov_2d(delta):
    v = X02**2 * np.exp(2.0 * MU2 * delta) * (np.exp(SIGMA2**2 * delta) - 1.0)
    return np.diag(v)


def make_gbm_2d():
    mu = jnp.asarray(MU2)
    sig = jnp.asarray(SIGMA2)

    def ito_drift(t, x):
        del t
        return mu * x

    def diffusion(t, x):
        del t
        return jnp.diag(sig * x)

    def qc_fn(t, x):
        del t, x
        return jnp.eye(2)

    return ito_drift, diffusion, qc_fn


ITO_DRIFT_2D, DIFFUSION_2D, QC_2D = make_gbm_2d()
STRAT_DRIFT_2D, STRAT_DIFF_2D = ito_to_stratonovich_dynamics(ITO_DRIFT_2D, DIFFUSION_2D)


def wsckf_predict_2d(delta, lyons_degree, k=1):
    """Point-mass WSC-KF predict on 2D independent GBM; returns (mean, cov)."""
    integration_dt = delta / INNER_STEPS
    if int(lyons_degree) == 7:
        if int(k) != 1:
            raise ValueError("lyons_degree=7 helper is implemented for k=1 only")
        ts, path_values, path_weights = _gauss_hermite_straight_paths(
            horizon=float(delta),
            n_nodes=4,
            integration_dt=integration_dt,
            noise_dim=2,
        )
    else:
        ts, path_values, path_weights = precompute_cubature_paths(
            horizon=delta,
            noise_dim=2,
            integration_dt=integration_dt,
            cubature_order=k,
            multidim_method="product",
            lyons_degree=lyons_degree,
        )
    mean = jnp.asarray(X02)
    chol_cov = jnp.zeros((2, 2))
    pred_mean, pred_chol, _, _ = predict(
        mean, chol_cov, ts, path_values, path_weights, STRAT_DRIFT_2D, STRAT_DIFF_2D
    )
    P = pred_chol @ pred_chol.T
    return np.asarray(pred_mean), np.asarray(P)


def cd_predict_2d(delta, which):
    """2D CD-EKF / UKF / CKF predict from ``P0 = 0``."""
    m = jnp.asarray(X02)
    P = jnp.zeros((2, 2))
    n_inner = INNER_STEPS
    dt0 = delta / n_inner
    if which == "ekf":
        m_pred, P_pred = _ekf_predict(
            m, P, ITO_DRIFT_2D, DIFFUSION_2D, QC_2D, 0.0, delta, dt0,
            state_order="first", solver="rk4", n_steps=n_inner,
        )
    else:
        n = 2
        p = CKF_PARAMS if which == "ckf" else UKFParams()
        lamb = p.alpha**2 * (n + p.kappa) - n
        w_mean, w_cov, W_matrix = _ukf_weights(n, p.alpha, p.beta, lamb)
        m_pred, P_pred = _ukf_predict(
            m, P, ITO_DRIFT_2D, DIFFUSION_2D, QC_2D, 0.0, delta, dt0,
            n, lamb, w_mean, w_cov, W_matrix,
            state_order="first", solver="rk4", n_steps=n_inner,
        )
    return np.asarray(m_pred), np.asarray(P_pred)

# ---------------------------------------------------------------------------
# Slope fitting
# ---------------------------------------------------------------------------
def fit_slope(deltas, errs, dmax=0.4, floor=1e-13):
    d = np.asarray(deltas)
    e = np.asarray(errs)
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
    color=PALETTE["neutral_gray"],
    ls="--",
):
    d = np.asarray(deltas)
    c = anchor_err / (anchor_d**slope)
    (line,) = ax.loglog(
        d, c * d**slope, ls, color=color, alpha=0.8, lw=1.1, label=label
    )
    return line


# ---------------------------------------------------------------------------
def main():
    deltas = np.logspace(np.log10(0.03), np.log10(3.5), 14)
    rng_floor = 1e-15

    # ---- Panel 1: WSC-KF mean weak-error, k sweep (1D) ----
    ks = [1, 3, 5, 7]
    mean_err_by_k = {}
    for k in ks:
        errs = []
        for d in deltas:
            m_pred, _ = wsckf_predict_1d(d, k, multidim_paths="product")
            errs.append(abs(m_pred - exact_mean(d)))
        mean_err_by_k[k] = np.array(errs)

    # ---- Panel 2: 2D WSC-KF (m=3,5,7) vs CD cov weak-error ----
    # Fixed k=1 so the only change is Lyons degree m.
    wsckf_degrees = (3, 5, 7)
    wsckf_cov_err_2d = {}
    for m in wsckf_degrees:
        errs = []
        for d in deltas:
            _, P_wsckf = wsckf_predict_2d(d, lyons_degree=m, k=1)
            errs.append(float(np.linalg.norm(P_wsckf - exact_cov_2d(d), ord="fro")))
        wsckf_cov_err_2d[m] = np.array(errs)

    ekf_cov_err, ukf_cov_err, ckf_cov_err = [], [], []
    for d in deltas:
        tP = exact_cov_2d(d)
        _, P_ekf = cd_predict_2d(d, "ekf")
        _, P_ukf = cd_predict_2d(d, "ukf")
        _, P_ckf = cd_predict_2d(d, "ckf")
        ekf_cov_err.append(float(np.linalg.norm(P_ekf - tP, ord="fro")))
        ukf_cov_err.append(float(np.linalg.norm(P_ukf - tP, ord="fro")))
        ckf_cov_err.append(float(np.linalg.norm(P_ckf - tP, ord="fro")))
    ekf_cov_err = np.array(ekf_cov_err)
    ukf_cov_err = np.array(ukf_cov_err)
    ckf_cov_err = np.array(ckf_cov_err)

    # ----- console -----
    print(f"GBM  mu={MU} sigma={SIGMA} x0={X0}  (point-mass initial: P0=0)\n")
    print("Panel 1 -- WSC-KF mean weak-error slope vs delta (should be ~(m+1)/2, m=5 -> 3):")
    for k in ks:
        print(f"    k={k}:  slope = {fit_slope(deltas, mean_err_by_k[k]):.2f}")
    print("\nPanel 2 -- 2D cov weak-error slope vs delta:")
    for m in wsckf_degrees:
        print(
            f"    WSC-KF m={m}: slope = {fit_slope(deltas, wsckf_cov_err_2d[m]):.2f}"
            f"   (theory {(m + 1) / 2:.1f})"
        )
    print(f"    CD-EKF  : slope = {fit_slope(deltas, ekf_cov_err):.2f}")
    print(f"    CD-UKF  : slope = {fit_slope(deltas, ukf_cov_err):.2f}")
    print(f"    CD-CKF  : slope = {fit_slope(deltas, ckf_cov_err):.2f}")

    # ----- CSV -----
    csv_path = os.path.join(OUTPUT_DIR, "weak_error_delta_scaling.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["delta"]
            + [f"wsckf_mean_err_k{k}" for k in ks]
            + [
                "crf2d_cov_err_m3",
                "crf2d_cov_err_m5",
                "crf2d_cov_err_m7",
                "ekf2d_cov_err",
                "ukf2d_cov_err",
                "ckf2d_cov_err",
            ]
        )
        for i, d in enumerate(deltas):
            w.writerow(
                [d]
                + [mean_err_by_k[k][i] for k in ks]
                + [
                    wsckf_cov_err_2d[3][i],
                    wsckf_cov_err_2d[5][i],
                    wsckf_cov_err_2d[7][i],
                    ekf_cov_err[i],
                    ukf_cov_err[i],
                    ckf_cov_err[i],
                ]
            )

    # ----- plot (two separate publication-size figures) -----
    mid = min(6, len(deltas) - 1)
    mean_colors = {
        1: PALETTE["blue"],
        3: PALETTE["teal_green"],
        5: PALETTE["orange"],
        7: PALETTE["warm_red"],
    }
    mean_markers = {1: "o", 3: "s", 5: "D", 7: "^"}
    wsckf_colors = {
        3: PALETTE["purple"],
        5: PALETTE["warm_red"],
        7: PALETTE["orange"],
    }
    wsckf_markers = {3: "o", 5: "D", 7: "^"}

    fig1, ax1 = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.3))
    h_ws = []
    for k in ks:
        (ln,) = ax1.loglog(
            deltas,
            np.maximum(mean_err_by_k[k], rng_floor),
            marker=mean_markers[k],
            color=mean_colors[k],
            markerfacecolor="white",
            markeredgewidth=0.75,
            markersize=3.5,
            linewidth=1.35,
            label=rf"WSC-KF, $k={k}$",
        )
        h_ws.append(ln)
    h_g3 = guide(
        ax1,
        deltas,
        mean_err_by_k[ks[-1]][mid],
        deltas[mid],
        3.0,
        r"$\propto\delta^{3}$",
        color=PALETTE["neutral_gray"],
        ls="--",
    )
    ax1.set_xlabel(r"Assimilation interval $\delta$")
    ax1.set_ylabel(r"$|\widehat{\mathrm{E}}[x_\delta]-\mathrm{E}[x_\delta]|$")
    ax1.set_xlim(deltas[0], deltas[-1])
    style_curve_axis(ax1)
    mean_legend_handles = [h_ws[0], h_ws[3], h_ws[1], h_g3, h_ws[2]]
    fig1.legend(
        handles=mean_legend_handles,
        labels=[h.get_label() for h in mean_legend_handles],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
    )
    fig1.subplots_adjust(left=0.19, right=0.98, bottom=0.2, top=0.7)
    png_mean = os.path.join(OUTPUT_DIR, "weak_error_delta_scaling_mean.png")
    mean_png, mean_pdf = save_publication_figure(fig1, png_mean)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    wsckf_fitted_slopes = {
        m: fit_slope(deltas, wsckf_cov_err_2d[m]) for m in wsckf_degrees
    }
    h_ws = []
    for m in wsckf_degrees:
        (ln,) = ax2.loglog(
            deltas,
            np.maximum(wsckf_cov_err_2d[m], rng_floor),
            color=wsckf_colors[m],
            marker=wsckf_markers[m],
            markerfacecolor="white",
            markeredgewidth=0.75,
            markersize=3.5,
            linewidth=1.35,
            label=rf"WSC-KF $m={m}$",
        )
        h_ws.append(ln)
    (h_ekf,) = ax2.loglog(
        deltas,
        np.maximum(ekf_cov_err, rng_floor),
        color=METHOD_COLORS["CD-EKF"],
        marker=METHOD_MARKERS["CD-EKF"],
        markerfacecolor="white",
        markeredgewidth=0.75,
        markersize=3.5,
        linewidth=1.35,
        label="CD-EKF",
    )
    (h_ukf,) = ax2.loglog(
        deltas,
        np.maximum(ukf_cov_err, rng_floor),
        color=METHOD_COLORS["CD-UKF"],
        marker=METHOD_MARKERS["CD-UKF"],
        markerfacecolor="white",
        markeredgewidth=0.75,
        markersize=3.5,
        linewidth=1.35,
        label="CD-UKF",
    )
    (h_ckf,) = ax2.loglog(
        deltas,
        np.maximum(ckf_cov_err, rng_floor),
        color=METHOD_COLORS["CD-CKF"],
        marker=METHOD_MARKERS["CD-CKF"],
        markerfacecolor="white",
        markeredgewidth=0.75,
        markersize=3.5,
        linewidth=1.35,
        label="CD-CKF",
    )
    h_g2 = guide(
        ax2, deltas, ekf_cov_err[mid], deltas[mid], 2.0, r"$\propto\delta^{2}$",
        color=PALETTE["neutral_gray"], ls="--",
    )
    h_g3 = guide(
        ax2, deltas, wsckf_cov_err_2d[5][mid], deltas[mid], 3.0, r"$\propto\delta^{3}$",
        color=PALETTE["neutral_gray"], ls=":",
    )
    h_g4 = guide(
        ax2,
        deltas,
        max(wsckf_cov_err_2d[7][mid], rng_floor),
        deltas[mid],
        4.0,
        r"$\propto\delta^{4}$",
        color=PALETTE["neutral_gray"],
        ls="-.",
    )
    ax2.set_xlabel(r"Assimilation interval $\delta$")
    ax2.set_ylabel(r"$\|\widehat P-P\|_F$")
    ax2.set_xlim(deltas[0], deltas[-1])
    style_curve_axis(ax2)
    ax2.text(
        0.03,
        0.96,
        (
            rf"$\hat p_{{m=3,5,7}}="
            rf"({wsckf_fitted_slopes[3]:.2f},\,"
            rf"{wsckf_fitted_slopes[5]:.2f},\,"
            rf"{wsckf_fitted_slopes[7]:.2f})$"
        ),
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=PALETTE["black"],
    )
    # Center the three semantic rows independently so their unequal label
    # widths do not make the shorter rows look left-shifted.
    legend_kwargs = dict(
        ncol=3,
        loc="upper center",
        fontsize=6.2,
        columnspacing=1.15,
        handlelength=1.55,
        handletextpad=0.45,
        borderaxespad=0.0,
        borderpad=0.0,
    )
    for handles, anchor_y in (
        (h_ws, 0.985),
        ([h_ekf, h_ukf, h_ckf], 0.910),
        ([h_g2, h_g3, h_g4], 0.835),
    ):
        fig2.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            bbox_to_anchor=(0.525, anchor_y),
            **legend_kwargs,
        )
    fig2.subplots_adjust(
        left=0.19,
        right=0.98,
        bottom=0.19,
        top=0.77,
    )
    png_cov = os.path.join(OUTPUT_DIR, "weak_error_delta_scaling_cov.png")
    cov_png, cov_pdf = save_publication_figure(fig2, png_cov)
    plt.close(fig2)

    print(f"\nSaved plot -> {mean_png}")
    print(f"Saved plot -> {mean_pdf}")
    print(f"Saved plot -> {cov_png}")
    print(f"Saved plot -> {cov_pdf}")
    print(f"Saved data -> {csv_path}")


if __name__ == "__main__":
    main()
