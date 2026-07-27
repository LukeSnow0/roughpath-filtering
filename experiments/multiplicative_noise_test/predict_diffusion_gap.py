"""Isolating the state-dependent (multiplicative) diffusion gap: CRF vs CD-EKF/UKF/CKF.

We compare the *predict* (time-update) step only, over a single assimilation
interval of length ``delta``, on a scalar geometric-Brownian-motion (GBM) SDE
with **linear drift** and **multiplicative diffusion**:

    Ito:  dx = mu * x dt + sigma * x dW,     x(0) ~ N(m0, P0).

GBM is the clean isolating case:

* The drift ``mu*x`` is *linear*, so the nonlinear-drift / flow-deformation gap
  is exactly zero. Any error in the predicted moments is therefore attributable
  purely to how each filter treats the *state-dependent diffusion*.
* The diffusion ``sigma*x`` is genuinely multiplicative, and ``b(x)^2 = sigma^2 x^2``
  is *quadratic*, so E[b^2] = sigma^2 (m^2 + P) depends on the state variance P.
  The CD moment-ODE filters freeze the diffusion at the mean (add sigma^2 m^2),
  missing the ``sigma^2 P`` contribution; the CRF integrates sigma*x along the
  actual propagated state, capturing it.  CD-CKF (spherical-radial) and CD-UKF
  share this freeze, so they do not escape the gap relative to CD-EKF here.
* The predicted moments over a Gaussian prior have a **closed form**, giving an
  exact reference with no Monte-Carlo noise:

      E[x_delta]   = m0 * exp(mu * delta)
      E[x_delta^2] = (m0^2 + P0) * exp((2 mu + sigma^2) delta)
      Var[x_delta] = E[x_delta^2] - E[x_delta]^2

A large Monte-Carlo simulation is included purely to validate the closed form.

Run:
    python experiments/multiplicative_noise_test/predict_diffusion_gap.py
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

# --- Make the CRF package and the cd_dynamax_mirror baselines importable -------
EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENTS_DIR, ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, ROOT_DIR)

from cd_dynamax_mirror.filters import UKFParams, _ekf_predict, _ukf_predict, _ukf_weights

from crf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# GBM model (Ito): dx = mu x dt + sigma x dW
# ---------------------------------------------------------------------------
MU = 0.5
SIGMA = 0.6
M0 = 1.0
P0 = 0.25  # initial state variance (std 0.5); drives the sigma^2 * P0 gap

CUBATURE_ORDER = 3  # CRF Lyons macro-segments (k); 3^k = 27 paths in 1D
INNER_DT = 0.005  # inner RK4 step; kept small so discretization error << closure error
MC_SAMPLES = 2_000_000


def ito_drift(t, x):
    del t
    return MU * x


def diffusion(t, x):
    del t
    return (SIGMA * x).reshape(1, 1)


def qc_fn(t, x):
    del t, x
    return jnp.eye(1)


# CRF integrates in Stratonovich form; convert the Ito drift once.
strat_drift, strat_diffusion = ito_to_stratonovich_dynamics(ito_drift, diffusion)


# ---------------------------------------------------------------------------
# Exact and Monte-Carlo references for the predicted moments
# ---------------------------------------------------------------------------
def exact_moments(delta: float):
    mean = M0 * np.exp(MU * delta)
    second = (M0**2 + P0) * np.exp((2.0 * MU + SIGMA**2) * delta)
    var = second - mean**2
    return mean, var


def mc_moments(delta: float, rng: np.random.Generator):
    x0 = M0 + np.sqrt(P0) * rng.standard_normal(MC_SAMPLES)
    w = np.sqrt(delta) * rng.standard_normal(MC_SAMPLES)
    x = x0 * np.exp((MU - 0.5 * SIGMA**2) * delta + SIGMA * w)
    return float(np.mean(x)), float(np.var(x))


# ---------------------------------------------------------------------------
# Filter predict steps (single interval [0, delta])
# ---------------------------------------------------------------------------
def crf_predict(delta: float):
    n_inner = max(1, int(round(delta / INNER_DT)))
    integration_dt = delta / n_inner
    ts, path_values, path_weights = precompute_cubature_paths(
        horizon=delta,
        noise_dim=1,
        integration_dt=integration_dt,
        cubature_order=CUBATURE_ORDER,
    )
    mean = jnp.array([M0])
    chol_cov = jnp.array([[np.sqrt(P0)]])
    pred_mean, pred_chol, _, _ = predict(
        mean, chol_cov, ts, path_values, path_weights, strat_drift, strat_diffusion
    )
    return float(pred_mean[0]), float((pred_chol @ pred_chol.T)[0, 0])


def ekf_predict(delta: float):
    n_inner = max(1, int(round(delta / INNER_DT)))
    m = jnp.array([M0])
    P = jnp.array([[P0]])
    m_pred, P_pred = _ekf_predict(
        m, P, ito_drift, diffusion, qc_fn, 0.0, delta, INNER_DT,
        state_order="first", solver="rk4", n_steps=n_inner,
    )
    return float(m_pred[0]), float(P_pred[0, 0])


def ukf_predict(delta: float, ukf_params: UKFParams | None = None):
    n_inner = max(1, int(round(delta / INNER_DT)))
    n = 1
    p = UKFParams() if ukf_params is None else ukf_params
    lamb = p.alpha**2 * (n + p.kappa) - n
    w_mean, w_cov, W_matrix = _ukf_weights(n, p.alpha, p.beta, lamb)
    m = jnp.array([M0])
    P = jnp.array([[P0]])
    m_pred, P_pred = _ukf_predict(
        m, P, ito_drift, diffusion, qc_fn, 0.0, delta, INNER_DT,
        n, lamb, w_mean, w_cov, W_matrix,
        state_order="first", solver="rk4", n_steps=n_inner,
    )
    return float(m_pred[0]), float(P_pred[0, 0])


def ckf_predict(delta: float):
    """Spherical-radial CD-CKF ≡ UKF with λ = 0."""
    return ukf_predict(delta, UKFParams(alpha=1.0, beta=0.0, kappa=0.0))


# ---------------------------------------------------------------------------
# Sweep over the assimilation interval
# ---------------------------------------------------------------------------
def main():
    deltas = np.logspace(np.log10(0.05), np.log10(1.5), 14)
    rng = np.random.default_rng(0)

    rows = []
    for d in deltas:
        ex_m, ex_v = exact_moments(d)
        mc_m, mc_v = mc_moments(d, rng)
        crf_m, crf_v = crf_predict(d)
        ekf_m, ekf_v = ekf_predict(d)
        ukf_m, ukf_v = ukf_predict(d)
        ckf_m, ckf_v = ckf_predict(d)
        rows.append(
            dict(
                delta=d,
                exact_mean=ex_m, exact_var=ex_v,
                mc_mean=mc_m, mc_var=mc_v,
                crf_mean=crf_m, crf_var=crf_v,
                ekf_mean=ekf_m, ekf_var=ekf_v,
                ukf_mean=ukf_m, ukf_var=ukf_v,
                ckf_mean=ckf_m, ckf_var=ckf_v,
            )
        )

    def rel(a, b):
        return abs(a - b) / abs(b)

    # ----- console summary -----
    print(f"GBM  mu={MU}  sigma={SIGMA}  m0={M0}  P0={P0}  (CRF k={CUBATURE_ORDER})")
    print("MC validates the closed form (rel err of MC vs exact var, largest delta):")
    print(f"  {rel(rows[-1]['mc_var'], rows[-1]['exact_var']):.2e}\n")
    hdr = (
        f"{'delta':>7} | {'exact var':>10} | "
        f"{'CRF':>9} {'CD-EKF':>9} {'CD-UKF':>9} {'CD-CKF':>9}  (rel err in var)"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['delta']:7.3f} | {r['exact_var']:10.4f} | "
            f"{rel(r['crf_var'], r['exact_var']):9.2e} "
            f"{rel(r['ekf_var'], r['exact_var']):9.2e} "
            f"{rel(r['ukf_var'], r['exact_var']):9.2e} "
            f"{rel(r['ckf_var'], r['exact_var']):9.2e}"
        )

    # ----- CSV -----
    csv_path = os.path.join(OUTPUT_DIR, "predict_diffusion_gap.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ----- plot -----
    d_arr = np.array([r["delta"] for r in rows])
    var_err = {
        "CRF (Wiener cubature)": np.array([rel(r["crf_var"], r["exact_var"]) for r in rows]),
        "CD-EKF (moment ODE)": np.array([rel(r["ekf_var"], r["exact_var"]) for r in rows]),
        "CD-UKF (moment ODE)": np.array([rel(r["ukf_var"], r["exact_var"]) for r in rows]),
        "CD-CKF (moment ODE)": np.array([rel(r["ckf_var"], r["exact_var"]) for r in rows]),
    }
    mean_err = {
        "CRF (Wiener cubature)": np.array([rel(r["crf_mean"], r["exact_mean"]) for r in rows]),
        "CD-EKF (moment ODE)": np.array([rel(r["ekf_mean"], r["exact_mean"]) for r in rows]),
        "CD-UKF (moment ODE)": np.array([rel(r["ukf_mean"], r["exact_mean"]) for r in rows]),
        "CD-CKF (moment ODE)": np.array([rel(r["ckf_mean"], r["exact_mean"]) for r in rows]),
    }
    colors = {
        "CRF (Wiener cubature)": "C2",
        "CD-EKF (moment ODE)": "C3",
        "CD-UKF (moment ODE)": "C0",
        "CD-CKF (moment ODE)": "#2E86AB",
    }
    markers = {
        "CRF (Wiener cubature)": "o",
        "CD-EKF (moment ODE)": "s",
        "CD-UKF (moment ODE)": "^",
        "CD-CKF (moment ODE)": "D",
    }

    fig, (axv, axm) = plt.subplots(1, 2, figsize=(12, 5))
    floor = 1e-16
    for label in var_err:
        axv.loglog(d_arr, np.maximum(var_err[label], floor), marker=markers[label],
                   color=colors[label], label=label)
    axv.set_xlabel(r"assimilation interval $\delta$")
    axv.set_ylabel("relative error in predicted variance")
    axv.set_title("Predicted variance: multiplicative-diffusion gap")
    axv.grid(True, which="both", alpha=0.3)
    axv.legend()

    for label in mean_err:
        axm.loglog(d_arr, np.maximum(mean_err[label], floor), marker=markers[label],
                   color=colors[label], label=label)
    axm.set_xlabel(r"assimilation interval $\delta$")
    axm.set_ylabel("relative error in predicted mean")
    axm.set_title("Predicted mean: linear drift (no diffusion gap)")
    axm.grid(True, which="both", alpha=0.3)
    axm.legend()

    fig.suptitle(
        f"CRF vs CD moment-ODE predict on GBM  "
        f"dx = {MU}x dt + {SIGMA}x dW,  x(0)~N({M0},{P0})",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "predict_diffusion_gap.png")
    fig.savefig(png_path, dpi=150)
    print(f"\nSaved plot -> {png_path}")
    print(f"Saved data -> {csv_path}")


if __name__ == "__main__":
    main()
