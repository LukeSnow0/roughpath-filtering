"""Sanity checks for 2D TME against closed Heston affine means and Euler cov."""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "heston_test"))

from heston_model import HestonParams, exact_heston_sv_mean, make_heston_sv_dynamics
from tme import tme_mean_cov


def test_tme_mean_matches_cir_taylor():
    p = HestonParams()
    drift, L, *_ = make_heston_sv_dynamics(p, smooth_vol_eps=float(p.theta))
    x0 = np.array([np.exp(p.x0), 0.08], dtype=np.float64)
    tau = 0.2
    m_exact = exact_heston_sv_mean(tau, x0, p)
    # Affine mean: S is exact at every TME order (μ=0); v is the CIR Taylor series.
    kappa, theta, v0 = p.kappa, p.theta, x0[1]
    rem_factor = np.exp(-kappa * tau)
    taylor = 0.0
    term = 1.0
    for r in range(0, 4):
        if r:
            term *= (-kappa * tau) / r
        taylor += term
    m_v_tme3_theory = theta + (v0 - theta) * taylor

    m2, _ = tme_mean_cov(x0, tau, 2, drift, L)
    m3, P3 = tme_mean_cov(x0, tau, 3, drift, L)
    assert abs(m2[0] - m_exact[0]) < 1e-12
    assert abs(m3[0] - m_exact[0]) < 1e-12
    assert abs(m3[1] - m_v_tme3_theory) < 1e-10
    # Order-2 remainder on v is O(τ^3); order-3 is O(τ^4).
    err2 = abs(m2[1] - m_exact[1])
    err3 = abs(m3[1] - m_exact[1])
    assert err3 < err2
    # Covariance at small τ is PD and O(τ).
    assert np.min(np.linalg.eigvalsh(P3)) > 0.0


def test_tme2_cov_leading_term_is_gamma_dt():
    p = HestonParams()
    drift, L, *_ = make_heston_sv_dynamics(p, smooth_vol_eps=float(p.theta))
    x0 = np.array([np.exp(p.x0), float(p.theta)], dtype=np.float64)
    tau = 1e-4
    _, P2 = tme_mean_cov(x0, tau, 2, drift, L)
    G = np.asarray(L(jnp.asarray(0.0), jnp.asarray(x0)) @ L(jnp.asarray(0.0), jnp.asarray(x0)).T)
    rel = np.linalg.norm(P2 - G * tau) / (tau * np.linalg.norm(G))
    assert rel < 5e-3


def test_tme3_filter_stays_finite_on_stress_logprice():
    """Degree-3 Φ truncation can lose PD; the filter must eigen-clip and finish."""
    from dataclasses import replace

    from cd_dynamax_mirror.filters import UKFParams
    from heston_model import init_estimate, make_heston_dynamics, simulate_heston
    from predictive_nll_stepwise import run_tme_predictive_nll

    p = replace(HestonParams(xi=1.0, rho=-0.9, meas_std=0.02), diff_scale=1.75)
    _, ys, _ = simulate_heston(
        horizon=32 * 0.05, n_fine=256, p=p, seed=100, obs_stride=8
    )
    m0, P0 = init_estimate(p, true_state=np.array([0.0, 0.04]), cov_scale=1.0)
    drift, L, _Qc, h, R = make_heston_dynamics(p)
    ckf = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)
    pred, filt, means = run_tme_predictive_nll(
        drift=drift,
        L_fn=L,
        h=h,
        R=R,
        ys=jnp.asarray(ys),
        m0=jnp.asarray(m0),
        P0=jnp.asarray(P0),
        dt=0.05,
        order=3,
        ukf_params=ckf,
    )
    assert np.isfinite(float(pred))
    assert np.isfinite(float(filt))
    assert np.isfinite(np.asarray(means)).all()

