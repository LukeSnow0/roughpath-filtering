"""Stepwise filter passes with mixture predictive observation NLL.

At each assimilation step, after the predict phase and before update:

    log p(y_k | y_{1:k-1}) ≈ log Σ_j w_j N(y_k; h(x_j), R)

using the forecast cloud (WSC cubature paths, EnKF/PF particles, UKF sigma
points). This differs from the default filter score when the update collapses
that cloud to a Gaussian innovation (WSC, EnKF).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.scipy.special import logsumexp

from cd_dynamax_mirror.filters import (
    UKFParams,
    _ekf_predict,
    _ekf_update,
    _enkf_analysis,
    _enkf_forecast,
    _rk4_n_steps,
    _systematic_resample,
    _ukf_predict,
    _ukf_update,
    _ukf_sigmas,
    _ukf_weights,
    gaussian_logpdf,
    psd,
)
from wsckf.filter.filtering import predict, update
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths


def mixture_obs_logpdf(y, samples, weights, h, R):
    """Log density of y under Σ_j w_j N(h(x_j), R)."""
    weights = jnp.asarray(weights, dtype=jnp.float64)
    weights = weights / jnp.sum(weights)
    z = vmap(h)(samples)
    y = jnp.reshape(y, z.shape[1:])

    def one(zj):
        return gaussian_logpdf(y, zj, R)

    log_terms = vmap(one)(z)
    return logsumexp(jnp.log(weights) + log_terms)


def _scan_nlls(body_fn, init, n_obs: int):
    def step(carry, k):
        carry, pred_ll_k, filt_ll_k, mean_k = body_fn(carry, k)
        return carry, (pred_ll_k, filt_ll_k, mean_k)

    _, (pred_lls, filt_lls, means) = lax.scan(step, init, jnp.arange(n_obs))
    pred_nll = -jnp.sum(pred_lls) / float(n_obs)
    filt_nll = -jnp.sum(filt_lls) / float(n_obs)
    return pred_nll, filt_nll, means


def run_wsc_predictive_nll(
    *,
    drift,
    L_fn,
    h,
    R,
    chol_R,
    ys,
    m0,
    chol_P0,
    dt,
    dt0,
    lyons_degree: int,
    k: int,
):
    strat_drift, strat_L = ito_to_stratonovich_dynamics(drift, L_fn)
    ts, path_values, path_weights = precompute_cubature_paths(
        horizon=float(dt),
        noise_dim=2,
        integration_dt=float(dt0),
        cubature_order=int(k),
        multidim_method="lv",
        lyons_degree=int(lyons_degree),
    )

    def body(carry, step_k):
        m, chol = carry
        pred_mean, pred_chol, samples, weights = predict(
            m,
            chol,
            ts,
            path_values,
            path_weights,
            strat_drift,
            strat_L,
        )
        ll_pred = mixture_obs_logpdf(ys[step_k], samples, weights, h, R)
        m, chol, ll_filt, _, _ = update(pred_mean, samples, weights, h, chol_R, ys[step_k])
        return (m, chol), ll_pred, ll_filt, m

    return _scan_nlls(body, (m0, chol_P0), ys.shape[0])


def run_ekf_predictive_nll(
    *,
    drift,
    L_fn,
    Qc_fn,
    h,
    R,
    ys,
    m0,
    P0,
    dt,
    dt0,
):
    n_inner = _rk4_n_steps(dt, dt0)

    def body(carry, step_k):
        m, P = carry
        t0 = step_k * dt
        m_pred, P_pred = _ekf_predict(
            m, P, drift, L_fn, Qc_fn, t0, t0 + dt, dt0,
            "flowmap", solver="rk4", n_steps=n_inner,
        )
        H = jax.jacfwd(h)(m_pred)
        S = psd(H @ P_pred @ H.T + R)
        ll_pred = gaussian_logpdf(ys[step_k], h(m_pred), S)
        m, P = _ekf_update(m_pred, P_pred, h, R, ys[step_k], 1)
        return (m, P), ll_pred, ll_pred, m

    return _scan_nlls(body, (m0, P0), ys.shape[0])


def _ukf_mixture_ll(y, m_pred, P_pred, h, R, n, lamb, w_mean):
    sigmas = _ukf_sigmas(m_pred, P_pred, n, lamb)
    return mixture_obs_logpdf(y, sigmas, w_mean, h, R)


def run_ukf_predictive_nll(
    *,
    drift,
    L_fn,
    Qc_fn,
    h,
    R,
    ys,
    m0,
    P0,
    dt,
    dt0,
    ukf_params: UKFParams,
):
    n = m0.shape[0]
    alpha, beta, kappa = ukf_params.alpha, ukf_params.beta, ukf_params.kappa
    lamb = alpha**2 * (n + kappa) - n
    w_mean, w_cov, W_matrix = _ukf_weights(n, alpha, beta, lamb)
    n_inner = _rk4_n_steps(dt, dt0)

    def body(carry, step_k):
        m, P = carry
        t0 = step_k * dt
        m_pred, P_pred = _ukf_predict(
            m, P, drift, L_fn, Qc_fn, t0, t0 + dt, dt0, n, lamb,
            w_mean, w_cov, W_matrix, "flowmap", solver="rk4", n_steps=n_inner,
        )
        ll_pred = _ukf_mixture_ll(ys[step_k], m_pred, P_pred, h, R, n, lamb, w_mean)
        ll_filt, m, P, _, _ = _ukf_update(m_pred, P_pred, h, R, ys[step_k], n, lamb, w_mean, w_cov, 1)
        return (m, P), ll_pred, ll_filt, m

    return _scan_nlls(body, (m0, P0), ys.shape[0])


def run_enkf_predictive_nll(
    *,
    drift,
    L_fn,
    Qc_fn,
    h,
    R,
    ys,
    m0,
    P0,
    dt,
    dt0,
    key,
    n_particles: int,
):
    n_inner = _rk4_n_steps(dt, dt0)
    key, init_key = jr.split(key)
    particles = jr.multivariate_normal(init_key, mean=m0, cov=psd(P0), shape=(n_particles,))
    weights = jnp.full(n_particles, 1.0 / n_particles)

    def body(carry, step_k):
        parts, key = carry
        key, k_pred, k_filt = jr.split(key, 3)
        t0 = step_k * dt
        x_pred = _enkf_forecast(
            k_pred, parts, drift, L_fn, Qc_fn, t0, t0 + dt, dt0,
            2, 1.0, solver="rk4", n_steps=n_inner, sde_form="ito",
        )
        ll_pred = mixture_obs_logpdf(ys[step_k], x_pred, weights, h, R)
        x_filt, y_mean, S = _enkf_analysis(k_filt, x_pred, h, R, ys[step_k], True, 0.0)
        ll_filt = gaussian_logpdf(ys[step_k], y_mean, S)
        return (x_filt, key), ll_pred, ll_filt, jnp.mean(x_filt, axis=0)

    return _scan_nlls(body, (particles, key), ys.shape[0])


def run_pf_predictive_nll(
    *,
    drift,
    L_fn,
    Qc_fn,
    h,
    R,
    ys,
    m0,
    P0,
    dt,
    dt0,
    key,
    n_particles: int,
):
    """Equal-weight mixture NLL (matches bootstrap PF evidence increment)."""
    n_inner = _rk4_n_steps(dt, dt0)
    key, init_key = jr.split(key)
    particles = jr.multivariate_normal(init_key, mean=m0, cov=psd(P0), shape=(n_particles,))
    weights = jnp.full(n_particles, 1.0 / n_particles)

    def body(carry, step_k):
        parts, key = carry
        key, k_pred, k_res = jr.split(key, 3)
        t0 = step_k * dt
        x_pred = _enkf_forecast(
            k_pred, parts, drift, L_fn, Qc_fn, t0, t0 + dt, dt0,
            2, 1.0, solver="rk4", n_steps=n_inner, sde_form="ito",
        )
        ll_pred = mixture_obs_logpdf(ys[step_k], x_pred, weights, h, R)
        log_w = vmap(lambda x: gaussian_logpdf(ys[step_k], h(x), R))(x_pred)
        ll_filt = logsumexp(log_w) - jnp.log(n_particles)
        w = jax.nn.softmax(log_w)
        idx = _systematic_resample(k_res, w)
        return (x_pred[idx], key), ll_pred, ll_filt, jnp.sum(w[:, None] * x_pred, axis=0)

    return _scan_nlls(body, (particles, key), ys.shape[0])
