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


def _spd(A, jitter: float = 1e-12):
    """Symmetric eigen-clip onto the SPD cone (TME truncations need not be PD)."""
    A = 0.5 * (A + A.T)
    w, V = jnp.linalg.eigh(A)
    return (V * jnp.maximum(w, jitter)) @ V.T


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


def run_tme_predictive_nll(
    *,
    drift,
    L_fn,
    h,
    R,
    ys,
    m0,
    P0,
    dt,
    order: int,
    ukf_params: UKFParams,
):
    """TME Gaussian filter (Zhao et al., TAC 2021) with mixture predictive NLL."""
    from tme import assemble_tme_jax, cached_eval_fn

    n = m0.shape[0]
    alpha, beta, kappa = ukf_params.alpha, ukf_params.beta, ukf_params.kappa
    lamb = alpha**2 * (n + kappa) - n
    w_mean, w_cov, _W = _ukf_weights(n, alpha, beta, lamb)
    eval_all = cached_eval_fn(drift, L_fn, int(order))

    def moments_at(x):
        return assemble_tme_jax(eval_all(x), float(dt), int(order))

    def body(carry, step_k):
        m, P = carry
        X = _ukf_sigmas(m, P, n, lamb)
        a_i, S_i = vmap(moments_at)(X)
        S_i = vmap(_spd)(S_i)
        m_pred = w_mean @ a_i
        second = vmap(lambda a, S: S + jnp.outer(a, a))(a_i, S_i)
        P_pred = _spd(jnp.tensordot(w_mean, second, axes=1) - jnp.outer(m_pred, m_pred))

        def one_ll(a, S, w):
            z = h(a)
            H = jax.jacfwd(h)(a)
            Sy = psd(H @ S @ H.T + R)
            ll = gaussian_logpdf(ys[step_k], z, Sy)
            return jnp.where(w > 0, jnp.log(w) + ll, -jnp.inf)

        ll_pred = logsumexp(vmap(one_ll)(a_i, S_i, w_mean))
        m, P = _ekf_update(m_pred, P_pred, h, R, ys[step_k], 1)
        S_y = psd(jax.jacfwd(h)(m_pred) @ P_pred @ jax.jacfwd(h)(m_pred).T + R)
        ll_filt = gaussian_logpdf(ys[step_k], h(m_pred), S_y)
        return (m, P), ll_pred, ll_filt, m

    return _scan_nlls(body, (m0, P0), ys.shape[0])


def _kl_wdot(t, Z, horizon: float):
    """Truncated KL / Fourier derivative of Brownian motion on ``[0, horizon]``.

    ``Z`` has shape ``(n_modes, noise_dim)``. ``Ẇ_t = Σ_k Z_k √(2/T) cos(ω_k t)``.
    """
    n_modes = Z.shape[0]
    k = jnp.arange(1, n_modes + 1, dtype=t.dtype)
    omega = (k - 0.5) * jnp.pi / horizon
    coef = jnp.sqrt(2.0 / horizon) * jnp.cos(omega * t)
    return coef @ Z


def run_seukf_predictive_nll(
    *,
    drift,
    L_fn,
    h,
    R,
    ys,
    m0,
    P0,
    dt,
    dt0,
    n_modes: int,
    ukf_params: UKFParams,
):
    """Series-expansion UKF (Lyons, Särkkä, Storkey, TSP 2014).

    Truncated Fourier / KL Brownian motion; one unscented transform of the joint
    ``(x_0, Z_{1:N})`` per assimilation interval; Wong–Zakai ODE in the
    Stratonovich interpretation.
    """
    from wsckf.library import ito_to_stratonovich_dynamics

    strat_drift, strat_L = ito_to_stratonovich_dynamics(drift, L_fn)
    n_x = int(m0.shape[0])
    n_modes = int(n_modes)
    noise_dim = 2
    n_z = n_modes * noise_dim
    n_aug = n_x + n_z
    alpha, beta, kappa = ukf_params.alpha, ukf_params.beta, ukf_params.kappa
    lamb = alpha**2 * (n_aug + kappa) - n_aug
    w_mean, w_cov, _W = _ukf_weights(n_aug, alpha, beta, lamb)
    n_inner = _rk4_n_steps(dt, dt0)
    horizon = jnp.asarray(dt, dtype=jnp.float64)

    def ode_rhs(t, x, Z):
        wdot = _kl_wdot(t, Z, horizon)
        return strat_drift(t, x) + strat_L(t, x) @ wdot

    def integrate_one(x0, Z):
        def body(x, i):
            ti = (i.astype(x.dtype)) * (horizon / n_inner)
            t1 = ti + horizon / n_inner
            dtp = horizon / n_inner
            t_mid = ti + 0.5 * dtp
            k1 = ode_rhs(ti, x, Z)
            k2 = ode_rhs(t_mid, x + 0.5 * dtp * k1, Z)
            k3 = ode_rhs(t_mid, x + 0.5 * dtp * k2, Z)
            k4 = ode_rhs(t1, x + dtp * k3, Z)
            return x + (dtp / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

        x1, _ = lax.scan(body, x0, jnp.arange(n_inner))
        return x1

    def body(carry, step_k):
        m, P = carry
        m_aug = jnp.concatenate([m, jnp.zeros(n_z, dtype=m.dtype)])
        P_aug = jnp.block(
            [
                [P, jnp.zeros((n_x, n_z), dtype=P.dtype)],
                [jnp.zeros((n_z, n_x), dtype=P.dtype), jnp.eye(n_z, dtype=P.dtype)],
            ]
        )
        sigmas = _ukf_sigmas(m_aug, P_aug, n_aug, lamb)
        X0 = sigmas[:, :n_x]
        Z = sigmas[:, n_x:].reshape(sigmas.shape[0], n_modes, noise_dim)
        Xf = vmap(integrate_one)(X0, Z)
        m_pred = w_mean @ Xf
        P_pred = psd(jnp.tensordot(w_cov, _outer_pts(Xf, m_pred), axes=1))
        ll_pred = mixture_obs_logpdf(ys[step_k], Xf, w_mean, h, R)
        n = n_x
        lamb_x = ukf_params.alpha**2 * (n + ukf_params.kappa) - n
        w_mx, w_cx, _ = _ukf_weights(n, ukf_params.alpha, ukf_params.beta, lamb_x)
        ll_filt, m, P, _, _ = _ukf_update(
            m_pred, P_pred, h, R, ys[step_k], n, lamb_x, w_mx, w_cx, 1
        )
        return (m, P), ll_pred, ll_filt, m

    return _scan_nlls(body, (m0, P0), ys.shape[0])


def _outer_pts(X, m):
    d = X - m
    return vmap(lambda e: jnp.outer(e, e))(d)
