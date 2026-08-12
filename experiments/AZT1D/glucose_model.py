"""Minimal glucose–insulin SDE for AZT1D continuous–discrete filtering.

State ``x = [G, X]``:
  * ``G`` — plasma / interstitial glucose [mg/dL]
  * ``X`` — remote insulin action [1/min]

Known inputs (interpolated from pump / meal logs):
  * ``I(t)`` — insulin delivery rate [U/min]
  * ``C(t)`` — carbohydrate appearance rate [g/min]

Itô SDE (Bergman-style with multiplicative glucose noise)::

    dG = ( -p1 (G - Gb) - X G + k_c C(t) ) dt + σ_G √(max(G, ε)) dW₁
    dX = ( -p2 X + p3 I(t) ) dt + σ_X dW₂

    y_k = G_k + ε_k,   ε ~ N(0, R)

The ``X G`` bilinear term and √G diffusion are where moment-ODE EKF/UKF
freeze-at-mean approximations lose fidelity — the same structure as the MFTS /
Heston Pareto story, with Brownian dim ``d = 2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GlucoseParams:
    p1: float = 0.02  # 1/min glucose effectiveness
    p2: float = 0.03  # 1/min insulin-action clearance
    p3: float = 5e-4  # 1/min per (U/min) — insulin sensitivity scale
    Gb: float = 100.0  # mg/dL basal glucose
    k_c: float = 8.0  # (mg/dL) / g  appearance gain (spread via C(t))
    sigma_g: float = 1.5  # √(mg/dL) / √min  (√G diffusion)
    sigma_x: float = 2e-4  # 1/min / √min
    g_eps: float = 1.0  # mg/dL floor inside √G
    meas_std: float = 8.0  # mg/dL CGM noise
    x0: float = 0.0  # initial insulin action


def make_glucose_dynamics(
    p: GlucoseParams,
    *,
    t_min: np.ndarray | jnp.ndarray,
    insulin_u_per_min: np.ndarray | jnp.ndarray,
    carb_g_per_min: np.ndarray | jnp.ndarray,
    p3=None,
    sigma_g=None,
):
    """Return ``(drift, L, Qc, obs, R)`` with time-varying insulin/carb inputs.

    ``drift(t, x)`` / ``L(t, x)`` expect **absolute** time ``t`` in the same units
    as ``t_min`` (minutes from series start). The CD-EKF/UKF/EnKF/PF baselines
    pass absolute interval times; the CRF integrates each assimilation window in
    *relative* time ``[0, Δt]``, so wrap with :func:`shift_sde_time` using
    ``t0 = (step_index - 1) * Δt`` (see :func:`crf_time_offset`).

    ``p3`` (insulin sensitivity) and ``sigma_g`` (√G diffusion scale) may be JAX
    scalars so the filter likelihood is differentiable w.r.t. them; defaults
    fall back to ``p.p3`` / ``p.sigma_g``.
    """
    t_grid = jnp.asarray(t_min, dtype=jnp.float64)
    i_grid = jnp.asarray(insulin_u_per_min, dtype=jnp.float64)
    c_grid = jnp.asarray(carb_g_per_min, dtype=jnp.float64)
    t_lo = t_grid[0]
    t_hi = t_grid[-1]
    p1, p2 = p.p1, p.p2
    p3_ = p.p3 if p3 is None else p3
    Gb, k_c = p.Gb, p.k_c
    sigma_g_ = p.sigma_g if sigma_g is None else sigma_g
    sigma_x, g_eps = p.sigma_x, p.g_eps

    def _inputs(t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        tt = jnp.clip(t, t_lo, t_hi)
        return jnp.interp(tt, t_grid, i_grid), jnp.interp(tt, t_grid, c_grid)

    def drift_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        G, X = x[0], x[1]
        I_t, C_t = _inputs(t)
        dG = -p1 * (G - Gb) - X * G + k_c * C_t
        dX = -p2 * X + p3_ * I_t
        return jnp.array([dG, dX])

    def diffusion_coefficient_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        del t
        G = jnp.maximum(x[0], g_eps)
        return jnp.array(
            [
                [sigma_g_ * jnp.sqrt(G), 0.0],
                [0.0, sigma_x],
            ]
        )

    def diffusion_cov_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        del t, x
        return jnp.eye(2)

    def obs_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0]])

    R = jnp.array([[p.meas_std**2]])
    return drift_fn, diffusion_coefficient_fn, diffusion_cov_fn, obs_fn, R


def shift_sde_time(drift_fn, diffusion_fn, t0):
    """Map CRF relative time ``τ ∈ [0, Δt]`` to absolute ``t = τ + t0``.

    The cubature filter integrates each assimilation interval on ``[0, Δt]``.
    Pump / meal inputs are tabulated in absolute series time, so the SDE
    callbacks used by CRF must shift the integrator clock by the window start.
    """

    def drift_abs(t, x):
        return drift_fn(t + t0, x)

    def diffusion_abs(t, x):
        return diffusion_fn(t + t0, x)

    return drift_abs, diffusion_abs


def crf_time_offset(model_inputs, dt: float):
    """Absolute start time of the CRF assimilation window for ``model_inputs``.

    AZT1D wires ``model_inputs`` as step indices ``0..N`` (or a pytree with
    ``t_idx``), observing ``y[k-1]`` on step ``k≥1``. That matches the CD
    baselines' interval ``[(k-1)Δt, kΔt]``, so ``t0 = (k - 1) Δt``.
    """
    idx = model_inputs.t_idx if hasattr(model_inputs, "t_idx") else model_inputs
    return (idx - 1) * dt


def crf_glucose_dynamics(drift_fn, diffusion_fn, model_inputs, dt: float):
    """CRF ``get_dynamics`` helper: absolute-time drift/diffusion for one step."""
    return shift_sde_time(drift_fn, diffusion_fn, crf_time_offset(model_inputs, dt))


def init_estimate(cgm: np.ndarray, p: GlucoseParams) -> tuple[np.ndarray, np.ndarray]:
    """Prior at first CGM: G ≈ y₀, X ≈ 0, modest covariance."""
    g0 = float(cgm[0])
    m0 = np.array([g0, p.x0], dtype=np.float64)
    P0 = np.diag([p.meas_std**2, (5e-3) ** 2]).astype(np.float64)
    return m0, P0


def mape_glucose(est_G: np.ndarray, cgm: np.ndarray) -> float:
    """Mean absolute percentage error of filtered glucose vs CGM [%]."""
    y = np.asarray(cgm, dtype=np.float64)
    g = np.asarray(est_G, dtype=np.float64)
    # Align lengths: estimates often include t0 prior mean.
    n = min(g.shape[0], y.shape[0])
    if g.shape[0] == y.shape[0] + 1:
        g = g[1:]
        n = y.shape[0]
    denom = np.maximum(np.abs(y[:n]), 1.0)
    return float(np.mean(np.abs(g[:n] - y[:n]) / denom) * 100.0)


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Elementwise CRPS for univariate N(μ, σ²) forecasts (Gneiting & Raftery).

    ``CRPS = σ [ z (2Φ(z)-1) + 2φ(z) - 1/√π ]`` with ``z = (y-μ)/σ``.
    """
    from scipy.stats import norm

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64).reshape(-1), 1e-12)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _eabs_gaussian(mu: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
    """Elementwise ``E|N(μ, σ²)|``."""
    from scipy.stats import norm

    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    mu = np.asarray(mu, dtype=np.float64)
    z = mu / sigma
    return sigma * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z * z) + mu * (2.0 * norm.cdf(z) - 1.0)


def crps_gaussian_mixture(
    y: float | np.ndarray,
    means: np.ndarray,
    sigma: float,
    weights: np.ndarray | None = None,
) -> float:
    """CRPS of a univariate Gaussian mixture ``∑ w_i N(m_i, σ²)`` at observation ``y``.

    Uses the energy identity ``E|X-y| - ½ E|X-X'|`` with closed-form Gaussian
    absolute moments (proper for the mixture, unlike moment-matched Gaussian CRPS).
    """
    means = np.asarray(means, dtype=np.float64).reshape(-1)
    y = float(np.asarray(y, dtype=np.float64).reshape(-1)[0])
    sigma = float(max(sigma, 1e-12))
    n = means.shape[0]
    if weights is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w = w / np.sum(w)
    e1 = float(np.sum(w * _eabs_gaussian(means - y, sigma)))
    d = means[:, None] - means[None, :]
    e2 = float(np.sum((w[:, None] * w[None, :]) * _eabs_gaussian(d, sigma * np.sqrt(2.0))))
    return e1 - 0.5 * e2


def heldout_crps_per_obs(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_train: int,
) -> float:
    """Mean Gaussian CRPS on the held-out suffix ``y[n_train:]``."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    # Align optional t0 prior entry on mu/sigma (length n_obs+1).
    if mu.shape[0] == y.shape[0] + 1:
        mu = mu[1:]
        sigma = sigma[1:]
    n = min(y.shape[0], mu.shape[0], sigma.shape[0])
    if not (0 <= n_train < n):
        raise ValueError(f"n_train={n_train} invalid for n={n}")
    return float(np.mean(crps_gaussian(y[n_train:n], mu[n_train:n], sigma[n_train:n])))


def heldout_mean_score(scores: np.ndarray, n_train: int, n_obs: int | None = None) -> float:
    """Mean of a per-observation score on the held-out suffix."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if n_obs is None:
        n_obs = int(s.shape[0])
    if s.shape[0] == n_obs + 1:
        s = s[1:]
    n = min(int(s.shape[0]), n_obs)
    if not (0 <= n_train < n):
        raise ValueError(f"n_train={n_train} invalid for n={n}")
    return float(np.mean(s[n_train:n]))


def rmse(est: np.ndarray, truth: np.ndarray, cols: slice | None = None) -> float:
    a = est if cols is None else est[:, cols]
    b = truth if cols is None else truth[:, cols]
    n = min(a.shape[0], b.shape[0])
    if a.shape[0] == b.shape[0] + 1:
        a = a[1:]
        n = b.shape[0]
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def simulate_glucose_twin(
    p: GlucoseParams,
    *,
    t_min: np.ndarray,
    insulin_u_per_min: np.ndarray,
    carb_g_per_min: np.ndarray,
    n_substeps: int = 10,
    seed: int = 0,
    g0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler–Maruyama twin of the glucose Itô SDE.

    Returns
    -------
    traj : (T, 2)
        True state ``[G, X]`` at each ``t_min`` grid point.
    ys : (T, 1)
        Noisy CGM of ``G`` at the same times (matches real AZT1D layout).
    """
    t = np.asarray(t_min, dtype=np.float64)
    I = np.asarray(insulin_u_per_min, dtype=np.float64)
    C = np.asarray(carb_g_per_min, dtype=np.float64)
    if t.ndim != 1 or I.shape != t.shape or C.shape != t.shape:
        raise ValueError("t_min, insulin, carb must be 1D and aligned")
    if t.size < 2:
        raise ValueError("need at least 2 time points")

    rng = np.random.default_rng(seed)
    g = float(p.Gb if g0 is None else g0)
    x = float(p.x0)
    traj = [np.array([g, x], dtype=np.float64)]
    ys = [np.array([g + p.meas_std * rng.standard_normal()], dtype=np.float64)]

    for k in range(t.size - 1):
        dt_obs = float(t[k + 1] - t[k])
        dt = dt_obs / n_substeps
        for s in range(n_substeps):
            tau = t[k] + (s + 0.5) * dt
            I_t = float(np.interp(tau, t, I))
            C_t = float(np.interp(tau, t, C))
            g_pos = max(g, p.g_eps)
            dW = rng.normal(scale=np.sqrt(dt), size=2)
            dG = (-p.p1 * (g - p.Gb) - x * g + p.k_c * C_t) * dt + p.sigma_g * np.sqrt(
                g_pos
            ) * dW[0]
            dX = (-p.p2 * x + p.p3 * I_t) * dt + p.sigma_x * dW[1]
            g = float(g + dG)
            x = float(x + dX)
        traj.append(np.array([g, x], dtype=np.float64))
        ys.append(np.array([g + p.meas_std * rng.standard_normal()], dtype=np.float64))

    return np.asarray(traj, dtype=np.float64), np.stack(ys)
