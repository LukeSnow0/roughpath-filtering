"""Heston stochastic-volatility twin for continuous–discrete filtering.

State ``x = [X, v]`` with ``X = log S``:

    dX = (μ − v/2) dt + √v_ε dW¹
    dv = κ(θ − v) dt + ξ √v_ε (ρ dW¹ + √(1−ρ²) dW²)

    y_k = X_k + ε_k,   ε ~ N(0, R)

Correlation is encoded in the diffusion factor ``L`` with ``Qc = I₂``.
The floor ``v_ε = max(v, ε)`` keeps √v defined for filters that wander
slightly negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class HestonParams:
    mu: float = 0.0
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    v_eps: float = 1e-8
    meas_std: float = 0.02  # log-price observation noise
    diff_scale: float = 1.0  # multiplies L (process-noise power)
    x0: float = 0.0  # initial log-price
    v0: float | None = None  # default: theta


def make_heston_dynamics(p: HestonParams):
    """Return ``(drift, L, Qc, obs, R)`` for CD / CRF filters."""
    mu, kappa, theta, xi, rho = p.mu, p.kappa, p.theta, p.xi, p.rho
    v_eps = p.v_eps
    diff_scale = p.diff_scale
    rho_bar = float(np.sqrt(max(0.0, 1.0 - rho * rho)))

    def drift_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        v = x[1]
        return jnp.array([mu - 0.5 * v, kappa * (theta - v)])

    def L_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        sqrt_v = jnp.sqrt(jnp.maximum(x[1], v_eps))
        return diff_scale * jnp.array(
            [
                [sqrt_v, 0.0],
                [xi * rho * sqrt_v, xi * rho_bar * sqrt_v],
            ]
        )

    def Qc_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(2)

    def obs_fn(x: jnp.ndarray) -> jnp.ndarray:
        return x[:1]

    R = jnp.array([[p.meas_std**2]], dtype=jnp.float64)
    return drift_fn, L_fn, Qc_fn, obs_fn, R


def simulate_heston(
    *,
    horizon: float,
    n_fine: int,
    p: HestonParams,
    seed: int = 0,
    obs_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euler–Maruyama twin on a fine grid.

    Returns ``(true_fine, ys, times_fine)`` where
      * ``true_fine`` has shape ``(n_fine + 1, 2)`` (includes t=0),
      * ``ys`` has shape ``(n_obs, 1)`` at times ``stride, 2*stride, …``,
      * ``times_fine`` has shape ``(n_fine + 1,)``.
    """
    if n_fine < 1:
        raise ValueError("n_fine must be >= 1")
    if obs_stride < 1 or n_fine % obs_stride != 0:
        raise ValueError("obs_stride must divide n_fine")

    rng = np.random.default_rng(seed)
    dt = horizon / n_fine
    v0 = float(p.theta if p.v0 is None else p.v0)
    x = np.array([p.x0, v0], dtype=np.float64)
    rho, xi = p.rho, p.xi
    rho_bar = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    v_eps = p.v_eps

    true_fine = [x.copy()]
    times_fine = np.arange(n_fine + 1, dtype=np.float64) * dt
    ys_list: list[np.ndarray] = []

    for step in range(n_fine):
        v = max(x[1], v_eps)
        sqrt_v = np.sqrt(v)
        scale = p.diff_scale
        dW = rng.standard_normal(2) * np.sqrt(dt)
        dX = (p.mu - 0.5 * x[1]) * dt + scale * sqrt_v * dW[0]
        dv = p.kappa * (p.theta - x[1]) * dt + scale * xi * sqrt_v * (
            rho * dW[0] + rho_bar * dW[1]
        )
        x = np.array([x[0] + dX, x[1] + dv], dtype=np.float64)
        true_fine.append(x.copy())
        if (step + 1) % obs_stride == 0:
            y = np.array(
                [x[0] + p.meas_std * rng.standard_normal()],
                dtype=np.float64,
            )
            ys_list.append(y)

    return (
        np.asarray(true_fine, dtype=np.float64),
        np.stack(ys_list),
        times_fine,
    )


def init_estimate(
    p: HestonParams,
    *,
    true_state: np.ndarray | None = None,
    cov_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior mean/cov; slightly overdispersed on variance when starting from truth."""
    if true_state is not None:
        m0 = np.asarray(true_state, dtype=np.float64).copy()
    else:
        v0 = float(p.theta if p.v0 is None else p.v0)
        m0 = np.array([p.x0, v0], dtype=np.float64)
    # Log-price prior loose; variance prior ~ (0.5 θ)² scaled.
    P0 = np.diag([0.05**2, (0.5 * p.theta) ** 2]).astype(np.float64) * cov_scale
    return m0, P0


def rmse(estimates: np.ndarray, truth: np.ndarray) -> float:
    """Overall RMSE over all state components and time."""
    return float(np.sqrt(np.mean((estimates - truth) ** 2)))


def rmse_components(estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-component RMSE, shape ``(x_dim,)``."""
    err2 = (estimates - truth) ** 2
    return np.sqrt(np.mean(err2, axis=0))
