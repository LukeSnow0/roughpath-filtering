"""Modified four-tank system (MFTS) from Nielsen et al. (2022), Sec. IV.

State x = [m1, m2, m3, m4, F3, F4]^T  (tank masses [g], disturbance flows [cm^3/s]).
Observations: tank heights h_i = beta_i * m_i [cm], additive Gaussian noise.

Truth simulation uses a piecewise F-bar schedule (100 -> 200 -> 300 cm^3/s); estimators
use constant F-bar = 150 (model mismatch, as in the Nielsen benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MFTSParams:
    gamma1: float = 0.45
    gamma2: float = 0.40
    rho: float = 1.0
    g: float = 981.0
    A: tuple[float, float, float, float] = (380.1327, 380.1327, 380.1327, 380.1327)
    a_out: tuple[float, float, float, float] = (1.2272, 1.2272, 1.2272, 1.2272)
    meas_std: float = 2.0  # cm
    F_nominal_est: float = 150.0
    mass_eps: float = 1e-6


def betas(p: MFTSParams) -> np.ndarray:
    return np.array([1.0 / (p.rho * A_i) for A_i in p.A])


def alphas(p: MFTSParams) -> np.ndarray:
    beta = betas(p)
    return np.array(
        [a_i * np.sqrt(2.0 * p.g * beta_i) for a_i, beta_i in zip(p.a_out, beta)]
    )


def mfts_steady_masses(F1: float, F2: float, F3: float, F4: float, p: MFTSParams) -> np.ndarray:
    alpha = alphas(p)
    m3 = (((1.0 - p.gamma2) * F2 + F3) / alpha[2]) ** 2
    m4 = (((1.0 - p.gamma1) * F1 + F4) / alpha[3]) ** 2
    q3 = alpha[2] * np.sqrt(max(m3, p.mass_eps))
    q4 = alpha[3] * np.sqrt(max(m4, p.mass_eps))
    m1 = ((p.gamma1 * F1 + q3) / alpha[0]) ** 2
    m2 = ((p.gamma2 * F2 + q4) / alpha[1]) ** 2
    return np.array([m1, m2, m3, m4], dtype=np.float64)


def bar_F_schedule(t_min: float, p: MFTSParams) -> float:
    if t_min < 10.0:
        return 100.0
    if t_min < 20.0:
        return 200.0
    return 300.0


def make_mfts_dynamics(
    p: MFTSParams,
    *,
    lambda_dist: float,
    sigma_dist: float,
    bar_F3: Callable[[float], float],
    bar_F4: Callable[[float], float],
    F1: float = 5.0,
    F2: float = 5.0,
):
    """Return drift, diffusion L, Qc, observation map, and R for continuous-discrete filters."""
    alpha = jnp.array(alphas(p))
    beta = jnp.array(betas(p))
    g1, g2 = p.gamma1, p.gamma2
    eps = p.mass_eps

    def _q_out(m):
        return alpha * jnp.sqrt(jnp.maximum(m, eps))

    def drift_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        m = x[:4]
        F3, F4 = x[4], x[5]
        t_min = t / 60.0
        q = _q_out(m)
        dm1 = g1 * F1 + (1.0 - g2) * F2 + q[2] - q[0]
        dm2 = g2 * F2 + (1.0 - g1) * F1 + q[3] - q[1]
        dm3 = (1.0 - g2) * F2 + F3 - q[2]
        dm4 = (1.0 - g1) * F1 + F4 - q[3]
        b3 = bar_F3(t_min)
        b4 = bar_F4(t_min)
        dF3 = lambda_dist * (b3 - F3)
        dF4 = lambda_dist * (b4 - F4)
        return jnp.array([dm1, dm2, dm3, dm4, dF3, dF4])

    def diffusion_coefficient_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [sigma_dist, 0.0],
                [0.0, sigma_dist],
            ]
        )

    def diffusion_cov_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(2)

    def obs_fn(x: jnp.ndarray) -> jnp.ndarray:
        return beta * x[:4]

    R = jnp.diag(jnp.full(4, p.meas_std**2))
    return drift_fn, diffusion_coefficient_fn, diffusion_cov_fn, obs_fn, R


def simulate_mfts(
    *,
    horizon_min: float,
    n_obs: int,
    n_substeps_truth: int,
    p: MFTSParams,
    lambda_sim: float = 0.1,
    sigma_sim: float = 5.0,
    F1: float = 5.0,
    F2: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, jnp.ndarray]:
    """Euler-Maruyama truth; returns (true_traj, observations, R)."""
    rng = np.random.default_rng(seed)
    dt_obs_sec = (horizon_min / n_obs) * 60.0
    dt_sub = dt_obs_sec / n_substeps_truth

    m0 = mfts_steady_masses(F1, F2, p.F_nominal_est, p.F_nominal_est, p)
    x = np.array([*m0, p.F_nominal_est, p.F_nominal_est], dtype=np.float64)
    alpha = alphas(p)
    true_traj = [x.copy()]
    observations: list[np.ndarray] = []
    R = np.diag(np.full(4, p.meas_std**2))

    def drift_np(t_sec: float, state: np.ndarray) -> np.ndarray:
        m = state[:4]
        F3, F4 = state[4], state[5]
        q = alpha * np.sqrt(np.maximum(m, p.mass_eps))
        dm1 = p.gamma1 * F1 + (1 - p.gamma2) * F2 + q[2] - q[0]
        dm2 = p.gamma2 * F2 + (1 - p.gamma1) * F1 + q[3] - q[1]
        dm3 = (1 - p.gamma2) * F2 + F3 - q[2]
        dm4 = (1 - p.gamma1) * F1 + F4 - q[3]
        t_min = t_sec / 60.0
        b3 = bar_F_schedule(t_min, p)
        b4 = bar_F_schedule(t_min, p)
        dF3 = lambda_sim * (b3 - F3)
        dF4 = lambda_sim * (b4 - F4)
        return np.array([dm1, dm2, dm3, dm4, dF3, dF4])

    t_sec = 0.0
    for _ in range(n_obs):
        for _ in range(n_substeps_truth):
            f = drift_np(t_sec, x)
            dW = rng.normal(scale=np.sqrt(dt_sub), size=2)
            x[:4] += f[:4] * dt_sub
            x[4] += f[4] * dt_sub + sigma_sim * dW[0]
            x[5] += f[5] * dt_sub + sigma_sim * dW[1]
            x[:4] = np.maximum(x[:4], p.mass_eps)
            t_sec += dt_sub
        y = betas(p) * x[:4] + rng.multivariate_normal(np.zeros(4), R)
        observations.append(y.copy())
        true_traj.append(x.copy())

    return np.array(true_traj), np.stack(observations), jnp.asarray(R)


def init_estimate(
    p: MFTSParams,
    F1: float,
    F2: float,
    *,
    true_state: np.ndarray | None = None,
    cov_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if true_state is not None:
        x0 = np.asarray(true_state, dtype=np.float64)
    else:
        m0 = mfts_steady_masses(F1, F2, p.F_nominal_est, p.F_nominal_est, p)
        x0 = np.array([*m0, p.F_nominal_est, p.F_nominal_est])
    P0 = (np.diag([1.0, 1.0, 1.0, 1.0, 25.0, 25.0]).astype(np.float64)) * cov_scale
    return x0, P0


def mape_pct(
    estimates: np.ndarray,
    true_traj: np.ndarray,
    state_idx: slice,
    *,
    eps: float = 1e-3,
) -> float:
    err = estimates[:, state_idx] - true_traj[:, state_idx]
    denom = np.maximum(np.abs(true_traj[:, state_idx]), eps)
    return float(100.0 * np.mean(np.abs(err / denom)))
