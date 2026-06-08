#!/usr/bin/env python3
"""
Benchmark Wiener cubature (CRF) against Nielsen et al. (2022) continuous-discrete filters.

Reference: https://arxiv.org/abs/2205.02730
  State Estimation Methods for Continuous-Discrete Nonlinear Systems
  involving Stochastic Differential Equations (MFTS example, Sec. IV).

Model equations (modified four-tank system, Azam 2015 / Nielsen Sec. IV)
------------------------------------------------------------------------
State x = [m1, m2, m3, m4, F3, F4]^T with tank masses mi [g] (rho=1) and disturbance
flows F3, F4 [cm^3/s]. Known pump inputs F1, F2 [cm^3/s] (defaults 5, 5).

Geometry (tank i = 1..4, parameters in MFTSParams / Table II):
  beta_i = 1 / (rho * A_i)           # mass -> height [cm]
  alpha_i = a_i * sqrt(2 * g * beta_i)
  q_i(m_i) = alpha_i * sqrt(max(m_i, eps))   # Torricelli outflow [cm^3/s]

Valve split gamma1, gamma2 (Table I) routes F1, F2 between tanks.

SDE drift f(t, x) — mass balances integrated in **seconds**; bar_F evaluated at
t_min = t / 60 (minutes):
  dm1/dt = gamma1*F1 + (1-gamma2)*F2 + q3(m3) - q1(m1)
  dm2/dt = gamma2*F2 + (1-gamma1)*F1 + q4(m4) - q2(m2)
  dm3/dt = (1-gamma2)*F2 + F3 - q3(m3)
  dm4/dt = (1-gamma1)*F1 + F4 - q4(m4)
  dF3/dt = lambda * (bar_F3(t_min) - F3)
  dF4/dt = lambda * (bar_F4(t_min) - F4)

Diffusion (two independent Wiener drivers on F3, F4 only):
  dx = f(t,x) dt + L dW,   L in R^{6x2} with L[5,0]=L[6,1]=sigma, else zero.

Observations at discrete times k (heights [cm], Azam eq. 21):
  y_k = [h1, h2, h3, h4]^T + v_k,   hi = beta_i * mi,
  v_k ~ N(0, meas_std^2 * I_4).

Truth simulation: bar_F3 = bar_F4 = F_bar(t) piecewise 100 -> 200 -> 300 cm^3/s
on [0,10), [10,20), [20,30) minutes. Estimators use constant bar_F = 150.

Continuous-discrete filtering: predict by integrating the SDE over each interval
Dt; update with the linear Gaussian measurement above (see make_mfts_dynamics).

Simulation environment (paper Sec. IV-A):
  - Modified four-tank system with OU disturbances F3, F4 as explicit states
  - 30 min horizon, 120 equidistant observations
  - Truth integration: 1000 steps per observation interval; filters: 100 steps
  - Methods: CD-EKF (Joseph), CD-UKF (stable UT), cubature KF (spatial nodes + SDE predict),
    perturbed-observation EnKF (N=250), bootstrap PF (N=1000), Wiener CRF
  - Metrics: MAPE on tank masses (MAPEx) and disturbances (MAPEd), plus TU/MU times

Usage:
    export JAX_ENABLE_X64=1
    python nielsen_mfts_filter_comparison.py
    python nielsen_mfts_filter_comparison.py --fast
    python nielsen_mfts_filter_comparison.py --noisy --show-plot
    python nielsen_mfts_filter_comparison.py --meas-std 6 --sigma-sim 10 --no-twin
    python nielsen_mfts_filter_comparison.py --cubature-paths 32 --cubature-order 2
    python nielsen_mfts_filter_comparison.py --cubature-loop --no-plot
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jacfwd, vmap
from jax.scipy.special import logsumexp

from cd_dynamax_enkf import (
    CDDynamaxEnKF,
    EnKFHyperParams,
    diffeqsolve,
    enkf_condition_on,
    enkf_predict,
    psd,
    psd_solve,
)
from jax_crf import JaxCubatureRecursiveFilter
from wiener_cubature import (
    default_recombined_num_paths,
    spatial_cubature_nodes,
    weighted_mean_cov,
)

jax.config.update("jax_enable_x64", True)

_outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)


# ---------------------------------------------------------------------------
# Modified four-tank system (Azam & Jørgensen / Nielsen et al.)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MFTSParams:
    """Table II (Azam 2015) + RHP valve split (Table I)."""

    gamma1: float = 0.45
    gamma2: float = 0.40
    rho: float = 1.0
    g: float = 981.0
    A: tuple[float, float, float, float] = (380.1327, 380.1327, 380.1327, 380.1327)
    a_out: tuple[float, float, float, float] = (1.2272, 1.2272, 1.2272, 1.2272)
    meas_std: float = 2.0  # cm, diagonal R (Azam eq. 21)
    F_nominal_est: float = 150.0  # constant mean for F3,F4 in estimators
    mass_eps: float = 1e-6


def _betas(p: MFTSParams) -> np.ndarray:
    return np.array([1.0 / (p.rho * A_i) for A_i in p.A])


def _alphas(p: MFTSParams) -> np.ndarray:
    beta = _betas(p)
    return np.array(
        [a_i * np.sqrt(2.0 * p.g * beta_i) for a_i, beta_i in zip(p.a_out, beta)]
    )


def mfts_steady_masses(
    F1: float,
    F2: float,
    F3: float,
    F4: float,
    p: MFTSParams,
) -> np.ndarray:
    """Steady-state tank masses [m1..m4] from mass balances."""
    alpha = _alphas(p)
    m3 = (((1.0 - p.gamma2) * F2 + F3) / alpha[2]) ** 2
    m4 = (((1.0 - p.gamma1) * F1 + F4) / alpha[3]) ** 2
    q3 = alpha[2] * np.sqrt(max(m3, p.mass_eps))
    q4 = alpha[3] * np.sqrt(max(m4, p.mass_eps))
    m1 = ((p.gamma1 * F1 + q3) / alpha[0]) ** 2
    m2 = ((p.gamma2 * F2 + q4) / alpha[1]) ** 2
    return np.array([m1, m2, m3, m4], dtype=np.float64)


def bar_F_schedule(t_min: float, p: MFTSParams) -> float:
    """Truth simulation: F̄_i cycles {100, 200, 300} over 30 min (paper Fig. 2)."""
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
    """
    Augmented state x = [m1, m2, m3, m4, F3, F4].
    Tank dynamics (mass balances, rho=1). Disturbances: dF_i = lambda*(bar_F_i(t)-F_i) dt + sigma_i dW_i.
    """
    alpha = jnp.array(_alphas(p))
    beta = jnp.array(_betas(p))
    g1, g2 = p.gamma1, p.gamma2
    eps = p.mass_eps

    def _q_out(m):
        return alpha * jnp.sqrt(jnp.maximum(m, eps))

    def drift_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        m = x[:4]
        F3, F4 = x[4], x[5]
        t_min = t / 60.0  # state equations use minute-scale inputs in the paper
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
        # Noise acts on disturbance states only (2 Wiener drivers).
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
) -> tuple[np.ndarray, list[np.ndarray], jnp.ndarray]:
    """
    Euler–Maruyama truth with explicit F̄(t) schedule (1000 steps / interval).

    Mass balances use cm³/s flows integrated in **seconds** (consistent with the
    JAX filters). Time for F̄ scheduling is in minutes (paper Fig. 2).
    """
    rng = np.random.default_rng(seed)
    dt_obs_sec = (horizon_min / n_obs) * 60.0
    dt_sub = dt_obs_sec / n_substeps_truth

    m0 = mfts_steady_masses(F1, F2, p.F_nominal_est, p.F_nominal_est, p)
    x = np.array([*m0, p.F_nominal_est, p.F_nominal_est], dtype=np.float64)
    alpha = _alphas(p)
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
        beta = _betas(p)
        y = beta * x[:4] + rng.multivariate_normal(np.zeros(4), R)
        observations.append(y.copy())
        true_traj.append(x.copy())

    return np.array(true_traj), observations, jnp.asarray(R)


# ---------------------------------------------------------------------------
# Metrics (Nielsen MAPE, eq. 51)
# ---------------------------------------------------------------------------


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


def timing_row(name: str, tu: float, mu: float, mapex: float, maped: float) -> dict:
    return {
        "name": name,
        "time_tu_s": tu,
        "time_mu_s": mu,
        "mapex_pct": mapex,
        "maped_pct": maped,
    }


# ---------------------------------------------------------------------------
# CD-EKF with Joseph stabilising covariance update (paper eq. 9c)
# ---------------------------------------------------------------------------


def ekf_predict(
    m: jnp.ndarray,
    P: jnp.ndarray,
    *,
    drift_fn: Callable,
    L_fn: Callable,
    Qc_fn: Callable,
    t0: float,
    t1: float,
    dt0: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = m.shape[0]

    def rhs(t, y, args):
        m_t, P_t = y
        F_t = jacfwd(lambda x: drift_fn(t, x))(m_t)
        L_t = L_fn(t, m_t)
        Qc_t = Qc_fn(t, m_t)
        dmdt = drift_fn(t, m_t)
        dPdt = F_t @ P_t + P_t @ F_t.T + L_t @ Qc_t @ L_t.T
        return (dmdt, dPdt)

    sol = diffeqsolve(rhs, t0, t1, (m, P), dt0=dt0)
    return sol[0][0], psd(sol[1][0])


def ekf_update_joseph(
    m: jnp.ndarray,
    P: jnp.ndarray,
    y: jnp.ndarray,
    obs_fn: Callable,
    R: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    H = jacfwd(obs_fn)(m)
    z = obs_fn(m)
    S = psd(R + H @ P @ H.T)
    K = psd_solve(S, H @ P).T
    I = jnp.eye(m.shape[0])
    m_new = m + K @ (y - z)
    IK = I - K @ H
    P_new = psd(IK @ P @ IK.T + K @ R @ K.T)
    return m_new, P_new


# ---------------------------------------------------------------------------
# CD-UKF first-order (Nielsen Sec. III-B; augmented noise sigma-points)
# ---------------------------------------------------------------------------


def ukf_nielsen_weights(n: int, n_omega: int, alpha: float, beta: float, kappa: float):
    n_bar = n + n_omega
    c = alpha**2 * (n + kappa)
    lamb = alpha**2 * (n_bar + kappa) - n_bar
    factor = 1.0 / (2.0 * (n_bar + lamb))
    w_mean = jnp.concatenate((jnp.array([lamb / (n_bar + lamb)]), jnp.ones(2 * n_bar) * factor))
    w_cov = jnp.concatenate(
        (
            jnp.array([lamb / (n_bar + lamb) + (1.0 - alpha**2 + beta)]),
            jnp.ones(2 * n_bar) * factor,
        )
    )
    return float(c), float(lamb), w_mean, w_cov


def ukf_nielsen_sigmas(m: jnp.ndarray, P: jnp.ndarray, n: int, c: float) -> jnp.ndarray:
    """Sigma points via eigendecomposition (stable for alpha=0.001, kappa=0)."""
    P = psd(P)
    scale = jnp.sqrt(jnp.maximum(n + c, 1e-12))
    w, V = jnp.linalg.eigh(P)
    w = jnp.maximum(w, 1e-12)
    axes = V * (scale * jnp.sqrt(w))
    plus = jnp.array([m + axes[:, i] for i in range(n)])
    minus = jnp.array([m - axes[:, i] for i in range(n)])
    return jnp.concatenate((jnp.array([m]), plus, minus))


def ukf_nielsen_predict(
    m: jnp.ndarray,
    P: jnp.ndarray,
    *,
    drift_fn: Callable,
    L_fn: Callable,
    Qc_fn: Callable,
    t0: float,
    t1: float,
    c: float,
    w_mean: jnp.ndarray,
    w_cov: jnp.ndarray,
    n_omega: int,
    dt0: float,
    key,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Nielsen Sec. III-B1: 2n+1 points through drift ODE (eq. 16);
    2 n_omega pairs through SDE with injected dω (eq. 17).
    """
    n = m.shape[0]
    X_det = ukf_nielsen_sigmas(m, P, n, c)  # 2n+1

    def propagate_ode(x0):
        sol = diffeqsolve(lambda t, y, args: drift_fn(t, y), t0, t1, x0, dt0=dt0)
        return sol[0, :]

    X_det_pred = vmap(propagate_ode)(X_det)

    # Stochastic sigma-points at fixed mean; noise enters via diffusion over [t0, t1]
    X_stoch = jnp.tile(m[None, :], (2 * n_omega, 1))

    def combined_diff(t, y, args):
        return L_fn(t, y) @ jnp.linalg.cholesky(psd(Qc_fn(t, y)))

    key, *subkeys = jr.split(key, 2 * n_omega + 1)

    def propagate_sde(x0, k0):
        sol = diffeqsolve(
            lambda t, y, args: drift_fn(t, y),
            t0,
            t1,
            x0,
            diffusion=combined_diff,
            key=k0,
            noise_dim=n_omega,
            dt0=dt0,
        )
        return sol[0, :]

    X_stoch_pred = vmap(propagate_sde)(X_stoch, jnp.stack(subkeys))
    X_pred = jnp.concatenate((X_det_pred, X_stoch_pred), axis=0)
    m_pred = jnp.tensordot(w_mean, X_pred, axes=1)
    P_pred = jnp.tensordot(w_cov, _outer(X_pred - m_pred, X_pred - m_pred), axes=1)
    return m_pred, psd(P_pred)


def ukf_nielsen_update(
    m: jnp.ndarray,
    P: jnp.ndarray,
    y: jnp.ndarray,
    obs_fn: Callable,
    R: jnp.ndarray,
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = m.shape[0]
    c = alpha**2 * (n + kappa)
    lamb = alpha**2 * (n + kappa) - n
    factor = 1.0 / (2.0 * (n + lamb))
    w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
    w_cov = jnp.concatenate(
        (
            jnp.array([lamb / (n + lamb) + (1.0 - alpha**2 + beta)]),
            jnp.ones(2 * n) * factor,
        )
    )
    sigmas = ukf_nielsen_sigmas(m, P, n, c)
    z_sig = vmap(obs_fn)(sigmas)
    z_mean = jnp.tensordot(w_mean, z_sig, axes=1)
    dz = z_sig - z_mean
    S = psd(R + jnp.tensordot(w_cov, _outer(dz, dz), axes=1))
    cross = jnp.tensordot(w_cov, _outer(sigmas - m, dz), axes=1)
    K = psd_solve(S, cross.T).T
    m_upd = m + K @ (y - z_mean)
    H = jacfwd(obs_fn)(m)
    I = jnp.eye(n)
    IK = I - K @ H
    P_upd = psd(IK @ P @ IK.T + K @ R @ K.T)
    return m_upd, P_upd


def ukf_stable_weights(n: int, alpha: float = jnp.sqrt(3.0), beta: float = 2.0, kappa: float = 1.0):
    """Standard scaled UT weights (well-conditioned for n=6)."""
    lamb = alpha**2 * (n + kappa) - n
    factor = 1.0 / (2.0 * (n + lamb))
    w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
    w_cov = jnp.concatenate(
        (
            jnp.array([lamb / (n + lamb) + (1.0 - alpha**2 + beta)]),
            jnp.ones(2 * n) * factor,
        )
    )
    return float(lamb), w_mean, w_cov


def ukf_stable_sigmas(m: jnp.ndarray, P: jnp.ndarray, n: int, lamb: float) -> jnp.ndarray:
    dist = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(psd(P))
    plus = jnp.array([m + dist[:, i] for i in range(n)])
    minus = jnp.array([m - dist[:, i] for i in range(n)])
    return jnp.concatenate((jnp.array([m]), plus, minus))


def ukf_stable_predict(
    m: jnp.ndarray,
    P: jnp.ndarray,
    *,
    drift_fn: Callable,
    L_fn: Callable,
    Qc_fn: Callable,
    t0: float,
    t1: float,
    lamb: float,
    w_mean: jnp.ndarray,
    w_cov: jnp.ndarray,
    dt0: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Zeroth-order CD-UKF: ODE sigma points + additive diffusion (cd-dynamax style)."""
    n = m.shape[0]
    X_t = ukf_stable_sigmas(m, P, n, lamb)

    def propagate(x0):
        sol = diffeqsolve(lambda t, y, args: drift_fn(t, y), t0, t1, x0, dt0=dt0)
        return sol[0, :]

    X_pred = vmap(propagate)(X_t)
    m_pred = jnp.tensordot(w_mean, X_pred, axes=1)
    P_pred = jnp.tensordot(w_cov, _outer(X_pred - m_pred, X_pred - m_pred), axes=1)
    dt = t1 - t0
    L_t = L_fn(t0, m)
    Qc_t = Qc_fn(t0, m)
    return m_pred, psd(P_pred + dt * L_t @ Qc_t @ L_t.T)


def ukf_stable_update(
    m: jnp.ndarray,
    P: jnp.ndarray,
    y: jnp.ndarray,
    obs_fn: Callable,
    R: jnp.ndarray,
    lamb: float,
    w_mean: jnp.ndarray,
    w_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = m.shape[0]
    sigmas = ukf_stable_sigmas(m, P, n, lamb)
    z_sig = vmap(obs_fn)(sigmas)
    z_mean = jnp.tensordot(w_mean, z_sig, axes=1)
    dz = z_sig - z_mean
    S = psd(R + jnp.tensordot(w_cov, _outer(dz, dz), axes=1))
    cross = jnp.tensordot(w_cov, _outer(sigmas - m, dz), axes=1)
    K = psd_solve(S, cross.T).T
    m_upd = m + K @ (y - z_mean)
    H = jacfwd(obs_fn)(m)
    I = jnp.eye(n)
    IK = I - K @ H
    P_upd = psd(IK @ P @ IK.T + K @ R @ K.T)
    return m_upd, P_upd


def _sanitize_mfts_state(m: jnp.ndarray, mass_eps: float) -> jnp.ndarray:
    """Keep tank masses positive; leave disturbances unconstrained."""
    m = m.at[:4].set(jnp.maximum(m[:4], mass_eps))
    return m

# ---------------------------------------------------------------------------
# Particle filter — systematic resampling (paper eq. 47)
# ---------------------------------------------------------------------------

def gaussian_logpdf(y, mean, R):
    d = y.shape[0]
    S = psd(R)
    diff = y - mean
    sign, logdet = jnp.linalg.slogdet(S)
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet + diff.T @ jnp.linalg.solve(S, diff))

def systematic_resample(key, particles: jnp.ndarray, log_w: jnp.ndarray) -> jnp.ndarray:
    n = particles.shape[0]
    w = jnp.exp(log_w - logsumexp(log_w))
    cdf = jnp.cumsum(w)
    u0 = jr.uniform(key, ())
    positions = (jnp.arange(n) + u0) / n
    idx = jnp.searchsorted(cdf, positions)
    return particles[idx]


# ---------------------------------------------------------------------------
# Filter runners
# ---------------------------------------------------------------------------


@dataclass
class NielsenConfig:
    horizon_min: float = 30.0
    n_obs: int = 120
    n_substeps_truth: int = 1000
    n_substeps_filter: int = 100
    enkf_particles: int = 250
    pf_particles: int = 1000
    ukf_alpha: float = 0.001
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    ukf_use_paper_weights: bool = False
    meas_std: float = 2.0  # observation noise σ [cm]; paper Azam eq. 21
    lambda_sim: float = 0.1
    sigma_sim: float = 5.0
    lambda_ekf_ukf: float = 0.0
    lambda_enkf_pf: float = 2.0e-3
    sigma_ekf_enkf_pf: float = 5.0
    sigma_ukf: float = 1.0
    init_cov_scale: float = 1.0
    twin_experiment: bool = True
    seed: int = 42
    cubature_order: int = 2
    recombined_num_paths: Optional[int] = None
    F1: float = 5.0
    F2: float = 5.0
    plot_path: Optional[str] = "nielsen_mfts_tracking.png"
    show_plot: bool = False


# --cubature-loop: fixed Lyons order, sweep recombined Wiener path count P
CUBATURE_LOOP_ORDER: int = 3
CUBATURE_LOOP_PATHS: tuple[int, ...] = (2, 8, 16, 27)


@dataclass
class MFTSBenchmarkContext:
    """Shared simulation + dynamics for one benchmark seed (cubature-loop reuse)."""

    p: MFTSParams
    beta: np.ndarray
    dt_obs: float
    dt0: float
    true_traj: np.ndarray
    observations: list
    R: jnp.ndarray
    x0: np.ndarray
    P0: np.ndarray
    drift_ekf: Callable
    L_ekf: Callable
    Qc: Callable
    obs_fn: Callable
    drift_ukf: Callable
    L_ukf: Callable
    drift_enkf: Callable
    L_enkf: Callable
    diffusion_fn: Callable
    cfg: NielsenConfig


def _init_estimate(
    p: MFTSParams,
    F1: float,
    F2: float,
    *,
    true_state: Optional[np.ndarray] = None,
    cov_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Twin experiment: start at true x₀ with modest P₀ (paper: known initial state)."""
    if true_state is not None:
        x0 = np.asarray(true_state, dtype=np.float64)
    else:
        m0 = mfts_steady_masses(F1, F2, p.F_nominal_est, p.F_nominal_est, p)
        x0 = np.array([*m0, p.F_nominal_est, p.F_nominal_est])
    # Masses ~10²–10³ g; disturbances ~10² cm³/s scale
    P0 = (np.diag([1.0, 1.0, 1.0, 1.0, 25.0, 25.0]).astype(np.float64)) * cov_scale
    return x0, P0


# ---------------------------------------------------------------------------
# Cubature KF: degree-3 spatial cubature + per-node SDE propagation (no Wiener paths)
# ---------------------------------------------------------------------------


def cubature_kf_predict(
    m: jnp.ndarray,
    P: jnp.ndarray,
    *,
    drift_fn: Callable,
    L_fn: Callable,
    Qc_fn: Callable,
    t0: float,
    t1: float,
    dt0: float,
    key: jax.Array,
    noise_dim: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate 2n spatial cubature nodes with cd-dynamax Heun SDE; weighted mean/cov."""
    nodes, weights = spatial_cubature_nodes(m, P)
    n_nodes = nodes.shape[0]

    def combined_diff(t, y, args):
        return L_fn(t, y) @ jnp.linalg.cholesky(psd(Qc_fn(t, y)))

    def propagate(x0, k0):
        sol = diffeqsolve(
            lambda t, y, args: drift_fn(t, y),
            t0,
            t1,
            x0,
            diffusion=combined_diff,
            key=k0,
            noise_dim=noise_dim,
            dt0=dt0,
        )
        return sol[0, :]

    keys = jr.split(key, n_nodes)
    X_pred = vmap(propagate)(nodes, keys)
    m_pred, P_pred = weighted_mean_cov(X_pred, weights)
    return m_pred, psd(P_pred)


def cubature_kf_update(
    m: jnp.ndarray,
    P: jnp.ndarray,
    y: jnp.ndarray,
    obs_fn: Callable,
    R: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Joseph update on spatial cubature nodes (same statistics as CRF Gaussian update)."""
    nodes, weights = spatial_cubature_nodes(m, P)
    z_sig = vmap(obs_fn)(nodes)
    z_mean = weights @ z_sig
    dz = z_sig - z_mean
    S = psd(R + dz.T @ (weights[:, None] * dz))
    cross = (nodes - m).T @ (weights[:, None] * dz)
    K = psd_solve(S, cross.T).T
    m_new = m + K @ (y - z_mean)
    H = jacfwd(obs_fn)(m)
    I = jnp.eye(m.shape[0])
    IK = I - K @ H
    P_new = psd(IK @ P @ IK.T + K @ R @ K.T)
    return m_new, P_new


def run_cubature_kf_nielsen(
    drift_fn,
    L_fn,
    Qc_fn,
    obs_fn,
    R,
    x0,
    P0,
    observations,
    true_traj,
    dt_obs_min: float,
    dt0: float,
    seed: int,
    noise_dim: int = 2,
) -> dict:
    m = jnp.asarray(x0, dtype=jnp.float64)
    P = jnp.asarray(P0, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64)
    mass_eps = MFTSParams().mass_eps
    estimates = [np.asarray(m)]
    tu = mu = 0.0
    key = jr.PRNGKey(seed + 23)

    for k, y in enumerate(observations):
        t0 = k * dt_obs_min * 60.0
        t1 = (k + 1) * dt_obs_min * 60.0
        key, k_pred = jr.split(key)
        t_pred = time.perf_counter()
        m, P = cubature_kf_predict(
            m,
            P,
            drift_fn=drift_fn,
            L_fn=L_fn,
            Qc_fn=Qc_fn,
            t0=t0,
            t1=t1,
            dt0=dt0,
            key=k_pred,
            noise_dim=noise_dim,
        )
        m = _sanitize_mfts_state(jax.block_until_ready(m), mass_eps)
        tu += time.perf_counter() - t_pred

        t_upd = time.perf_counter()
        m, P = cubature_kf_update(m, P, jnp.asarray(y), obs_fn, R_j)
        m = _sanitize_mfts_state(jax.block_until_ready(m), mass_eps)
        mu += time.perf_counter() - t_upd
        estimates.append(np.asarray(m))

    est = np.array(estimates)
    n_nodes = 2 * int(x0.shape[0])
    return {
        **timing_row(
            f"Cubature KF (2n={n_nodes}, SDE)",
            tu,
            mu,
            mape_pct(est, true_traj, slice(0, 4)),
            mape_pct(est, true_traj, slice(4, 6)),
        ),
        "estimates": est,
    }


def run_cd_ekf_nielsen(
    drift_fn,
    L_fn,
    Qc_fn,
    obs_fn,
    R,
    x0,
    P0,
    observations,
    true_traj,
    dt_obs_min: float,
    dt0: float,
) -> dict:
    m = jnp.asarray(x0, dtype=jnp.float64)
    P = jnp.asarray(P0, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64)
    estimates = [np.asarray(m)]
    tu = mu = 0.0

    for k, y in enumerate(observations):
        t0 = k * dt_obs_min * 60.0
        t1 = (k + 1) * dt_obs_min * 60.0
        t_pred = time.perf_counter()
        m, P = ekf_predict(
            m, P, drift_fn=drift_fn, L_fn=L_fn, Qc_fn=Qc_fn, t0=t0, t1=t1, dt0=dt0
        )
        m = jax.block_until_ready(m)
        tu += time.perf_counter() - t_pred

        t_upd = time.perf_counter()
        m, P = ekf_update_joseph(m, P, jnp.asarray(y), obs_fn, R_j)
        m = _sanitize_mfts_state(jax.block_until_ready(m), MFTSParams().mass_eps)
        mu += time.perf_counter() - t_upd
        estimates.append(np.asarray(m))

    est = np.array(estimates)
    return {
        **timing_row(
            "CD-EKF (Joseph)",
            tu,
            mu,
            mape_pct(est, true_traj, slice(0, 4)),
            mape_pct(est, true_traj, slice(4, 6)),
        ),
        "estimates": est,
    }


def run_cd_ukf_nielsen(
    drift_fn,
    L_fn,
    Qc_fn,
    obs_fn,
    R,
    x0,
    P0,
    observations,
    true_traj,
    dt_obs_min: float,
    dt0: float,
    cfg: NielsenConfig,
) -> dict:
    m = jnp.asarray(x0, dtype=jnp.float64)
    P = jnp.asarray(P0, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64)
    n = m.shape[0]
    mass_eps = MFTSParams().mass_eps
    estimates = [np.asarray(m)]
    tu = mu = 0.0
    key = jr.PRNGKey(cfg.seed + 17)

    if cfg.ukf_use_paper_weights:
        label = "CD-UKF (paper α=0.001)"
        n_omega = 2
        c, lamb_p, w_mean, w_cov = ukf_nielsen_weights(
            n, n_omega, cfg.ukf_alpha, cfg.ukf_beta, cfg.ukf_kappa
        )
        if lamb_p + (n + n_omega) < 0.1:
            print(
                "WARNING: paper UKF weights are ill-conditioned for n=6 "
                f"(λ={lamb_p:.4g}); expect obs over-fitting and corrupted F states."
            )
    else:
        label = "CD-UKF (stable UT)"
        lamb_s, w_mean, w_cov = ukf_stable_weights(n)

    for k, y in enumerate(observations):
        t0 = k * dt_obs_min * 60.0
        t1 = (k + 1) * dt_obs_min * 60.0
        t_pred = time.perf_counter()
        if cfg.ukf_use_paper_weights:
            key, k_pred = jr.split(key)
            m, P = ukf_nielsen_predict(
                m,
                P,
                drift_fn=drift_fn,
                L_fn=L_fn,
                Qc_fn=Qc_fn,
                t0=t0,
                t1=t1,
                c=c,
                w_mean=w_mean,
                w_cov=w_cov,
                n_omega=n_omega,
                dt0=dt0,
                key=k_pred,
            )
        else:
            m, P = ukf_stable_predict(
                m,
                P,
                drift_fn=drift_fn,
                L_fn=L_fn,
                Qc_fn=Qc_fn,
                t0=t0,
                t1=t1,
                lamb=lamb_s,
                w_mean=w_mean,
                w_cov=w_cov,
                dt0=dt0,
            )
        m = _sanitize_mfts_state(jax.block_until_ready(m), mass_eps)
        tu += time.perf_counter() - t_pred

        t_upd = time.perf_counter()
        if cfg.ukf_use_paper_weights:
            m, P = ukf_nielsen_update(
                m,
                P,
                jnp.asarray(y),
                obs_fn,
                R_j,
                cfg.ukf_alpha,
                cfg.ukf_beta,
                cfg.ukf_kappa,
            )
        else:
            m, P = ukf_stable_update(
                m, P, jnp.asarray(y), obs_fn, R_j, lamb_s, w_mean, w_cov
            )
        m = _sanitize_mfts_state(jax.block_until_ready(m), mass_eps)
        mu += time.perf_counter() - t_upd
        estimates.append(np.asarray(m))

    est = np.array(estimates)
    return {
        **timing_row(
            label,
            tu,
            mu,
            mape_pct(est, true_traj, slice(0, 4)),
            mape_pct(est, true_traj, slice(4, 6)),
        ),
        "estimates": est,
    }


def run_enkf_nielsen(
    drift_fn,
    L_fn,
    Qc_fn,
    obs_fn,
    R,
    x0,
    P0,
    observations,
    true_traj,
    dt_obs_min: float,
    dt0: float,
    n_particles: int,
    seed: int,
) -> dict:
    hp = EnKFHyperParams(
        N_particles=n_particles,
        dt0=dt0,
        dt_average=dt_obs_min * 60.0,
        state_order="first",
        perturb_measurements=True,
    )
    enkf = CDDynamaxEnKF(
        drift_fn,
        obs_fn,
        jnp.asarray(R, dtype=jnp.float64),
        jnp.asarray(x0, dtype=jnp.float64),
        jnp.asarray(P0, dtype=jnp.float64),
        hp,
        L_fn,
        Qc_fn,
        noise_dim=2,
    )
    key = jr.PRNGKey(seed)
    key, k_init = jr.split(key)
    ensemble = enkf.init_ensemble(k_init)
    estimates = [np.asarray(jnp.mean(ensemble, axis=0))]
    tu = mu = 0.0
    dt_s = dt_obs_min * 60.0

    for k, y in enumerate(observations):
        t0 = k * dt_s
        t1 = (k + 1) * dt_s
        key, k_pred, k_filt = jr.split(key, 3)
        t_pred = time.perf_counter()
        ensemble = enkf_predict(
            k_pred,
            ensemble,
            drift_fn=drift_fn,
            diffusion_coefficient_fn=L_fn,
            diffusion_cov_fn=Qc_fn,
            t0=t0,
            t1=t1,
            dt0=dt0,
            noise_dim=2,
        )
        ensemble = jax.block_until_ready(ensemble)
        tu += time.perf_counter() - t_pred

        t_upd = time.perf_counter()
        ensemble = enkf_condition_on(
            k_filt, ensemble, obs_fn, jnp.asarray(R), jnp.asarray(y)
        )
        ensemble = jax.block_until_ready(ensemble)
        mu += time.perf_counter() - t_upd
        estimates.append(np.asarray(jnp.mean(ensemble, axis=0)))

    est = np.array(estimates)
    return {
        **timing_row(
            f"EnKF (N={n_particles})",
            tu,
            mu,
            mape_pct(est, true_traj, slice(0, 4)),
            mape_pct(est, true_traj, slice(4, 6)),
        ),
        "estimates": est,
    }


def run_pf_nielsen(
    drift_fn,
    L_fn,
    Qc_fn,
    obs_fn,
    R,
    x0,
    P0,
    observations,
    true_traj,
    dt_obs_min: float,
    dt0: float,
    n_particles: int,
    seed: int,
) -> dict:
    key = jr.PRNGKey(seed)
    R_j = jnp.asarray(R, dtype=jnp.float64)
    key, k_init = jr.split(key)
    particles = jr.multivariate_normal(
        k_init,
        mean=jnp.asarray(x0),
        cov=psd(jnp.asarray(P0)),
        shape=(n_particles,),
    )
    log_w = jnp.full((n_particles,), -jnp.log(n_particles))
    estimates = [np.asarray(x0)]
    tu = mu = 0.0
    dt_s = dt_obs_min * 60.0

    for k, y in enumerate(observations):
        t0 = k * dt_s
        t1 = (k + 1) * dt_s
        key, k_pred, k_res = jr.split(key, 3)
        t_pred = time.perf_counter()
        particles = enkf_predict(
            k_pred,
            particles,
            drift_fn=drift_fn,
            diffusion_coefficient_fn=L_fn,
            diffusion_cov_fn=Qc_fn,
            t0=t0,
            t1=t1,
            dt0=dt0,
            noise_dim=2,
        )
        particles = jax.block_until_ready(particles)
        tu += time.perf_counter() - t_pred

        t_upd = time.perf_counter()
        y_j = jnp.asarray(y)
        log_lik = vmap(lambda p: gaussian_logpdf(y_j, obs_fn(p), R_j))(particles)
        log_w = log_w + log_lik
        log_w = log_w - logsumexp(log_w)
        particles = systematic_resample(k_res, particles, log_w)
        log_w = jnp.full((n_particles,), -jnp.log(n_particles))
        particles = jax.block_until_ready(particles)
        mu += time.perf_counter() - t_upd
        estimates.append(np.asarray(jnp.mean(particles, axis=0)))

    est = np.array(estimates)
    return {
        **timing_row(
            f"PF (N={n_particles})",
            tu,
            mu,
            mape_pct(est, true_traj, slice(0, 4)),
            mape_pct(est, true_traj, slice(4, 6)),
        ),
        "estimates": est,
    }


def run_crf_nielsen(
    drift_fn,
    diffusion_fn,
    obs_fn,
    R,
    x0,
    P0,
    observations,
    true_traj,
    dt_obs_min: float,
    dt0: float,
    cfg: NielsenConfig,
) -> dict:
    recomb_P = cfg.recombined_num_paths or default_recombined_num_paths(2, cfg.cubature_order)
    crf = JaxCubatureRecursiveFilter(
        drift_fn,
        diffusion_fn,
        obs_fn,
        jnp.asarray(R, dtype=jnp.float64),
        jnp.asarray(x0, dtype=jnp.float64),
        jnp.asarray(P0, dtype=jnp.float64),
        noise_dim=2,
        cubature_order=cfg.cubature_order,
        integration_dt=dt0,
        predict_horizon=dt_obs_min * 60.0,
        path_sampling="full",
        multidim_paths="recombined",
        recombined_num_paths=recomb_P,
        seed=cfg.seed,
    )
    key = jr.PRNGKey(cfg.seed)
    estimates = [np.asarray(crf.x)]
    tu = mu = 0.0
    dt_s = dt_obs_min * 60.0

    for k, y in enumerate(observations):
        t0 = k * dt_s
        t1 = (k + 1) * dt_s
        key, sub = jr.split(key)
        t_pred = time.perf_counter()
        x_pred, P_pred, samples, weights = crf.predict(t0, t1, key=sub)
        x_pred = jax.block_until_ready(x_pred)
        tu += time.perf_counter() - t_pred

        t_upd = time.perf_counter()
        x_upd, P_upd = crf.update(jnp.asarray(y), x_pred, P_pred, samples, weights)
        x_upd = jax.block_until_ready(x_upd)
        mu += time.perf_counter() - t_upd
        crf.x, crf.P = x_upd, P_upd
        estimates.append(np.asarray(x_upd))

    est = np.array(estimates)
    return {
        **timing_row(
            f"CRF cubature (P={recomb_P})",
            tu,
            mu,
            mape_pct(est, true_traj, slice(0, 4)),
            mape_pct(est, true_traj, slice(4, 6)),
        ),
        "estimates": est,
        "recombined_num_paths": recomb_P,
    }

# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def _height_spread_cm(results: list[dict], beta: np.ndarray) -> tuple[float, float]:
    """Mean and max pairwise max-tank |Δh| [cm] across filter means."""
    est_h = [r["estimates"][:, :4] * beta for r in results]
    n = min(len(e) for e in est_h)
    est_h = [e[:n] for e in est_h]
    pair_max = []
    for i in range(len(est_h)):
        for j in range(i + 1, len(est_h)):
            d = np.max(np.abs(est_h[i] - est_h[j]), axis=1)
            pair_max.append(d)
    stack = np.stack(pair_max, axis=0)
    return float(np.mean(stack)), float(np.max(stack))


def print_results_table(
    results: list[dict],
    *,
    beta: Optional[np.ndarray] = None,
    paper_table: bool = False,
) -> None:
    print()
    print("=" * 95)
    print(
        f"{'Method':<26} {'TU [s]':>10} {'MU [s]':>10} {'MAPEx [%]':>12} {'MAPEd [%]':>12}"
    )
    print("-" * 95)
    for r in results:
        print(
            f"{r['name']:<26} {r['time_tu_s']:>10.3e} {r['time_mu_s']:>10.3e} "
            f"{r['mapex_pct']:>12.3f} {r['maped_pct']:>12.3f}"
        )
    print("=" * 95)
  #  if beta is not None and len(results) >= 2:
  #      mean_spread, max_spread = _height_spread_cm(results, beta)
  #      print(
  #           f"Filter disagreement (pairwise |Δh| over tanks): "
  #           f"mean={mean_spread:.3f} cm, max={max_spread:.3f} cm"
  #       )
    if paper_table:
        print()
        print("Paper Table I (Nielsen et al. 2022, indicative):")
        print("  EKF   TU~3.1e-01  MU~1.2e-02  MAPEx~2.55  MAPEd~15.7")
        print("  UKF   TU~2.9e+00  MU~4.1e-02  MAPEx~2.97  MAPEd~17.5")
        print("  EnKF  TU~3.4e+01  MU~2.3e-01  MAPEx~2.35  MAPEd~14.7")
        print("  PF    TU~1.4e+02  MU~1.1e+00  MAPEx~2.40  MAPEd~13.7")
        print("  (MATLAB / different hardware; use as qualitative reference.)")


def _input_trajectory(
    t: np.ndarray,
    F1: float | np.ndarray,
    F2: float | np.ndarray,
    n_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pump flows F₁, F₂ on the plot time grid (constant or per-observation vector)."""
    t_in = t[1 : n_obs + 1] if len(t) > n_obs else t

    def _expand(u):
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 0 or u.size == 1:
            return np.full(len(t_in), float(u))
        if u.shape[0] == n_obs:
            return u
        raise ValueError(f"input length {u.shape[0]} must be 1 or n_obs={n_obs}")

    return _expand(F1), _expand(F2)


def _pump_inputs_vary(F1: float | np.ndarray, F2: float | np.ndarray, *, atol: float = 1e-9) -> bool:
    """True when F₁ or F₂ is a non-constant time series (not fixed pump flow)."""
    for u in (F1, F2):
        arr = np.asarray(u, dtype=np.float64).ravel()
        if arr.size > 1 and np.ptp(arr) > atol:
            return True
    return False


def plot_mfts_results(
    results: list[dict],
    true_traj: np.ndarray,
    dt_obs_min: float,
    save_path: Optional[str],
    show: bool,
    observations: Optional[list] = None,
    F1: float | np.ndarray = 5.0,
    F2: float | np.ndarray = 5.0,
) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(len(true_traj)) * dt_obs_min
    n_obs = len(observations) if observations is not None else max(len(t) - 1, 0)
    beta = _betas(MFTSParams())
    true_h = true_traj[:, :4] * beta
    t_in = t[1 : n_obs + 1]
    show_pumps = _pump_inputs_vary(F1, F2)
    n_rows = 4 if show_pumps else 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 13 if show_pumps else 10), sharex=True)
    fig.suptitle("Modified four-tank system (Nielsen et al. benchmark)", fontsize=13)

    labels = ("h₁", "h₂", "h₃", "h₄")
    palette = ["#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#e377c2", "#2ca02c", "#8c564b"]
    colors = {r["name"]: palette[i % len(palette)] for i, r in enumerate(results)}

    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.plot(t, true_h[:, i], "k-", lw=2.0, label="True state")
        if observations is not None:
            obs_h = np.array([y[i] for y in observations])
            ax.scatter(
                t_in,
                obs_h,
                s=12,
                c="0.45",
                alpha=0.4,
                marker=".",
                label="Observations" if i == 0 else None,
                zorder=4,
            )
        for res in results:
            est_h = res["estimates"][:, :4] * beta
            ax.plot(
                t[: len(est_h)],
                est_h[:, i],
                lw=1.2,
                alpha=0.9,
                color=colors.get(res["name"]),
                label=res["name"],
            )
        ax.set_ylabel(f"{labels[i]} [cm]")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, ncol=2)

    f_vals = [true_traj[:, 4], true_traj[:, 5]]
    for res in results:
        e = res["estimates"]
        f_vals.extend([e[:, 4], e[:, 5]])
    f_stack = np.concatenate([np.ravel(v) for v in f_vals])
    f_finite = f_stack[np.isfinite(f_stack)]
    if f_finite.size:
        lo, hi = np.percentile(f_finite, [2, 98])
        pad = max(10.0, 0.05 * (hi - lo))
        f_ylim = (lo - pad, hi + pad)
    else:
        f_ylim = (0.0, 300.0)

    ax_d = axes[2, 0]
    for res in results:
        ax_d.plot(
            t[: len(res["estimates"])],
            res["estimates"][:, 4],
            lw=1.0,
            color=colors.get(res["name"]),
            label=res["name"],
        )
    ax_d.plot(t, true_traj[:, 4], "k-", lw=1.8, label="True F₃")
    ax_d.axhline(MFTSParams().F_nominal_est, color="0.55", ls="--", lw=1.0, label="Model F̄=150")
    ax_d.set_ylabel("F₃ [cm³/s]")
    ax_d.set_ylim(f_ylim)
    ax_d.grid(True, alpha=0.3)

    ax_d2 = axes[2, 1]
    for res in results:
        ax_d2.plot(
            t[: len(res["estimates"])],
            res["estimates"][:, 5],
            lw=1.0,
            color=colors.get(res["name"]),
            label=res["name"],
        )
    ax_d2.plot(t, true_traj[:, 5], "k-", lw=1.8, label="True F₄")
    ax_d2.axhline(MFTSParams().F_nominal_est, color="0.55", ls="--", lw=1.0, label="Model F̄=150")
    ax_d2.set_ylabel("F₄ [cm³/s]")
    ax_d2.set_ylim(f_ylim)
    ax_d2.grid(True, alpha=0.3)
    if not show_pumps:
        ax_d2.set_xlabel("time [min]")

    if show_pumps:
        F1_t, F2_t = _input_trajectory(t, F1, F2, n_obs)
        u_ylim = (
            min(F1_t.min(), F2_t.min()) - 0.5,
            max(F1_t.max(), F2_t.max()) + 0.5,
        )
        ax_u1 = axes[3, 0]
        ax_u1.plot(t_in, F1_t, "b-", lw=2.0, label="Pump 1 inflow F₁")
        ax_u1.set_ylabel("F₁ [cm³/s]")
        ax_u1.set_xlabel("time [min]")
        ax_u1.set_ylim(u_ylim)
        ax_u1.grid(True, alpha=0.3)
        ax_u1.legend(fontsize=8)

        ax_u2 = axes[3, 1]
        ax_u2.plot(t_in, F2_t, "b-", lw=2.0, label="Pump 2 inflow F₂")
        ax_u2.set_ylabel("F₂ [cm³/s]")
        ax_u2.set_xlabel("time [min]")
        ax_u2.set_ylim(u_ylim)
        ax_u2.grid(True, alpha=0.3)
        ax_u2.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def prepare_mfts_benchmark(cfg: NielsenConfig) -> MFTSBenchmarkContext:
    p = MFTSParams(meas_std=cfg.meas_std)
    beta = np.array(_betas(p))
    dt_obs = cfg.horizon_min / cfg.n_obs
    dt0 = dt_obs * 60.0 / cfg.n_substeps_filter

    true_traj, observations, R = simulate_mfts(
        horizon_min=cfg.horizon_min,
        n_obs=cfg.n_obs,
        n_substeps_truth=cfg.n_substeps_truth,
        p=p,
        lambda_sim=cfg.lambda_sim,
        sigma_sim=cfg.sigma_sim,
        F1=cfg.F1,
        F2=cfg.F2,
        seed=cfg.seed,
    )
    x0, P0 = _init_estimate(
        p,
        F1=cfg.F1,
        F2=cfg.F2,
        true_state=true_traj[0] if cfg.twin_experiment else None,
        cov_scale=cfg.init_cov_scale,
    )

    bar_est = lambda t_min, nom=p.F_nominal_est: nom

    drift_ekf, L_ekf, Qc, obs_fn, _ = make_mfts_dynamics(
        p,
        lambda_dist=cfg.lambda_ekf_ukf,
        sigma_dist=cfg.sigma_ekf_enkf_pf,
        bar_F3=bar_est,
        bar_F4=bar_est,
    )
    drift_ukf, L_ukf, _, _, _ = make_mfts_dynamics(
        p,
        lambda_dist=cfg.lambda_ekf_ukf,
        sigma_dist=cfg.sigma_ukf,
        bar_F3=bar_est,
        bar_F4=bar_est,
    )
    drift_enkf, L_enkf, _, _, _ = make_mfts_dynamics(
        p,
        lambda_dist=cfg.lambda_enkf_pf,
        sigma_dist=cfg.sigma_ekf_enkf_pf,
        bar_F3=bar_est,
        bar_F4=bar_est,
    )

    def diffusion_fn(t, x):
        return L_enkf(t, x)

    return MFTSBenchmarkContext(
        p=p,
        beta=beta,
        dt_obs=dt_obs,
        dt0=dt0,
        true_traj=true_traj,
        observations=observations,
        R=R,
        x0=x0,
        P0=P0,
        drift_ekf=drift_ekf,
        L_ekf=L_ekf,
        Qc=Qc,
        obs_fn=obs_fn,
        drift_ukf=drift_ukf,
        L_ukf=L_ukf,
        drift_enkf=drift_enkf,
        L_enkf=L_enkf,
        diffusion_fn=diffusion_fn,
        cfg=cfg,
    )


def _print_benchmark_header(ctx: MFTSBenchmarkContext, *, cubature_loop: bool = False) -> None:
    cfg = ctx.cfg
    p = ctx.p
    print("=" * 95)
    print(
        f"Nielsen MFTS benchmark  |  T={cfg.horizon_min} min, N_obs={cfg.n_obs}, "
        f"EnKF={cfg.enkf_particles}, PF={cfg.pf_particles}"
    )
    init_mode = "twin (true x₀)" if cfg.twin_experiment else "steady-state x₀"
    print(
        f"  Truth: λ={cfg.lambda_sim}, σ_proc={cfg.sigma_sim}, σ_obs={cfg.meas_std} cm; "
        f"init={init_mode}, P₀ scale={cfg.init_cov_scale}"
    )
    print(
        f"  F̄ schedule 100→200→300; estimators F̄={p.F_nominal_est} "
        f"(σ_est EKF/EnKF/PF/CRF={cfg.sigma_ekf_enkf_pf}, UKF={cfg.sigma_ukf})"
    )
    if cubature_loop:
        paths = ", ".join(str(p) for p in CUBATURE_LOOP_PATHS)
        print(f"  CRF cubature-loop: k={CUBATURE_LOOP_ORDER}, P in {{{paths}}}")
    else:
        crf_P = cfg.recombined_num_paths or default_recombined_num_paths(2, cfg.cubature_order)
        print(f"  CRF: cubature order k={cfg.cubature_order}, Wiener paths P={crf_P}")
    print("=" * 95)


def compare_nielsen_mfts(cfg: NielsenConfig) -> list[dict]:
    ctx = prepare_mfts_benchmark(cfg)
    _print_benchmark_header(ctx)
    results = _run_standard_filters(ctx, include_crf=True)
    print_results_table(results, beta=ctx.beta, paper_table=False)
    if cfg.plot_path or cfg.show_plot:
        plot_mfts_results(
            results,
            ctx.true_traj,
            ctx.dt_obs,
            cfg.plot_path,
            cfg.show_plot,
            observations=ctx.observations,
            F1=cfg.F1,
            F2=cfg.F2,
        )
    return results


def _run_standard_filters(ctx: MFTSBenchmarkContext, *, include_crf: bool) -> list[dict]:
    cfg = ctx.cfg
    runners: list[Callable[[], dict]] = [
        lambda: run_cd_ekf_nielsen(
            ctx.drift_ekf,
            ctx.L_ekf,
            ctx.Qc,
            ctx.obs_fn,
            ctx.R,
            ctx.x0,
            ctx.P0,
            ctx.observations,
            ctx.true_traj,
            ctx.dt_obs,
            ctx.dt0,
        ),
        lambda: run_cd_ukf_nielsen(
            ctx.drift_ukf,
            ctx.L_ukf,
            ctx.Qc,
            ctx.obs_fn,
            ctx.R,
            ctx.x0,
            ctx.P0,
            ctx.observations,
            ctx.true_traj,
            ctx.dt_obs,
            ctx.dt0,
            cfg,
        ),
        lambda: run_enkf_nielsen(
            ctx.drift_enkf,
            ctx.L_enkf,
            ctx.Qc,
            ctx.obs_fn,
            ctx.R,
            ctx.x0,
            ctx.P0,
            ctx.observations,
            ctx.true_traj,
            ctx.dt_obs,
            ctx.dt0,
            cfg.enkf_particles,
            cfg.seed,
        ),
        lambda: run_pf_nielsen(
            ctx.drift_enkf,
            ctx.L_enkf,
            ctx.Qc,
            ctx.obs_fn,
            ctx.R,
            ctx.x0,
            ctx.P0,
            ctx.observations,
            ctx.true_traj,
            ctx.dt_obs,
            ctx.dt0,
            cfg.pf_particles,
            cfg.seed + 1,
        ),
        lambda: run_cubature_kf_nielsen(
            ctx.drift_enkf,
            ctx.L_enkf,
            ctx.Qc,
            ctx.obs_fn,
            ctx.R,
            ctx.x0,
            ctx.P0,
            ctx.observations,
            ctx.true_traj,
            ctx.dt_obs,
            ctx.dt0,
            cfg.seed,
        ),
    ]
    if include_crf:
        runners.append(
            lambda: run_crf_nielsen(
                ctx.drift_enkf,
                ctx.diffusion_fn,
                ctx.obs_fn,
                ctx.R,
                ctx.x0,
                ctx.P0,
                ctx.observations,
                ctx.true_traj,
                ctx.dt_obs,
                ctx.dt0,
                cfg,
            )
        )
    return [fn() for fn in runners]


def print_cubature_loop_table(baseline: list[dict], crf_sweep: list[dict]) -> None:
    print()
    print("=" * 105)
    print("Baseline filters (fixed parameters; not varied with cubature order)")
    print("-" * 105)
    print(
        f"{'Method':<26} {'TU [s]':>10} {'MU [s]':>10} {'MAPEx [%]':>12} {'MAPEd [%]':>12}"
    )
    print("-" * 105)
    for r in baseline:
        print(
            f"{r['name']:<26} {r['time_tu_s']:>10.3e} {r['time_mu_s']:>10.3e} "
            f"{r['mapex_pct']:>12.3f} {r['maped_pct']:>12.3f}"
        )
    print("-" * 105)
    print(f"CRF sweep (k={CUBATURE_LOOP_ORDER}, varying P)")
    print(
        f"{'P':>5} {'TU [s]':>10} {'MU [s]':>10} {'MAPEx [%]':>12} {'MAPEd [%]':>12}"
    )
    print("-" * 105)
    for r in crf_sweep:
        print(
            f"{r['recombined_num_paths']:>5} "
            f"{r['time_tu_s']:>10.3e} {r['time_mu_s']:>10.3e} "
            f"{r['mapex_pct']:>12.3f} {r['maped_pct']:>12.3f}"
        )
    print("=" * 105)


def plot_cubature_loop(
    baseline: list[dict],
    crf_sweep: list[dict],
    *,
    save_path: Optional[str],
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    paths = np.array([r["recombined_num_paths"] for r in crf_sweep], dtype=int)
    metrics = (
        ("time_tu_s", "TU [s]", True),
        ("time_mu_s", "MU [s]", True),
        ("mapex_pct", "MAPEx [%]", False),
        ("maped_pct", "MAPEd [%]", False),
    )
    palette = ["#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#e377c2"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"MFTS benchmark vs CRF Wiener paths P (cubature order k={CUBATURE_LOOP_ORDER})",
        fontsize=12,
    )

    for ax, (key, ylabel, log_y) in zip(axes.ravel(), metrics):
        crf_y = np.array([r[key] for r in crf_sweep], dtype=float)
        ax.plot(
            paths,
            crf_y,
            "o-",
            color="#2ca02c",
            lw=2,
            ms=8,
            label="CRF",
            zorder=3,
        )
        for i, r in enumerate(baseline):
            ax.axhline(
                r[key],
                color=palette[i % len(palette)],
                ls="--",
                lw=1.2,
                alpha=0.85,
                label=r["name"],
            )
        ax.set_xlabel("recombined Wiener paths P")
        ax.set_ylabel(ylabel)
        ax.set_xticks(paths)
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Cubature-loop plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_cubature_loop(cfg: NielsenConfig) -> tuple[list[dict], list[dict]]:
    ctx = prepare_mfts_benchmark(cfg)
    _print_benchmark_header(ctx, cubature_loop=True)

    print("\nRunning baseline filters (EKF, UKF, EnKF, PF)...")
    baseline = _run_standard_filters(ctx, include_crf=False)

    crf_sweep: list[dict] = []
    for P in CUBATURE_LOOP_PATHS:
        print(f"\nCRF cubature-loop: k={CUBATURE_LOOP_ORDER}, P={P} ...")
        crf_cfg = _apply_cfg_overrides(
            ctx.cfg,
            cubature_order=CUBATURE_LOOP_ORDER,
            recombined_num_paths=P,
        )
        res = run_crf_nielsen(
            ctx.drift_enkf,
            ctx.diffusion_fn,
            ctx.obs_fn,
            ctx.R,
            ctx.x0,
            ctx.P0,
            ctx.observations,
            ctx.true_traj,
            ctx.dt_obs,
            ctx.dt0,
            crf_cfg,
        )
        res["cubature_order"] = CUBATURE_LOOP_ORDER
        res["recombined_num_paths"] = P
        crf_sweep.append(res)

    print_cubature_loop_table(baseline, crf_sweep)
    if cfg.plot_path or cfg.show_plot:
        plot_cubature_loop(
            baseline,
            crf_sweep,
            save_path=cfg.plot_path,
            show=cfg.show_plot,
        )
    return baseline, crf_sweep


def _apply_cfg_overrides(cfg: NielsenConfig, **overrides) -> NielsenConfig:
    clean = {k: v for k, v in overrides.items() if v is not None}
    return NielsenConfig(**{**cfg.__dict__, **clean})


def main():
    parser = argparse.ArgumentParser(
        description="Nielsen et al. (2022) MFTS filter comparison + Wiener cubature CRF"
    )
    parser.add_argument("--fast", action="store_true", help="Shorter horizon / fewer particles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enkf-particles", type=int, default=None)
    parser.add_argument("--pf-particles", type=int, default=None)
    parser.add_argument("--n-obs", type=int, default=None)
    parser.add_argument(
        "--noisy",
        action="store_true",
        help="High-noise preset: σ_obs=8 cm, σ_proc=12, cold start, larger P₀",
    )
    parser.add_argument(
        "--meas-std",
        type=float,
        default=None,
        metavar="CM",
        help="Observation std per height [cm] (default 2; paper Azam)",
    )
    parser.add_argument(
        "--sigma-sim",
        type=float,
        default=None,
        help="Truth OU diffusion scale on F₃,F₄",
    )
    parser.add_argument(
        "--sigma-dist",
        type=float,
        default=None,
        help="Estimator diffusion on F₃,F₄ (EKF/EnKF/PF/CRF; UKF uses --sigma-ukf)",
    )
    parser.add_argument(
        "--sigma-ukf",
        type=float,
        default=None,
        help="UKF disturbance diffusion scale (default 1.0)",
    )
    parser.add_argument(
        "--init-cov-scale",
        type=float,
        default=None,
        help="Scale diagonal P₀",
    )
    parser.add_argument(
        "--no-twin",
        action="store_true",
        help="Initialize at steady-state (F̄=150) instead of true x₀",
    )
    parser.add_argument(
        "--ukf-paper-weights",
        action="store_true",
        help="Use Nielsen Table UKF α=0.001, κ=0 (ill-conditioned for n=6; debug only)",
    )
    parser.add_argument("--F1", type=float, default=5.0, help="Pump 1 inflow [cm³/s]")
    parser.add_argument("--F2", type=float, default=5.0, help="Pump 2 inflow [cm³/s]")
    parser.add_argument("--plot-path", type=str, default="nielsen_mfts_tracking.png")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--cubature-paths",
        type=int,
        default=None,
        metavar="P",
        help="CRF recombined Wiener paths per predict (default: auto from --cubature-order)",
    )
    parser.add_argument(
        "--cubature-order",
        type=int,
        default=None,
        metavar="K",
        help="Lyons macro-segment order k for Wiener cubature (default 2)",
    )
    parser.add_argument(
        "--cubature-loop",
        action="store_true",
        help=(
            f"Sweep CRF at k={CUBATURE_LOOP_ORDER} over P in {CUBATURE_LOOP_PATHS}; "
            "run EKF/UKF/EnKF/PF once; plot metrics vs P with baseline horizontals"
        ),
    )
    args = parser.parse_args()

    if args.fast:
        cfg = NielsenConfig(
            horizon_min=6.0,
            n_obs=24,
            n_substeps_truth=200,
            n_substeps_filter=40,
            enkf_particles=50,
            pf_particles=150,
            seed=args.seed,
            plot_path=None if args.no_plot else args.plot_path or None,
            show_plot=args.show_plot,
        )
    else:
        cfg = NielsenConfig(
            seed=args.seed,
            plot_path=None if args.no_plot else args.plot_path or None,
            show_plot=args.show_plot,
        )

    if args.noisy:
        cfg = _apply_cfg_overrides(
            cfg,
            meas_std=8.0,
            sigma_sim=12.0,
            sigma_ekf_enkf_pf=12.0,
            sigma_ukf=12.0,
            init_cov_scale=4.0,
            twin_experiment=False,
        )

    cfg = _apply_cfg_overrides(
        cfg,
        enkf_particles=args.enkf_particles,
        pf_particles=args.pf_particles,
        n_obs=args.n_obs,
        meas_std=args.meas_std,
        sigma_sim=args.sigma_sim,
        sigma_ekf_enkf_pf=args.sigma_dist,
        sigma_ukf=args.sigma_ukf if args.sigma_ukf is not None else args.sigma_dist,
        init_cov_scale=args.init_cov_scale,
        twin_experiment=False if args.no_twin else None,
        ukf_use_paper_weights=True if args.ukf_paper_weights else None,
        F1=args.F1,
        F2=args.F2,
        recombined_num_paths=args.cubature_paths,
        cubature_order=args.cubature_order,
    )

    if args.cubature_loop:
        if args.no_plot:
            cfg = _apply_cfg_overrides(cfg, plot_path=None, show_plot=False)
        elif cfg.plot_path == "nielsen_mfts_tracking.png":
            cfg = _apply_cfg_overrides(cfg, plot_path="nielsen_mfts_cubature_loop.png")
        run_cubature_loop(cfg)
        return

    compare_nielsen_mfts(cfg)


if __name__ == "__main__":
    main()
