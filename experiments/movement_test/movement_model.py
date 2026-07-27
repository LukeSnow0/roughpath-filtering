"""Continuous-time movement SDE for the dabbling-duck GPS track.

We use an Ornstein-Uhlenbeck *velocity* model with a weak home-range spring and
**speed-dependent (multiplicative) diffusion** -- the state-dependent-diffusion
structure that distinguishes the CRF from the moment-ODE Gaussian filters.

State (per individual):  s = [x, y, vx, vy]
    positions x, y in km (mean-centred UTM);  velocities vx, vy in km/h;  t in h.

Stratonovich/Ito SDE (velocity carries the noise; position is its integral):
    dx  = vx dt
    dy  = vy dt
    dvx = ( -beta_v * vx - omega2 * (x - cx) ) dt + g(v) dW1
    dvy = ( -beta_v * vy - omega2 * (y - cy) ) dt + g(v) dW2
    g(v) = sigma0 * sqrt(1 + (|v| / v_c)^2)     # multiplicative in speed

Observation: noisy GPS position only (velocity latent):
    z = [x, y] + N(0, meas_std^2 I)

Parameters:
    beta_v : velocity mean-reversion rate (1/h)  = 1 / (velocity autocorr time)
    omega2 : home-range spring stiffness (1/h^2)
    sigma0 : baseline velocity diffusion (km h^-1.5)
    v_c    : speed scale (km/h) above which stochasticity grows
    meas_std : GPS noise std (km)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MovementParams:
    beta_v: float = 0.5      # 1/h  (velocity autocorr time ~2 h)
    omega2: float = 0.0044   # 1/h^2 (home-range return time ~15 h)
    sigma0: float = 2.0      # km h^-1.5
    v_c: float = 5.0         # km/h
    meas_std: float = 0.03   # km (~30 m GPS error)
    center: tuple[float, float] = (0.0, 0.0)


def make_movement_dynamics(
    p: MovementParams,
    *,
    beta_v,
    sigma0,
):
    """Return (drift, diffusion_coeff, diffusion_cov, obs, R) for the CD filters.

    ``beta_v`` and ``sigma0`` may be JAX scalars (tracers) so the filter
    likelihood is differentiable w.r.t. them; the remaining params come from ``p``.
    """
    cx, cy = p.center
    omega2 = p.omega2
    v_c = p.v_c

    def drift_fn(t, s):
        del t
        x, y, vx, vy = s[0], s[1], s[2], s[3]
        return jnp.array(
            [
                vx,
                vy,
                -beta_v * vx - omega2 * (x - cx),
                -beta_v * vy - omega2 * (y - cy),
            ]
        )

    def diffusion_coefficient_fn(t, s):
        del t
        # Speed^2 directly (no intermediate sqrt): keeps g smooth/differentiable at
        # v=0, where d|v|/dv is singular and would poison the Ito->Stratonovich
        # jacobian at zero-velocity cubature nodes.
        speed_sq = s[2] ** 2 + s[3] ** 2
        g = sigma0 * jnp.sqrt(1.0 + speed_sq / v_c**2)
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [g, 0.0],
                [0.0, g],
            ]
        )

    def diffusion_cov_fn(t, s):
        del t, s
        return jnp.eye(2)

    def obs_fn(s):
        return s[:2]

    R = jnp.diag(jnp.array([p.meas_std**2, p.meas_std**2]))
    return drift_fn, diffusion_coefficient_fn, diffusion_cov_fn, obs_fn, R


def tau_sigma_to_beta_sigma0(tau_v, sigma_v):
    """Interpretable (τ_v, σ_v) → SDE coefficients (β_v, σ₀).

    τ_v = 1/β_v  (velocity autocorrelation time, hours)
    σ_v = asymptotic per-axis velocity std under *frozen* g≡σ₀ OU:
          σ_v = σ₀ / √(2 β_v)  ⇒  σ₀ = σ_v √(2/τ_v).

    With multiplicative g(v)=σ₀√(1+|v|²/v_c²) this σ_v is exact only at rest;
    it remains the natural ecological scale for the baseline diffusion.
    """
    beta_v = 1.0 / tau_v
    sigma0 = sigma_v * jnp.sqrt(2.0 / tau_v)
    return beta_v, sigma0


def beta_sigma0_to_tau_sigma(beta_v, sigma0):
    """Inverse of ``tau_sigma_to_beta_sigma0``."""
    tau_v = 1.0 / beta_v
    sigma_v = sigma0 * jnp.sqrt(tau_v / 2.0)
    return tau_v, sigma_v


def init_estimate(track_xy: np.ndarray, p: MovementParams):
    """Initial mean/cov: first observed position, zero velocity, broad prior."""
    x0 = np.array([track_xy[0, 0], track_xy[0, 1], 0.0, 0.0], dtype=np.float64)
    P0 = np.diag([p.meas_std**2, p.meas_std**2, 4.0, 4.0]).astype(np.float64)
    return x0, P0
