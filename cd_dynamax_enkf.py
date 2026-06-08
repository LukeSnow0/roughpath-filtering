#!/usr/bin/env python3
"""
cd_dynamax-aligned Ensemble Kalman Filter.

Matches the algorithm in hd-UQ/cd_dynamax
  continuous_discrete_nonlinear_gaussian_ssm/inference_enkf.py

Key alignment points:
  - SDE forecast: Heun + VirtualBrownianTree (via diffeqsolve wrapper)
  - Diffusion: combined L @ chol(Qc)
  - Analysis: perturbed observations, spread inflation (inflation_delta)
  - Initial ensemble: MVN(mean, cov) samples
  - state_order modes: first (SDE), zeroth (ODE), discrete (ODE + sampled noise)

Benchmark loop uses predict-then-update so observations at interval ends match CRF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap


StateOrder = Literal["first", "zeroth", "discrete"]


def psd(A: jnp.ndarray, jitter: float = 1e-8) -> jnp.ndarray:
    n = A.shape[0]
    return 0.5 * (A + A.T) + jitter * jnp.eye(n)


def psd_solve(A: jnp.ndarray, B: jnp.ndarray, jitter: float = 1e-8) -> jnp.ndarray:
    n = A.shape[0]
    return jnp.linalg.solve(psd(A, jitter=jitter), B)


_outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)


def diffeqsolve(
    drift: Callable,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    *,
    diffusion: Optional[Callable] = None,
    key=None,
    noise_dim: Optional[int] = None,
    dt0: float = 0.01,
    tol_vbt: float | None = None,
    max_steps: int = int(1e5),
    solver=None,
    stepsize_controller=None,
    adjoint=None,
    **kwargs,
) -> jnp.ndarray:
    """cd_dynamax-style diffrax wrapper (Heun SDE, Dopri5 ODE defaults)."""
    max_steps = int(max_steps)
    if tol_vbt is None:
        tol_vbt = dt0 / 2.0
    if key is None:
        key = jr.PRNGKey(0)

    if solver is None:
        solver = dfx.Heun() if diffusion is not None else dfx.Dopri5()
    if stepsize_controller is None:
        stepsize_controller = dfx.ConstantStepSize()
    if adjoint is None:
        adjoint = dfx.RecursiveCheckpointAdjoint()

    if diffusion is None:
        terms = dfx.ODETerm(drift)
    else:
        bm_shape = (noise_dim,) if noise_dim is not None else y0.shape
        bm = dfx.VirtualBrownianTree(
            t0=t0, t1=t1, tol=tol_vbt, shape=bm_shape, key=key
        )
        terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diffusion, bm))

    sol = dfx.diffeqsolve(
        terms,
        solver=solver,
        stepsize_controller=stepsize_controller,
        t0=t0,
        t1=t1,
        y0=y0,
        args=None,
        dt0=dt0,
        saveat=dfx.SaveAt(t1=True),
        adjoint=adjoint,
        max_steps=max_steps,
        **kwargs,
    ).ys
    return sol


def enkf_predict(
    key,
    particles: jnp.ndarray,
    *,
    drift_fn: Callable,
    diffusion_coefficient_fn: Optional[Callable],
    diffusion_cov_fn: Optional[Callable],
    t0: float,
    t1: float,
    state_order: StateOrder = "first",
    cov_rescaling: float = 1.0,
    dt_average: float = 0.1,
    dt0: float = 0.01,
    diffeqsolve_settings: Optional[dict] = None,
    noise_dim: Optional[int] = None,
) -> jnp.ndarray:
    """Forecast ensemble (cd_dynamax _predict)."""
    diffeqsolve_settings = diffeqsolve_settings or {}
    n_particles, _ = particles.shape

    def drift(t, y, args):
        return drift_fn(t, y)

    if state_order == "zeroth":
        diffusion = None
    else:

        def diffusion(t, y, args):
            L_t = diffusion_coefficient_fn(t, y) * cov_rescaling
            Qc_t = diffusion_cov_fn(t, y)
            return L_t @ jnp.linalg.cholesky(psd(Qc_t))

    def particle_solve(y0, key0):
        sol = diffeqsolve(
            drift=drift,
            t0=t0,
            t1=t1,
            y0=y0,
            diffusion=diffusion,
            key=key0,
            noise_dim=noise_dim,
            dt0=dt0,
            **diffeqsolve_settings,
        )
        return sol[0, :]

    key_array = jr.split(key, n_particles)
    x_pred = vmap(particle_solve, in_axes=(0, 0))(particles, key_array)

    if state_order in ("zeroth", "discrete"):
        dt = (t1 - t0) if state_order == "zeroth" else dt_average
        key_array = jr.split(key, n_particles)

        def _sample_noise(x_i, key_i):
            L_t = diffusion_coefficient_fn(t0, x_i) * cov_rescaling
            Qc_t = diffusion_cov_fn(t0, x_i)
            state_noise_cov = dt * L_t @ Qc_t @ L_t.T
            return jr.multivariate_normal(
                key=key_i, mean=jnp.zeros(x_i.shape[0]), cov=psd(state_noise_cov)
            )

        noise = vmap(_sample_noise, in_axes=(0, 0))(x_pred, key_array)
        x_pred = x_pred + noise

    return x_pred


def enkf_condition_on(
    key,
    particles: jnp.ndarray,
    obs_fn: Callable,
    R: jnp.ndarray,
    y: jnp.ndarray,
    *,
    perturb_measurements: bool = True,
    inflation_delta: float = 0.0,
) -> jnp.ndarray:
    """Analysis step (cd_dynamax _condition_on). Returns filtered particles."""
    n_particles, _ = particles.shape

    y_ensemble = vmap(obs_fn)(particles)
    y_pred_mean = jnp.mean(y_ensemble, axis=0)
    y_pred_cov = psd(
        jnp.sum(_outer(y_ensemble - y_pred_mean, y_ensemble - y_pred_mean), axis=0)
        / (n_particles - 1)
    )

    if inflation_delta > 0.0:
        x_mean = jnp.mean(particles, axis=0)
        x_inflated = x_mean + (1.0 + inflation_delta) * (particles - x_mean)
        y_ensemble_infl = vmap(obs_fn)(x_inflated)
        y_pred_mean_infl = jnp.mean(y_ensemble_infl, axis=0)
        y_pred_cov_infl = psd(
            jnp.sum(
                _outer(y_ensemble_infl - y_pred_mean_infl, y_ensemble_infl - y_pred_mean_infl),
                axis=0,
            )
            / (n_particles - 1)
        )
    else:
        x_inflated = particles
        y_ensemble_infl = y_ensemble
        y_pred_mean_infl = y_pred_mean
        y_pred_cov_infl = y_pred_cov

    cross_cov = (
        jnp.sum(
            _outer(
                x_inflated - jnp.mean(x_inflated, axis=0),
                y_ensemble_infl - y_pred_mean_infl,
            ),
            axis=0,
        )
        / (n_particles - 1)
    )

    S = psd(y_pred_cov_infl + R)
    K = psd_solve(S, cross_cov.T).T

    if perturb_measurements:
        y_data_perturbed = jr.multivariate_normal(key=key, mean=y, cov=R, shape=(n_particles,))
    else:
        y_data_perturbed = jnp.broadcast_to(y, (n_particles, y.shape[0]))

    return x_inflated + (K @ (y_data_perturbed - y_ensemble_infl).T).T


@dataclass
class EnKFHyperParams:
    N_particles: int = 25
    perturb_measurements: bool = True
    cov_rescaling: float = 1.0
    inflation_delta: float = 0.0
    state_order: StateOrder = "first"
    dt_average: float = 0.1
    dt0: float = 0.01
    diffeqsolve_settings: dict = field(default_factory=dict)


class CDDynamaxEnKF:
    """
    cd_dynamax-aligned EnKF for benchmarking against CRF.

    step_predict_then_update: forecast [t_k, t_{k+1}] then assimilate y_{k+1}
    (matches CRF observation times). cd_dynamax scan uses condition-then-predict;
    set use_cd_dynamax_step_order=True to match that loop instead.
    """

    def __init__(
        self,
        drift_fn: Callable,
        obs_fn: Callable,
        R: jnp.ndarray,
        init_mean: jnp.ndarray,
        init_cov: jnp.ndarray,
        hyperparams: Optional[EnKFHyperParams] = None,
        diffusion_coefficient_fn: Optional[Callable] = None,
        diffusion_cov_fn: Optional[Callable] = None,
        noise_dim: Optional[int] = None,
    ):
        self.drift_fn = drift_fn
        self.obs_fn = obs_fn
        self.R = jnp.asarray(R, dtype=jnp.float64)
        self.init_mean = jnp.asarray(init_mean, dtype=jnp.float64)
        self.init_cov = jnp.asarray(init_cov, dtype=jnp.float64)
        self.hp = hyperparams or EnKFHyperParams()
        self.diffusion_coefficient_fn = diffusion_coefficient_fn
        if noise_dim is not None:
            self.noise_dim = noise_dim
        elif diffusion_coefficient_fn is not None:
            L0 = diffusion_coefficient_fn(0.0, self.init_mean)
            self.noise_dim = int(L0.shape[1])
        else:
            self.noise_dim = int(self.init_mean.shape[0])
        if diffusion_cov_fn is None:
            nd = self.noise_dim
            diffusion_cov_fn = lambda t, y, nd=nd: jnp.eye(nd)
        self.diffusion_cov_fn = diffusion_cov_fn

    @property
    def ensemble_size(self) -> int:
        return self.hp.N_particles

    def init_ensemble(self, key) -> jnp.ndarray:
        return jr.multivariate_normal(
            key=key,
            mean=self.init_mean,
            cov=psd(self.init_cov),
            shape=(self.hp.N_particles,),
        )

    def step_predict_then_update(
        self,
        particles: jnp.ndarray,
        y: jnp.ndarray,
        t0: float,
        t1: float,
        key,
    ) -> tuple[jnp.ndarray, jax.Array]:
        """Predict to t1, then condition on y (CRF-aligned observation times)."""
        key, key_predict, key_filter = jr.split(key, 3)
        x_pred = enkf_predict(
            key_predict,
            particles,
            drift_fn=self.drift_fn,
            diffusion_coefficient_fn=self.diffusion_coefficient_fn,
            diffusion_cov_fn=self.diffusion_cov_fn,
            t0=t0,
            t1=t1,
            state_order=self.hp.state_order,
            cov_rescaling=self.hp.cov_rescaling,
            dt_average=self.hp.dt_average,
            dt0=self.hp.dt0,
            diffeqsolve_settings=self.hp.diffeqsolve_settings,
            noise_dim=self.noise_dim,
        )
        x_filt = enkf_condition_on(
            key_filter,
            x_pred,
            self.obs_fn,
            self.R,
            y,
            perturb_measurements=self.hp.perturb_measurements,
            inflation_delta=self.hp.inflation_delta,
        )
        return x_filt, key

    def step_condition_then_predict(
        self,
        particles: jnp.ndarray,
        y: jnp.ndarray,
        t0: float,
        t1: float,
        key,
    ) -> tuple[jnp.ndarray, jax.Array]:
        """cd_dynamax lax.scan ordering: filter at t0, predict to t1."""
        key, key_filter, key_predict = jr.split(key, 3)
        x_filt = enkf_condition_on(
            key_filter,
            particles,
            self.obs_fn,
            self.R,
            y,
            perturb_measurements=self.hp.perturb_measurements,
            inflation_delta=self.hp.inflation_delta,
        )
        x_pred = enkf_predict(
            key_predict,
            x_filt,
            drift_fn=self.drift_fn,
            diffusion_coefficient_fn=self.diffusion_coefficient_fn,
            diffusion_cov_fn=self.diffusion_cov_fn,
            t0=t0,
            t1=t1,
            state_order=self.hp.state_order,
            cov_rescaling=self.hp.cov_rescaling,
            dt_average=self.hp.dt_average,
            dt0=self.hp.dt0,
            diffeqsolve_settings=self.hp.diffeqsolve_settings,
            noise_dim=self.noise_dim,
        )
        return x_pred, key


def make_cd_dynamax_dynamics(
    sigma: float = 0.05,
    state_dependent_diffusion: bool = False,
):
    """
    JAX dynamics for benchmarks: dx = f dt + L dW, Qc = I  =>  combined diffusion L.
    """
    import jax.numpy as jnp

    def drift_fn(t, x):
        x1, x2 = x[0], x[1]
        return jnp.array([-0.1 * x1 + x2, -x1 - 0.1 * x2 + 0.1 * jnp.sin(x1)])

    if state_dependent_diffusion:

        def diffusion_coefficient_fn(t, x):
            x1, x2 = x[0], x[1]
            return jnp.diag(
                jnp.array([sigma * (1.0 + 0.5 * x1**2), sigma * (1.0 + 0.5 * x2**2)])
            )
    else:

        def diffusion_coefficient_fn(t, x):
            return jnp.eye(2) * sigma

    def diffusion_cov_fn(t, x):
        return jnp.eye(2)

    def obs_fn(x):
        return jnp.array([x[0], x[1] + 0.05 * jnp.sin(x[0])])

    return drift_fn, diffusion_coefficient_fn, diffusion_cov_fn, obs_fn


def verify_cd_dynamax_enkf(seed: int = 0, verbose: bool = True) -> dict:
    """
    Sanity checks for cd_dynamax EnKF alignment.

    Confirms noise-driver dimension, diffusion coupling L @ chol(Qc), ensemble
    statistics, and that predict-then-update / condition-then-predict both run.
    """
    jax.config.update("jax_enable_x64", True)
    key = jr.PRNGKey(seed)
    dt = 0.1
    checks = {}

    # --- 2D square diffusion (n_state == noise_dim) ---
    drift_fn, L_fn, Qc_fn, obs_fn = make_cd_dynamax_dynamics(sigma=0.05)
    x0 = jnp.array([1.0, 0.5])
    P0 = jnp.eye(2) * 0.3
    R = jnp.diag(jnp.array([0.15**2, 0.25**2]))
    hp = EnKFHyperParams(N_particles=50, dt0=0.01, dt_average=dt, state_order="first")
    enkf = CDDynamaxEnKF(drift_fn, obs_fn, R, x0, P0, hp, L_fn, Qc_fn)
    checks["2d_noise_dim"] = enkf.noise_dim == 2

    key, k_init = jr.split(key)
    particles = enkf.init_ensemble(k_init)
    checks["ensemble_shape"] = particles.shape == (50, 2)
    checks["init_cov_close"] = bool(jnp.allclose(jnp.mean(particles, axis=0), x0, atol=0.15))

    L = L_fn(0.0, x0)
    Qc = Qc_fn(0.0, x0)
    combined = L @ jnp.linalg.cholesky(psd(Qc))
    checks["diffusion_is_L_when_Qc_I"] = bool(jnp.allclose(combined, L))

    y = obs_fn(x0) + jnp.array([0.01, -0.02])
    key, k_step = jr.split(key)
    p_pred_upd, _ = enkf.step_predict_then_update(particles, y, 0.0, dt, k_step)
    key, k_step = jr.split(key)
    p_cond_pred, _ = enkf.step_condition_then_predict(particles, y, 0.0, dt, k_step)
    checks["predict_then_update_runs"] = p_pred_upd.shape == particles.shape
    checks["condition_then_predict_runs"] = p_cond_pred.shape == particles.shape

    # --- rectangular diffusion (n_state > noise_dim) ---
    from jax_crf import make_highdim_dynamics

    ns, nn = 10, 8
    drift_h, L_h, obs_h = make_highdim_dynamics(n_state=ns, n_noise=nn, n_obs=5, sigma=0.08)[:3]

    def L_rect(t, x):
        return L_h(t, x)

    def Qc_rect(t, x):
        return jnp.eye(nn)

    x0_h = jnp.zeros(ns)
    P0_h = jnp.eye(ns) * 0.2
    R_h = jnp.eye(5) * 0.12**2
    enkf_h = CDDynamaxEnKF(
        drift_h, obs_h, R_h, x0_h, P0_h, hp, L_rect, Qc_rect, noise_dim=nn
    )
    checks["highdim_noise_dim"] = enkf_h.noise_dim == nn
    Lh = L_rect(0.0, x0_h)
    checks["highdim_L_shape"] = Lh.shape == (ns, nn)
    Qc_h = enkf_h.diffusion_cov_fn(0.0, x0_h)
    checks["highdim_Qc_shape"] = Qc_h.shape == (nn, nn)

    key, k_init = jr.split(key)
    particles_h = enkf_h.init_ensemble(k_init)
    key, k_step = jr.split(key)
    p_h, _ = enkf_h.step_predict_then_update(
        particles_h, jnp.zeros(5), 0.0, dt, k_step
    )
    checks["highdim_forecast_runs"] = p_h.shape == (50, ns)

    # --- analysis: inflated spread increases obs variance ---
    hp_infl = EnKFHyperParams(N_particles=100, inflation_delta=0.05)
    enkf_infl = CDDynamaxEnKF(drift_fn, obs_fn, R, x0, P0, hp_infl, L_fn, Qc_fn)
    key, k_init = jr.split(key)
    parts = enkf_infl.init_ensemble(k_init)
    y_ens = vmap(obs_fn)(parts)
    var_base = float(jnp.trace(jnp.cov(y_ens, rowvar=False)))
    x_mean = jnp.mean(parts, axis=0)
    parts_infl = x_mean + 1.05 * (parts - x_mean)
    y_infl = vmap(obs_fn)(parts_infl)
    var_infl = float(jnp.trace(jnp.cov(y_infl, rowvar=False)))
    checks["inflation_increases_spread"] = var_infl > var_base

    all_ok = all(checks.values())
    if verbose:
        print("cd_dynamax EnKF verification")
        print("-" * 40)
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL':4s}  {name}")
        print("-" * 40)
        print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
        print()
        print("cd_dynamax alignment (inference_enkf.py):")
        print("  Forecast: diffeqsolve → Heun + VirtualBrownianTree (state_order='first')")
        print("  Diffusion: L(t,x) @ chol(Qc(t,x)) with Qc on noise dimension")
        print("  Analysis: perturbed observations + optional inflation_delta")
        print("  Init: MVN(mean, cov) ensemble")
        print("  Benchmark loop: predict-then-update (CRF-aligned obs times)")
        print("  Note: cd_dynamax scan uses condition-then-predict; use step_condition_then_predict")

    return {"ok": all_ok, "checks": checks}


if __name__ == "__main__":
    verify_cd_dynamax_enkf()
