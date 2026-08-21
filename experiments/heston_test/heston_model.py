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
    """Return ``(drift, L, Qc, obs, R)`` for CD / WSC-KF filters."""
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


def make_heston_sv_dynamics(p: HestonParams, *, smooth_vol_eps: float | None = None):
    """Textbook Heston in ``x = [S, v]`` (Itô).

    dS = μ S dt + s S φ(v) dW¹
    dv = κ(θ − v) dt + s ξ φ(v) (ρ dW¹ + √(1−ρ²) dW²)

    By default ``φ(v) = √max(v, ε)``. If ``smooth_vol_eps`` is set, ``φ`` is the
    globally ``C^∞`` map ``(v² + ε²)^{1/4}`` (``~√|v|`` for large ``|v|``), so
    every derivative is bounded for that fixed ``ε>0``. Drift is unchanged, so
    ``E[S]`` and ``E[v]`` remain the closed affine means.
    """
    mu, kappa, theta, xi, rho = p.mu, p.kappa, p.theta, p.xi, p.rho
    v_eps = p.v_eps
    diff_scale = p.diff_scale
    rho_bar = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    eps2 = None if smooth_vol_eps is None else float(smooth_vol_eps) ** 2

    def _phi(v: jnp.ndarray) -> jnp.ndarray:
        if eps2 is None:
            return jnp.sqrt(jnp.maximum(v, v_eps))
        return jnp.power(v * v + eps2, 0.25)

    def drift_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        S, v = x[0], x[1]
        return jnp.array([mu * S, kappa * (theta - v)])

    def L_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        S = x[0]
        s = diff_scale * _phi(x[1])
        return jnp.array(
            [
                [S * s, 0.0],
                [xi * rho * s, xi * rho_bar * s],
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


def exact_heston_mean_cov(
    delta: float,
    x0: np.ndarray,
    p: HestonParams,
    *,
    n_steps: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact second moments of Heston from a point mass via closed Itô ODEs.

    Integrates the closed system for
    ``(E[X], E[v], E[X²], E[v²], E[Xv])`` with classical RK4. Diffusion scale
    ``p.diff_scale`` is included (effective vol-of-vol ``ξ_eff = diff_scale·ξ``
    and quadratic variation of ``X`` scaled by ``diff_scale²``).

    Returns ``(mean, cov)`` with shapes ``(2,)`` and ``(2, 2)``.
    """
    X0, v0 = float(x0[0]), float(x0[1])
    mu, kappa, theta = p.mu, p.kappa, p.theta
    xi, rho = p.xi, p.rho
    s = p.diff_scale
    s2 = s * s
    xi2_s2 = s2 * (xi * xi)
    cross = s2 * xi * rho  # coefficient of E[v] in dE[Xv]/dt from d<X,v>
    # State z = [m_X, m_v, s_XX, s_vv, s_Xv]
    z = np.array([X0, v0, X0 * X0, v0 * v0, X0 * v0], dtype=np.float64)

    def rhs(z_: np.ndarray) -> np.ndarray:
        m_X, m_v, s_XX, s_vv, s_Xv = z_
        return np.array(
            [
                mu - 0.5 * m_v,
                kappa * (theta - m_v),
                2.0 * mu * m_X - s_Xv + s2 * m_v,
                2.0 * kappa * (theta * m_v - s_vv) + xi2_s2 * m_v,
                kappa * (theta * m_X - s_Xv) + mu * m_v - 0.5 * s_vv + cross * m_v,
            ],
            dtype=np.float64,
        )

    if delta <= 0.0:
        mean = np.array([X0, v0], dtype=np.float64)
        return mean, np.zeros((2, 2), dtype=np.float64)

    n_steps = max(1, int(n_steps))
    h = float(delta) / n_steps
    for _ in range(n_steps):
        k1 = rhs(z)
        k2 = rhs(z + 0.5 * h * k1)
        k3 = rhs(z + 0.5 * h * k2)
        k4 = rhs(z + h * k3)
        z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    m_X, m_v, s_XX, s_vv, s_Xv = z
    mean = np.array([m_X, m_v], dtype=np.float64)
    cov = np.array(
        [
            [s_XX - m_X * m_X, s_Xv - m_X * m_v],
            [s_Xv - m_X * m_v, s_vv - m_v * m_v],
        ],
        dtype=np.float64,
    )
    # Numerical PSD cleanup for tiny negative eigenvalues.
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def exact_heston_sv_mean(
    delta: float,
    x0: np.ndarray,
    p: HestonParams,
) -> np.ndarray:
    """Closed-form mean of textbook Heston ``x = [S, v]``.

    Drift is affine, so
    ``E[S_t] = S0 exp(μ t)`` and ``E[v_t] = θ + (v0 − θ) exp(−κ t)``
    are exact (independent of ``ξ``, ``ρ``).
    """
    S0, v0 = float(x0[0]), float(x0[1])
    t = float(delta)
    m_S = S0 * np.exp(p.mu * t)
    m_v = p.theta + (v0 - p.theta) * np.exp(-p.kappa * t)
    return np.array([m_S, m_v], dtype=np.float64)


def mc_heston_sv_mean_cov(
    delta: float,
    x0: np.ndarray,
    p: HestonParams,
    *,
    n_paths: int = 1_000_000,
    n_steps: int | None = None,
    seed: int = 0,
    smooth_vol_eps: float | None = None,
    batch_paths: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo Itô-Euler estimate of ``(E[x], Cov(x))`` for ``x=[S,v]``.

    Uses antithetic Brownian increments. ``n_paths`` is the number of
    independent ``Z`` draws; the estimator uses ``2 n_paths`` trajectories.
    Paths are processed in batches of ``batch_paths`` and moments are
    accumulated with a numerically stable streaming update.

    If ``smooth_vol_eps`` is set, diffusion uses ``φ(v)=(v²+ε²)^{1/4}`` instead
    of ``√max(v,ε)``, matching :func:`make_heston_sv_dynamics`.
    """
    import jax
    import jax.random as jr
    from jax import lax

    if n_steps is None:
        n_steps = max(256, int(np.ceil(float(delta) / 5e-4)))
    n_steps = int(n_steps)
    n_paths = int(n_paths)
    batch_paths = int(max(1, min(batch_paths, n_paths)))
    S0 = jnp.asarray(x0[0], dtype=jnp.float64)
    v0 = jnp.asarray(x0[1], dtype=jnp.float64)
    mu, kappa, theta = p.mu, p.kappa, p.theta
    xi, rho, v_eps, s = p.xi, p.rho, p.v_eps, p.diff_scale
    rho_bar = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    dt = jnp.asarray(delta / n_steps, dtype=jnp.float64)
    sqrt_dt = jnp.sqrt(dt)
    eps2 = None if smooth_vol_eps is None else float(smooth_vol_eps) ** 2

    def _phi(v):
        if eps2 is None:
            return jnp.sqrt(jnp.maximum(v, v_eps))
        return jnp.power(v * v + eps2, 0.25)

    def one_batch(key, n_batch: int):
        S = jnp.full((n_batch,), S0, dtype=jnp.float64)
        v = jnp.full((n_batch,), v0, dtype=jnp.float64)
        Sm = S.copy()
        vm = v.copy()

        def body(carry, k):
            S_, v_, Sm_, vm_ = carry
            z = jr.normal(k, (n_batch, 2), dtype=jnp.float64)
            dW = z * sqrt_dt
            dWm = -dW

            def step(S__, v__, dW__):
                sv = s * _phi(v__)
                dS = mu * S__ * dt + S__ * sv * dW__[:, 0]
                dv = kappa * (theta - v__) * dt + xi * sv * (
                    rho * dW__[:, 0] + rho_bar * dW__[:, 1]
                )
                return S__ + dS, v__ + dv

            S_, v_ = step(S_, v_, dW)
            Sm_, vm_ = step(Sm_, vm_, dWm)
            return (S_, v_, Sm_, vm_), None

        keys = jr.split(key, n_steps)
        (S, v, Sm, vm), _ = lax.scan(body, (S, v, Sm, vm), keys)
        xx = jnp.stack(
            [jnp.concatenate([S, Sm]), jnp.concatenate([v, vm])], axis=1
        )
        mean = jnp.mean(xx, axis=0)
        xc = xx - mean
        # Unnormalized scatter; caller converts with total count.
        scatter = xc.T @ xc
        return mean, scatter, xx.shape[0]

    # Stream batches: Chan-style mean/cov merge.
    total_n = 0
    mean = np.zeros(2, dtype=np.float64)
    scatter = np.zeros((2, 2), dtype=np.float64)
    key = jr.key(int(seed))
    remaining = n_paths
    batch_i = 0
    # Cache jitted kernels by batch size (last batch may be smaller).
    jitted: dict[int, object] = {}

    while remaining > 0:
        n_batch = min(batch_paths, remaining)
        if n_batch not in jitted:
            jitted[n_batch] = jax.jit(lambda k, nb=n_batch: one_batch(k, nb))
        key, sub = jr.split(key)
        b_mean, b_scatter, b_n = jitted[n_batch](sub)
        b_mean = np.asarray(b_mean)
        b_scatter = np.asarray(b_scatter)
        b_n = int(b_n)
        if total_n == 0:
            mean = b_mean
            scatter = b_scatter
            total_n = b_n
        else:
            delta_m = b_mean - mean
            mean = mean + delta_m * (b_n / (total_n + b_n))
            scatter = (
                scatter
                + b_scatter
                + np.outer(delta_m, delta_m) * (total_n * b_n / (total_n + b_n))
            )
            total_n += b_n
        remaining -= n_batch
        batch_i += 1

    cov = scatter / float(total_n)
    cov = 0.5 * (cov + cov.T)
    return mean, cov
