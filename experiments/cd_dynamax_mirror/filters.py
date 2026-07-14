"""Faithful, dependency-light reimplementations of the cd_dynamax continuous-discrete
nonlinear-Gaussian filters, for benchmarking against the cuthbert Cubature Recursive
Filter (CRF).

These mirror the algorithms in hd-UQ/cd_dynamax
(``src/continuous_discrete_nonlinear_gaussian_ssm/inference_{ekf,ukf,enkf}.py``) but
avoid that package's heavy dependencies (TensorFlow-Probability + the bundled dynamax
tree), which do not install on this environment's Python. We keep the exact moment 
equations and offer two continuous-time forecast backends via ``solver=``:
* ``"diffrax"`` (default): matches cd_dynamax's ``diffrax_utils.diffeqsolve``
  (Dopri5 for moment/flow ODEs; Heun + VirtualBrownianTree for SDEs).
* ``"rk4"``: fixed-step classical RK4 (ODEs) / RK4 with frozen Wiener increment per
  step (SDEs), matching the CRF's integrator for fair wall-clock comparisons.

Model (continuous-discrete, additive observation noise):
    dx = f(t, x) dt + L(t, x) dW,   Cov(dW) = Qc dt
    y_k = h(x(t_k)) + v_k,          v_k ~ N(0, R)

All filters use a **predict-then-update** loop so their observation times line up with
the cuthbert CRF: forecast over ``[t_k, t_{k+1}]`` then assimilate ``y_k``.

References:
    Sarkka (2006), *Recursive Bayesian inference on stochastic differential equations*.
    hd-UQ/cd_dynamax (continuous-discrete extension of dynamax).
"""

from __future__ import annotations

import math
from typing import Callable, Literal, NamedTuple, Optional

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jacfwd, jacrev, lax, vmap

Vector = jax.Array
Matrix = jax.Array
DriftFn = Callable[[jax.Array, jax.Array], jax.Array]  # (t, x) -> (x_dim,)
ObsFn = Callable[[jax.Array], jax.Array]  # x -> (y_dim,)
MatFn = Callable[[jax.Array, jax.Array], jax.Array]  # (t, x) -> matrix
SolverKind = Literal["diffrax", "rk4"]


# ---------------------------------------------------------------------------
# Shared numerical helpers (mirroring cd_dynamax.dynamax.utils + debug_utils)
# ---------------------------------------------------------------------------
def psd(A: Matrix, jitter: float = 1e-9) -> Matrix:
    """Symmetrize and add a small jitter for numerical positive-definiteness."""
    A = 0.5 * (A + A.T)
    return A + jitter * jnp.eye(A.shape[0], dtype=A.dtype)


def psd_solve(A: Matrix, B: Matrix, jitter: float = 1e-9) -> Matrix:
    """Solve ``A X = B`` for a (symmetrized) PSD ``A``."""
    return jnp.linalg.solve(psd(A, jitter), B)


def gaussian_logpdf(y: Vector, mean: Vector, cov: Matrix) -> jax.Array:
    """Full-covariance multivariate-normal log density (replaces TFP MVN.log_prob)."""
    y = jnp.atleast_1d(y)
    d = mean.shape[0]
    L = jnp.linalg.cholesky(psd(cov))
    diff = y - mean
    alpha = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + alpha @ alpha)


_outer = vmap(lambda a, b: jnp.outer(a, b), in_axes=0, out_axes=0)


def diffeqsolve(
    drift: Callable,
    t0: float,
    t1: float,
    y0,
    *,
    diffusion: Optional[Callable] = None,
    key=None,
    noise_dim: Optional[int] = None,
    dt0: float = 0.01,
    args=None,
    max_steps: int = 100_000,
) -> object:
    """cd_dynamax-style diffrax wrapper.

    ODE (``diffusion is None``): Dopri5 + constant step (used for the EKF/UKF moment
    ODEs). SDE: Heun + VirtualBrownianTree (used for EnKF/PF particle forecasts).
    Returns the ``diffrax`` solution's ``ys`` (saved at ``t1`` only).
    """
    if diffusion is None:
        terms = dfx.ODETerm(drift)
        solver = dfx.Dopri5()
    else:
        if key is None:
            key = jr.PRNGKey(0)
        bm_shape = (noise_dim,) if noise_dim is not None else jnp.shape(y0)
        # Pad the range slightly: fixed-step solvers can evaluate marginally past
        # t1 due to floating-point overshoot, which VirtualBrownianTree rejects.
        bm = dfx.VirtualBrownianTree(
            t0=t0 - dt0, t1=t1 + dt0, tol=dt0 / 2.0, shape=bm_shape, key=key
        )
        terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diffusion, bm))
        solver = dfx.Heun()

    sol = dfx.diffeqsolve(
        terms,
        solver=solver,
        stepsize_controller=dfx.ConstantStepSize(),
        t0=t0,
        t1=t1,
        y0=y0,
        args=args,
        dt0=dt0,
        saveat=dfx.SaveAt(t1=True),
        max_steps=int(max_steps),
    )
    return sol.ys


def _rk4_ode_step(y, t0, t1, drift, args):
    """ One classical RK4 step for a (possibly pytree) ODE state."""
    dt = t1-t0
    half = 0.5*dt
    t_mid = t0+half

    def add_scaled(state, k, scale):
        return jax.tree.map(lambda s, ks: s + scale * ks, state, k)
    
    k1 = drift(t0, y, args)
    k2 = drift(t_mid, add_scaled(y, k1, half), args)
    k3 = drift(t_mid, add_scaled(y, k2, half), args)
    k4 = drift(t1, add_scaled(y, k3, dt), args)
    return jax.tree.map(
        lambda yi, a, b, c, d: yi + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d),
        y,
        k1,
        k2,
        k3,
        k4,
    )

def _rk4_ode_solve(drift, t0, t1, y0, dt0, args=None, n_steps: Optional[int] = None):
    """Fixed-step RK4 ODE integrate; return value matches ``diffeqsolve`` ``ys``.

    ``n_steps`` should be a Python ``int`` (static under JIT) so the scan is
    reverse-mode differentiable — required for flow-map EKF Jacobians.
    """
    if n_steps is None:
        # Best-effort concrete count; prefer callers that pass a Python int.
        n_steps = max(1, int(jnp.ceil((t1 - t0) / dt0)))
    n_steps = int(n_steps)
    dt = (t1 - t0) / n_steps

    def body(y, i):
        ti = t0 + i.astype(dt.dtype) * dt
        return _rk4_ode_step(y, ti, ti + dt, drift, args), None

    y1, _ = lax.scan(body, y0, jnp.arange(n_steps))
    return jax.tree.map(lambda a: a[None, ...], y1)


def _rk4_sde_step(x, t0, t1, dW, drift, diffusion, args):
    """One RK4 step with Wiener increment frozen as ``omega_dot = dW / dt``."""
    dt = t1 - t0
    omega_dot = dW / dt
    t_mid = 0.5 * (t0 + t1)
    def rhs(t, state):
        return drift(t, state, args) + diffusion(t, state, args) @ omega_dot
    k1 = rhs(t0, x)
    k2 = rhs(t_mid, x + 0.5 * dt * k1)
    k3 = rhs(t_mid, x + 0.5 * dt * k2)
    k4 = rhs(t1, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_sde_solve(
    drift, diffusion, t0, t1, y0, key, noise_dim, dt0, args=None, n_steps: Optional[int] = None
):
    """Fixed-step RK4 SDE integrate; return value matches ``diffeqsolve`` ``ys``."""
    if n_steps is None:
        n_steps = max(1, int(jnp.ceil((t1 - t0) / dt0)))
    n_steps = int(n_steps)
    dt = (t1 - t0) / n_steps
    keys = jr.split(key, n_steps)

    def body(carry, rng):
        x, ti = carry
        dW = jnp.sqrt(dt) * jr.normal(rng, (noise_dim,))
        x_next = _rk4_sde_step(x, ti, ti + dt, dW, drift, diffusion, args)
        return (x_next, ti + dt), None

    (y1, _), _ = lax.scan(body, (y0, t0), keys)
    return y1[None, ...]


def integrate(
    drift: Callable,
    t0: float,
    t1: float,
    y0,
    *,
    diffusion: Optional[Callable] = None,
    key=None,
    noise_dim: Optional[int] = None,
    dt0: float = 0.01,
    args=None,
    max_steps: int = 100_000,
    solver: SolverKind = "diffrax",
    n_steps: Optional[int] = None,
) -> object:
    """Continuous-time forecast integrator used by all CD filters.
    ``solver="diffrax"``: Dopri5 (ODE) / Heun+VBT (SDE), matching cd_dynamax.
    ``solver="rk4"``: fixed-step RK4, matched to the CRF integrator for fair timing.
    Returns a ``ys`` pytree with a leading length-1 time axis (``SaveAt(t1=True)``
    convention). Treat ``solver`` as a static argument when JIT-compiling callers.
    """
    if solver == "diffrax":
        return diffeqsolve(
            drift,
            t0,
            t1,
            y0,
            diffusion=diffusion,
            key=key,
            noise_dim=noise_dim,
            dt0=dt0,
            args=args,
            max_steps=max_steps,
        )
    if solver == "rk4":
        if diffusion is None:
            return _rk4_ode_solve(drift, t0, t1, y0, dt0, args=args, n_steps=n_steps)
        if key is None:
            key = jr.PRNGKey(0)
        if noise_dim is None:
            noise_dim = y0.shape[-1]
        return _rk4_sde_solve(
            drift, diffusion, t0, t1, y0, key, noise_dim, dt0, args=args, n_steps=n_steps
        )
    raise ValueError(f"Unknown solver={solver!r}; expected 'diffrax' or 'rk4'.")

    
def _rk4_n_steps(dt: float, dt0: float) -> int:
    """Static inner-step count for one assimilation interval ``dt``."""
    return max(1, int(math.ceil(float(dt) / float(dt0))))

# ---------------------------------------------------------------------------
# CD-EKF  (cd_dynamax inference_ekf.py, state_order="first")
# ---------------------------------------------------------------------------
class FilterResult(NamedTuple):
    """Filtered estimates over the observation sequence."""

    means: jax.Array  # (T, x_dim)
    covs: jax.Array  # (T, x_dim, x_dim)  (PF: ensemble covariance)
    cum_loglik: jax.Array  # (T,) cumulative marginal log-likelihood (nan if N/A)


def _flow_map(m, f, t0, t1, dt0, solver: SolverKind = "diffrax", n_steps: Optional[int] = None):
    """Deterministic flow ``x(t1)`` of ``dx/dt = f(t, x)`` from ``x(t0) = m``."""
    ys = integrate(
        lambda t, y, a: f(t, y), t0, t1, m, dt0=dt0, solver=solver, n_steps=n_steps
    )
    return ys[-1]


def _ekf_predict(m, P, f, L_fn, Qc_fn, t0, t1, dt0, state_order="first", solver: SolverKind = "diffrax", n_steps: Optional[int] = None):
    if state_order == "flowmap":
        # Discrete EKF on the SDE flow map: linearize the flow via autodiff, add the
        # interval's process noise once. Bounded/stable on chaotic systems.
        m_pred = _flow_map(m, f, t0, t1, dt0, solver=solver, n_steps=n_steps)
        # jacrev (reverse mode) is required: diffrax's default adjoint is a custom_vjp,
        # which forward-mode autodiff (jacfwd) cannot traverse. RK4 is fine either way
        J = jacrev(lambda x: _flow_map(x, f, t0, t1, dt0, solver=solver, n_steps=n_steps))(m)
        L_t = L_fn(t0, m)
        Qc_t = Qc_fn(t0, m)
        Q = (t1 - t0) * L_t @ Qc_t @ L_t.T
        return m_pred, psd(J @ P @ J.T + Q)

    # state_order == "first": canonical continuous-discrete EKF moment ODEs.
    def rhs(t, y, args):
        m_t, P_t = y
        F_t = jacfwd(lambda x: f(t, x))(m_t)
        L_t = L_fn(t, m_t)
        Qc_t = Qc_fn(t, m_t)
        dm = f(t, m_t)
        dP = F_t @ P_t + P_t @ F_t.T + L_t @ Qc_t @ L_t.T
        return (dm, dP)

    ys = integrate(rhs, t0, t1, (m, P), dt0=dt0, solver=solver, n_steps=n_steps)
    return ys[0][-1], psd(ys[1][-1])


def _ekf_update(m, P, h, R, y, num_iter):
    def step(carry, _):
        mm, PP = carry
        H_x = jacfwd(h)(mm)
        S = psd(R + H_x @ PP @ H_x.T)
        K = psd_solve(S, H_x @ PP).T
        PP_new = psd(PP - K @ S @ K.T)
        mm_new = mm + K @ (y - h(mm))
        return (mm_new, PP_new), None

    (m_c, P_c), _ = lax.scan(step, (m, P), jnp.arange(num_iter))
    return m_c, P_c


def run_cd_ekf(
    f: DriftFn,
    h: ObsFn,
    L_fn: MatFn,
    Qc_fn: MatFn,
    R: Matrix,
    m0: Vector,
    P0: Matrix,
    ys: jax.Array,
    dt: float,
    dt0: float = 0.01,
    num_iter: int = 1,
    state_order: str = "first",
    solver: SolverKind = "diffrax",
) -> FilterResult:
    """Continuous-discrete Extended Kalman Filter (predict-then-update).

    ``state_order="first"`` integrates the canonical CD-EKF mean/covariance ODEs;
    ``state_order="flowmap"`` linearizes the deterministic flow map and adds the
    interval process noise (a stable discrete-EKF variant).
    ``solver`` selects the continuous-time integrator (``"diffrax"`` or ``"rk4"``).
    """
    n = ys.shape[0]
    n_inner = _rk4_n_steps(dt, dt0) if solver == "rk4" else None

    def step(carry, k):
        m, P, ll = carry
        t0 = k * dt
        m_pred, P_pred = _ekf_predict(
            m, P, f, L_fn, Qc_fn, t0, t0 + dt, dt0, state_order, solver=solver, n_steps=n_inner
        )
        H_x = jacfwd(h)(m_pred)
        ll = ll + gaussian_logpdf(ys[k], h(m_pred), H_x @ P_pred @ H_x.T + R)
        m_filt, P_filt = _ekf_update(m_pred, P_pred, h, R, ys[k], num_iter)
        return (m_filt, P_filt, ll), (m_filt, P_filt, ll)

    _, (means, covs, lls) = lax.scan(step, (m0, P0, 0.0), jnp.arange(n))
    return FilterResult(means, covs, lls)


# ---------------------------------------------------------------------------
# CD-UKF  (cd_dynamax inference_ukf.py, state_order="first")
# ---------------------------------------------------------------------------
class UKFParams(NamedTuple):
    alpha: float = jnp.sqrt(3.0)
    beta: float = 2.0
    kappa: float = 1.0


def _ukf_sigmas(m, P, n, lamb):
    distances = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(psd(P))
    sigma_plus = m[None, :] + distances.T
    sigma_minus = m[None, :] - distances.T
    return jnp.concatenate([m[None, :], sigma_plus, sigma_minus], axis=0)


def _ukf_weights(n, alpha, beta, lamb):
    factor = 1.0 / (2.0 * (n + lamb))
    w_mean = jnp.concatenate([jnp.array([lamb / (n + lamb)]), jnp.full(2 * n, factor)])
    w_cov = jnp.concatenate(
        [jnp.array([lamb / (n + lamb) + (1.0 - alpha**2 + beta)]), jnp.full(2 * n, factor)]
    )
    I_w = jnp.eye(2 * n + 1) - w_mean[:, None]
    W_matrix = I_w @ jnp.diag(w_cov) @ I_w.T
    return w_mean, w_cov, W_matrix


def _ukf_predict(
    m, P, f, L_fn, Qc_fn, t0, t1, dt0, n, lamb, w_mean, w_cov, W_matrix,
    state_order="first", solver: SolverKind = "diffrax", n_steps: Optional[int] = None,
):
    if state_order == "flowmap":
        # Push sigma points through the deterministic flow, rebuild moments by the
        # unscented transform, then add the interval process noise (cd_dynamax
        # "zeroth"/"discrete"). Bounded/stable on chaotic systems.
        X = _ukf_sigmas(m, P, n, lamb)
        Xf = vmap(lambda x: _flow_map(x, f, t0, t1, dt0, solver=solver, n_steps=n_steps))(X)
        m_pred = w_mean @ Xf
        L_t = L_fn(t0, m)
        Qc_t = Qc_fn(t0, m)
        Q = (t1 - t0) * L_t @ Qc_t @ L_t.T
        P_pred = jnp.tensordot(w_cov, _outer(Xf - m_pred, Xf - m_pred), axes=1) + Q
        return m_pred, psd(P_pred)

    # state_order == "first": canonical continuous-discrete UKF moment ODEs.
    def rhs(t, y, args):
        m_t, P_t = y
        X_t = _ukf_sigmas(m_t, P_t, n, lamb)
        f_X = vmap(lambda x: f(t, x))(X_t)
        L_t = L_fn(t, m_t)
        Qc_t = Qc_fn(t, m_t)
        dm = f_X.T @ w_mean
        foo = f_X.T @ W_matrix @ X_t
        dP = foo + foo.T + L_t @ Qc_t @ L_t.T
        return (dm, dP)

    ys = integrate(rhs, t0, t1, (m, P), dt0=dt0, solver=solver, n_steps=n_steps)
    return ys[0][-1], psd(ys[1][-1])


def _ukf_update(m, P, h, R, y, n, lamb, w_mean, w_cov):
    sigmas = _ukf_sigmas(m, P, n, lamb)
    sigmas_prop = vmap(h)(sigmas)
    y_pred = w_mean @ sigmas_prop
    S = psd(R + jnp.tensordot(w_cov, _outer(sigmas_prop - y_pred, sigmas_prop - y_pred), axes=1))
    C = jnp.tensordot(w_cov, _outer(sigmas - m, sigmas_prop - y_pred), axes=1)
    ll = gaussian_logpdf(y, y_pred, S)
    K = psd_solve(S, C.T).T
    m_cond = m + K @ (y - y_pred)
    P_cond = psd(P - K @ S @ K.T)
    return ll, m_cond, P_cond


def run_cd_ukf(
    f: DriftFn,
    h: ObsFn,
    L_fn: MatFn,
    Qc_fn: MatFn,
    R: Matrix,
    m0: Vector,
    P0: Matrix,
    ys: jax.Array,
    dt: float,
    dt0: float = 0.01,
    ukf_params: UKFParams = UKFParams(),
    state_order: str = "first",
    solver: SolverKind = "diffrax",
) -> FilterResult:
    """Continuous-discrete additive Unscented Kalman Filter (predict-then-update).

    ``state_order="first"`` integrates the CD-UKF sigma-point moment ODEs;
    ``state_order="flowmap"`` propagates sigma points through the deterministic flow
    and adds the interval process noise (a stable discrete-UKF variant).
    ``solver`` selects the continuous-time integrator (``"diffrax"`` or ``"rk4"``).
    """
    n_steps = ys.shape[0]
    n = m0.shape[0]
    alpha, beta, kappa = ukf_params.alpha, ukf_params.beta, ukf_params.kappa
    lamb = alpha**2 * (n + kappa) - n
    w_mean, w_cov, W_matrix = _ukf_weights(n, alpha, beta, lamb)
    n_inner = _rk4_n_steps(dt, dt0) if solver == "rk4" else None

    def step(carry, k):
        m, P, ll = carry
        t0 = k * dt
        m_pred, P_pred = _ukf_predict(
            m, P, f, L_fn, Qc_fn, t0, t0 + dt, dt0, n, lamb, w_mean, w_cov, W_matrix, state_order, solver=solver, n_steps=n_inner
        )
        ll_inc, m_filt, P_filt = _ukf_update(m_pred, P_pred, h, R, ys[k], n, lamb, w_mean, w_cov)
        ll = ll + ll_inc
        return (m_filt, P_filt, ll), (m_filt, P_filt, ll)

    _, (means, covs, lls) = lax.scan(step, (m0, P0, 0.0), jnp.arange(n_steps))
    return FilterResult(means, covs, lls)


# ---------------------------------------------------------------------------
# CD-EnKF  (cd_dynamax inference_enkf.py, state_order="first")
# ---------------------------------------------------------------------------
def _enkf_forecast(key, particles, f, L_fn, Qc_fn, t0, t1, dt0, noise_dim, cov_rescaling, solver: SolverKind = "diffrax", 
                   n_steps: Optional[int] = None):
    def drift(t, y, args):
        return f(t, y)

    def diffusion(t, y, args):
        L_t = L_fn(t, y) * cov_rescaling
        return L_t @ jnp.linalg.cholesky(psd(Qc_fn(t, y)))

    def one(y0, k):
        ys = integrate(
            drift, t0, t1, y0, diffusion=diffusion, key=k, noise_dim=noise_dim, dt0=dt0, solver=solver, n_steps=n_steps
        )
        return ys[0]

    keys = jr.split(key, particles.shape[0])
    return vmap(one)(particles, keys)


def _enkf_analysis(key, particles, h, R, y, perturb_obs, inflation_delta):
    n_p = particles.shape[0]
    if inflation_delta > 0.0:
        x_mean = jnp.mean(particles, axis=0)
        particles = x_mean + (1.0 + inflation_delta) * (particles - x_mean)

    y_ens = vmap(h)(particles)
    y_mean = jnp.mean(y_ens, axis=0)
    dy = y_ens - y_mean
    dx = particles - jnp.mean(particles, axis=0)
    y_cov = psd(jnp.sum(_outer(dy, dy), axis=0) / (n_p - 1))
    cross = jnp.sum(_outer(dx, dy), axis=0) / (n_p - 1)

    S = psd(y_cov + R)
    K = psd_solve(S, cross.T).T
    if perturb_obs:
        y_pert = jr.multivariate_normal(key, mean=y, cov=psd(R), shape=(n_p,))
    else:
        y_pert = jnp.broadcast_to(y, (n_p, y.shape[0]))
    return particles + (K @ (y_pert - y_ens).T).T, y_mean, S


def run_cd_enkf(
    f: DriftFn,
    h: ObsFn,
    L_fn: MatFn,
    Qc_fn: MatFn,
    R: Matrix,
    m0: Vector,
    P0: Matrix,
    ys: jax.Array,
    dt: float,
    key: jax.Array,
    n_particles: int = 50,
    noise_dim: Optional[int] = None,
    dt0: float = 0.01,
    cov_rescaling: float = 1.0,
    inflation_delta: float = 0.0,
    perturb_obs: bool = True,
    solver: SolverKind = "diffrax",
) -> FilterResult:
    """Continuous-discrete Ensemble Kalman Filter (perturbed-obs, predict-then-update).

    Forecast uses per-particle SDE integration (``solver="diffrax"``: Heun + VBT;
    ``solver="rk4"``: RK4 with frozen Wiener increments). A Gaussian marginal
    log-likelihood is accumulated from the ensemble innovation statistics
    (diagnostic only; the EnKF is not a likelihood-exact filter).
    """
    n_steps = ys.shape[0]
    if noise_dim is None:
        noise_dim = m0.shape[0]
    key, init_key = jr.split(key)
    particles = jr.multivariate_normal(init_key, mean=m0, cov=psd(P0), shape=(n_particles,))
    n_inner = _rk4_n_steps(dt, dt0) if solver == "rk4" else None


    def step(carry, k):
        parts, ll, key = carry
        key, k_pred, k_filt = jr.split(key, 3)
        t0 = k * dt
        x_pred = _enkf_forecast(
            k_pred, parts, f, L_fn, Qc_fn, t0, t0 + dt, dt0, noise_dim, cov_rescaling, solver=solver, n_steps=n_inner
        )
        x_filt, y_mean, S = _enkf_analysis(
            k_filt, x_pred, h, R, ys[k], perturb_obs, inflation_delta
        )
        ll = ll + gaussian_logpdf(ys[k], y_mean, S)
        mean = jnp.mean(x_filt, axis=0)
        cov = jnp.cov(x_filt, rowvar=False)
        return (x_filt, ll, key), (mean, cov, ll)

    _, (means, covs, lls) = lax.scan(step, (particles, 0.0, key), jnp.arange(n_steps))
    return FilterResult(means, covs, lls)


# ---------------------------------------------------------------------------
# Bootstrap particle filter (continuous-discrete, SDE proposal)
# ---------------------------------------------------------------------------
def _systematic_resample(key, weights):
    n = weights.shape[0]
    positions = (jr.uniform(key) + jnp.arange(n)) / n
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, positions)
    return jnp.clip(idx, 0, n - 1)


def run_bootstrap_pf(
    f: DriftFn,
    h: ObsFn,
    L_fn: MatFn,
    Qc_fn: MatFn,
    R: Matrix,
    m0: Vector,
    P0: Matrix,
    ys: jax.Array,
    dt: float,
    key: jax.Array,
    n_particles: int = 200,
    noise_dim: Optional[int] = None,
    dt0: float = 0.01,
    solver: SolverKind = "diffrax",
) -> FilterResult:
    """Bootstrap particle filter with an SDE proposal, resampling each step.

    The transition prior is the SDE forecast (``solver="diffrax"``: Heun + VBT;
    ``solver="rk4"``: RK4 with frozen Wiener increments); weights are the Gaussian
    observation likelihood. Accumulates the standard particle-filter log-evidence
    estimate.
    """
    n_steps = ys.shape[0]
    if noise_dim is None:
        noise_dim = m0.shape[0]
    key, init_key = jr.split(key)
    particles = jr.multivariate_normal(init_key, mean=m0, cov=psd(P0), shape=(n_particles,))
    n_inner = _rk4_n_steps(dt, dt0) if solver == "rk4" else None

    def step(carry, k):
        parts, ll, key = carry
        key, k_pred, k_res = jr.split(key, 3)
        t0 = k * dt
        x_pred = _enkf_forecast(
            k_pred, parts, f, L_fn, Qc_fn, t0, t0 + dt, dt0, noise_dim, 1.0, solver=solver, n_steps=n_inner,
        )
        log_w = vmap(lambda x: gaussian_logpdf(ys[k], h(x), R))(x_pred)
        ll_inc = jax.scipy.special.logsumexp(log_w) - jnp.log(n_particles)
        ll = ll + ll_inc
        weights = jax.nn.softmax(log_w)
        mean = weights @ x_pred
        dx = x_pred - mean
        cov = (weights[:, None] * dx).T @ dx
        idx = _systematic_resample(k_res, weights)
        return (x_pred[idx], ll, key), (mean, cov, ll)

    _, (means, covs, lls) = lax.scan(step, (particles, 0.0, key), jnp.arange(n_steps))
    return FilterResult(means, covs, lls)
