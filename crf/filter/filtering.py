"""Implements the Cubature Recursive Filter (CRF) predict and update steps.

The CRF treats the filtering distribution as **Gaussian** (mean + Cholesky
covariance) but performs the *predict* step by propagating spatial cubature
nodes along a small set of deterministic Wiener-space cubature paths through the
nonlinear Stratonovich SDE, with **no Jacobian and no Monte-Carlo sampling**.
The resulting weighted cloud is collapsed back to Gaussian moments. The *update*
is a statistical-linearization (sigma-point) Kalman update in square-root form,
sharing the cross-covariance structure of the EnKF/UKF update.

System (Stratonovich):
    dx = drift(t, x) dt + diffusion(t, x) o dW
    y  = h(x) + v,   v ~ N(0, R)

References:
    Lyons & Victoir (2004), *Cubature on Wiener space*.
    Arasaratnam & Haykin (2009), *Cubature Kalman Filters*.
"""

from typing import Callable, Tuple

from jax import checkpoint, lax, vmap
from jax import numpy as jnp

from cuthbertlib.linalg import tria
from cuthbertlib.stats import multivariate_normal
from cuthbertlib.types import Array, ScalarArray

# (t, x) -> drift vector of shape (x_dim,)
DriftFn = Callable[[ScalarArray, Array], Array]
# (t, x) -> diffusion matrix of shape (x_dim, noise_dim)
DiffusionFn = Callable[[ScalarArray, Array], Array]
# x -> observation mean of shape (y_dim,)
ObservationFn = Callable[[Array], Array]


def spatial_cubature_nodes(mean: Array, chol_cov: Array) -> Tuple[Array, Array]:
    """Degree-3 spherical-cubature nodes for a Gaussian ``N(mean, cov)``.

    Args:
        mean: Mean vector, shape ``(x_dim,)``.
        chol_cov: Generalised Cholesky factor of the covariance, shape ``(x_dim, x_dim)``.

    Returns:
        Tuple ``(nodes, weights)`` with ``nodes`` of shape ``(2 * x_dim, x_dim)``
        and uniform ``weights`` of shape ``(2 * x_dim,)``.
    """
    n = mean.shape[0]
    scale = jnp.sqrt(jnp.asarray(n, dtype=chol_cov.dtype))
    nodes = jnp.concatenate([mean + scale * chol_cov.T, mean - scale * chol_cov.T], axis=0)
    weights = jnp.full(2 * n, 1.0 / (2 * n), dtype=chol_cov.dtype)
    return nodes, weights


def _rk4_stratonovich_step(
    x: Array,
    t0: ScalarArray,
    t1: ScalarArray,
    omega_dot: Array,
    drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
) -> Array:
    """One RK4 step for ``dx/dt = drift(t, x) + diffusion(t, x) @ omega_dot``."""
    dt = t1 - t0
    t_mid = 0.5 * (t0 + t1)

    def rhs(t, state):
        return drift_fn(t, state) + diffusion_fn(t, state) @ omega_dot

    k1 = rhs(t0, x)
    k2 = rhs(t_mid, x + 0.5 * dt * k1)
    k3 = rhs(t_mid, x + 0.5 * dt * k2)
    k4 = rhs(t1, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_stratonovich_cde(
    x0: Array,
    ts: Array,
    omega_path: Array,
    drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
    *,
    checkpoint_steps: bool = False,
) -> Array:
    """Integrate the Stratonovich SDE along a fixed cubature path ``omega(t)``.

    Args:
        x0: Initial state, shape ``(x_dim,)``.
        ts: Integration time grid, shape ``(num_steps + 1,)``.
        omega_path: Driver path values on ``ts``, shape ``(num_steps + 1, noise_dim)``.
        drift_fn: Drift function ``(t, x) -> (x_dim,)``.
        diffusion_fn: Diffusion function ``(t, x) -> (x_dim, noise_dim)``.
        checkpoint_steps: If ``True``, rematerialize each RK4 step under reverse-mode
            AD (``jax.checkpoint``). Because the Wiener driver is a frozen ODE path,
            this yields an ``O(1)``-memory (in the number of inner steps) adjoint
            through the Stratonovich CDE, at the cost of recomputing the forward
            pass during the backward pass.

    Returns:
        Terminal state, shape ``(x_dim,)``.
    """
    if omega_path.ndim == 1:
        omega_path = omega_path[:, None]

    dts = ts[1:] - ts[:-1]
    omega_dot = (omega_path[1:] - omega_path[:-1]) / dts[:, None]

    def step(x, inputs):
        t0, t1, od = inputs
        return _rk4_stratonovich_step(x, t0, t1, od, drift_fn, diffusion_fn), None

    if checkpoint_steps:
        step = checkpoint(step)

    x_final, _ = lax.scan(step, x0, (ts[:-1], ts[1:], omega_dot))
    return x_final


def _propagate(
    nodes: Array,
    path_values: Array,
    ts: Array,
    drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
    *,
    checkpoint_steps: bool = False,
) -> Array:
    """Propagate every (spatial node, Wiener path) pair to its terminal state.

    Returns an array of shape ``(num_nodes, num_paths, x_dim)``.
    """

    def one(node, path):
        return integrate_stratonovich_cde(
            node,
            ts,
            path,
            drift_fn,
            diffusion_fn,
            checkpoint_steps=checkpoint_steps,
        )

    return vmap(vmap(one, in_axes=(None, 0)), in_axes=(0, None))(nodes, path_values)


def predict(
    mean: Array,
    chol_cov: Array,
    ts: Array,
    path_values: Array,
    path_weights: Array,
    drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
    *,
    checkpoint_steps: bool = False,
) -> Tuple[Array, Array, Array, Array]:
    """CRF predict: propagate cubature nodes along Wiener paths, collapse to moments.

    Args:
        mean: Filtering mean from the previous step, shape ``(x_dim,)``.
        chol_cov: Generalised Cholesky factor of the covariance, shape ``(x_dim, x_dim)``.
        ts: Integration time grid, shape ``(num_steps + 1,)``.
        path_values: Wiener cubature paths, shape ``(num_paths, num_steps + 1, noise_dim)``.
        path_weights: Wiener cubature weights, shape ``(num_paths,)``.
        drift_fn: Drift function ``(t, x) -> (x_dim,)``.
        diffusion_fn: Diffusion function ``(t, x) -> (x_dim, noise_dim)``.
        checkpoint_steps: Rematerialize each inner RK4 step under reverse-mode AD
            (see :func:`integrate_stratonovich_cde`).

    Returns:
        Tuple ``(predicted_mean, predicted_chol_cov, samples, weights)`` where
        ``samples`` of shape ``(num_nodes * num_paths, x_dim)`` and ``weights``
        of shape ``(num_nodes * num_paths,)`` are the propagated weighted cloud
        (reused by :func:`update`).
    """
    x_dim = mean.shape[0]
    nodes, node_weights = spatial_cubature_nodes(mean, chol_cov)

    terminals = _propagate(
        nodes,
        path_values,
        ts,
        drift_fn,
        diffusion_fn,
        checkpoint_steps=checkpoint_steps,
    )
    samples = terminals.reshape(-1, x_dim)
    weights = (node_weights[:, None] * path_weights[None, :]).reshape(-1)
    weights = weights / jnp.sum(weights)

    predicted_mean = weights @ samples
    dev = (samples - predicted_mean).T * jnp.sqrt(weights)[None, :]
    predicted_chol_cov = tria(dev)

    return predicted_mean, predicted_chol_cov, samples, weights


def update(
    predicted_mean: Array,
    samples: Array,
    weights: Array,
    observation_fn: ObservationFn,
    chol_R: Array,
    y: Array,
) -> Tuple[Array, Array, ScalarArray]:
    """Statistical-linearization (sigma-point) Kalman update in square-root form.

    Uses the array square-root measurement update: a single triangularization of
    the pre-array yields the innovation Cholesky factor, the Kalman gain, and the
    updated state Cholesky factor simultaneously.

    Args:
        predicted_mean: Predicted mean, shape ``(x_dim,)``.
        samples: Propagated weighted cloud, shape ``(N, x_dim)``.
        weights: Cloud weights, shape ``(N,)``.
        observation_fn: Observation function ``x -> (y_dim,)``.
        chol_R: Cholesky factor of the observation noise covariance, shape ``(y_dim, y_dim)``.
        y: Observation vector, shape ``(y_dim,)``.

    Returns:
        Tuple ``(updated_mean, updated_chol_cov, log_likelihood, innov_mean, chol_S)``.
    """
    weights = weights / jnp.sum(weights)
    x_dim = predicted_mean.shape[0]
    y_dim = chol_R.shape[0]

    z = vmap(observation_fn)(samples)
    z_mean = weights @ z

    sqrt_w = jnp.sqrt(weights)
    z_dev = (z - z_mean).T * sqrt_w[None, :]
    x_dev = (samples - predicted_mean).T * sqrt_w[None, :]

    # Pre-array (block lower triangular after triangularization):
    #   [[ chol_R , z_dev ],     ->   [[ chol_S , 0           ],
    #    [   0    , x_dev ]]            [ L21    , chol_cov_upd ]]
    top = jnp.concatenate([chol_R, z_dev], axis=1)
    bottom = jnp.concatenate([jnp.zeros((x_dim, y_dim), dtype=chol_R.dtype), x_dev], axis=1)
    pre_array = jnp.concatenate([top, bottom], axis=0)
    post = tria(pre_array)

    chol_S = post[:y_dim, :y_dim]
    L21 = post[y_dim:, :y_dim]
    updated_chol_cov = post[y_dim:, y_dim:]

    # K = C_xz S^{-1} = L21 @ chol_S^{-1}; solve K @ chol_S = L21.
    gain = lax.linalg.triangular_solve(chol_S, L21, left_side=False, lower=True)
    updated_mean = predicted_mean + gain @ (y - z_mean)

    log_likelihood = multivariate_normal.logpdf(y, z_mean, chol_S, nan_support=False)

    return (
        updated_mean,
        updated_chol_cov,
        jnp.asarray(log_likelihood),
        z_mean,
        chol_S,
    )
