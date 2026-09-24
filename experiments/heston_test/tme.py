"""Taylor moment expansion (TME) of SDE transition mean and covariance.

Implements Definition 1 and the degree-M covariance truncation of Zhao,
Karvonen, Hostettler, and Särkkä, IEEE TAC 2021: the maps
``E[φ(x_τ)|x_0]`` for ``φ(x)=x_i`` and ``φ(x)=x_i x_j`` are expanded in
time via the Itô generator, then the covariance polynomial is kept at
degree ``M`` (their Remark 2 / ``Φ_{x,r}`` form).

Used as a point-mass predictor (``P0=0``), matching the Heston weak-error
scaling protocol.
"""

from __future__ import annotations

from math import comb
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

DriftFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DiffusionFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

_EVAL_CACHE: dict[tuple[int, int, int], object] = {}


def _moment_monomials_2d(x: jnp.ndarray) -> jnp.ndarray:
    """``[x0, x1, x0², x0 x1, x1²]``."""
    return jnp.array(
        [x[0], x[1], x[0] * x[0], x[0] * x[1], x[1] * x[1]],
        dtype=x.dtype,
    )


def _make_eval_generator_iterates(
    drift_fn: DriftFn,
    L_fn: DiffusionFn,
    max_order: int,
):
    """Return ``eval(x) -> (max_order+1, 5)`` array of ``A^r`` monomials."""

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return drift_fn(jnp.asarray(0.0, dtype=x.dtype), x)

    def Gamma(x: jnp.ndarray) -> jnp.ndarray:
        L = L_fn(jnp.asarray(0.0, dtype=x.dtype), x)
        return L @ L.T

    def apply_A(g):
        def Ag(x: jnp.ndarray) -> jnp.ndarray:
            J = jax.jacfwd(g)(x)
            H = jax.jacfwd(jax.jacfwd(g))(x)
            return J @ f(x) + 0.5 * jnp.einsum("ijk,jk->i", H, Gamma(x))

        return Ag

    gens = [_moment_monomials_2d]
    g = _moment_monomials_2d
    for _ in range(int(max_order)):
        g = apply_A(g)
        gens.append(g)

    def eval_all(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([gi(x) for gi in gens], axis=0)

    return jax.jit(eval_all)


def cached_eval_fn(drift_fn: DriftFn, L_fn: DiffusionFn, max_order: int):
    """Jitted ``x -> (max_order+1, 5)`` generator iterates, cached by function id."""
    key = (id(drift_fn), id(L_fn), int(max_order))
    eval_all = _EVAL_CACHE.get(key)
    if eval_all is None:
        eval_all = _make_eval_generator_iterates(drift_fn, L_fn, int(max_order))
        _EVAL_CACHE[key] = eval_all
    return eval_all


def _eval_iterates(
    x: jnp.ndarray,
    max_order: int,
    drift_fn: DriftFn,
    L_fn: DiffusionFn,
) -> np.ndarray:
    eval_all = cached_eval_fn(drift_fn, L_fn, max_order)
    return np.asarray(eval_all(x), dtype=np.float64)


def tme_mean_cov(
    x0: np.ndarray | jnp.ndarray,
    dt: float,
    order: int,
    drift_fn: DriftFn,
    L_fn: DiffusionFn,
) -> tuple[np.ndarray, np.ndarray]:
    """``M``-th order TME estimates of ``(E[x_τ], Cov(x_τ))`` from a point ``x0``.

    Parameters
    ----------
    x0
        Initial state, shape ``(2,)``.
    dt
        Expansion horizon ``τ``.
    order
        Expansion order ``M`` (2 or 3 in the Heston comparison).
    drift_fn, L_fn
        Itô drift and diffusion ``(t, x) -> ...`` with ``Qc = I``.
    """
    if order < 1:
        raise ValueError("TME order must be >= 1")
    x = jnp.asarray(x0, dtype=jnp.float64)
    if x.shape != (2,):
        raise ValueError(f"TME helper is 2D; got x0 shape {tuple(x.shape)}")
    vals = _eval_iterates(x, int(order), drift_fn, L_fn)
    return _assemble_tme(vals, float(dt), int(order))


def _assemble_tme(
    vals: np.ndarray,
    dt: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``a_M`` and degree-``M`` ``Σ_M`` from generator iterates."""
    Ar_x = vals[:, :2]
    a = np.zeros(2, dtype=np.float64)
    fact = 1.0
    for r in range(order + 1):
        if r > 0:
            fact *= r
        a = a + (dt**r / fact) * Ar_x[r]

    Sigma = np.zeros((2, 2), dtype=np.float64)
    fact = 1.0
    for r in range(1, order + 1):
        fact *= r
        Phi = np.array(
            [[vals[r, 2], vals[r, 3]], [vals[r, 3], vals[r, 4]]],
            dtype=np.float64,
        )
        for s in range(r + 1):
            Phi = Phi - comb(r, s) * np.outer(Ar_x[s], Ar_x[r - s])
        Sigma = Sigma + (dt**r / fact) * Phi
    Sigma = 0.5 * (Sigma + Sigma.T)
    return a, Sigma


def assemble_tme_jax(vals: jnp.ndarray, dt: float, order: int):
    """JAX ``(a_M, Σ_M)`` from generator iterates; ``order`` must be static."""
    order = int(order)
    dt = jnp.asarray(dt, dtype=vals.dtype)
    Ar_x = vals[:, :2]
    a = jnp.zeros(2, dtype=vals.dtype)
    fact = 1.0
    for r in range(order + 1):
        if r > 0:
            fact *= r
        a = a + (dt**r / fact) * Ar_x[r]

    Sigma = jnp.zeros((2, 2), dtype=vals.dtype)
    fact = 1.0
    for r in range(1, order + 1):
        fact *= r
        Phi = jnp.array(
            [[vals[r, 2], vals[r, 3]], [vals[r, 3], vals[r, 4]]],
            dtype=vals.dtype,
        )
        for s in range(r + 1):
            Phi = Phi - comb(r, s) * jnp.outer(Ar_x[s], Ar_x[r - s])
        Sigma = Sigma + (dt**r / fact) * Phi
    Sigma = 0.5 * (Sigma + Sigma.T)
    return a, Sigma
