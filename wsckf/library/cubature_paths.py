"""Wiener-space cubature path construction for the Wiener Space Cubature Kalman Filter (WSC-KF).

These routines build a small set of deterministic Brownian "cubature" paths
omega^{j}(t)` together with weights w_j that match the moments
of the Wiener measure up to a chosen degree. They are computed **host-side with
NumPy** because they depend only on the assimilation interval, the Brownian
dimension and the cubature order -- never on the (traced) filter state. The
resulting arrays are therefore treated as static constants and closed over by the
JAX filtering kernels in wsckf.filtering.

Multidimensional constructions:

- ``"lv"``: explicit Lyons–Victoir *multi-D* formulae (Lyons & Victoir 2004).
  Degree ``m=5`` is a degree-5 Gaussian cubature on ``N(0,I_d)`` times a
  Bernoulli sign ``η=±1`` that realises the Lévy-area brackets
  ``A_{ij}=(η/2) z_i z_j`` (``2·3^d`` paths per macro-segment). For ``d=2``
  each path is the paper's planar construction: the 1D √22 path along the
  increment, plus a triangular perpendicular excursion that produces the
  area (equivariant under swapping driving axes). For ``d>2`` the same
  area targets are realised by coordinate-plane bumps. Degree ``m=3`` is
  the Stroud spherical rule (straight lines to ``±√d e_i``). This is **not**
  the tensor product of 1D path shapes.
- ``"product"``: tensor product of Lyons–Victoir *1D* cubature paths. Degree
  ``m=5`` uses the three-path √22 formula (``(3^k)^d`` paths); ``m=3`` uses the
  two-path straight-line formula (``(2^k)^d``). Misses mixed Lévy-area tensors.
- ``"recombined"``: recursive KLV expansion (always the m=5 3-branch) reduced by
  moment-matching (RMP) to a sparse measure.

``multidim_method="auto"`` (and ``build_filter(..., multidim_paths="auto")``)
selects ``"lv"`` for ``lyons_degree=5`` and ``d>1`` when the LV support fits
the path-count threshold, otherwise ``"recombined"``. 1D always uses the
classical 1D formulae.

References:
    Lyons & Victoir (2004), *Cubature on Wiener space*.
    Litterer & Lyons (2012), *High order recombination and an application to
        cubature on Wiener space*.
"""

from __future__ import annotations

import math
from itertools import combinations_with_replacement, product
from typing import List, Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np

__all__ = [
    "build_cubature_paths_1d",
    "build_lyons_victoir_multid_paths",
    "build_product_lyons_paths",
    "build_recombined_cubature_paths",
    "build_cubature_driver",
    "default_num_paths",
    "default_recombined_num_paths",
    "lv_path_count",
    "precompute_cubature_paths",
    "product_path_count",
    "select_multidim_paths",
    "resolve_multidim_method",
]

MultidimMethod = Literal["lv", "product", "recombined"]
MultidimMethodInput = Literal["lv", "product", "recombined", "auto"]

# Path counts above this threshold switch ``multidim_method="auto"`` to
# ``"recombined"``. Sized so degree-5 LV on ``d=2, k=2`` (324 paths) stays LV.
_AUTO_PRODUCT_PATH_THRESHOLD = 512

# Upper bound on the auto-selected post-recombination support when the caller
# does not specify ``recombined_num_paths`` but a high target degree is requested.
_RECOMBINE_BUDGET_CAP = 4096


# ---------------------------------------------------------------------------
# Lyons–Victoir 1D cubature paths (Proc. R. Soc. A 2004)
#
# The three-path formula with weights (1/6, 2/3, 1/6) and the √22 piecewise
# linear drivers is the explicit **degree m = 5** construction.  The library
# historically labelled it "degree-3" (confusing the branch count with m);
# callers should treat it as m = 5.
#
# Degree m = 3 is the two-path straight-line formula ω±(t) = ±t on [0, 1]
# with equal weights 1/2 (Gaussian cubature nodes ±1, lifted to paths).
# ---------------------------------------------------------------------------


def _cubature_path_1(t: np.ndarray) -> np.ndarray:
    left = -np.sqrt(3.0) * (0.5 * (4.0 - np.sqrt(22.0)) * t)
    middle = -np.sqrt(3.0) * (
        1.0 / 6.0 * (4.0 - np.sqrt(22.0)) + (-1.0 + np.sqrt(22.0)) * (t - 1.0 / 3.0)
    )
    right = -np.sqrt(3.0) * (
        1.0 / 6.0 * (2.0 + np.sqrt(22.0)) + 0.5 * (4.0 - np.sqrt(22.0)) * (t - 2.0 / 3.0)
    )
    return np.where(t <= 1.0 / 3.0, left, np.where(t <= 2.0 / 3.0, middle, right))


def _cubature_path_2(t: np.ndarray) -> np.ndarray:
    left = np.sqrt(3.0) * (0.5 * (4.0 - np.sqrt(22.0)) * t)
    middle = np.sqrt(3.0) * (
        1.0 / 6.0 * (4.0 - np.sqrt(22.0)) + (-1.0 + np.sqrt(22.0)) * (t - 1.0 / 3.0)
    )
    right = np.sqrt(3.0) * (
        1.0 / 6.0 * (2.0 + np.sqrt(22.0)) + 0.5 * (4.0 - np.sqrt(22.0)) * (t - 2.0 / 3.0)
    )
    return np.where(t <= 1.0 / 3.0, left, np.where(t <= 2.0 / 3.0, middle, right))


def _cubature_path_3(t: np.ndarray) -> np.ndarray:
    return np.zeros_like(t)



def build_cubature_paths_1d(
    k: int = 3,
    gamma: float = 0.6,
    steps_per_unit: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """Build Lyons–Victoir **degree m = 5** Wiener-space cubature paths on [0, 1].

    The base (k = 1) support has three paths with weights ``(1/6, 2/3, 1/6)``.
    Macro-segment count ``k`` composes the formula, giving ``3^k`` paths that
    remain degree-5 cubature on each scaled timeframe.

    Args:
        k: Number of macro segments.
        gamma: Macro-segment time-grid concentration exponent.
        steps_per_unit: Native time-grid resolution.

    Returns:
        Tuple (paths, weights, t_native) with paths of shape (3^k, T).
    """

    t_d = np.array(
        [1.0 - (1.0 - i / k) ** gamma for i in range(k + 1)], dtype=np.float64
    )
    p = np.zeros((3, 0), dtype=np.float64)
    w = np.zeros(3, dtype=np.float64)
    t_native = np.zeros(0, dtype=np.float64)
    dtt = 0

    for segment in range(k):
        dt = t_d[segment + 1] - t_d[segment]
        dtn = max(2, int(np.ceil(dt * steps_per_unit)))
        if segment == 0:
            x_loc = np.linspace(0.0, dt, dtn)
            p = np.stack(
                [
                    np.sqrt(dt) * _cubature_path_1(x_loc / dt),
                    _cubature_path_3(x_loc),
                    np.sqrt(dt) * _cubature_path_2(x_loc / dt),
                ],
                axis=0,
            )
            w[:] = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=np.float64)
            t_native = np.linspace(0.0, dt, dtn)
            dtt = dtn
        else:
            x_loc = np.linspace(dt / dtn, dt, dtn)
            q = p
            p = np.empty((q.shape[0] * 3, q.shape[1] + dtn), dtype=np.float64)
            w_old = w
            w = np.empty(q.shape[0] * 3, dtype=np.float64)

            for n in range(p.shape[0]):
                parent = n // 3
                if n % 3 == 0:
                    p[n, : q.shape[1]] = q[parent]
                else:
                    p[n, : q.shape[1]] = p[n - 1, : q.shape[1]]

                x0 = p[n, dtt - 1]
                if n % 3 == 0:
                    p[n, dtt:] = x0 + np.sqrt(dt) * _cubature_path_1(x_loc / dt)
                    w[n] = w_old[parent] * (1.0 / 6.0)
                elif (n - 1) % 3 == 0:
                    p[n, dtt:] = x0 + _cubature_path_3(x_loc)
                    w[n] = w_old[parent] * (2.0 / 3.0)
                else:
                    p[n, dtt:] = x0 + np.sqrt(dt) * _cubature_path_2(x_loc / dt)
                    w[n] = w_old[parent] * (1.0 / 6.0)

            t_native = np.concatenate(
                [t_native, t_native[-1] + np.linspace(dt / dtn, dt, dtn)]
            )
            dtt += dtn

    return p, w, t_native


def build_cubature_paths_1d_m3(
    k: int = 3,
    gamma: float = 0.6,
    steps_per_unit: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Lyons–Victoir **degree m = 3** Wiener-space cubature paths on [0, 1].

    Base (k = 1) support: two straight-line paths ``ω±(t) = ±t`` with weights
    ``1/2`` (Gaussian nodes ±1 lifted to paths). Composing ``k`` macro segments
    yields ``2^k`` paths that remain degree-3 cubature on each scaled timeframe.
    """
    t_d = np.array(
        [1.0 - (1.0 - i / k) ** gamma for i in range(k + 1)], dtype=np.float64
    )
    p = np.zeros((2, 0), dtype=np.float64)
    w = np.zeros(2, dtype=np.float64)
    t_native = np.zeros(0, dtype=np.float64)
    dtt = 0

    for segment in range(k):
        dt = t_d[segment + 1] - t_d[segment]
        dtn = max(2, int(np.ceil(dt * steps_per_unit)))
        if segment == 0:
            x_loc = np.linspace(0.0, dt, dtn)
            # ω±(s) = ±s on [0, 1] scaled to segment length dt → ±sqrt(dt)*(x/dt)*sqrt(dt)
            # wait: on [0,1], ω±(t)=±t has ω(1)=±1. For time dt, ω_dt(t)=√dt · ω(t/dt)
            # so ω_dt(t) = ±√dt · (t/dt) = ± t/√dt ... terminal = ±√dt. Correct for Var=dt.
            s = x_loc / dt  # in [0, 1]
            p = np.stack(
                [
                    np.sqrt(dt) * s,
                    -np.sqrt(dt) * s,
                ],
                axis=0,
            )
            w[:] = np.array([0.5, 0.5], dtype=np.float64)
            t_native = np.linspace(0.0, dt, dtn)
            dtt = dtn
        else:
            x_loc = np.linspace(dt / dtn, dt, dtn)
            s = x_loc / dt
            q = p
            p = np.empty((q.shape[0] * 2, q.shape[1] + dtn), dtype=np.float64)
            w_old = w
            w = np.empty(q.shape[0] * 2, dtype=np.float64)

            for n in range(p.shape[0]):
                parent = n // 2
                if n % 2 == 0:
                    p[n, : q.shape[1]] = q[parent]
                else:
                    p[n, : q.shape[1]] = p[n - 1, : q.shape[1]]

                x0 = p[n, dtt - 1]
                sign = 1.0 if n % 2 == 0 else -1.0
                p[n, dtt:] = x0 + sign * np.sqrt(dt) * s
                w[n] = w_old[parent] * 0.5

            t_native = np.concatenate(
                [t_native, t_native[-1] + np.linspace(dt / dtn, dt, dtn)]
            )
            dtt += dtn

    return p, w, t_native


def _resample_paths_1d(
        paths: np.ndarray, t_native: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
    t_target = np.asarray(t_target, dtype = np.float64)
    out = np.zeros((paths.shape[0], len(t_target)), dtype=np.float64)
    for i in range(paths.shape[0]):
        out[i] = np.interp(t_target, t_native, paths[i])
    return out


# ---------------------------------------------------------------------------
# Lyons–Victoir explicit multi-D formulae (Proc. R. Soc. A 2004)
#
# Not the tensor product of 1D path shapes. Degree m=5 realises the Lévy-area
# brackets via a Bernoulli sign on top of a degree-5 Gaussian cubature;
# degree m=3 is the Stroud spherical rule (straight lines).
# ---------------------------------------------------------------------------


def _gaussian_cubature_degree3(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Degree-3 cubature for ``N(0, I_d)``: Stroud points ``±√d e_i``."""
    d = int(d)
    nodes = np.zeros((2 * d, d), dtype=np.float64)
    for i in range(d):
        nodes[2 * i, i] = np.sqrt(d)
        nodes[2 * i + 1, i] = -np.sqrt(d)
    weights = np.full(2 * d, 1.0 / (2 * d), dtype=np.float64)
    return nodes, weights


def _gaussian_cubature_degree5(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Degree-5 cubature for ``N(0, I_d)``: product 3-point Gauss–Hermite."""
    d = int(d)
    gh_nodes = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)], dtype=np.float64)
    gh_w = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=np.float64)
    nodes = np.array(list(product(gh_nodes, repeat=d)), dtype=np.float64)
    idx = list(product(range(3), repeat=d))
    weights = np.array(
        [float(np.prod([gh_w[j] for j in js])) for js in idx],
        dtype=np.float64,
    )
    return nodes, weights


def _oned_m5_sqrt22(t: np.ndarray, terminal: float) -> np.ndarray:
    """Unit-time Lyons–Victoir 1D degree-5 driver ending at ``terminal``."""
    if abs(terminal) < 1e-15:
        return np.zeros_like(t)
    left = 0.5 * (4.0 - np.sqrt(22.0)) * t
    middle = (
        1.0 / 6.0 * (4.0 - np.sqrt(22.0)) + (-1.0 + np.sqrt(22.0)) * (t - 1.0 / 3.0)
    )
    right = (
        1.0 / 6.0 * (2.0 + np.sqrt(22.0))
        + 0.5 * (4.0 - np.sqrt(22.0)) * (t - 2.0 / 3.0)
    )
    base = np.where(t <= 1.0 / 3.0, left, np.where(t <= 2.0 / 3.0, middle, right))
    return terminal * base


def _levy_area_pl(omega: np.ndarray) -> np.ndarray:
    """Lévy area matrix of a piecewise-linear path (trapezoid / PL formula)."""
    n, d = omega.shape
    a = np.zeros((d, d), dtype=np.float64)
    for k in range(n - 1):
        w0, w1 = omega[k], omega[k + 1]
        dw = w1 - w0
        mid = 0.5 * (w0 + w1)
        a += np.outer(mid, dw) - np.outer(dw, mid)
    return 0.5 * a


def _perp_triangle_excursion(t: np.ndarray) -> np.ndarray:
    """Unit-height triangular bump on the three √22 thirds: rise, hold, fall.

    This is the planar d=2 Lévy-area excursion (Lyons–Victoir 2004,
    construction of the path for d=2): a linear detour orthogonal to the
    increment, supported on the same breakpoints as the 1D √22 path.
    """
    t = np.asarray(t, dtype=np.float64)
    g = np.empty_like(t)
    left = t <= 1.0 / 3.0
    mid = (t > 1.0 / 3.0) & (t <= 2.0 / 3.0)
    g[left] = 3.0 * t[left]
    g[mid] = 1.0
    g[~left & ~mid] = 3.0 * (1.0 - t[~left & ~mid])
    return g


def _lv_d2_unit_path(z: np.ndarray, eta: float, t: np.ndarray) -> np.ndarray:
    """Unit-time LV d=2 path for Gaussian node ``z`` and Bernoulli sign ``η``.

    Along the increment: the classical 1D √22 path (so the time–space
    brackets ``[[ε0, ε_u], ε_u]`` match the 1D formula). Orthogonal to it:
    a triangular excursion scaled so the Lévy area is ``(η/2) z1 z2``.
    The construction is equivariant under orthogonal transformations of
    the driving plane (in particular under swapping the two axes).
    """
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(2)
    rho = float(np.linalg.norm(z))
    if rho < 1e-15:
        return np.zeros((t.shape[0], 2), dtype=np.float64)
    u = z / rho
    n = np.array([-u[1], u[0]], dtype=np.float64)
    omega = _oned_m5_sqrt22(t, rho)[:, None] * u[None, :]
    a_tgt = 0.5 * float(eta) * float(z[0] * z[1])
    if abs(a_tgt) < 1e-15:
        return omega
    g = _perp_triangle_excursion(t)
    trial = omega + g[:, None] * n[None, :]
    sens = _levy_area_pl(trial)[0, 1] - _levy_area_pl(omega)[0, 1]
    if abs(sens) < 1e-15:
        return omega
    return omega + (a_tgt / sens) * g[:, None] * n[None, :]


def _add_area_bumps(
    omega: np.ndarray,
    area_targets: dict[tuple[int, int], float],
    t: np.ndarray,
    n_correct: int = 3,
) -> np.ndarray:
    """Pairwise planar bumps for ``d>2`` (d=2 uses :func:`_lv_d2_unit_path`)."""
    g = _perp_triangle_excursion(t)
    omega = omega.copy()
    eps = 1e-7
    for _ in range(n_correct):
        a_now = _levy_area_pl(omega)
        for (i, j), a_tgt in area_targets.items():
            da = a_tgt - a_now[i, j]
            if abs(da) < 1e-14:
                continue
            dz = omega[-1, j] - omega[0, j]
            di = omega[-1, i] - omega[0, i]
            # Perpendicular in the (i, j)-plane, matching the d=2 construction.
            rho_ij = float(np.hypot(di, dz))
            if rho_ij < 1e-15:
                continue
            n_i, n_j = -dz / rho_ij, di / rho_ij
            trial = omega.copy()
            trial[:, i] = trial[:, i] + eps * g * n_i
            trial[:, j] = trial[:, j] + eps * g * n_j
            sens = (_levy_area_pl(trial)[i, j] - a_now[i, j]) / eps
            if abs(sens) < 1e-12:
                continue
            omega[:, i] = omega[:, i] + (da / sens) * g * n_i
            omega[:, j] = omega[:, j] + (da / sens) * g * n_j
    return omega


def _unit_lv_paths(
    noise_dim: int, lyons_degree: int, t_unit: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-time LV multi-D paths on ``t_unit ⊂ [0, 1]``.

    Returns ``(paths, weights)`` with ``paths`` of shape ``(P, len(t_unit), d)``
    and ``E[ω(1)ω(1)^T] = I`` (no Brownian ``√T`` scaling).
    """
    d = int(noise_dim)
    t_unit = np.asarray(t_unit, dtype=np.float64)
    lyons_degree = int(lyons_degree)
    if lyons_degree == 3:
        nodes, gauss_w = _gaussian_cubature_degree3(d)
        paths = np.stack([t_unit[:, None] * z[None, :] for z in nodes], axis=0)
        weights = gauss_w.copy()
    elif lyons_degree == 5:
        nodes, gauss_w = _gaussian_cubature_degree5(d)
        path_list = []
        weight_list = []
        for z, lw in zip(nodes, gauss_w):
            for eta in (+1.0, -1.0):
                if d == 2:
                    omega = _lv_d2_unit_path(z, eta, t_unit)
                else:
                    primary = np.stack(
                        [_oned_m5_sqrt22(t_unit, float(z[i])) for i in range(d)],
                        axis=1,
                    )
                    areas = {
                        (i, j): 0.5 * eta * z[i] * z[j]
                        for i in range(d)
                        for j in range(i + 1, d)
                    }
                    omega = (
                        primary
                        if not areas
                        else _add_area_bumps(primary, areas, t_unit)
                    )
                path_list.append(omega)
                weight_list.append(0.5 * lw)
        paths = np.stack(path_list, axis=0)
        weights = np.asarray(weight_list, dtype=np.float64)
    else:
        raise ValueError("lyons_degree must be 3 or 5")
    weights = weights / weights.sum()
    return paths, weights


def lv_path_count(noise_dim: int, k: int = 1, lyons_degree: int = 5) -> int:
    """Number of paths in the explicit multi-D Lyons–Victoir construction.

    ``m=5``: ``(2 · 3^d)^k`` (Gauss–Hermite × Bernoulli, composed ``k`` times).
    ``m=3``: ``(2d)^k`` (Stroud spherical rule).
    """
    d = max(1, int(noise_dim))
    k = max(1, int(k))
    lyons_degree = int(lyons_degree)
    if lyons_degree == 5:
        base = 2 * (3**d)
    elif lyons_degree == 3:
        base = 2 * d
    else:
        raise ValueError("lyons_degree must be 3 or 5")
    return int(base**k)


def build_lyons_victoir_multid_paths(
    ts: np.ndarray,
    noise_dim: int,
    lyons_degree: int = 5,
    k: int = 1,
    gamma: float = 0.6,
    steps_per_unit: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit multi-D Lyons–Victoir cubature, sampled on ``ts``.

    Returns *unit-time* path values (``E[ω(1)ω(1)^T] = I``). Callers that
    pass a horizon-``T`` grid must scale by ``√T`` (as
    :func:`precompute_cubature_paths` does).

    Args:
        ts: Time grid, typically ``[0, T]``. Values are interpolated in
            normalised time ``t/T``.
        noise_dim: Brownian dimension ``d > 1`` (1D uses the classical
            1D formulae instead).
        lyons_degree: ``3`` or ``5``.
        k: Number of macro-segments; the one-step formula is composed
            ``k`` times on a ``gamma``-concentrated partition of ``[0, 1]``.
        gamma: Macro-segment concentration exponent.
        steps_per_unit: Native resolution when ``k > 1``.
    """
    ts = np.asarray(ts, dtype=np.float64)
    d = int(noise_dim)
    k = max(1, int(k))
    lyons_degree = int(lyons_degree)
    t_span = float(ts[-1] - ts[0])
    t_target = (ts - ts[0]) / max(t_span, 1e-15)

    if k == 1:
        return _unit_lv_paths(d, lyons_degree, t_target)

    t_d = np.array(
        [1.0 - (1.0 - i / k) ** gamma for i in range(k + 1)], dtype=np.float64
    )
    t_ref = np.linspace(0.0, 1.0, max(32, int(steps_per_unit)))
    unit_paths, unit_w = _unit_lv_paths(d, lyons_degree, t_ref)
    n_unit = unit_paths.shape[0]

    cur_paths: Optional[np.ndarray] = None
    cur_w: Optional[np.ndarray] = None
    t_native = np.zeros(0, dtype=np.float64)

    for segment in range(k):
        dt = t_d[segment + 1] - t_d[segment]
        dtn = max(2, int(np.ceil(dt * steps_per_unit)))
        if segment == 0:
            x_loc = np.linspace(0.0, dt, dtn)
            t_native = np.linspace(0.0, dt, dtn)
        else:
            x_loc = np.linspace(dt / dtn, dt, dtn)
            t_native = np.concatenate([t_native, t_native[-1] + x_loc])
        s_unit = x_loc / max(dt, 1e-15)
        child = np.empty((n_unit, dtn, d), dtype=np.float64)
        for p in range(n_unit):
            for j in range(d):
                child[p, :, j] = np.interp(s_unit, t_ref, unit_paths[p, :, j])
        child *= np.sqrt(dt)

        if cur_paths is None:
            cur_paths = child
            cur_w = unit_w.copy()
            continue

        m_par, t_prev, _ = cur_paths.shape
        x0 = cur_paths[:, -1, :]
        tail = x0[:, None, None, :] + child[None, :, :, :]
        head = np.broadcast_to(
            cur_paths[:, None, :, :], (m_par, n_unit, t_prev, d)
        ).copy()
        cur_paths = np.concatenate([head, tail], axis=2).reshape(
            m_par * n_unit, t_prev + dtn, d
        )
        cur_w = (cur_w[:, None] * unit_w[None, :]).reshape(-1)

    assert cur_paths is not None and cur_w is not None
    cur_w = cur_w / cur_w.sum()
    out = np.zeros((cur_paths.shape[0], len(t_target), d), dtype=np.float64)
    for i in range(cur_paths.shape[0]):
        for j in range(d):
            out[i, :, j] = np.interp(t_target, t_native, cur_paths[i, :, j])
    return out, cur_w


def build_product_lyons_paths(
    ts: np.ndarray,
    noise_dim: int,
    k: int = 3,
    gamma: float = 0.6,
    lyons_degree: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tensor product of 1D Lyons–Victoir paths.

    ``lyons_degree=5`` (default): ``P = (3^k)^d`` from the explicit m=5 formula.
    ``lyons_degree=3``: ``P = (2^k)^d`` from the explicit m=3 straight-line formula.
    """
    lyons_degree = int(lyons_degree)
    if lyons_degree == 5:
        paths_1d, weights_1d, t_native = build_cubature_paths_1d(k=k, gamma=gamma)
    elif lyons_degree == 3:
        paths_1d, weights_1d, t_native = build_cubature_paths_1d_m3(k=k, gamma=gamma)
    else:
        raise ValueError("lyons_degree must be 3 or 5 (Lyons–Victoir explicit formulae)")
    ts = np.asarray(ts, dtype=np.float64)
    t_target = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-15)
    paths_1d = _resample_paths_1d(paths_1d, t_native, t_target)
    p1 = paths_1d.shape[0]
    t_len = len(ts)

    num_paths = p1**noise_dim
    path_values = np.zeros((num_paths, t_len, noise_dim), dtype=np.float64)
    weights = np.zeros(num_paths, dtype=np.float64)

    for flat in range(num_paths):
        w = 1.0
        rem = flat
        for d in range(noise_dim):
            j = rem % p1
            rem //= p1
            path_values[flat, :, d] = paths_1d[j]
            w *= weights_1d[j]
        weights[flat] = w

    return path_values, weights


# ---------------------------------------------------------------------------
# Path-budget heuristics
# ---------------------------------------------------------------------------


def product_path_count(noise_dim: int, k: int, lyons_degree: int = 5) -> int:
    """Number of paths in the tensor-product Lyons construction."""
    d = max(1, int(noise_dim))
    k = max(1, int(k))
    base = 3 if int(lyons_degree) == 5 else 2
    return base**k if d == 1 else base ** (k * d)


def select_multidim_paths(
    noise_dim: int,
    k: int = 2,
    *,
    product_path_threshold: int = _AUTO_PRODUCT_PATH_THRESHOLD,
    lyons_degree: int = 5,
) -> MultidimMethod:
    """Choose ``lv`` / ``product`` vs ``recombined`` from dimension and order.

    For ``lyons_degree=5`` and ``d>1``, prefers explicit multi-D LV when
    ``lv_path_count`` is below ``product_path_threshold``. Degree ``m=3``
    still uses the 1D-product (hypercube) formula.
    """
    d = max(1, int(noise_dim))
    k = max(1, int(k))
    lyons_degree = int(lyons_degree)
    if d > 1 and lyons_degree == 5:
        if lv_path_count(d, k, lyons_degree=5) <= product_path_threshold:
            return "lv"
        return "recombined"
    if product_path_count(d, k, lyons_degree=lyons_degree) <= product_path_threshold:
        return "product"
    return "recombined"


def resolve_multidim_method(
    multidim_method: MultidimMethodInput,
    noise_dim: int,
    k: int,
    *,
    product_path_threshold: int = _AUTO_PRODUCT_PATH_THRESHOLD,
    lyons_degree: int = 5,
) -> MultidimMethod:
    """Resolve ``"auto"`` to ``"lv"``, ``"product"``, or ``"recombined"``."""
    if multidim_method == "auto":
        return select_multidim_paths(
            noise_dim,
            k,
            product_path_threshold=product_path_threshold,
            lyons_degree=lyons_degree,
        )
    if multidim_method not in ("lv", "product", "recombined"):
        raise ValueError(
            'multidim_method must be "lv", "product", "recombined", or "auto".'
        )
    return multidim_method


def default_recombined_num_paths(
    noise_dim: int, k: int = 2, *, cap: int = 256, floor: int = 16
) -> int:
    """Dimension-aware default for the post-recombination Wiener path budget.

    Sized to fit the degree-3 monomial system on the ``k * noise_dim``
    terminal-increment features used by recombination. A smaller budget forces
    the matcher to drop moment degree and can collapse to a handful of paths
    with grossly wrong Brownian second moments (e.g. ``k=3, d=2, n=27`` → 2
    perfectly correlated paths).
    """
    d = max(1, int(noise_dim))
    k = max(1, int(k))
    by_order = 3**k
    by_dim = 4 * d * k
    hypercube = 2 ** min(d, 8)
    # combinations_with_replacement count for monomials up to degree 3 in k*d.
    feat_dim = k * d
    by_moments = sum(
        math.comb(feat_dim + total - 1, total) if feat_dim > 0 else 1
        for total in range(4)
    )
    return int(min(cap, max(floor, by_order, by_dim, by_moments, min(hypercube, cap))))


def default_num_paths(
    noise_dim: int,
    k: int,
    multidim_method: MultidimMethodInput = "auto",
    num_paths: Optional[int] = None,
    lyons_degree: int = 5,
) -> int:
    """Default number of Wiener paths for a given method."""
    if multidim_method == "recombined":
        if num_paths is not None:
            return max(1, int(num_paths))
        return default_recombined_num_paths(noise_dim, k)
    if noise_dim == 1:
        return (3 if int(lyons_degree) == 5 else 2) ** max(1, int(k))
    method = resolve_multidim_method(
        multidim_method, noise_dim, k, lyons_degree=lyons_degree
    )
    if method == "lv":
        return lv_path_count(noise_dim, k, lyons_degree=lyons_degree)
    if method == "product":
        return product_path_count(noise_dim, k, lyons_degree=lyons_degree)
    return default_recombined_num_paths(noise_dim, k)



# ---------------------------------------------------------------------------
# High-order recombination (RMP / moment matching)
# ---------------------------------------------------------------------------


def _segment_branch_shapes(
    k: int, gamma: float, steps_per_unit: int = 200
) -> Tuple[np.ndarray, List[int], List[np.ndarray], np.ndarray]:
    """Per-macro-segment 1D degree-3 Lyons cubature branch shapes on ``[0, 1]``.

    Returns ``(t_native, seg_lengths, branch_shapes, branch_w)`` where
    ``branch_shapes[s]`` has shape ``(3, seg_lengths[s])`` and holds the
    within-segment path increment (to be added to that segment's start value)
    for each of the 3 cubature branches. ``branch_w`` are the branch weights
    ``[1/6, 2/3, 1/6]``. The native grid mirrors :func:`build_cubature_paths_1d`
    so the resulting paths share its time discretisation.
    """
    t_d = np.array(
        [1.0 - (1.0 - i / k) ** gamma for i in range(k + 1)], dtype=np.float64
    )
    t_native = np.zeros(0, dtype=np.float64)
    seg_lengths: List[int] = []
    branch_shapes: List[np.ndarray] = []
    for segment in range(k):
        dt = t_d[segment + 1] - t_d[segment]
        dtn = max(2, int(np.ceil(dt * steps_per_unit)))
        if segment == 0:
            x_loc = np.linspace(0.0, dt, dtn)
            t_native = np.linspace(0.0, dt, dtn)
        else:
            x_loc = np.linspace(dt / dtn, dt, dtn)
            t_native = np.concatenate(
                [t_native, t_native[-1] + np.linspace(dt / dtn, dt, dtn)]
            )
        shapes = np.stack(
            [
                np.sqrt(dt) * _cubature_path_1(x_loc / dt),
                _cubature_path_3(x_loc),
                np.sqrt(dt) * _cubature_path_2(x_loc / dt),
            ],
            axis=0,
        )
        seg_lengths.append(dtn)
        branch_shapes.append(shapes)

    branch_w = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=np.float64)
    return t_native, seg_lengths, branch_shapes, branch_w


def _monomial_exponents(dim: int, degree: int) -> List[Tuple[int, ...]]:
    exponents: List[Tuple[int, ...]] = []
    for total in range(degree + 1):
        for combo in combinations_with_replacement(range(dim), total):
            exp = [0] * dim
            for i in combo:
                exp[i] += 1
            exponents.append(tuple(exp))
    return exponents


def _eval_monomials(x: np.ndarray, exponents: List[Tuple[int, ...]]) -> np.ndarray:
    out = np.ones((x.shape[0], len(exponents)), dtype=np.float64)
    for j, exp in enumerate(exponents):
        for i, p in enumerate(exp):
            if p:
                out[:, j] *= x[:, i] ** p
    return out


def _rmp_moment_match(
    features: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    max_support: int,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import linprog, nnls

    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = weights / np.sum(weights)

    P = features.shape[0]
    max_support = int(min(max_support, P))

    lam_dense, _ = nnls(features.T, targets)
    lam = lam_dense.copy()
    if np.sum(lam) > tol:
        lam = lam / np.sum(lam)
    else:
        lam = weights.copy()

    for _ in range(5):
        idx = np.where(lam > tol)[0]
        if idx.size <= max_support:
            break
        idx = idx[np.argsort(lam[idx])[::-1][:max_support]]
        sub = features[idx]
        A_eq = np.vstack([sub.T, np.ones((1, idx.size))])
        b_eq = np.append(targets, 1.0)
        c = np.zeros(idx.size)
        bounds = [(0.0, None)] * idx.size
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success and res.x is not None:
            lam_new = np.zeros(P)
            lam_new[idx] = res.x
            lam = lam_new
        else:
            lam = np.zeros(P)
            lam[idx] = weights[idx]
            if lam.sum() > tol:
                lam /= lam.sum()
            break

    active = np.where(lam > tol)[0]
    if active.size == 0:
        active = np.array([int(np.argmax(weights))])
        lam = np.zeros(P)
        lam[active[0]] = 1.0

    return active, lam[active] / np.sum(lam[active])


def build_recombined_cubature_paths(
    ts: np.ndarray,
    noise_dim: int,
    k: int = 2,
    gamma: float = 0.6,
    recombined_num_paths: Optional[int] = None,
    candidate_paths: Optional[int] = None,
    candidate_factor: float = 8.0,
    degree: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recombined Wiener cubature paths via recursive KLV expansion + RMP.

    Implements Algorithm 1 of Snow & Krishnamurthy (2025). Starting from the
    trivial (single-point) measure, then for each macro-segment:

    1. **Expand** -- push every current support path through one step of the
       exact, degree-3 tensor-product Kusuoka-Lyons-Victoir cubature, branching
       each path into ``3**noise_dim`` children with the product cubature
       weights.
    2. **Recombine** -- collapse the expanded measure with the RMP moment-match
       back to cardinality ``<= recombined_num_paths``, preserving the measure's
       own monomial moments up to ``degree``.

    Because each expansion is an *exact* cubature step and the recombination
    target is the expanded measure's own moments (so a feasible solution always
    exists), the surviving sparse measure reproduces the joint Wiener increment
    moments up to ``degree`` exactly. The working support stays bounded by
    ``recombined_num_paths * 3**noise_dim`` instead of the ``(3**k)**noise_dim``
    of the full product construction.

    Args:
        ts: Integration time grid (1D). Paths are returned sampled on this grid.
        noise_dim: Brownian dimension ``d``.
        k: Cubature order (number of macro segments).
        gamma: Macro-segment time-grid concentration exponent.
        recombined_num_paths: Target post-recombination support size. When
            ``None`` it is sized to comfortably fit the degree-``degree``
            monomial system (capped at :data:`_RECOMBINE_BUDGET_CAP`).
        candidate_paths: Unused; retained for API compatibility.
        candidate_factor: Unused; retained for API compatibility.
        degree: Target moment degree preserved by recombination.
        seed: Unused; the construction is fully deterministic.

    Returns:
        Tuple ``(path_values, weights)`` with ``path_values`` of shape
        ``(P, len(ts), noise_dim)`` and ``weights`` of shape ``(P,)`` where
        ``P <= recombined_num_paths``.

    Note:
        Each macro-segment expansion costs a factor ``3**noise_dim``, so this
        construction targets low-to-moderate ``noise_dim``. 
    """
    del candidate_paths, candidate_factor, seed  # deterministic; no candidate pool

    ts = np.asarray(ts, dtype=np.float64)
    d = max(1, int(noise_dim))
    k = max(1, int(k))

    if recombined_num_paths is not None:
        n_out = max(1, int(recombined_num_paths))
    else:
        n_out = default_recombined_num_paths(d, k)
        n_mom_full = len(_monomial_exponents(k * d, max(1, int(degree))))
        n_out = max(n_out, min(n_mom_full, _RECOMBINE_BUDGET_CAP))

    t_native, seg_lengths, branch_shapes, branch_w = _segment_branch_shapes(k, gamma)

    # Largest moment degree whose full monomial system still fits the budget.
    degree_eff = max(1, int(degree))
    while degree_eff > 1 and len(_monomial_exponents(k * d, degree_eff)) > n_out:
        degree_eff -= 1

    branch_combos = np.array(list(product(range(3), repeat=d)), dtype=np.int64)
    combo_w = np.prod(branch_w[branch_combos], axis=1)
    n_combo = branch_combos.shape[0]

    cur_paths = np.zeros((1, 0, d), dtype=np.float64)
    cur_incs = np.zeros((1, 0), dtype=np.float64)
    cur_w = np.ones(1, dtype=np.float64)

    for s in range(k):
        seg_len = seg_lengths[s]
        shapes_s = branch_shapes[s]  # (3, seg_len)
        m = cur_paths.shape[0]
        prev_len = cur_paths.shape[1]
        x0 = cur_paths[:, -1, :] if prev_len > 0 else np.zeros((m, d), dtype=np.float64)

        # Expand: each path branches into 3**d KLV one-step children.
        seg_block = np.transpose(shapes_s[branch_combos], (0, 2, 1))  # (C, seg_len, d)
        new_paths = np.zeros((m * n_combo, prev_len + seg_len, d), dtype=np.float64)
        new_paths[:, :prev_len, :] = np.repeat(cur_paths, n_combo, axis=0)
        x0_rep = np.repeat(x0, n_combo, axis=0)  # (m*C, d)
        new_paths[:, prev_len:, :] = np.tile(seg_block, (m, 1, 1)) + x0_rep[:, None, :]

        combo_inc = shapes_s[:, -1][branch_combos]  # (C, d) terminal increments
        new_incs = np.concatenate(
            [np.repeat(cur_incs, n_combo, axis=0), np.tile(combo_inc, (m, 1))], axis=1
        )

        new_w = np.repeat(cur_w, n_combo) * np.tile(combo_w, m)
        new_w = new_w / new_w.sum()

        # Recombine: collapse to <= n_out points preserving the expanded
        # measure's own monomial moments (exact, hence always feasible).
        exps = _monomial_exponents(new_incs.shape[1], degree_eff)
        feats = _eval_monomials(new_incs, exps)
        targets = feats.T @ new_w
        idx, lam = _rmp_moment_match(feats, new_w, targets, max_support=n_out)

        cur_paths = new_paths[idx]
        cur_incs = new_incs[idx]
        cur_w = lam

    # Resample from the native [0, 1] grid onto the requested integration grid.
    t_target = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-15) * t_native[-1]
    out = np.zeros((cur_paths.shape[0], len(ts), d), dtype=np.float64)
    for i in range(cur_paths.shape[0]):
        for j in range(d):
            out[i, :, j] = np.interp(t_target, t_native, cur_paths[i, :, j])
    return out, cur_w


    # ---------------------------------------------------------------------------
# Driver and precompute entry point
# ---------------------------------------------------------------------------


def build_cubature_driver(
    ts: np.ndarray,
    noise_dim: int,
    num_paths: Optional[int] = None,
    k: int = 3,
    gamma: float = 0.6,
    seed: int = 0,
    multidim_method: MultidimMethodInput = "auto",
    candidate_factor: float = 8.0,
    recombination_degree: int = 3,
    lyons_degree: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build Wiener cubature driver paths ``omega(t)`` and weights on grid ``ts``.

    Args:
        ts: Integration time grid (1D, length ``num_steps + 1``).
        noise_dim: Brownian dimension.
        num_paths: Number of paths. Meaning depends on ``multidim_method``.
        k: Cubature order (number of macro segments).
        gamma: Macro-segment time-grid concentration exponent.
        multidim_method: ``"lv"`` | ``"product"`` | ``"recombined"`` | ``"auto"``.
        candidate_factor: Candidate-pool scaling for recombination.
        recombination_degree: Target moment degree for recombination (only for
            ``"recombined"``; not the Lyons–Victoir degree ``m``).
        lyons_degree: Explicit Lyons–Victoir degree ``m ∈ {3, 5}``. Default
            ``5`` is the three-path 1D construction, or the multi-D LV lift
            (Gaussian cubature × Bernoulli Lévy area) when ``d>1``.

    Returns:
        Tuple ``(path_values, weights)`` with ``path_values`` of shape
        ``(P, len(ts), noise_dim)`` and ``weights`` of shape ``(P,)``.
        Path values are unit-time (``E[ω(1)ω(1)^T] = I``).
    """
    ts = np.asarray(ts, dtype=np.float64)
    lyons_degree = int(lyons_degree)
    if lyons_degree not in (3, 5):
        raise ValueError("lyons_degree must be 3 or 5")

    if noise_dim == 1:
        if lyons_degree == 5:
            paths_native, weights, t_native = build_cubature_paths_1d(k=k, gamma=gamma)
        else:
            paths_native, weights, t_native = build_cubature_paths_1d_m3(k=k, gamma=gamma)
        t_target = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-15)
        paths = _resample_paths_1d(paths_native, t_native, t_target)
        return paths[:, :, np.newaxis], weights

    # Explicit m=3 product is the 1D-tensor hypercube rule unless the caller
    # asked for LV (Stroud) or recombination.
    if lyons_degree == 3 and multidim_method == "auto":
        multidim_method = "product"
    else:
        multidim_method = resolve_multidim_method(
            multidim_method, noise_dim, k, lyons_degree=lyons_degree
        )

    if multidim_method == "lv":
        return build_lyons_victoir_multid_paths(
            ts - ts[0],
            noise_dim,
            lyons_degree=lyons_degree,
            k=k,
            gamma=gamma,
        )

    if multidim_method == "product":
        return build_product_lyons_paths(
            ts - ts[0], noise_dim, k=k, gamma=gamma, lyons_degree=lyons_degree
        )

    if multidim_method == "recombined":
        return build_recombined_cubature_paths(
            ts,
            noise_dim,
            k=k,
            gamma=gamma,
            recombined_num_paths=num_paths,
            candidate_factor=candidate_factor,
            degree=recombination_degree,
            seed=seed,
        )

    raise ValueError(f"Unknown multidim_method {multidim_method!r}")



def precompute_cubature_paths(
    horizon: float,
    noise_dim: int,
    integration_dt: float = 0.01,
    cubature_order: int = 2,
    cubature_gamma: float = 0.6,
    num_paths: Optional[int] = None,
    multidim_method: MultidimMethodInput = "auto",
    seed: int = 0,
    candidate_factor: float = 8.0,
    recombination_degree: int = 3,
    lyons_degree: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Precompute the integration grid, cubature paths, and weights as JAX arrays.

        Computed once at filter-build time (host-side NumPy) and closed over by the
        JAX kernels, so they enter the traced computation as static constants.

        Args:
            horizon: Length of one assimilation interval (relative time ``[0, horizon]``).
            noise_dim: Brownian dimension.
            integration_dt: Inner integration step size.
            cubature_order: Cubature order ``k``.
            cubature_gamma: Macro-segment time-grid concentration exponent.
            num_paths: Number of paths (method-dependent meaning).
            multidim_method: ``"lv"`` | ``"product"`` | ``"recombined"`` | ``"auto"``.
            candidate_factor: Candidate-pool scaling for recombination.
            recombination_degree: Target moment degree for recombination.
            lyons_degree: Explicit Lyons–Victoir degree ``m ∈ {3, 5}`` (default 5).

        Returns:
            Tuple ``(ts, path_values, path_weights)`` as ``float64`` JAX arrays, with
            ``ts`` of shape ``(num_steps + 1,)`` (relative times starting at 0).
        """
        horizon = max(float(horizon), 1e-15)
        num_steps = max(2, int(np.ceil(horizon / integration_dt)))
        ts = np.linspace(0.0, horizon, num_steps + 1)
        path_values, path_weights = build_cubature_driver(
            ts,
            noise_dim,
            num_paths=num_paths,
            k=cubature_order,
            gamma=cubature_gamma,
            seed=seed,
            multidim_method=multidim_method,
            candidate_factor=candidate_factor,
            recombination_degree=recombination_degree,
            lyons_degree=lyons_degree,
        )
        # Paths are constructed as unit-time Wiener cubature (E[ω(1)ω(1)^T] = I)
        # and resampled onto [0, horizon] in normalized time. Brownian scaling
        # ω_horizon(t) = √horizon · ω_1(t/horizon) restores E[ω(T)ω(T)^T] = T I.
        path_values = path_values * np.sqrt(horizon)
        return (
            jnp.asarray(ts, dtype=jnp.float64),
            jnp.asarray(path_values, dtype=jnp.float64),
            jnp.asarray(path_weights, dtype=jnp.float64),
        )
