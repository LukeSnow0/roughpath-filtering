#!/usr/bin/env python3
"""
Wiener-space cubature and JAX Cubature-Recursive Filter (CRF).

Self-contained module for running the differentiable CRF:
  - Degree-3 Lyons / KLV 1D cubature and tensor-product multidim paths
  - Rademacher simultaneous-noise paths (Snow–Krishnamurthy style)
  - RMP moment-matching recombination (Litterer–Lyons / Snow–Krishnamurthy)
  - Stratonovich SDE integration (RK4 + lax.scan) and JaxCubatureRecursiveFilter

Run: ``python wiener_cubature.py``  (requires jax, numpy, scipy)

System (Stratonovich):
    dx = f(t, x) dt + g(t, x) ∘ dW
    y  = h(x) + v,  v ~ N(0, R)
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np

__all__ = [
    "build_cubature_paths_1d",
    "resample_cubature_paths_1d",
    "build_multidim_cubature_paths",
    "build_product_lyons_paths",
    "default_num_paths",
    "default_recombined_num_paths",
    "resolve_recombined_num_paths",
    "build_cubature_driver",
    "macro_segment_times",
    "path_segment_increments",
    "recombine_path_measure",
    "build_recombined_cubature_paths",
    "default_recombination_candidates",
    "rmp_moment_match",
    # JAX CRF
    "PathSampling",
    "spatial_cubature_nodes",
    "weighted_mean_cov",
    "rk4_stratonovich_step",
    "integrate_stratonovich_cde",
    "JaxCubatureRecursiveFilter",
    "make_benchmark_dynamics",
    "make_highdim_dynamics",
    "make_l63_dynamics",
    "demo_autodiff_sigma",
]


# ---------------------------------------------------------------------------
# Degree-3 Lyons 1D cubature
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
    """
    Build degree-3 Wiener-space cubature paths on [0, 1] for a scalar Brownian driver.

    Returns paths (3^k, T), weights (3^k,), native time grid t_native.
    """
    t_d = np.array([1.0 - (1.0 - i / k) ** gamma for i in range(k + 1)], dtype=np.float64)
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


def resample_cubature_paths_1d(
    paths: np.ndarray,
    t_native: np.ndarray,
    t_target: np.ndarray,
) -> np.ndarray:
    """Linearly resample 1D cubature paths onto a uniform integration grid."""
    t_target = np.asarray(t_target, dtype=np.float64)
    out = np.zeros((paths.shape[0], len(t_target)), dtype=np.float64)
    for i in range(paths.shape[0]):
        out[i] = np.interp(t_target, t_native, paths[i])
    return out


def _build_rademacher_signs(num_paths: int, noise_dim: int, seed: int = 0) -> np.ndarray:
    if num_paths <= (1 << noise_dim):
        signs = np.stack(
            [((np.arange(num_paths) >> d) & 1) * 2 - 1 for d in range(noise_dim)],
            axis=1,
        )
    else:
        rng = np.random.default_rng(seed)
        signs = rng.integers(0, 2, size=(num_paths, noise_dim), endpoint=False) * 2 - 1
    return signs.astype(np.float64)


def build_multidim_cubature_paths(
    ts: np.ndarray,
    num_paths: int,
    noise_dim: int,
    k: int = 3,
    gamma: float = 0.6,
    seed: int = 0,
) -> np.ndarray:
    """Rademacher macro-segment cubature paths; shape (P, len(ts), noise_dim)."""
    ts = np.asarray(ts, dtype=np.float64)
    ts_rel = ts - ts[0]
    T = ts_rel[-1]
    t_macro = np.array([1.0 - (1.0 - i / k) ** gamma for i in range(k + 1)], dtype=np.float64) * T
    paths = np.zeros((num_paths, len(ts), noise_dim), dtype=np.float64)
    signs = _build_rademacher_signs(num_paths, noise_dim, seed)
    cum = np.zeros((num_paths, noise_dim), dtype=np.float64)

    for s in range(k):
        dt_macro = t_macro[s + 1] - t_macro[s]
        if dt_macro <= 0.0:
            continue
        inc = signs * np.sqrt(dt_macro)
        segment_indices = np.where((ts_rel[:-1] >= t_macro[s]) & (ts_rel[:-1] < t_macro[s + 1]))[0]
        for idx in segment_indices:
            alpha = (ts_rel[idx] - t_macro[s]) / dt_macro
            paths[:, idx, :] = cum + inc * alpha
        cum = cum + inc

    paths[:, -1, :] = cum
    return paths


def build_product_lyons_paths(
    ts: np.ndarray,
    noise_dim: int,
    k: int = 3,
    gamma: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tensor product of 1D Lyons paths; P = (3^k)^d paths."""
    paths_1d, weights_1d, t_native = build_cubature_paths_1d(k=k, gamma=gamma)
    ts = np.asarray(ts, dtype=np.float64)
    t_target = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-15)
    paths_1d = resample_cubature_paths_1d(paths_1d, t_native, t_target)
    p1 = paths_1d.shape[0]
    t_len = len(ts)

    num_paths = p1 ** noise_dim
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


def default_recombined_num_paths(
    noise_dim: int,
    k: int = 2,
    *,
    cap: int = 256,
    floor: int = 16,
) -> int:
    """
    Dimension-aware default for the post-recombination Wiener path budget.

    Scales with Brownian dimension ``d`` and macro-segment order ``k`` so that
    low-dimensional problems stay cheap and high-dimensional problems get more
    paths (up to ``cap``).

    Heuristic: ``max(3^k, 4·d·k, min(2^min(d,8), cap))``, clamped to ``[floor, cap]``.
    """
    d = max(1, int(noise_dim))
    k = max(1, int(k))
    by_order = 3 ** k
    by_dim = 4 * d * k
    hypercube = 2 ** min(d, 8)
    return int(min(cap, max(floor, by_order, by_dim, min(hypercube, cap))))


def resolve_recombined_num_paths(
    noise_dim: int,
    k: int,
    recombined_num_paths: Optional[int] = None,
    num_paths: Optional[int] = None,
) -> int:
    """
    Resolve the active-path budget for ``multidim_method='recombined'``.

    Priority: explicit ``recombined_num_paths`` → explicit ``num_paths`` →
    ``default_recombined_num_paths(noise_dim, k)``.
    """
    if recombined_num_paths is not None:
        return max(1, int(recombined_num_paths))
    if num_paths is not None:
        return max(1, int(num_paths))
    return default_recombined_num_paths(noise_dim, k)


def default_num_paths(
    noise_dim: int,
    k: int,
    multidim_method: Literal["rademacher", "product", "recombined"] = "product",
    num_paths: Optional[int] = None,
) -> int:
    if multidim_method == "recombined":
        return resolve_recombined_num_paths(noise_dim, k, num_paths=num_paths)
    if noise_dim == 1:
        return 3 ** k
    if multidim_method == "product":
        return 3 ** (k * noise_dim)
    return max(3 ** k, 2 ** noise_dim)


def build_cubature_driver(
    ts: np.ndarray,
    noise_dim: int,
    num_paths: Optional[int] = None,
    k: int = 3,
    gamma: float = 0.6,
    seed: int = 0,
    multidim_method: Literal["rademacher", "product", "recombined"] = "product",
    candidate_paths: Optional[int] = None,
    recombination_degree: int = 3,
    recombined_num_paths: Optional[int] = None,
    recombination_candidate_factor: float = 8.0,
    recombination_fill_to_support: bool = False,
    return_info: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build Wiener cubature driver paths ω(t) and weights on the grid ``ts``.

    ``multidim_method``: ``product`` | ``rademacher`` | ``recombined``.

    For ``recombined``, set ``recombined_num_paths`` (or ``num_paths``) to trade
    accuracy vs cost; ``recombination_candidate_factor`` scales the candidate pool.
    """
    ts = np.asarray(ts, dtype=np.float64)
    if noise_dim == 1:
        paths_native, weights, t_native = build_cubature_paths_1d(k=k, gamma=gamma)
        if num_paths is not None and num_paths != paths_native.shape[0]:
            raise ValueError(
                f"For noise_dim=1, num_paths must be 3^k={paths_native.shape[0]} (k={k})."
            )
        t_target = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-15)
        paths_resampled = resample_cubature_paths_1d(paths_native, t_native, t_target)
        out = paths_resampled[:, :, np.newaxis], weights
        return (*out, {}) if return_info else out

    if multidim_method == "product":
        out = build_product_lyons_paths(ts - ts[0], noise_dim, k=k, gamma=gamma)
        return (*out, {}) if return_info else out

    if multidim_method == "recombined":
        n_out = resolve_recombined_num_paths(
            noise_dim, k, recombined_num_paths=recombined_num_paths, num_paths=num_paths
        )
        paths, weights, info = build_recombined_cubature_paths(
            ts,
            noise_dim,
            k=k,
            gamma=gamma,
            num_paths=n_out,
            recombined_num_paths=n_out,
            candidate_paths=candidate_paths,
            candidate_factor=recombination_candidate_factor,
            degree=recombination_degree,
            seed=seed,
            candidate_method="rademacher",
            fill_to_support=recombination_fill_to_support,
        )
        return (paths, weights, info) if return_info else (paths, weights)

    if num_paths is None:
        num_paths = default_num_paths(noise_dim, k, multidim_method="rademacher")
    path_values = build_multidim_cubature_paths(
        ts - ts[0], num_paths, noise_dim, k=k, gamma=gamma, seed=seed
    )
    out = path_values, np.full(num_paths, 1.0 / num_paths)
    return (*out, {}) if return_info else out


# ---------------------------------------------------------------------------
# High-order recombination (RMP)
# ---------------------------------------------------------------------------


def macro_segment_times(T: float, k: int, gamma: float) -> np.ndarray:
    return np.array([T * (1.0 - (1.0 - i / k) ** gamma) for i in range(k + 1)], dtype=np.float64)


def path_segment_increments(
    path_values: np.ndarray,
    ts: np.ndarray,
    k: int,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    path_values = np.asarray(path_values, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    T = ts[-1] - ts[0]
    t_macro = macro_segment_times(T, k, gamma)
    t_abs = ts - ts[0]

    P, _, d = path_values.shape
    increments = np.zeros((P, k, d), dtype=np.float64)
    segment_dts = np.diff(t_macro)

    for s in range(k):
        t_end = t_macro[s + 1]
        idx_end = int(np.clip(np.searchsorted(t_abs, t_end, side="right") - 1, 0, len(ts) - 1))
        if s == 0:
            increments[:, s, :] = path_values[:, idx_end, :]
        else:
            t_prev = t_macro[s]
            idx_prev = int(np.clip(np.searchsorted(t_abs, t_prev, side="right") - 1, 0, len(ts) - 1))
            increments[:, s, :] = path_values[:, idx_end, :] - path_values[:, idx_prev, :]

    return increments, segment_dts


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


def _gaussian_moment(exponent: Tuple[int, ...], dt: float) -> float:
    if any(e % 2 == 1 for e in exponent):
        return 0.0
    result = 1.0
    for e in exponent:
        if e == 0:
            continue
        double_fact = 1.0
        for m in range(1, e, 2):
            double_fact *= m
        result *= double_fact * (dt ** (e / 2))
    return result


def path_segment_moment_features(
    increments: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    P, k, d = increments.shape
    exponents = _monomial_exponents(d, degree)
    n_mom = len(exponents)
    feats = np.zeros((P, k * n_mom), dtype=np.float64)
    for s in range(k):
        seg_feats = _eval_monomials(increments[:, s, :], exponents)
        feats[:, s * n_mom : (s + 1) * n_mom] = seg_feats
    return feats, exponents


def wiener_path_moment_targets(
    k: int,
    d: int,
    segment_dts: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    exponents = _monomial_exponents(d, degree)
    n_mom = len(exponents)
    targets = np.zeros(k * n_mom, dtype=np.float64)
    for s in range(k):
        for j, exp in enumerate(exponents):
            targets[s * n_mom + j] = _gaussian_moment(exp, float(segment_dts[s]))
    return targets, exponents


def rmp_moment_match(
    features: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    max_support: int,
    tol: float = 1e-10,
    fill_to_support: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse non-negative weights matching target moments (RMP / recombination).

    If ``fill_to_support`` is True and the LP solution uses fewer than ``max_support``
    paths, pad to uniform weights on top NNLS candidates (approximate moments).
    Default False: sparse exact/approx LP support usually beats forced full budget.
    """
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

    idx = np.where(lam > tol)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(weights))])
        lam = np.zeros(P)
        lam[idx[0]] = 1.0

    if fill_to_support and idx.size < max_support:
        ranked = np.argsort(lam_dense)[::-1][:max_support]
        lam = np.zeros(P, dtype=np.float64)
        lam[ranked] = 1.0 / max_support
        idx = ranked

    active = np.where(lam > tol)[0]
    if active.size == 0:
        active = np.array([int(np.argmax(weights))])
        lam = np.zeros(P)
        lam[active[0]] = 1.0

    return active, lam[active] / np.sum(lam[active])


def default_recombination_candidates(
    recombined_num_paths: int,
    noise_dim: int,
    *,
    candidate_factor: float = 8.0,
    k: int = 2,
    candidate_method: str = "rademacher",
) -> int:
    """
    Default candidate-pool size for recombination.

    Larger pools give RMP more choice when ``recombined_num_paths`` is high.
    """
    recombined_num_paths = max(1, int(recombined_num_paths))
    if candidate_method == "product" and noise_dim <= 3:
        return default_num_paths(noise_dim, k, multidim_method="product")
    return int(
        max(
            recombined_num_paths * candidate_factor,
            2 ** min(noise_dim, 12),
            64,
        )
    )


def recombine_path_measure(
    path_values: np.ndarray,
    weights: np.ndarray,
    ts: np.ndarray,
    k: int,
    gamma: float = 0.6,
    degree: int = 3,
    max_paths: Optional[int] = None,
    fill_to_support: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    path_values = np.asarray(path_values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    P = path_values.shape[0]
    max_paths = P if max_paths is None else int(min(max_paths, P))

    increments, segment_dts = path_segment_increments(path_values, ts, k, gamma)
    _, k_seg, d = increments.shape

    n_mom_at = lambda deg: k_seg * len(_monomial_exponents(d, deg))
    degree_eff = degree
    while degree_eff >= 1:
        if n_mom_at(degree_eff) <= max(max_paths * 3, degree_eff + 2):
            break
        degree_eff -= 1

    targets, _ = wiener_path_moment_targets(k_seg, d, segment_dts, degree_eff)
    feats, _ = path_segment_moment_features(increments, degree_eff)

    pool_mom = feats.T @ weights
    idx, lam = rmp_moment_match(
        feats, weights, targets, max_support=max_paths, fill_to_support=fill_to_support
    )

    reduced_paths = path_values[idx]
    reduced_weights = lam

    info = {
        "degree_used": degree_eff,
        "n_candidates": P,
        "n_active": idx.size,
        "max_paths_requested": max_paths,
        "fill_to_support": fill_to_support,
        "moment_error_before": float(np.linalg.norm(pool_mom - targets)),
        "moment_error_after": float(np.linalg.norm(feats[idx].T @ lam - targets)),
    }
    return reduced_paths, reduced_weights, info


def build_recombined_cubature_paths(
    ts: np.ndarray,
    noise_dim: int,
    k: int = 2,
    gamma: float = 0.6,
    num_paths: int = 32,
    recombined_num_paths: Optional[int] = None,
    candidate_paths: Optional[int] = None,
    candidate_factor: float = 8.0,
    degree: int = 3,
    seed: int = 0,
    candidate_method: str = "rademacher",
    fill_to_support: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build recombined Wiener cubature paths for high-dimensional noise.

    Parameters
    ----------
    recombined_num_paths : int, optional
        Target number of active paths after RMP. If unset, uses
        ``default_recombined_num_paths(noise_dim, k)`` (scales with ``d`` and ``k``).
    candidate_factor : float
        When ``candidate_paths`` is unset, pool size is
        ``max(recombined_num_paths * candidate_factor, 2^min(d,12), 64)``.
    fill_to_support : bool
        If True, use the full ``recombined_num_paths`` budget via NNLS when the
        sparse LP solution would leave paths unused.
    """
    ts = np.asarray(ts, dtype=np.float64)
    n_out = resolve_recombined_num_paths(
        noise_dim, k, recombined_num_paths=recombined_num_paths, num_paths=num_paths
    )

    if candidate_paths is None:
        candidate_paths = default_recombination_candidates(
            n_out, noise_dim, candidate_factor=candidate_factor, k=k, candidate_method=candidate_method
        )
    candidate_paths = int(max(candidate_paths, n_out))

    if candidate_method == "product" and noise_dim <= 3:
        candidates, cand_w = build_product_lyons_paths(ts - ts[0], noise_dim, k=k, gamma=gamma)
    else:
        candidates = build_multidim_cubature_paths(
            ts - ts[0], candidate_paths, noise_dim, k=k, gamma=gamma, seed=seed
        )
        cand_w = np.full(candidate_paths, 1.0 / candidate_paths)

    reduced, red_w, info = recombine_path_measure(
        candidates,
        cand_w,
        ts,
        k=k,
        gamma=gamma,
        degree=degree,
        max_paths=n_out,
        fill_to_support=fill_to_support,
    )
    info["candidate_method"] = candidate_method
    info["candidate_paths"] = candidate_paths
    info["recombined_num_paths"] = n_out
    info["candidate_factor"] = candidate_factor
    return reduced, red_w, info


# ---------------------------------------------------------------------------
# JAX Cubature-Recursive Filter
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from jax import lax, vmap

PathSampling = Literal["full", "random", "multinomial"]


def spatial_cubature_nodes(x: jnp.ndarray, P: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Degree-3 spherical cubature nodes for an n-dimensional Gaussian."""
    n = x.shape[0]
    U = jnp.linalg.cholesky(P)
    scale = jnp.sqrt(n)
    nodes = jnp.concatenate([x + scale * U.T, x - scale * U.T], axis=0)
    weights = jnp.full(2 * n, 1.0 / (2 * n))
    return nodes, weights


def weighted_mean_cov(
    samples: jnp.ndarray,
    weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    weights = weights / jnp.sum(weights)
    mean = weights @ samples
    diff = samples - mean
    cov = diff.T @ (weights[:, None] * diff)
    return mean, cov


def rk4_stratonovich_step(
    x: jnp.ndarray,
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    omega_dot: jnp.ndarray,
    drift_fn: Callable,
    diffusion_fn: Callable,
) -> jnp.ndarray:
    """One RK4 step for dx/dt = f(t,x) + g(t,x) @ omega_dot."""
    dt = t1 - t0
    t_mid = 0.5 * (t0 + t1)

    def rhs(t, state):
        G = diffusion_fn(t, state)
        return drift_fn(t, state) + G @ omega_dot

    k1 = rhs(t0, x)
    k2 = rhs(t_mid, x + 0.5 * dt * k1)
    k3 = rhs(t_mid, x + 0.5 * dt * k2)
    k4 = rhs(t1, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_stratonovich_cde(
    x0: jnp.ndarray,
    ts: jnp.ndarray,
    omega_path: jnp.ndarray,
    drift_fn: Callable,
    diffusion_fn: Callable,
) -> jnp.ndarray:
    """Integrate Stratonovich SDE along a fixed cubature path ω(t)."""
    if omega_path.ndim == 1:
        omega_path = omega_path[:, None]

    dts = ts[1:] - ts[:-1]
    omega_dot = (omega_path[1:] - omega_path[:-1]) / dts[:, None]
    t0s = ts[:-1]
    t1s = ts[1:]

    def step(x, inputs):
        t0, t1, od = inputs
        return rk4_stratonovich_step(x, t0, t1, od, drift_fn, diffusion_fn), None

    x_final, _ = lax.scan(step, x0, (t0s, t1s, omega_dot))
    return x_final


def _propagate_full(
    spatial_nodes: jnp.ndarray,
    path_values: jnp.ndarray,
    ts: jnp.ndarray,
    drift_fn: Callable,
    diffusion_fn: Callable,
) -> jnp.ndarray:
    def one(node, path):
        return integrate_stratonovich_cde(node, ts, path, drift_fn, diffusion_fn)

    return vmap(vmap(one, in_axes=(None, 0)), in_axes=(0, None))(spatial_nodes, path_values)


def _propagate_sampled(
    spatial_nodes: jnp.ndarray,
    path_values: jnp.ndarray,
    path_indices: jnp.ndarray,
    ts: jnp.ndarray,
    drift_fn: Callable,
    diffusion_fn: Callable,
) -> jnp.ndarray:
    selected = path_values[path_indices]

    def one(node, path):
        return integrate_stratonovich_cde(node, ts, path, drift_fn, diffusion_fn)

    return vmap(one)(spatial_nodes, selected)


def _numpy_precompute_paths(
    horizon: float,
    noise_dim: int,
    integration_dt: float,
    cubature_order: int,
    cubature_gamma: float,
    num_paths: Optional[int],
    multidim_paths: str,
    seed: int,
    candidate_paths: Optional[int] = None,
    recombination_degree: int = 3,
    recombined_num_paths: Optional[int] = None,
    recombination_candidate_factor: float = 8.0,
    recombination_fill_to_support: bool = False,
):
    num_steps = max(2, int(np.ceil(horizon / integration_dt)))
    ts = np.linspace(0.0, horizon, num_steps + 1)
    path_values, path_weights, recomb_info = build_cubature_driver(
        ts,
        noise_dim,
        num_paths=num_paths,
        k=cubature_order,
        gamma=cubature_gamma,
        seed=seed,
        multidim_method=multidim_paths,
        candidate_paths=candidate_paths,
        recombination_degree=recombination_degree,
        recombined_num_paths=recombined_num_paths,
        recombination_candidate_factor=recombination_candidate_factor,
        recombination_fill_to_support=recombination_fill_to_support,
        return_info=True,
    )
    return (
        jnp.asarray(ts, dtype=jnp.float64),
        jnp.asarray(path_values, dtype=jnp.float64),
        jnp.asarray(path_weights, dtype=jnp.float64),
        num_steps,
        float(horizon),
        recomb_info if multidim_paths == "recombined" else None,
    )


class JaxCubatureRecursiveFilter:
    """
    Differentiable CRF with JAX dynamics and batched cubature propagation.

    Drift ``drift_fn(t, x)``, diffusion ``diffusion_fn(t, x) -> (n, m)``,
    observation ``obs_fn(x)`` must be JAX-compatible for autodiff.
    """

    def __init__(
        self,
        drift_fn: Callable,
        diffusion_fn: Callable,
        obs_fn: Callable,
        R: jnp.ndarray,
        x0: jnp.ndarray,
        P0: jnp.ndarray,
        *,
        noise_dim: Optional[int] = None,
        cubature_order: int = 2,
        cubature_gamma: float = 0.6,
        num_paths: Optional[int] = None,
        integration_dt: float = 0.01,
        path_sampling: PathSampling = "full",
        multidim_paths: Literal["rademacher", "product", "recombined"] = "product",
        candidate_paths: Optional[int] = None,
        recombination_degree: int = 3,
        recombined_num_paths: Optional[int] = None,
        recombination_candidate_factor: float = 8.0,
        recombination_fill_to_support: bool = False,
        predict_horizon: Optional[float] = None,
        seed: int = 0,
    ):
        jax.config.update("jax_enable_x64", True)

        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.obs_fn = obs_fn
        self.R = jnp.asarray(R, dtype=jnp.float64)
        self.x = jnp.asarray(x0, dtype=jnp.float64)
        self.P = jnp.asarray(P0, dtype=jnp.float64)

        self.n = int(self.x.shape[0])
        self.m = int(self.R.shape[0])
        self.noise_dim = noise_dim or int(diffusion_fn(0.0, self.x).shape[1])
        self.cubature_order = cubature_order
        self.cubature_gamma = cubature_gamma
        self.num_paths = num_paths
        self.integration_dt = integration_dt
        self.path_sampling = path_sampling
        self.multidim_paths = multidim_paths
        self.candidate_paths = candidate_paths
        self.recombination_degree = recombination_degree
        self.recombined_num_paths = recombined_num_paths
        self.recombination_candidate_factor = recombination_candidate_factor
        self.recombination_fill_to_support = recombination_fill_to_support
        self.seed = seed
        self.last_num_integrations = 0
        self.num_wiener_paths = 0
        self.last_recombination_info: Optional[dict] = None

        self._cached_horizon: Optional[float] = None
        self._cached_num_steps: Optional[int] = None
        self._cached_ts_rel: Optional[jnp.ndarray] = None
        self._cached_path_values: Optional[jnp.ndarray] = None
        self._cached_path_weights: Optional[jnp.ndarray] = None

        if predict_horizon is not None:
            self.precompute_cubature_paths(predict_horizon)

    def precompute_cubature_paths(self, horizon: float) -> None:
        ts, paths, weights, num_steps, horizon_f, recomb_info = _numpy_precompute_paths(
            max(float(horizon), 1e-15),
            self.noise_dim,
            self.integration_dt,
            self.cubature_order,
            self.cubature_gamma,
            self.num_paths,
            self.multidim_paths,
            self.seed,
            self.candidate_paths,
            self.recombination_degree,
            self.recombined_num_paths,
            self.recombination_candidate_factor,
            self.recombination_fill_to_support,
        )
        self._cached_ts_rel = ts
        self._cached_path_values = paths
        self._cached_path_weights = weights
        self._cached_num_steps = num_steps
        self._cached_horizon = horizon_f
        self.num_wiener_paths = int(paths.shape[0])
        self.last_recombination_info = recomb_info

    def _get_paths(self, t_current: float, t_next: float):
        horizon = max(t_next - t_current, 1e-15)
        if (
            self._cached_path_values is None
            or self._cached_horizon is None
            or abs(horizon - self._cached_horizon) > 1e-12
        ):
            self.precompute_cubature_paths(horizon)

        ts = t_current + self._cached_ts_rel
        return ts, self._cached_path_values, self._cached_path_weights

    def predict(
        self,
        t_current: float,
        t_next: float,
        key: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ts, path_values, path_weights = self._get_paths(t_current, t_next)
        spatial_nodes, spatial_weights = spatial_cubature_nodes(self.x, self.P)
        num_spatial = spatial_nodes.shape[0]
        num_path = path_values.shape[0]

        if self.path_sampling == "full":
            terminals = _propagate_full(
                spatial_nodes, path_values, ts, self.drift_fn, self.diffusion_fn
            )
            samples_pred = terminals.reshape(-1, self.n)
            joint_w = (spatial_weights[:, None] * path_weights[None, :]).reshape(-1)
            self.last_num_integrations = num_spatial * num_path
        else:
            if key is None:
                key = jax.random.PRNGKey(self.seed)
            if self.path_sampling == "random":
                path_indices = jax.random.choice(
                    key,
                    num_path,
                    shape=(num_spatial,),
                    p=path_weights,
                )
            else:
                counts = jax.random.multinomial(key, num_spatial, path_weights)
                path_indices = jnp.repeat(
                    jnp.arange(num_path), counts, total_repeat_length=num_spatial
                )
                key, sub = jax.random.split(key)
                path_indices = jax.random.permutation(sub, path_indices)

            samples_pred = _propagate_sampled(
                spatial_nodes,
                path_values,
                path_indices,
                ts,
                self.drift_fn,
                self.diffusion_fn,
            )
            joint_w = spatial_weights
            self.last_num_integrations = num_spatial

        x_pred, P_pred = weighted_mean_cov(samples_pred, joint_w)
        return x_pred, P_pred, samples_pred, joint_w

    def update(
        self,
        y: jnp.ndarray,
        x_pred: jnp.ndarray,
        P_pred: jnp.ndarray,
        samples_pred: jnp.ndarray,
        sample_weights: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sample_weights = sample_weights / jnp.sum(sample_weights)
        z_samples = vmap(self.obs_fn)(samples_pred)
        z_pred = sample_weights @ z_samples
        dz = z_samples - z_pred
        P_zz = dz.T @ (sample_weights[:, None] * dz) + self.R
        dx = samples_pred - x_pred
        P_xz = dx.T @ (sample_weights[:, None] * dz)
        K = P_xz @ jnp.linalg.inv(P_zz)
        x_upd = x_pred + K @ (y - z_pred)
        P_upd = P_pred - K @ P_zz @ K.T
        return x_upd, P_upd

    def filter_step(
        self,
        y: jnp.ndarray,
        t_current: float,
        t_next: float,
        key: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_pred, P_pred, samples, weights = self.predict(t_current, t_next, key=key)
        x_upd, P_upd = self.update(y, x_pred, P_pred, samples, weights)
        self.x = x_upd
        self.P = P_upd
        return x_upd, P_upd


def make_benchmark_dynamics(
    sigma=0.05,
    state_dependent_diffusion: bool = False,
):
    """JAX dynamics matching ``_make_sde_system`` in unscented_kalman_filter.py."""
    sigma = jnp.asarray(sigma)

    def drift_fn(t, x):
        x1, x2 = x[0], x[1]
        return jnp.array([-0.1 * x1 + x2, -x1 - 0.1 * x2 + 0.1 * jnp.sin(x1)])

    if state_dependent_diffusion:

        def diffusion_fn(t, x):
            x1, x2 = x[0], x[1]
            return jnp.diag(
                jnp.array([sigma * (1.0 + 0.5 * x1**2), sigma * (1.0 + 0.5 * x2**2)])
            )
    else:

        def diffusion_fn(t, x):
            return jnp.eye(2) * sigma

    def obs_fn(x):
        return jnp.array([x[0], x[1] + 0.05 * jnp.sin(x[0])])

    return drift_fn, diffusion_fn, obs_fn


def make_highdim_dynamics(
    n_state: int = 10,
    n_noise: int = 8,
    n_obs: int = 5,
    sigma=0.08,
    coupling=0.04,
):
    """JAX dynamics for coupled high-dimensional benchmark."""
    sigma = jnp.asarray(sigma)
    coupling = jnp.asarray(coupling)
    n_obs = min(n_obs, n_state)

    G_base = jnp.zeros((n_state, n_noise))
    for j in range(n_noise):
        G_base = G_base.at[j, j].set(sigma)
        if j + 1 < n_state:
            G_base = G_base.at[j + 1, j].set(0.35 * sigma)

    def drift_fn(t, x):
        out = -0.12 * x
        out = out.at[:-1].add(coupling * x[1:])
        out = out.at[1:].add(coupling * x[:-1])
        out = out.at[0].add(0.06 * jnp.sin(x[0]))
        return out

    def diffusion_fn(t, x):
        return G_base

    def obs_fn(x):
        return x[:n_obs] + 0.03 * jnp.sin(x[:n_obs])

    return drift_fn, diffusion_fn, obs_fn


def make_l63_dynamics(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    diffusion_sigma: float = 0.1,
    obs_noise_std: float = 1.0,
):
    """
    Lorenz 63 SDE aligned with cd-dynamax ``l63_mech_x1`` / dynestyx L63 fixtures.

    State x = (x, y, z). Observe x[0] only with Gaussian noise.
    Diffusion: L = diffusion_sigma * I, Qc = I  (additive, Stratonovich = Itô).
    """
    sigma = jnp.asarray(sigma)
    rho = jnp.asarray(rho)
    beta = jnp.asarray(beta)
    diffusion_sigma = jnp.asarray(diffusion_sigma)

    def drift_fn(t, x):
        foo = jnp.array(
            [
                sigma * (x[1] - x[0]),
                rho * x[0] - x[1] - x[0] * x[2],
                -beta * x[2] + x[0] * x[1],
            ]
        )
        return jnp.clip(foo, -1000.0, 1000.0)

    def diffusion_fn(t, x):
        return diffusion_sigma * jnp.eye(3)

    def obs_fn(x):
        return jnp.array([x[0]])

    R = jnp.array([[obs_noise_std**2]])
    return drift_fn, diffusion_fn, obs_fn, R


def demo_autodiff_sigma(
    x0: jnp.ndarray,
    ts: jnp.ndarray,
    omega_path: jnp.ndarray,
    sigma_init: float = 0.03,
) -> Tuple[float, float]:
    """Gradient of terminal Stratonovich state w.r.t. diffusion scale σ."""

    def terminal_x0(sigma):
        drift_fn, diffusion_fn, _ = make_benchmark_dynamics(sigma)
        xT = integrate_stratonovich_cde(x0, ts, omega_path, drift_fn, diffusion_fn)
        return xT[0]

    val = float(terminal_x0(jnp.array(sigma_init)))
    grad = float(jax.grad(terminal_x0)(jnp.array(sigma_init)))
    return val, grad


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    x0 = jnp.array([1.0, 0.5])
    ts_np = np.linspace(0.0, 0.1, 11)
    paths, _ = build_cubature_driver(ts_np, noise_dim=2, k=2, multidim_method="product")
    ts = jnp.asarray(ts_np)
    omega = jnp.asarray(paths[0])
    val, grad = demo_autodiff_sigma(x0, ts, omega, sigma_init=0.05)
    print(f"Autodiff (nonzero cubature path): x_T[0]={val:.6f}, d/dσ={grad:.6f}")
