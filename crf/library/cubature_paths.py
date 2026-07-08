"""Wiener-space cubature path construction for the Cubature Recursive Filter (CRF).

These routines build a small set of deterministic Brownian "cubature" paths
omega^{j}(t)` together with weights w_j that match the moments
of the Wiener measure up to a chosen degree. They are computed **host-side with
NumPy** because they depend only on the assimilation interval, the Brownian
dimension and the cubature order -- never on the (traced) filter state. The
resulting arrays are therefore treated as static constants and closed over by the
JAX filtering kernels in crf.filtering.

Two multidimensional constructions are provided:

- ``"product"``: tensor product of degree-3 Lyons / Kusuoka-Lyons-Victoir 1D
  cubature paths. Exact to degree 3 but grows as (3^k)^d.
- ``"recombined"``: a large candidate pool reduced by moment-matching (RMP /
  recombination) to a sparse, low-cardinality measure with the same moments.

References:
    Lyons & Victoir (2004), *Cubature on Wiener space*.
    Litterer & Lyons (2012), *High order recombination and an application to
        cubature on Wiener space*.
"""

from __future__ import annotations

from itertools import combinations_with_replacement, product
from typing import List, Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np

__all__ = [
    "build_cubature_paths_1d",
    "build_product_lyons_paths",
    "build_recombined_cubature_paths",
    "build_cubature_driver",
    "default_num_paths",
    "default_recombined_num_paths",
    "precompute_cubature_paths",
    "product_path_count",
    "select_multidim_paths",
    "resolve_multidim_method",
]

MultidimMethod = Literal["product", "recombined"]
MultidimMethodInput = Literal["product", "recombined", "auto"]

# Product path counts above this threshold switch ``multidim_method="auto"`` to
# ``"recombined"``.
_AUTO_PRODUCT_PATH_THRESHOLD = 256

# Upper bound on the auto-selected post-recombination support when the caller
# does not specify ``recombined_num_paths`` but a high target degree is requested.
_RECOMBINE_BUDGET_CAP = 4096


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
    
    """Build degree-3 Wiener-space cubature paths on [0, 1] for a scalar driver.

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


def _resample_paths_1d(
        paths: np.ndarray, t_native: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
    t_target = np.asarray(t_target, dtype = np.float64)
    out = np.zeros((paths.shape[0], len(t_target)), dtype=np.float64)
    for i in range(paths.shape[0]):
        out[i] = np.interp(t_target, t_native, paths[i])
    return out



def build_product_lyons_paths(
    ts: np.ndarray,
    noise_dim: int,
    k: int = 3,
    gamma: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tensor product of 1D Lyons paths; ``P = (3^k)^d`` paths."""
    paths_1d, weights_1d, t_native = build_cubature_paths_1d(k=k, gamma=gamma)
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


def product_path_count(noise_dim: int, k: int) -> int:
    """Number of paths in the tensor-product Lyons construction ``(3^k)^d``."""
    d = max(1, int(noise_dim))
    k = max(1, int(k))
    return 3**k if d == 1 else 3 ** (k * d)


def select_multidim_paths(
    noise_dim: int,
    k: int = 2,
    *,
    product_path_threshold: int = _AUTO_PRODUCT_PATH_THRESHOLD,
) -> MultidimMethod:
    """Choose ``product`` vs ``recombined`` from Brownian dimension and order.

    Uses exact degree-3 product cubature when ``(3^k)^d`` is below
    ``product_path_threshold``; otherwise recombined cubature with the default
    path budget.
    """
    if product_path_count(noise_dim, k) <= product_path_threshold:
        return "product"
    return "recombined"


def resolve_multidim_method(
    multidim_method: MultidimMethodInput,
    noise_dim: int,
    k: int,
    *,
    product_path_threshold: int = _AUTO_PRODUCT_PATH_THRESHOLD,
) -> MultidimMethod:
    """Resolve ``"auto"`` to ``"product"`` or ``"recombined"``."""
    if multidim_method == "auto":
        return select_multidim_paths(
            noise_dim, k, product_path_threshold=product_path_threshold
        )
    if multidim_method not in ("product", "recombined"):
        raise ValueError('multidim_method must be "product", "recombined", or "auto".')
    return multidim_method


def default_recombined_num_paths(
    noise_dim: int, k: int = 2, *, cap: int = 256, floor: int = 16
) -> int:
    """Dimension-aware default for the post-recombination Wiener path budget."""
    d = max(1, int(noise_dim))
    k = max(1, int(k))
    by_order = 3**k
    by_dim = 4 * d * k
    hypercube = 2 ** min(d, 8)
    return int(min(cap, max(floor, by_order, by_dim, min(hypercube, cap))))


def default_num_paths(
    noise_dim: int,
    k: int,
    multidim_method: MultidimMethodInput = "product",
    num_paths: Optional[int] = None,
) -> int:
    """Default number of Wiener paths for a given method."""
    if multidim_method == "recombined":
        if num_paths is not None:
            return max(1, int(num_paths))
        return default_recombined_num_paths(noise_dim, k)
    if noise_dim == 1:
        return 3**k
    if multidim_method == "product":
        return product_path_count(noise_dim, k)
    #return max(3**k, 2**noise_dim)



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
    multidim_method: MultidimMethodInput = "product",
    candidate_factor: float = 8.0,
    recombination_degree: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build Wiener cubature driver paths ``omega(t)`` and weights on grid ``ts``.

    Args:
        ts: Integration time grid (1D, length ``num_steps + 1``).
        noise_dim: Brownian dimension.
        num_paths: Number of paths. Meaning depends on ``multidim_method``.
        k: Cubature order (number of macro segments).
        gamma: Macro-segment time-grid concentration exponent.
        multidim_method: ``"product"`` | ``"recombined"`` | ``"auto"``.
        candidate_factor: Candidate-pool scaling for recombination.
        recombination_degree: Target moment degree for recombination.

    Returns:
        Tuple ``(path_values, weights)`` with ``path_values`` of shape
        ``(P, len(ts), noise_dim)`` and ``weights`` of shape ``(P,)``.
    """
    ts = np.asarray(ts, dtype=np.float64)
    multidim_method = resolve_multidim_method(multidim_method, noise_dim, k)

    if noise_dim == 1:
        paths_native, weights, t_native = build_cubature_paths_1d(k=k, gamma=gamma)
        t_target = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-15)
        paths = _resample_paths_1d(paths_native, t_native, t_target)
        return paths[:, :, np.newaxis], weights

    if multidim_method == "product":
        return build_product_lyons_paths(ts - ts[0], noise_dim, k=k, gamma=gamma)

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

    # rademacher
   # if num_paths is None:
   #     num_paths = default_num_paths(noise_dim, k, multidim_method="rademacher")
   # paths = build_multidim_cubature_paths(
   #     ts - ts[0], num_paths, noise_dim, k=k, gamma=gamma, seed=seed
   # )
    return paths, np.full(num_paths, 1.0 / num_paths)



def precompute_cubature_paths(
    horizon: float,
    noise_dim: int,
    integration_dt: float = 0.01,
    cubature_order: int = 2,
    cubature_gamma: float = 0.6,
    num_paths: Optional[int] = None,
    multidim_method: MultidimMethodInput = "product",
    seed: int = 0,
    candidate_factor: float = 8.0,
    recombination_degree: int = 3,
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
            multidim_method: ``"product"`` | ``"recombined"`` | ``"auto"``.
            candidate_factor: Candidate-pool scaling for recombination.
            recombination_degree: Target moment degree for recombination.

        Returns:
            Tuple ``(ts, path_values, path_weights)`` as ``float64`` JAX arrays, with
            ``ts`` of shape ``(num_steps + 1,)`` (relative times starting at 0).
        """
        horizon = max(float(horizon), 1e-15)
        num_steps = max(2, int(np.ceil(horizon / integration_dt)))
        ts = np.linspace(0.0, horizon, num_steps + 1)
        multidim_method = resolve_multidim_method(
            multidim_method, noise_dim, cubature_order
        )
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
        )
        return (
            jnp.asarray(ts, dtype=jnp.float64),
            jnp.asarray(path_values, dtype=jnp.float64),
            jnp.asarray(path_weights, dtype=jnp.float64),
        )
