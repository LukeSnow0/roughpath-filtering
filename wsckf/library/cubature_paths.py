"""Wiener-space cubature path construction for the Wiener Space Cubature Kalman Filter (WSC-KF).

These routines build a small set of deterministic Brownian "cubature" paths
omega^{j}(t)` together with weights w_j that match the moments
of the Wiener measure up to a chosen degree. They are computed **host-side with
NumPy** because they depend only on the assimilation interval, the Brownian
dimension and the cubature order -- never on the (traced) filter state. The
resulting arrays are therefore treated as static constants and closed over by the
JAX filtering kernels in wsckf.filtering.

Multidimensional constructions:

- ``"lv"``: explicit Lyons–Victoir *multi-D* formulae (Lyons & Victoir 2004,
  §5). Degree ``m=5`` is their Theorem 5.9: a fixed piecewise-linear path
  ``ω^[d]`` whose log-signature matches (5.5) is scaled coordinate-wise by
  each node ``z`` of a degree-5 Gaussian cubature on ``N(0,I_d)``, once as is
  and once with its coordinates reversed, which flips the Lévy area
  (``2·3^d`` paths per macro-segment). ``ω^[d]`` is tabulated for ``d ≤ 2``
  only (Table 2 for ``d=2``); ``d ≥ 3`` raises ``NotImplementedError``.
  Degree ``m=3`` is the Stroud spherical rule (straight lines to
  ``±√d e_i``). This is **not** the tensor product of 1D path shapes.
- ``"product"``: tensor product of Lyons–Victoir *1D* cubature paths. Degree
  ``m=5`` uses the three-path √22 formula (``(3^k)^d`` paths); ``m=3`` uses the
  two-path straight-line formula (``(2^k)^d``). Misses mixed Lévy-area tensors.
- ``"recombined"``: recursive KLV expansion (always the m=5 3-branch) reduced by
  moment-matching (RMP) to a sparse measure.

``multidim_method="auto"`` (and ``build_filter(..., multidim_paths="auto")``)
selects ``"lv"`` for ``lyons_degree=5`` and ``d=2`` when the LV support fits
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
# Lyons–Victoir degree-5 paths (Proc. R. Soc. A 2004, §5(c))
#
# Theorem 5.9: let ω^[d] : [0, 1] → R^d have log-signature projecting onto
# (5.5), and let (z_k, λ_k) be a degree-5 cubature for N(0, I_d). Then the
# paths (t, z_k^1 ω^1, ..., z_k^d ω^d) and their coordinate reversals
# (t, z_k^1 ω^d, ..., z_k^d ω^1), each with weight λ_k / 2, define a degree-5
# cubature on Wiener space. ω^[d] is piecewise linear with L equal pieces and
# slope L·θ_ij on [(j-1)/L, j/L]; θ is from §5(c)(i) for d = 1 (the √22 path,
# L = 3) and from Table 2 for d = 2 (L = 4).
#
# In 1D the Gaussian nodes are (-√3, 0, √3) with weights (1/6, 2/3, 1/6), so
# the three paths are z·ω^[1]. The library historically labelled this formula
# "degree-3" (confusing the branch count with m); it is m = 5.
#
# Degree m = 3 is the two-path straight-line formula ω±(t) = ±t on [0, 1]
# with equal weights 1/2 (Gaussian cubature nodes ±1, lifted to paths).
# ---------------------------------------------------------------------------

_SQRT22 = np.sqrt(22.0)
_LV_DEG5_THETA = {
    1: np.array(
        [[(4.0 - _SQRT22) / 6.0, (-2.0 + 2.0 * _SQRT22) / 6.0, (4.0 - _SQRT22) / 6.0]]
    ),
    2: np.array(
        [
            [0.048077867969717, 0.089335677121310, 1.177095041847874, -0.314508586938901],
            [1.012368507495347, -1.619604763795000, 1.702104005103968, -0.094867748804311],
        ]
    ),
}

# 3-point Gauss–Hermite rule: degree-5 cubature for N(0, 1).
_GH3_NODES = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
_GH3_WEIGHTS = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])


def _lv_deg5_theta(d: int) -> np.ndarray:
    """Coefficients ``θ`` of ``ω^[d]``; raises for dimensions LV (2004) does not tabulate."""
    if d not in _LV_DEG5_THETA:
        raise NotImplementedError(
            f"Lyons–Victoir degree-5 paths are implemented for noise_dim in "
            f"{sorted(_LV_DEG5_THETA)}, got {d}; use multidim_method='recombined'."
        )
    return _LV_DEG5_THETA[d]


def _lv_unit_knots(d: int, lyons_degree: int) -> np.ndarray:
    """Corner times ``j/L`` of the one-segment LV formula (straight lines for m = 3)."""
    if int(lyons_degree) == 3:
        return np.array([0.0, 1.0])
    return np.linspace(0.0, 1.0, _lv_deg5_theta(d).shape[1] + 1)


def _lv_deg5_path(d: int, t: np.ndarray) -> np.ndarray:
    """Lyons–Victoir path ``ω^[d]`` sampled on ``t ⊂ [0, 1]``, shape ``(len(t), d)``."""
    theta = _lv_deg5_theta(d)
    values = np.concatenate([np.zeros((d, 1)), np.cumsum(theta, axis=1)], axis=1)
    knots = _lv_unit_knots(d, 5)
    return _sample_paths(t, knots, values.T[None])[0]


def _lv_deg5_1d_branches(t: np.ndarray) -> np.ndarray:
    """The three 1D degree-5 paths ``z·ω^[1](t)`` for ``z ∈ (-√3, 0, √3)``, shape ``(3, len(t))``."""
    return _GH3_NODES[:, None] * _lv_deg5_path(1, t)[None, :, 0]


# ---------------------------------------------------------------------------
# Exact piecewise-linear path representation
#
# Every cubature path here is piecewise linear, and the RK4 integrator holds
# ω̇ constant on each grid step, so a path is integrated exactly only if all
# of its corners ("knots") are grid points. Paths are therefore built as their
# values at the knots, and the integration grid is built around those knots.
# ---------------------------------------------------------------------------


def _macro_segment_bounds(k: int, gamma: float) -> np.ndarray:
    """Macro-segment boundaries ``1 - (1 - i/k)^γ`` for ``i = 0, ..., k``."""
    k = max(1, int(k))
    return np.array([1.0 - (1.0 - i / k) ** gamma for i in range(k + 1)], dtype=np.float64)


def _concatenate_macro_segments(
    unit_values: np.ndarray, unit_weights: np.ndarray, k: int, gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose a one-segment cubature formula over ``k`` macro-segments, exactly.

    ``unit_values`` holds the unit-time paths at their knots ``j/L``, shape
    ``(P, L + 1, d)`` and starting at 0. Segment ``s`` of length ``Δ_s`` uses
    the formula under Brownian scaling (``√Δ_s``), offset by the parent path's
    value, giving ``P**k`` paths (parent-major) on ``k·L + 1`` knots.

    Returns:
        ``(knots, values, weights)`` with ``values`` of shape ``(P**k, k·L + 1, d)``.
    """
    unit_values = np.asarray(unit_values, dtype=np.float64)
    unit_weights = np.asarray(unit_weights, dtype=np.float64)
    n_unit, n_knots, d = unit_values.shape
    unit_knots = np.linspace(0.0, 1.0, n_knots)
    knots = [np.zeros(1)]
    values = np.zeros((1, 1, d))
    weights = np.ones(1)
    bounds = _macro_segment_bounds(k, gamma)
    for start, end in zip(bounds[:-1], bounds[1:]):
        span = end - start
        knots.append(start + span * unit_knots[1:])
        head = np.broadcast_to(values[:, None], (values.shape[0], n_unit) + values.shape[1:])
        tail = values[:, None, -1:, :] + np.sqrt(span) * unit_values[None, :, 1:, :]
        values = np.concatenate([head, tail], axis=2).reshape(
            -1, values.shape[1] + n_knots - 1, d
        )
        weights = (weights[:, None] * unit_weights[None, :]).reshape(-1)
    return np.concatenate(knots), values, weights / weights.sum()


def _sample_paths(t_unit: np.ndarray, knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Evaluate piecewise-linear paths given at ``knots`` on ``t_unit`` (both in ``[0, 1]``).

    Exact at every ``t_unit`` point; the sampled polyline equals the path only if
    ``t_unit`` contains every knot.
    """
    t = np.clip(np.asarray(t_unit, dtype=np.float64), knots[0], knots[-1])
    i = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, len(knots) - 2)
    frac = ((t - knots[i]) / (knots[i + 1] - knots[i]))[None, :, None]
    return values[:, i, :] * (1.0 - frac) + values[:, i + 1, :] * frac


def _unit_time(ts: np.ndarray) -> np.ndarray:
    """Map a grid on ``[ts[0], ts[-1]]`` to ``[0, 1]``."""
    ts = np.asarray(ts, dtype=np.float64)
    return (ts - ts[0]) / max(float(ts[-1] - ts[0]), 1e-15)


def _integration_grid(knots: np.ndarray, horizon: float, integration_dt: float) -> np.ndarray:
    """Unit-time integration grid containing every knot.

    Each piece between consecutive knots is split into equal steps of physical
    length at most ``integration_dt``, so every RK4 step lies on one linear piece
    of every path. At least two steps in total.
    """
    pieces = np.diff(knots)
    n_steps = np.maximum(1, np.ceil(pieces * horizon / integration_dt - 1e-9).astype(int))
    if n_steps.sum() < 2:
        n_steps[np.argmax(pieces)] += 1
    grid = [knots[:1]] + [
        np.linspace(a, b, n + 1)[1:] for a, b, n in zip(knots[:-1], knots[1:], n_steps)
    ]
    return np.concatenate(grid)


def build_cubature_paths_1d(
    k: int = 3,
    gamma: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Lyons–Victoir **degree m = 5** Wiener-space cubature paths on [0, 1].

    The base (k = 1) support has three paths with weights ``(1/6, 2/3, 1/6)``.
    Macro-segment count ``k`` composes the formula, giving ``3^k`` paths that
    remain degree-5 cubature on each scaled timeframe.

    Args:
        k: Number of macro segments.
        gamma: Macro-segment time-grid concentration exponent.

    Returns:
        Tuple ``(paths, weights, knots)``: the paths' exact values at their
        ``3k + 1`` corner times ``knots``, with ``paths`` of shape ``(3^k, 3k + 1)``.
    """
    unit = _lv_deg5_1d_branches(_lv_unit_knots(1, 5))[:, :, None]
    knots, values, weights = _concatenate_macro_segments(unit, _GH3_WEIGHTS, k, gamma)
    return values[:, :, 0], weights, knots


def build_cubature_paths_1d_m3(
    k: int = 3,
    gamma: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Lyons–Victoir **degree m = 3** Wiener-space cubature paths on [0, 1].

    Base (k = 1) support: two straight-line paths ``ω±(t) = ±t`` with weights
    ``1/2`` (Gaussian nodes ±1 lifted to paths). Composing ``k`` macro segments
    yields ``2^k`` paths that remain degree-3 cubature on each scaled timeframe.

    Returns:
        Tuple ``(paths, weights, knots)`` as for :func:`build_cubature_paths_1d`,
        with ``k + 1`` knots.
    """
    unit = np.array([[[0.0], [1.0]], [[0.0], [-1.0]]])
    knots, values, weights = _concatenate_macro_segments(unit, np.array([0.5, 0.5]), k, gamma)
    return values[:, :, 0], weights, knots


# ---------------------------------------------------------------------------
# Lyons–Victoir explicit multi-D formulae (Proc. R. Soc. A 2004, §5)
#
# Not the tensor product of 1D path shapes. Degree m=5 is Theorem 5.9 applied
# to ω^[d] (see the degree-5 path section above) with the product Gauss–Hermite
# rule; degree m=3 is the Stroud spherical rule (straight lines, Prop. 5.3).
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
    nodes = np.array(list(product(_GH3_NODES, repeat=d)), dtype=np.float64)
    weights = np.array(
        [float(np.prod(ws)) for ws in product(_GH3_WEIGHTS, repeat=d)],
        dtype=np.float64,
    )
    return nodes, weights


def _unit_lv_paths(
    noise_dim: int, lyons_degree: int, t_unit: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-time LV multi-D paths on ``t_unit ⊂ [0, 1]``.

    Degree 5 is Theorem 5.9 of Lyons & Victoir (2004): for each Gauss–Hermite
    node ``z`` the paths ``z ∘ ω^[d]`` and ``z ∘ reverse(ω^[d])``, each with
    half the node weight. It is available for ``d ∈ {1, 2}``.

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
        omega = _lv_deg5_path(d, t_unit)
        nodes, gauss_w = _gaussian_cubature_degree5(d)
        path_list = []
        weight_list = []
        for z, lw in zip(nodes, gauss_w):
            # Theorem 5.9: ω_k and its coordinate reversal ω_{n+k}.
            path_list += [omega * z, omega[:, ::-1] * z]
            weight_list += [0.5 * lw, 0.5 * lw]
        paths = np.stack(path_list, axis=0)
        weights = np.asarray(weight_list, dtype=np.float64)
    else:
        raise ValueError("lyons_degree must be 3 or 5")
    weights = weights / weights.sum()
    return paths, weights


def lv_path_count(noise_dim: int, k: int = 1, lyons_degree: int = 5) -> int:
    """Number of paths in the explicit multi-D Lyons–Victoir construction.

    ``m=5``: ``(2 · 3^d)^k`` (Gauss–Hermite nodes × {path, coordinate reversal},
    composed ``k`` times); the construction itself exists for ``d ≤ 2``.
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


def _lv_knot_paths(
    noise_dim: int, lyons_degree: int, k: int, gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Knot representation of the explicit multi-D LV paths (unit time)."""
    unit_knots = _lv_unit_knots(int(noise_dim), lyons_degree)
    unit_paths, unit_w = _unit_lv_paths(noise_dim, lyons_degree, unit_knots)
    return _concatenate_macro_segments(unit_paths, unit_w, k, gamma)


def build_lyons_victoir_multid_paths(
    ts: np.ndarray,
    noise_dim: int,
    lyons_degree: int = 5,
    k: int = 1,
    gamma: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit multi-D Lyons–Victoir cubature, sampled on ``ts``.

    Returns *unit-time* path values (``E[ω(1)ω(1)^T] = I``). Callers that
    pass a horizon-``T`` grid must scale by ``√T`` (as
    :func:`precompute_cubature_paths` does). The samples reproduce the paths
    exactly only if ``ts`` contains their corners, as the grid built by
    :func:`precompute_cubature_paths` does.

    Args:
        ts: Time grid, typically ``[0, T]``. Values are interpolated in
            normalised time ``t/T``.
        noise_dim: Brownian dimension ``d`` (``d ∈ {1, 2}`` for degree 5).
            :func:`build_cubature_driver` routes ``d = 1`` to the 1D builders.
        lyons_degree: ``3`` or ``5``.
        k: Number of macro-segments; the one-step formula is composed
            ``k`` times on a ``gamma``-concentrated partition of ``[0, 1]``.
        gamma: Macro-segment concentration exponent.
    """
    knots, values, weights = _lv_knot_paths(noise_dim, int(lyons_degree), k, gamma)
    return _sample_paths(_unit_time(ts), knots, values), weights


def _product_knot_paths(
    noise_dim: int, k: int, gamma: float, lyons_degree: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Knot representation of the tensor product of 1D Lyons–Victoir paths."""
    if int(lyons_degree) == 5:
        paths_1d, weights_1d, knots = build_cubature_paths_1d(k=k, gamma=gamma)
    elif int(lyons_degree) == 3:
        paths_1d, weights_1d, knots = build_cubature_paths_1d_m3(k=k, gamma=gamma)
    else:
        raise ValueError("lyons_degree must be 3 or 5 (Lyons–Victoir explicit formulae)")
    p1 = paths_1d.shape[0]
    # Row = flat path index; coordinate 0 is the fastest-varying digit.
    digits = (np.arange(p1**noise_dim)[:, None] // p1 ** np.arange(noise_dim)[None, :]) % p1
    values = np.transpose(paths_1d[digits], (0, 2, 1))
    return knots, values, np.prod(weights_1d[digits], axis=1)


def build_product_lyons_paths(
    ts: np.ndarray,
    noise_dim: int,
    k: int = 3,
    gamma: float = 0.6,
    lyons_degree: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tensor product of 1D Lyons–Victoir paths, sampled on ``ts``.

    ``lyons_degree=5`` (default): ``P = (3^k)^d`` from the explicit m=5 formula.
    ``lyons_degree=3``: ``P = (2^k)^d`` from the explicit m=3 straight-line formula.
    """
    knots, values, weights = _product_knot_paths(noise_dim, k, gamma, lyons_degree)
    return _sample_paths(_unit_time(ts), knots, values), weights


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

    For ``lyons_degree=5`` and ``d>1``, prefers explicit multi-D LV when it is
    implemented for ``d`` (``d = 2``) and ``lv_path_count`` is within
    ``product_path_threshold``. Degree ``m=3`` still uses the 1D-product
    (hypercube) formula.
    """
    d = max(1, int(noise_dim))
    k = max(1, int(k))
    lyons_degree = int(lyons_degree)
    if d > 1 and lyons_degree == 5:
        if (
            d in _LV_DEG5_THETA
            and lv_path_count(d, k, lyons_degree=5) <= product_path_threshold
        ):
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
    k: int, gamma: float
) -> Tuple[np.ndarray, List[int], List[np.ndarray], np.ndarray]:
    """Per-macro-segment 1D degree-5 Lyons branch shapes at the segment's knots.

    Returns ``(knots, seg_lengths, branch_shapes, branch_w)`` where
    ``branch_shapes[s]`` has shape ``(3, seg_lengths[s])`` and holds the
    within-segment path increment (to be added to that segment's start value)
    for each of the 3 cubature branches, at the segment's knots (the first
    segment includes ``t = 0``). ``branch_w`` are the branch weights
    ``[1/6, 2/3, 1/6]``.
    """
    unit_knots = _lv_unit_knots(1, 5)
    unit = _lv_deg5_1d_branches(unit_knots)
    bounds = _macro_segment_bounds(k, gamma)
    knots: List[np.ndarray] = []
    seg_lengths: List[int] = []
    branch_shapes: List[np.ndarray] = []
    for s, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        first = 0 if s == 0 else 1
        span = end - start
        knots.append(start + span * unit_knots[first:])
        branch_shapes.append(np.sqrt(span) * unit[:, first:])
        seg_lengths.append(len(unit_knots) - first)
    return np.concatenate(knots), seg_lengths, branch_shapes, _GH3_WEIGHTS.copy()


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
    knots, values, weights = _recombined_knot_paths(
        noise_dim, k, gamma, recombined_num_paths, degree
    )
    return _sample_paths(_unit_time(ts), knots, values), weights


def _recombined_knot_paths(
    noise_dim: int,
    k: int,
    gamma: float,
    recombined_num_paths: Optional[int],
    degree: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Knot representation of :func:`build_recombined_cubature_paths` (unit time)."""
    d = max(1, int(noise_dim))
    k = max(1, int(k))

    if recombined_num_paths is not None:
        n_out = max(1, int(recombined_num_paths))
    else:
        n_out = default_recombined_num_paths(d, k)
        n_mom_full = len(_monomial_exponents(k * d, max(1, int(degree))))
        n_out = max(n_out, min(n_mom_full, _RECOMBINE_BUDGET_CAP))

    knots, seg_lengths, branch_shapes, branch_w = _segment_branch_shapes(k, gamma)

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

    return knots, cur_paths, cur_w


# ---------------------------------------------------------------------------
# Driver and precompute entry point
# ---------------------------------------------------------------------------


def _cubature_knot_paths(
    noise_dim: int,
    *,
    k: int,
    gamma: float,
    multidim_method: MultidimMethodInput,
    lyons_degree: int,
    num_paths: Optional[int],
    recombination_degree: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Knot representation ``(knots, values, weights)`` of the selected construction.

    ``values`` has shape ``(P, len(knots), noise_dim)`` and holds the unit-time
    paths' exact values at their corner times ``knots ⊂ [0, 1]``.
    """
    lyons_degree = int(lyons_degree)
    if lyons_degree not in (3, 5):
        raise ValueError("lyons_degree must be 3 or 5")

    if noise_dim == 1:
        builder = build_cubature_paths_1d if lyons_degree == 5 else build_cubature_paths_1d_m3
        paths, weights, knots = builder(k=k, gamma=gamma)
        return knots, paths[:, :, np.newaxis], weights

    # Explicit m=3 product is the 1D-tensor hypercube rule unless the caller
    # asked for LV (Stroud) or recombination.
    if lyons_degree == 3 and multidim_method == "auto":
        multidim_method = "product"
    else:
        multidim_method = resolve_multidim_method(
            multidim_method, noise_dim, k, lyons_degree=lyons_degree
        )

    if multidim_method == "lv":
        return _lv_knot_paths(noise_dim, lyons_degree, k, gamma)
    if multidim_method == "product":
        return _product_knot_paths(noise_dim, k, gamma, lyons_degree)
    if multidim_method == "recombined":
        return _recombined_knot_paths(noise_dim, k, gamma, num_paths, recombination_degree)
    raise ValueError(f"Unknown multidim_method {multidim_method!r}")


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
            ``5`` is the three-path 1D construction, or the multi-D LV
            construction (Theorem 5.9, ``d = 2`` only) for ``"lv"``.

    Returns:
        Tuple ``(path_values, weights)`` with ``path_values`` of shape
        ``(P, len(ts), noise_dim)`` and ``weights`` of shape ``(P,)``.
        Path values are unit-time (``E[ω(1)ω(1)^T] = I``).
    """
    del seed, candidate_factor  # recombination is deterministic
    knots, values, weights = _cubature_knot_paths(
        noise_dim,
        k=k,
        gamma=gamma,
        multidim_method=multidim_method,
        lyons_degree=lyons_degree,
        num_paths=num_paths,
        recombination_degree=recombination_degree,
    )
    return _sample_paths(_unit_time(ts), knots, values), weights



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

    Every cubature path is piecewise linear and the RK4 integrator holds ``ω̇``
    constant on each step, so the grid is built around the paths' corners: each
    linear piece is split into equal steps of at most ``integration_dt``. The
    integrated driver then equals the cubature path exactly.

    Args:
        horizon: Length of one assimilation interval (relative time ``[0, horizon]``).
        noise_dim: Brownian dimension.
        integration_dt: Maximum inner integration step size.
        cubature_order: Cubature order ``k``.
        cubature_gamma: Macro-segment time-grid concentration exponent.
        num_paths: Number of paths (method-dependent meaning).
        multidim_method: ``"lv"`` | ``"product"`` | ``"recombined"`` | ``"auto"``.
        seed: Unused; the constructions are deterministic.
        candidate_factor: Unused; retained for API compatibility.
        recombination_degree: Target moment degree for recombination.
        lyons_degree: Explicit Lyons–Victoir degree ``m ∈ {3, 5}`` (default 5).

    Returns:
        Tuple ``(ts, path_values, path_weights)`` as ``float64`` JAX arrays.
        ``ts`` is a piecewise-uniform grid on ``[0, horizon]`` that contains
        every path corner.
    """
    del seed, candidate_factor  # recombination is deterministic
    horizon = max(float(horizon), 1e-15)
    knots, values, path_weights = _cubature_knot_paths(
        noise_dim,
        k=cubature_order,
        gamma=cubature_gamma,
        multidim_method=multidim_method,
        lyons_degree=lyons_degree,
        num_paths=num_paths,
        recombination_degree=recombination_degree,
    )
    grid = _integration_grid(knots, horizon, integration_dt)
    # Paths are unit-time Wiener cubature (E[ω(1)ω(1)^T] = I); Brownian scaling
    # ω_horizon(t) = √horizon · ω_1(t/horizon) restores E[ω(T)ω(T)^T] = T I.
    path_values = _sample_paths(grid, knots, values) * np.sqrt(horizon)
    return (
        jnp.asarray(grid * horizon, dtype=jnp.float64),
        jnp.asarray(path_values, dtype=jnp.float64),
        jnp.asarray(path_weights, dtype=jnp.float64),
    )
