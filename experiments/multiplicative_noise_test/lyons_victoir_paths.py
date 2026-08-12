"""Lyons–Victoir explicit multi-D cubature paths (Proc. R. Soc. A, 2004).

This is a **standalone** path builder for the paper's degree ``m = 3`` and
``m = 5`` constructions. It does **not** modify ``wsckf.library.cubature_paths``
(the repo default is the 1D-product KLV grid, which is not the LV multi-D lift).

Constructions (cf. Lyons–Victoir Thms for Wiener-space cubature via Gaussian
cubature + Bernoulli):

* **m = 3.** Let ``(z_k, λ_k)`` be a degree-3 cubature for ``N(0, I_d)``.
  Lie polynomial ``ℓ_k = ε_0 + Σ_i z_k^i ε_i``. Realising path: the straight
  line ``ω(t) = z_k · t`` on ``[0, 1]``.

* **m = 5.** Let ``(z_k, λ_k)`` be a degree-5 cubature for ``N(0, I_d)`` and
  ``η ∈ {±1}``. Lie polynomial includes the Lévy-area terms
  ``(η/2) Σ_{i<j} z_i z_j [ε_i, ε_j]`` (and weight ``λ_k / 2``). Realising
  path: straight line to ``z`` plus pairwise area bumps so that
  ``A_{ij} = (η/2) z_i z_j``.

Gaussian cubatures used here (efficient, standard):

* deg 3: Stroud / spherical — ``±√d e_i``, weight ``1/(2d)`` (``2d`` points).
* deg 5: product of the 1D 3-point Gauss–Hermite rule
  ``{±√3, 0}`` with weights ``{1/6, 2/3, 1/6}`` (``3^d`` points).

Paths are returned on an arbitrary time grid ``ts ⊂ [0, T]`` with Brownian
scaling ``ω_T(t) = √T · ω_1(t/T)``.
"""

from __future__ import annotations

from itertools import product
from typing import Literal, Tuple

import numpy as np

LyonsDegree = Literal[3, 5]


# ---------------------------------------------------------------------------
# Gaussian cubature formulae
# ---------------------------------------------------------------------------


def gaussian_cubature_degree3(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Degree-3 cubature for ``N(0, I_d)``: Stroud points ``±√d e_i``."""
    d = int(d)
    nodes = np.zeros((2 * d, d), dtype=np.float64)
    for i in range(d):
        nodes[2 * i, i] = np.sqrt(d)
        nodes[2 * i + 1, i] = -np.sqrt(d)
    weights = np.full(2 * d, 1.0 / (2 * d), dtype=np.float64)
    return nodes, weights


def gaussian_cubature_degree5(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Degree-5 cubature for ``N(0, I_d)``: product 3-point Gauss–Hermite."""
    d = int(d)
    gh_nodes_1d = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)], dtype=np.float64)
    gh_w_1d = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=np.float64)
    nodes = np.array(list(product(gh_nodes_1d, repeat=d)), dtype=np.float64)
    # weights = ⊗_i w_{j_i}
    idx = list(product(range(3), repeat=d))
    weights = np.array(
        [float(np.prod([gh_w_1d[j] for j in js])) for js in idx],
        dtype=np.float64,
    )
    return nodes, weights


# ---------------------------------------------------------------------------
# Unit-time path realisations
# ---------------------------------------------------------------------------


def _straight_line_path(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``ω(t) = z · t`` on ``[0, 1]``, shape ``(len(t), d)``."""
    return t[:, None] * z[None, :]


def _oned_m5_sqrt22(t: np.ndarray, terminal: float) -> np.ndarray:
    """Lyons–Victoir 1D degree-5 path on ``[0, 1]`` ending at ``terminal``.

    The classic √22 piecewise-linear drivers end at ``±√3``; we scale so the
    path ends at an arbitrary Gauss–Hermite node (``0, ±√3`` in the deg-5 rule).
    """
    if abs(terminal) < 1e-15:
        return np.zeros_like(t)
    # Path shape ending at +1 (multiply by terminal to hit Gauss–Hermite node).
    # Classical LV path is ``√3 * base`` ending at ``+√3``.
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
    A = np.zeros((d, d), dtype=np.float64)
    for k in range(n - 1):
        w0, w1 = omega[k], omega[k + 1]
        dw = w1 - w0
        mid = 0.5 * (w0 + w1)
        A += np.outer(mid, dw) - np.outer(dw, mid)
    return 0.5 * A


def _add_area_bumps(
    omega: np.ndarray,
    z: np.ndarray,
    area_targets: dict[tuple[int, int], float],
    t: np.ndarray,
    n_correct: int = 3,
) -> np.ndarray:
    """Adjust a path by zero-endpoint bumps so pairwise Lévy areas match targets.

    Preserves ``ω(0)=0`` and ``ω(1)=z``. Sensitivity of ``A_ij`` to a bump on
    coordinate ``j`` is measured by a finite-difference probe so the correction
    works for non-linear primary paths (√22 drivers), not just straight lines.
    """
    del z  # terminal already fixed by zero-endpoint bumps
    g = 4.0 * t * (1.0 - t)
    omega = omega.copy()
    eps = 1e-7
    for _ in range(n_correct):
        A_now = _levy_area_pl(omega)
        for (i, j), A_tgt in area_targets.items():
            dA = A_tgt - A_now[i, j]
            if abs(dA) < 1e-14:
                continue
            trial = omega.copy()
            trial[:, j] = trial[:, j] + eps * g
            sens = (_levy_area_pl(trial)[i, j] - A_now[i, j]) / eps
            if abs(sens) < 1e-12:
                trial = omega.copy()
                trial[:, i] = trial[:, i] + eps * g
                sens_i = (_levy_area_pl(trial)[i, j] - A_now[i, j]) / eps
                if abs(sens_i) < 1e-12:
                    continue
                omega[:, i] = omega[:, i] + (dA / sens_i) * g
            else:
                omega[:, j] = omega[:, j] + (dA / sens) * g
    return omega


def unit_path_m3(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    return _straight_line_path(z, t)


def unit_path_m5(z: np.ndarray, eta: float, t: np.ndarray) -> np.ndarray:
    """Unit-time path for LV degree 5 at Gaussian node ``z`` and Bernoulli ``η``.

    Primary motion: coordinate-wise Lyons–Victoir √22 degree-5 drivers (so the
    construction reduces to the classical 1D formula when ``d=1``). Cross terms:
    Lévy-area bumps enforcing ``A_ij = (η/2) z_i z_j``.
    """
    d = int(z.shape[0])
    omega = np.stack([_oned_m5_sqrt22(t, float(z[i])) for i in range(d)], axis=1)
    areas = {
        (i, j): 0.5 * eta * z[i] * z[j]
        for i in range(d)
        for j in range(i + 1, d)
    }
    if areas:
        omega = _add_area_bumps(omega, z, areas, t)
    return omega


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_lyons_victoir_paths(
    ts: np.ndarray,
    noise_dim: int,
    lyons_degree: LyonsDegree,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build LV multi-D cubature paths on an arbitrary time grid.

    Args:
        ts: Time grid, shape ``(N,)``, covering ``[0, T]`` (``ts[0]`` may be 0).
        noise_dim: Brownian dimension ``d``.
        lyons_degree: ``3`` or ``5``.

    Returns:
        ``path_values`` of shape ``(P, N, d)`` and ``weights`` of shape ``(P,)``.
    """
    ts = np.asarray(ts, dtype=np.float64)
    d = int(noise_dim)
    T = float(ts[-1] - ts[0])
    if T <= 0.0:
        raise ValueError("ts must span a positive horizon")
    t_unit = (ts - ts[0]) / T  # in [0, 1]

    if lyons_degree == 3:
        nodes, gauss_w = gaussian_cubature_degree3(d)
        paths = np.stack([unit_path_m3(z, t_unit) for z in nodes], axis=0)
        weights = gauss_w.copy()
    elif lyons_degree == 5:
        nodes, gauss_w = gaussian_cubature_degree5(d)
        path_list = []
        weight_list = []
        for z, lw in zip(nodes, gauss_w):
            for eta in (+1.0, -1.0):
                path_list.append(unit_path_m5(z, eta, t_unit))
                weight_list.append(0.5 * lw)
        paths = np.stack(path_list, axis=0)
        weights = np.asarray(weight_list, dtype=np.float64)
    else:
        raise ValueError("lyons_degree must be 3 or 5")

    # Brownian scaling: ω_T(t) = √T · ω_1(t/T)
    paths = paths * np.sqrt(T)
    weights = weights / weights.sum()
    return paths, weights


def precompute_lv_paths(
    horizon: float,
    noise_dim: int,
    integration_dt: float,
    lyons_degree: LyonsDegree,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the ``(ts, path_values, path_weights)`` layout of ``precompute_cubature_paths``."""
    horizon = max(float(horizon), 1e-15)
    n_steps = max(2, int(np.ceil(horizon / integration_dt)))
    ts = np.linspace(0.0, horizon, n_steps + 1)
    paths, weights = build_lyons_victoir_paths(ts, noise_dim, lyons_degree)
    return ts, paths, weights


def describe(noise_dim: int = 2) -> str:
    n3 = gaussian_cubature_degree3(noise_dim)[0].shape[0]
    n5 = gaussian_cubature_degree5(noise_dim)[0].shape[0]
    return (
        f"LV explicit multi-D (d={noise_dim}): "
        f"m=3 → {n3} paths (Stroud deg-3 Gauss, straight lines); "
        f"m=5 → {2 * n5} paths (product GH deg-5 × Bernoulli η, area bumps). "
        f"No recombination; no 1D-product KLV grid."
    )
