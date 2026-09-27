"""Explicit multi-D Lyons–Victoir cubature (not 1D-product)."""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from wsckf.library import (
    build_cubature_driver,
    ito_to_stratonovich_dynamics,
    lv_path_count,
    precompute_cubature_paths,
    predict,
    product_path_count,
    select_multidim_paths,
)
from wsckf.library.cubature_paths import build_lyons_victoir_multid_paths


def _unit_paths(noise_dim, *, method, lyons_degree=5, k=1, n_steps=40):
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    return build_cubature_driver(
        ts,
        noise_dim,
        k=k,
        multidim_method=method,
        lyons_degree=lyons_degree,
    )


def test_lv_path_counts():
    assert lv_path_count(2, 1, lyons_degree=5) == 18
    assert lv_path_count(2, 2, lyons_degree=5) == 18**2
    assert lv_path_count(2, 1, lyons_degree=3) == 4
    assert product_path_count(2, 1, lyons_degree=5) == 9


def test_auto_selects_lv_for_deg5_multid():
    assert select_multidim_paths(2, 1, lyons_degree=5) == "lv"
    assert select_multidim_paths(2, 2, lyons_degree=5) == "lv"
    assert select_multidim_paths(2, 3, lyons_degree=5) == "recombined"
    assert select_multidim_paths(1, 2, lyons_degree=5) == "product"
    assert select_multidim_paths(3, 1, lyons_degree=5) == "recombined"


def test_lv_deg5_not_available_for_three_dims():
    with pytest.raises(NotImplementedError):
        _unit_paths(3, method="lv", lyons_degree=5)


def test_precompute_lv_vs_product_support():
    paths_lv, w_lv = _unit_paths(2, method="lv", lyons_degree=5)
    paths_pr, w_pr = _unit_paths(2, method="product", lyons_degree=5)
    assert paths_lv.shape[0] == 18
    assert paths_pr.shape[0] == 9
    np.testing.assert_allclose(np.sum(w_lv), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(w_pr), 1.0, atol=1e-12)


def _signature(path, depth):
    """Signature levels ``0..depth`` of a piecewise-linear path (Chen's identity)."""
    dim = path.shape[1]
    sig = [np.ones(())] + [np.zeros((dim,) * k) for k in range(1, depth + 1)]
    for dx in np.diff(path, axis=0):
        seg = [np.ones(())]
        for k in range(1, depth + 1):
            seg.append(np.multiply.outer(seg[-1], dx) / k)
        sig = [
            sum(np.multiply.outer(sig[i], seg[k - i]) for i in range(k + 1))
            for k in range(depth + 1)
        ]
    return sig


def _brownian_expected_signature(noise_dim, depth):
    """``E[S(t, B)] = exp(e_0 + ½ Σ e_i ⊗ e_i)`` on ``[0, 1]``, letter 0 being time.

    Lyons & Victoir (2004), Proposition 4.10.
    """
    dim = noise_dim + 1
    gen = [np.zeros(())] + [np.zeros((dim,) * k) for k in range(1, depth + 1)]
    gen[1][0] = 1.0
    for i in range(1, dim):
        gen[2][i, i] = 0.5
    out = [np.ones(())] + [np.zeros((dim,) * k) for k in range(1, depth + 1)]
    term = [t.copy() for t in out]
    for n in range(1, depth + 1):
        term = [
            sum(np.multiply.outer(term[i], gen[k - i]) for i in range(k + 1)) / n
            for k in range(depth + 1)
        ]
        out = [a + b for a, b in zip(out, term)]
    return out


@pytest.mark.parametrize("noise_dim", [1, 2])
@pytest.mark.parametrize("lyons_degree", [3, 5])
def test_lv_paths_match_brownian_expected_signature(noise_dim, lyons_degree):
    """Degree-m Wiener cubature matches E[S(t, B)] on every word of degree <= m.

    The time letter counts twice towards the degree. The grid holds every multiple
    of 1/12, so the path corners at j/3 (d=1) and j/4 (d=2) are sampled exactly.
    """
    ts = np.linspace(0.0, 1.0, 13)
    paths, weights = build_lyons_victoir_multid_paths(
        ts, noise_dim, lyons_degree=lyons_degree, k=1
    )
    time = np.broadcast_to(ts[None, :, None], (len(weights), len(ts), 1))
    sigs = [_signature(p, lyons_degree) for p in np.concatenate([time, paths], axis=2)]
    expected = _brownian_expected_signature(noise_dim, lyons_degree)
    for level in range(1, lyons_degree + 1):
        cubature = sum(w * s[level] for w, s in zip(weights, sigs))
        for word in itertools.product(range(noise_dim + 1), repeat=level):
            if level + word.count(0) <= lyons_degree:
                assert cubature[word] == pytest.approx(
                    expected[level][word], abs=1e-9
                ), word


def test_oned_deg5_still_three_paths():
    paths, w = _unit_paths(1, method="auto", lyons_degree=5, k=1)
    assert paths.shape[0] == 3
    assert paths.shape[2] == 1
    np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-12)


def test_lv_m5_matches_levy_area_sde_second_moment():
    """dx¹=dW¹, dx²=x¹ dW² ⇒ E[(x²_δ)²]=δ²/2 is a degree-4 tensor LV m=5 hits."""
    delta = 0.4

    def ito_drift(t, x):
        del t
        return jnp.zeros_like(x)

    def diffusion(t, x):
        del t
        return jnp.array([[1.0, 0.0], [0.0, x[0]]], dtype=x.dtype)

    strat_drift, strat_l = ito_to_stratonovich_dynamics(ito_drift, diffusion)
    ts, paths, weights = precompute_cubature_paths(
        horizon=delta,
        noise_dim=2,
        integration_dt=delta / 64,
        cubature_order=1,
        multidim_method="lv",
        lyons_degree=5,
    )
    mean = jnp.zeros(2)
    chol = jnp.zeros((2, 2))
    m_pred, chol_pred, _, _ = predict(
        mean, chol, ts, paths, weights, strat_drift, strat_l
    )
    p = np.asarray(chol_pred @ chol_pred.T)
    m = np.asarray(m_pred)
    second = float(p[1, 1] + m[1] ** 2)
    np.testing.assert_allclose(second, 0.5 * delta**2, atol=1e-10)


def test_lv_m5_weak_order_on_nonnilpotent_sde():
    """dx¹=dW¹, dx²=sin(x¹) dW² from (1, 0): LV m=5 has local weak error O(δ³).

    Unlike dx²=x¹ dW², this SDE is not nilpotent, so every level of the expected
    signature enters E[(x²_δ)²] = δ/2 − cos(2)(1 − e^{−2δ})/4, and paths that miss
    level-4 words give O(δ²) instead. Itô and Stratonovich forms coincide here.
    """

    def drift(t, x):
        del t
        return jnp.zeros_like(x)

    def diffusion(t, x):
        del t
        return jnp.array([[1.0, 0.0], [0.0, jnp.sin(x[0])]], dtype=x.dtype)

    def second_moment_error(delta):
        ts, paths, weights = precompute_cubature_paths(
            horizon=delta,
            noise_dim=2,
            integration_dt=delta / 128,
            cubature_order=1,
            multidim_method="lv",
            lyons_degree=5,
        )
        m_pred, chol_pred, _, _ = predict(
            jnp.array([1.0, 0.0]), jnp.zeros((2, 2)), ts, paths, weights, drift, diffusion
        )
        second = float((chol_pred @ chol_pred.T)[1, 1] + m_pred[1] ** 2)
        exact = delta / 2 - np.cos(2.0) * (1.0 - np.exp(-2.0 * delta)) / 4
        return abs(second - exact)

    ratio = second_moment_error(0.1) / second_moment_error(0.05)
    assert 6.5 < ratio < 9.5  # 2³ = 8
