"""Explicit multi-D Lyons–Victoir cubature (not 1D-product)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

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


def test_precompute_lv_vs_product_support():
    paths_lv, w_lv = _unit_paths(2, method="lv", lyons_degree=5)
    paths_pr, w_pr = _unit_paths(2, method="product", lyons_degree=5)
    assert paths_lv.shape[0] == 18
    assert paths_pr.shape[0] == 9
    np.testing.assert_allclose(np.sum(w_lv), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(w_pr), 1.0, atol=1e-12)


def test_lv_m5_matches_brownian_second_moment():
    paths, w = _unit_paths(2, method="lv", lyons_degree=5, n_steps=80)
    term = paths[:, -1, :]
    mean = w @ term
    cov = (term - mean).T @ ((term - mean) * w[:, None])
    np.testing.assert_allclose(mean, 0.0, atol=1e-12)
    np.testing.assert_allclose(cov, np.eye(2), atol=1e-10)


def _levy_area_12(omega):
    a = 0.0
    for k in range(omega.shape[0] - 1):
        w0, w1 = omega[k], omega[k + 1]
        dw = w1 - w0
        mid = 0.5 * (w0 + w1)
        a += 0.5 * (mid[0] * dw[1] - mid[1] * dw[0])
    return a


def test_lv_m5_has_nonzero_levy_area_mean_zero():
    paths, w = _unit_paths(2, method="lv", lyons_degree=5, n_steps=100)
    areas = np.array([_levy_area_12(omega) for omega in paths])
    assert np.max(np.abs(areas)) > 0.05
    np.testing.assert_allclose(float(w @ areas), 0.0, atol=1e-12)
    np.testing.assert_allclose(float(w @ areas**2), 0.25, atol=5e-3)


def test_lv_d2_axis_swap_symmetry():
    """d=2 construction is equivariant: the two driving axes are interchangeable."""
    paths, w = _unit_paths(2, method="lv", lyons_degree=5, n_steps=120)
    dt = 1.0 / (paths.shape[1] - 1)
    qvar = np.array(
        [np.sum(0.5 * (p[:-1] ** 2 + p[1:] ** 2) * dt, axis=0) for p in paths]
    )
    eq = w @ qvar
    np.testing.assert_allclose(eq[0], eq[1], rtol=1e-8, atol=1e-10)
    areas = np.array([_levy_area_12(p) for p in paths])
    areas_swap = np.array([_levy_area_12(p[:, ::-1]) for p in paths])
    np.testing.assert_allclose(areas_swap, -areas, atol=1e-12)
    np.testing.assert_allclose(float(w @ qvar[:, 0]), 0.5, atol=5e-2)


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
