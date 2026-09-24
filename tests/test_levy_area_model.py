"""Lévy-area SDE: TME-2 moments exact; degree-3 cubature misses x²."""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "heston_test"))

from levy_area_model import exact_char, exact_second_moment, gaussian_char, ito_L, ito_drift
from tme import tme_mean_cov
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict


def test_exact_char_is_cameron_martin_not_gaussian():
    delta, omega = 0.5, 3.0
    true = exact_char(delta, omega)
    gauss = gaussian_char(delta, omega)
    assert abs(true - np.cosh(omega * delta) ** -0.5) < 1e-15
    assert abs(true - gauss) > 0.05
    assert 0.6 < true < 0.7


def test_tme2_second_moment_is_exact():
    delta = 0.5
    m, P = tme_mean_cov(np.zeros(2), delta, 2, ito_drift, ito_L)
    assert abs(m[0]) < 1e-12
    assert abs(m[1]) < 1e-12
    assert abs(P[1, 1] - exact_second_moment(delta)) < 1e-12


def test_wsc_m3_from_origin_has_vanishing_x2():
    delta = 0.5
    strat_d, strat_L = ito_to_stratonovich_dynamics(ito_drift, ito_L)
    ts, paths, w = precompute_cubature_paths(
        horizon=delta,
        noise_dim=2,
        integration_dt=delta / 32.0,
        cubature_order=1,
        multidim_method="lv",
        lyons_degree=3,
    )
    _m, _c, samples, weights = predict(
        jnp.zeros(2), jnp.zeros((2, 2)), ts, paths, w, strat_d, strat_L
    )
    x2 = np.asarray(samples)[:, 1]
    wt = np.asarray(weights)
    wt = wt / wt.sum()
    assert float(np.max(np.abs(x2))) < 1e-9
    assert abs(float(np.sum(wt * x2**2))) < 1e-18


def test_wsc_m5_matches_second_moment():
    delta = 0.5
    strat_d, strat_L = ito_to_stratonovich_dynamics(ito_drift, ito_L)
    ts, paths, w = precompute_cubature_paths(
        horizon=delta,
        noise_dim=2,
        integration_dt=delta / 32.0,
        cubature_order=1,
        multidim_method="lv",
        lyons_degree=5,
    )
    _m, _c, samples, weights = predict(
        jnp.zeros(2), jnp.zeros((2, 2)), ts, paths, w, strat_d, strat_L
    )
    x2 = np.asarray(samples)[:, 1]
    wt = np.asarray(weights)
    wt = wt / wt.sum()
    second = float(np.sum(wt * x2**2))
    assert abs(second - exact_second_moment(delta)) < 1e-8
