"""Exact references and simulator for the area SDEs in experiments/heston_test/area_models.py."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HESTON = os.path.join(os.path.dirname(__file__), "..", "experiments", "heston_test")
if _HESTON not in sys.path:
    sys.path.insert(0, _HESTON)

import jax.numpy as jnp
from area_models import LEVY_AREA, SIN_AREA, _sin_char
from wsckf.library import ito_to_stratonovich_dynamics


@pytest.mark.parametrize("delta", [0.05, 0.5, 2.0])
def test_sin_char_small_omega_matches_second_moment(delta):
    """E[cos(ω x²)] = 1 − ω² E[(x²)²]/2 + O(ω⁴) ties the Feynman–Kac solve to the closed form."""
    omega = 1e-3
    implied = 2.0 * (1.0 - SIN_AREA.exact_char(delta, omega)) / omega**2
    assert implied == pytest.approx(SIN_AREA.exact_second_moment(delta), rel=1e-5)


def test_sin_char_converged_in_fourier_modes():
    """Truncation error is below expm rounding (~1e-14) already at 10 modes."""
    assert _sin_char(0.8, 3.0, 1.0, n_modes=10) == pytest.approx(
        _sin_char(0.8, 3.0, 1.0, n_modes=80), abs=1e-12
    )


def test_sin_char_matches_monte_carlo():
    """Conditionally Gaussian Monte Carlo: E[exp(−ω²/2 ∫ sin²(1 + W_s) ds)]."""
    delta, omega, n_paths, n_steps = 0.5, 3.0, 20_000, 400
    rng = np.random.default_rng(0)
    dt = delta / n_steps
    w = 1.0 + np.cumsum(rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps)), axis=1)
    w = np.concatenate([np.ones((n_paths, 1)), w], axis=1)
    quad_var = np.trapezoid(np.sin(w) ** 2, dx=dt, axis=1)
    samples = np.exp(-0.5 * omega**2 * quad_var)
    se = samples.std(ddof=1) / np.sqrt(n_paths)
    assert abs(samples.mean() - SIN_AREA.exact_char(delta, omega)) < 4.0 * se


@pytest.mark.parametrize("model", [LEVY_AREA, SIN_AREA], ids=lambda m: m.name)
def test_ito_and_stratonovich_forms_coincide(model):
    strat_drift, _ = ito_to_stratonovich_dynamics(model.ito_drift, model.ito_L)
    for x in ([0.3, -1.2], [1.0, 0.0], [-2.0, 0.7]):
        np.testing.assert_allclose(strat_drift(0.0, jnp.asarray(x)), 0.0, atol=1e-15)


def test_levy_simulator_matches_sequential_euler_maruyama():
    """The vectorised simulator reproduces the original step-by-step Itô loop exactly."""
    horizon, n_fine, stride, seed, omega, meas_std = 4.0, 512, 64, 7, 3.0, 0.25
    true_fine, ys, _ = LEVY_AREA.simulate(
        horizon=horizon, n_fine=n_fine, omega=omega, meas_std=meas_std,
        seed=seed, obs_stride=stride,
    )
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(horizon / n_fine), size=(n_fine, 2))
    x = np.zeros(2)
    xs = [x.copy()]
    for i in range(n_fine):
        x = x + np.array([dW[i, 0], x[0] * dW[i, 1]])
        xs.append(x.copy())
    xs = np.stack(xs)
    obs_idx = np.arange(stride, n_fine + 1, stride)
    ys_ref = np.cos(omega * xs[obs_idx, 1])[:, None] + rng.normal(
        0.0, meas_std, size=(obs_idx.shape[0], 1)
    )
    np.testing.assert_array_equal(true_fine, xs)
    np.testing.assert_array_equal(ys, ys_ref)
