"""Tests for Ito ↔ Stratonovich drift conversion helpers."""

import jax.numpy as jnp

from crf.library.conversions import ito_to_stratonovich_drift


def test_constant_diffusion_leaves_drift_unchanged():
    """When G does not depend on x, the Stratonovich correction vanishes."""

    def ito_drift(t, x):
        del t
        return -0.5 * x

    def diffusion(t, x):
        del t, x
        return jnp.array([[0.3]])

    x = jnp.array([1.2])
    f_ito = ito_drift(0.0, x)
    f_strato = ito_to_stratonovich_drift(ito_drift, diffusion)(0.0, x)
    assert jnp.allclose(f_ito, f_strato)


def test_multiplicative_noise_applies_half_variation_correction():
    """For G(x) = sigma * x, f_S = f_I - 0.5 * sigma^2 * x."""

    sigma = 0.4

    def ito_drift(t, x):
        del t
        return jnp.array([0.2 - 0.7 * x[0]])

    def diffusion(t, x):
        del t
        return sigma * x.reshape(1, 1)

    x = jnp.array([0.5])
    f_ito = ito_drift(0.0, x)
    f_strato = ito_to_stratonovich_drift(ito_drift, diffusion)(0.0, x)
    expected = f_ito - 0.5 * (sigma**2) * x
    assert jnp.allclose(f_strato, expected)
