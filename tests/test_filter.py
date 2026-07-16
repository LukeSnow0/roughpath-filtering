"""Tests for CRF construction and a small end-to-end filtering run."""

import jax.numpy as jnp
import pytest
from cuthbert import filter as run_filter
from jax import jit

from crf import build_filter


def _linear_gaussian_1d_filter(*, sde_form: str = "stratonovich"):
    """Minimal 1D OU + linear observation model (see examples/linear_gaussian_1d.py)."""
    kappa = 0.8
    sigma = 0.25
    obs_std = 0.15
    dt = 0.1
    noise_dim = 1

    m0 = jnp.array([0.0])
    chol_P0 = jnp.linalg.cholesky(jnp.array([[1.0]]))
    chol_R = jnp.array([[obs_std]])
    ys = jnp.array([0.12, 0.08, -0.05, 0.02, -0.11])

    def get_init_params(model_inputs):
        del model_inputs
        return m0, chol_P0

    def get_dynamics(model_inputs):
        del model_inputs

        def drift(t, x):
            del t
            return -kappa * x

        def diffusion(t, x):
            del t, x
            return sigma * jnp.ones((1, noise_dim))

        return drift, diffusion

    def get_observations(model_inputs):
        return lambda x: x, chol_R, ys[model_inputs - 1]

    return build_filter(
        get_init_params,
        get_dynamics,
        get_observations,
        noise_dim=noise_dim,
        dt=dt,
        integration_dt=0.02,
        cubature_order=2,
        sde_form=sde_form,
    ), ys


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"noise_dim": 0}, "noise_dim"),
        ({"dt": 0.0}, "dt"),
        ({"sde_form": "wrong"}, "sde_form"),
    ],
)
def test_build_filter_rejects_invalid_arguments(kwargs, match):
    base = dict(
        get_init_params=lambda mi: (jnp.array([0.0]), jnp.array([[1.0]])),
        get_dynamics=lambda mi: (lambda t, x: -x, lambda t, x: jnp.ones((1, 1))),
        get_observations=lambda mi: (lambda x: x, jnp.array([[0.1]]), jnp.array([0.0])),
        noise_dim=1,
        dt=0.1,
    )
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        build_filter(**base)


def test_linear_gaussian_1d_regression():
    """End-to-end CRF run matches the reference output from the example script."""
    filter_obj, ys = _linear_gaussian_1d_filter()
    model_inputs = jnp.arange(ys.shape[0] + 1)
    states = jit(run_filter, static_argnames=["filter_obj"])(filter_obj, model_inputs)

    final_mean = float(states.mean[-1, 0])
    final_var = float((states.chol_cov[-1] @ states.chol_cov[-1].T)[0, 0])
    final_loglik = float(states.log_normalizing_constant[-1])

    # References recomputed after fixing Wiener-path Brownian scaling
    # (ω_T = √T · ω_1) in ``precompute_cubature_paths``.
    assert final_mean == pytest.approx(-0.0257306255, rel=1e-5, abs=1e-5)
    assert final_var == pytest.approx(0.0082392945, rel=1e-4, abs=1e-5)
    assert final_loglik == pytest.approx(1.4505350276, rel=1e-3, abs=1e-3)
