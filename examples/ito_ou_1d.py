"""CRF example with an Ito-specified model (automatic conversion via sde_form).

Ito model (1D, multiplicative noise):
    dx = (theta - kappa * x) dt + sigma * x dW

Run:
    python examples/ito_ou_1d.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from cuthbert import filter as run_filter
from jax import jit

from crf import build_filter

jax.config.update("jax_enable_x64", True)

theta = 0.2
kappa = 0.8
sigma = 0.25
obs_std = 0.12
dt = 0.1
noise_dim = 1

m0 = jnp.array([0.3])
chol_P0 = jnp.linalg.cholesky(jnp.array([[0.05**2]]))
chol_R = jnp.array([[obs_std]])

ys = jnp.array([0.28, 0.24, 0.19, 0.17, 0.15])
model_inputs = jnp.arange(ys.shape[0] + 1)


def get_init_params(model_inputs):
    del model_inputs
    return m0, chol_P0


def get_dynamics(model_inputs):
    del model_inputs

    def drift(t, x):
        del t
        return theta - kappa * x

    def diffusion(t, x):
        del t
        return sigma * x.reshape(1, noise_dim)

    return drift, diffusion


def get_observations(model_inputs):
    return lambda x: x, chol_R, ys[model_inputs - 1]


filter_obj = build_filter(
    get_init_params,
    get_dynamics,
    get_observations,
    noise_dim=noise_dim,
    dt=dt,
    integration_dt=0.02,
    cubature_order=2,
    multidim_paths="product",
    sde_form="ito",  # write your model in Ito form; conversion is automatic
)

states = jit(run_filter, static_argnames=["filter_obj"])(filter_obj, model_inputs)
print(f"Final mean:       {states.mean[-1]}")
print(f"Log-likelihood:   {float(states.log_normalizing_constant[-1]):.4f}")
