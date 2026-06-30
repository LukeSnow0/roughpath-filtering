"""Minimal CRF example: 1D Ornstein–Uhlenbeck process with linear observations.

Model (Stratonovich):
    dx = -kappa * x dt + sigma dW
    y_k = x_k + v_k,   v_k ~ N(0, obs_std^2)

Run from the repo root (after ``pip install -e .``):

    python examples/linear_gaussian_1d.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from cuthbert import filter as run_filter
from jax import jit

from crf import build_filter

jax.config.update("jax_enable_x64", True)

# Model parameters
kappa = 0.8
sigma = 0.25
obs_std = 0.15
dt = 0.1
noise_dim = 1

m0 = jnp.array([0.0])
P0 = jnp.array([[1.0]])
chol_P0 = jnp.linalg.cholesky(P0)
chol_R = jnp.array([[obs_std]])

# Synthetic observations (fixed for reproducibility)
ys = jnp.array([0.12, 0.08, -0.05, 0.02, -0.11])
num_steps = ys.shape[0]
model_inputs = jnp.arange(num_steps + 1)


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


filter_obj = build_filter(
    get_init_params,
    get_dynamics,
    get_observations,
    noise_dim=noise_dim,
    dt=dt,
    integration_dt=0.02,
    cubature_order=2,
)

states = jit(run_filter, static_argnames=["filter_obj"])(filter_obj, model_inputs)

final_mean = states.mean[-1]
final_chol = states.chol_cov[-1]
final_var = float((final_chol @ final_chol.T)[0, 0])
final_loglik = float(states.log_normalizing_constant[-1])

print(f"Final mean:       {final_mean}")
print(f"Final variance:   {final_var:.6f}")
print(f"Log-likelihood:   {final_loglik:.4f}")
