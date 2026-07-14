"""Compare the Cubature Recursive Filter (CRF) against cd_dynamax-style
continuous-discrete filters (CD-EKF, CD-UKF, CD-EnKF) and a bootstrap particle filter,
on a stochastic Lorenz-63 tracking problem.

- The CRF is built with the filter structure
  (``crf.filter.cubature_filter.build_filter``)
- The competing filters are cd_dynamax-style continuous-discrete nonlinear-Gaussian
  filters (see ``experiments/cd_dynamax_mirror/filters.py``). Default ``--solver rk4``
  matches the CRF integrator for fair wall-clock timing; use ``--solver diffrax``
  for Dopri5 / Heun+VBT.

Run:  python3 experiments/l63_test/run_comparison.py
      python3 experiments/l63_test/run_comparison.py --solver rk4
      python3 experiments/l63_test/run_comparison.py --solver diffrax
Outputs (plots + metrics) are written to ``experiments/l63_test/output/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

import jax
import jax.numpy as jnp
import matplotlib
from jax import jit, random

matplotlib.use("Agg")
import matplotlib.pyplot as plt  

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPERIMENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

sys.path.insert(0, EXPERIMENTS_DIR)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from cd_dynamax_mirror.filters import (  
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)
from cuthbert import filter as run_filter  
from crf import build_filter, CRFState


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_parser = argparse.ArgumentParser(
    description="CRF vs cd_dynamax filters on stochastic Lorenz-63"
)
_parser.add_argument(
    "--solver",
    choices=("rk4", "diffrax"),
    default="rk4",
    help="Continuous-time integrator for CD filters (default: rk4, matched to CRF)",
)
_args = _parser.parse_args()
SOLVER = _args.solver

# ---------------------------------------------------------------------------
# Lorenz-63 stochastic model (matching Cuthbert_CRF.py)
# ---------------------------------------------------------------------------
lorenz_sigma = 10.0
lorenz_rho = 28.0
lorenz_beta = 8.0 / 3.0

dt_inner = 0.01
n_inner_steps = 35
dt = dt_inner * n_inner_steps  # assimilation interval (0.35)
diff_std = 1.0
obs_std = 0.3

x_dim = 3
y_dim = 1
noise_dim = x_dim


def lorenz_drift(x):
    return jnp.array(
        [
            lorenz_sigma * (x[1] - x[0]),
            x[0] * (lorenz_rho - x[2]) - x[1],
            x[0] * x[1] - lorenz_beta * x[2],
        ]
    )


def lorenz_step(x):
    """Advance one assimilation interval with Heun integration (ground-truth model)."""

    def body(_, x_curr):
        k1 = lorenz_drift(x_curr)
        x_pred = x_curr + dt_inner * k1
        k2 = lorenz_drift(x_pred)
        return x_curr + 0.5 * dt_inner * (k1 + k2)

    return jax.lax.fori_loop(0, n_inner_steps, body, x)


Q = (diff_std**2 * dt) * jnp.eye(x_dim)
chol_Q = jnp.linalg.cholesky(Q)
R = (obs_std**2) * jnp.eye(y_dim)
chol_R = jnp.linalg.cholesky(R)

H = jnp.array([[1.0, 0.0, 0.0]])
d_obs = jnp.zeros(y_dim)


# Continuous-time model functions for the cd_dynamax-style filters.
def f_drift(t, x):
    return lorenz_drift(x)


def h_obs(x):
    return H @ x + d_obs


def L_fn(t, x):
    return diff_std * jnp.eye(x_dim)


def Qc_fn(t, x):
    return jnp.eye(x_dim)


# ---------------------------------------------------------------------------
# Simulate ground truth + observations
# ---------------------------------------------------------------------------
num_time_steps = 200
key = random.key(0)

x = jnp.array([1.0, 1.0, 1.0])
for _ in range(1_000):  # spin up onto the attractor
    key, dyn_key = random.split(key)
    x = lorenz_step(x) + chol_Q @ random.normal(dyn_key, (x_dim,))

m0 = x
P0 = 2.0 * jnp.eye(x_dim)
chol_P0 = jnp.linalg.cholesky(P0)

xs, ys_list = [], []
for _ in range(num_time_steps):
    key, dyn_key, obs_key = random.split(key, 3)
    x = lorenz_step(x) + chol_Q @ random.normal(dyn_key, (x_dim,))
    y = H @ x + d_obs + chol_R @ random.normal(obs_key, (y_dim,))
    xs.append(x)
    ys_list.append(y)

true_states = jnp.stack(xs)
ys = jnp.stack(ys_list)
times = jnp.arange(num_time_steps) * dt
model_inputs = jnp.arange(num_time_steps + 1)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
N_TIMING = 10


def time_call(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return out, float(jnp.median(jnp.array(samples)))


# ---------------------------------------------------------------------------
#CRF
# ---------------------------------------------------------------------------
crf = build_filter(
    get_init_params=lambda mi: (m0, chol_P0),
    get_dynamics=lambda mi: (
        lambda t, x: lorenz_drift(x),
        lambda t, x: diff_std * jnp.eye(x_dim),
    ),
    get_observations=lambda mi: (lambda x: H @ x + d_obs, chol_R, ys[mi - 1]),
    noise_dim=noise_dim,
    dt=dt,
    integration_dt=dt_inner,
    cubature_order=2, #    multidim_paths="recombined",
    num_paths=18,
)
jitted_filter = jit(run_filter, static_argnames=["filter_obj"])
crf_states, crf_time = time_call(lambda: jitted_filter(crf, model_inputs))
crf_means = crf_states.mean[1:]
crf_loglik = crf_states.log_normalizing_constant[1:]


# ---------------------------------------------------------------------------
# cd_dynamax-style filters
# ---------------------------------------------------------------------------
# The flow-map predict (dynamax's discrete EKF/UKF applied to the SDE flow over each
# interval) is used because the continuous moment-ODE variants are unstable on this
# chaotic, sparsely observed, long-interval problem: the moment-ODE UKF diverges and
# the moment-ODE EKF has no skill. See README for details.

ekf_jit = jit(
    lambda: run_cd_ekf(
        f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt_inner,
        state_order="flowmap", solver=SOLVER,
    )
)
ekf_res, ekf_time = time_call(ekf_jit)

ukf_jit = jit(
    lambda: run_cd_ukf(
        f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt_inner, UKFParams(),
        state_order="flowmap",
        solver=SOLVER,
    )
)
ukf_res, ukf_time = time_call(ukf_jit)

key, enkf_key = random.split(key)
enkf_jit = jit(
    lambda k: run_cd_enkf(
        f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, k,
        n_particles=50, noise_dim=noise_dim, dt0=dt_inner, inflation_delta=0.02,
        solver=SOLVER,
    )
)
enkf_res, enkf_time = time_call(enkf_jit, enkf_key)

key, pf_key = random.split(key)
pf_jit = jit(
    lambda k: run_bootstrap_pf(
        f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, k,
        n_particles=200, noise_dim=noise_dim, dt0=dt_inner, solver=SOLVER,
    )
)
pf_res, pf_time = time_call(pf_jit, pf_key)


# ---------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------
RESULTS = {
    "CD-EKF": dict(means=ekf_res.means, loglik=ekf_res.cum_loglik, time=ekf_time, color="#2E86AB"),
    "CD-UKF": dict(means=ukf_res.means, loglik=ukf_res.cum_loglik, time=ukf_time, color="#3DA35D"),
    "CD-EnKF": dict(means=enkf_res.means, loglik=enkf_res.cum_loglik, time=enkf_time, color="#F6AE2D"),
    "PF": dict(means=pf_res.means, loglik=pf_res.cum_loglik, time=pf_time, color="#F24236"),
    "CRF": dict(means=crf_means, loglik=crf_loglik, time=crf_time, color="#6A4C93"),
}


def rmse(estimates, truth):
    return jnp.sqrt(jnp.nanmean((estimates - truth) ** 2, axis=0))


for name, r in RESULTS.items():
    r["rmse"] = rmse(r["means"], true_states)


# ---------------------------------------------------------------------------
# Plot 1: state trajectories
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
dim_labels = ["$x_1$ (observed)", "$x_2$ (unobserved)", "$x_3$ (unobserved)"]
for i, ax in enumerate(axes):
    ax.plot(times, true_states[:, i], "k-", linewidth=1.2, label="True", alpha=0.7)
    if i == 0:
        ax.scatter(times, ys[:, 0], s=6, color="gray", alpha=0.3, label="Obs", zorder=1)
    for name, r in RESULTS.items():
        ax.plot(times, r["means"][:, i], color=r["color"], linewidth=1.0, label=name, alpha=0.9)
    ax.set_ylabel(dim_labels[i], fontsize=12)
    ax.grid(True, alpha=0.2)
axes[0].legend(loc="upper right", fontsize=9, ncol=6)
axes[2].set_xlabel("Time", fontsize=13)
fig.suptitle("CRF vs cd_dynamax filters — Lorenz-63 tracking", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "trajectories.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Plot 2: RMSE bars
# ---------------------------------------------------------------------------
component_names = ["$x_1$", "$x_2$", "$x_3$", "Overall"]
xpos = jnp.arange(len(component_names))
names = list(RESULTS.keys())
width = 0.8 / len(names)

fig, ax = plt.subplots(figsize=(9, 4.5))
for j, name in enumerate(names):
    vals = [*RESULTS[name]["rmse"].tolist(), float(jnp.mean(RESULTS[name]["rmse"]))]
    ax.bar(xpos + (j - (len(names) - 1) / 2) * width, vals, width, label=name, color=RESULTS[name]["color"])
ax.set_xticks(xpos)
ax.set_xticklabels(component_names, fontsize=12)
ax.set_ylabel("RMSE", fontsize=13)
ax.set_title("Per-component and overall RMSE", fontsize=14)
ax.legend(fontsize=10, ncol=5)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rmse.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Plot 3: cumulative log-likelihood
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
for name, r in RESULTS.items():
    ax.plot(times, r["loglik"], color=r["color"], linewidth=1.8, label=name)
ax.set_xlabel("Time", fontsize=13)
ax.set_ylabel("Cumulative log-likelihood - log scale", fontsize=13)
ax.set_title("Cumulative marginal log-likelihood (EnKF value is a Gaussian diagnostic)", fontsize=12)
ax.legend(fontsize=10, ncol=5)
ax.grid(True, alpha=0.2)
ax.set_yscale("symlog")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "loglik.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Plot 4: runtime
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(names, [RESULTS[n]["time"] for n in names], color=[RESULTS[n]["color"] for n in names])
ax.set_ylabel("Wall-clock time (s) - log scale", fontsize=13)
ax.set_title(
    f"Filter runtime (median of {N_TIMING}, excl. JIT; CD solver={SOLVER})",
    fontsize=12,
)
ax.grid(True, axis="y", alpha=0.3)
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "timing.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print(f"{'Filter':10s} {'RMSE(overall)':>14s} {'final LL':>12s} {'time (ms)':>12s}")
print("-" * 64)
for name, r in RESULTS.items():
    print(
        f"{name:10s} {float(jnp.mean(r['rmse'])):>14.4f} "
        f"{float(r['loglik'][-1]):>12.2f} {r['time'] * 1e3:>12.2f}"
    )
print("=" * 64)
print(f"Plots written to {OUTPUT_DIR}")
