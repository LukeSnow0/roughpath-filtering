"""Compare cuthbert CRF vs cd_dynamax-style filters on the Nielsen four-tank (MFTS) benchmark.

Mirrors ``crf_vs_dynamax/run_comparison.py`` but uses the modified four-tank system from
Nielsen et al. (2022), Sec. IV (Azam 2015 geometry).

Run:  python experiments/mfts_test/run_comparison.py
Outputs: ``experiments/mfts_test/output/``
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

EXPERIMENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

sys.path.insert(0, EXPERIMENTS_DIR)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import jit, random

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "crf_vs_dynamax"))

from experiments.cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)
from cuthbert import filter as run_filter  
from crf.filter import cubature_filter  
from crf.library import default_recombined_num_paths  

from mfts_model import (
    MFTSParams,
    betas,
    init_estimate,
    make_mfts_dynamics,
    mape_pct,
    simulate_mfts,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Nielsen MFTS benchmark settings (defaults from nielsen_mfts_filter_comparison.py)
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="CRF vs cd_dynamax filters on Nielsen MFTS")
_parser.add_argument(
    "--seed",
    type=int,
    default=100,
    help="RNG seed for truth simulation and stochastic filters (EnKF, PF)",
)
_parser.add_argument(
    "--cubature-order",
    type=int,
    default=3,
    dest="cubature_order",
    help="CRF Lyons macro-segment count k (Wiener cubature order)",
)
_args = _parser.parse_args()
SEED = _args.seed
CUBATURE_ORDER = _args.cubature_order


HORIZON_MIN = 30.0
N_OBS = 120
N_SUBSTEPS_TRUTH = 1000
N_SUBSTEPS_FILTER = 100

LAMBDA_SIM = 0.1
SIGMA_SIM = 5.0
LAMBDA_EKF_UKF = 0.0
LAMBDA_ENKF_PF = 2.0e-3
SIGMA_EKF_ENKF_PF = 5.0
SIGMA_UKF = 1.0

ENKF_PARTICLES = 250
PF_PARTICLES = 1000
CRF_NUM_PATHS = default_recombined_num_paths(2, CUBATURE_ORDER)

F1, F2 = 5.0, 5.0
N_TIMING = 5

COLORS = {
    "CD-EKF": "#2E86AB",
    "CD-UKF": "#3DA35D",
    "CD-EnKF": "#F6AE2D",
    "PF": "#F24236",
    "CRF": "#6A4C93",
}


# ---------------------------------------------------------------------------
# Prepare simulation + filter models
# ---------------------------------------------------------------------------
p = MFTSParams()
beta = betas(p)
dt_obs_min = HORIZON_MIN / N_OBS
dt = dt_obs_min * 60.0  # assimilation interval [s]
dt0 = dt / N_SUBSTEPS_FILTER

true_traj, ys_np, R = simulate_mfts(
    horizon_min=HORIZON_MIN,
    n_obs=N_OBS,
    n_substeps_truth=N_SUBSTEPS_TRUTH,
    p=p,
    lambda_sim=LAMBDA_SIM,
    sigma_sim=SIGMA_SIM,
    F1=F1,
    F2=F2,
    seed=SEED,
)
ys = jnp.asarray(ys_np)
x0, P0 = init_estimate(p, F1, F2, true_state=true_traj[0], cov_scale=1.0)
m0 = jnp.asarray(x0, dtype=jnp.float64)
P0_j = jnp.asarray(P0, dtype=jnp.float64)
chol_P0 = jnp.linalg.cholesky(P0_j)
chol_R = jnp.linalg.cholesky(R)

bar_est = lambda t_min, nom=p.F_nominal_est: nom

drift_ekf, L_ekf, Qc, obs_fn, _ = make_mfts_dynamics(
    p, lambda_dist=LAMBDA_EKF_UKF, sigma_dist=SIGMA_EKF_ENKF_PF, bar_F3=bar_est, bar_F4=bar_est, F1=F1, F2=F2
)
drift_ukf, L_ukf, _, _, _ = make_mfts_dynamics(
    p, lambda_dist=LAMBDA_EKF_UKF, sigma_dist=SIGMA_UKF, bar_F3=bar_est, bar_F4=bar_est, F1=F1, F2=F2
)
drift_enkf, L_enkf, _, _, _ = make_mfts_dynamics(
    p, lambda_dist=LAMBDA_ENKF_PF, sigma_dist=SIGMA_EKF_ENKF_PF, bar_F3=bar_est, bar_F4=bar_est, F1=F1, F2=F2
)

times_min = np.arange(N_OBS + 1) * dt_obs_min
model_inputs = jnp.arange(N_OBS + 1)


def time_call(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return out, float(np.median(samples))


def stack_estimates(m0_arr, means):
    """Prepend initial state so estimates align with true_traj (length N_OBS+1)."""
    return np.vstack([np.asarray(m0_arr), np.asarray(means)])


# ---------------------------------------------------------------------------
# cuthbert CRF (EnKF/PF/CRF dynamics: lambda > 0 on disturbance states)
# ---------------------------------------------------------------------------
crf_drift, crf_L, _, _, _ = make_mfts_dynamics(
    p, lambda_dist=LAMBDA_ENKF_PF, sigma_dist=SIGMA_EKF_ENKF_PF, bar_F3=bar_est, bar_F4=bar_est, F1=F1, F2=F2
)

crf = cubature_filter.build_filter(
    get_init_params=lambda mi: (m0, chol_P0),
    get_dynamics=lambda mi: (
        lambda t, x: crf_drift(t, x),
        lambda t, x: crf_L(t, x),
    ),
    get_observations=lambda mi: (obs_fn, chol_R, ys[mi - 1]),
    noise_dim=2,
    dt=dt,
    integration_dt=dt0,
    cubature_order=CUBATURE_ORDER,
    multidim_paths="recombined",
    num_paths=CRF_NUM_PATHS,
)
jitted_filter = jit(run_filter, static_argnames=["filter_obj"])
crf_states, crf_time = time_call(lambda: jitted_filter(crf, model_inputs))
crf_est = stack_estimates(m0, crf_states.mean[1:])
crf_loglik = crf_states.log_normalizing_constant[1:]


# ---------------------------------------------------------------------------
# cd_dynamax-style filters
# ---------------------------------------------------------------------------
# Moment-ODE predict (state_order="first") is stable on this plant benchmark.
ekf_jit = jit(
    lambda: run_cd_ekf(
        drift_ekf, obs_fn, L_ekf, Qc, R, m0, P0_j, ys, dt, dt0, state_order="first"
    )
)
ekf_res, ekf_time = time_call(ekf_jit)
ekf_est = stack_estimates(m0, ekf_res.means)

ukf_jit = jit(
    lambda: run_cd_ukf(
        drift_ukf, obs_fn, L_ukf, Qc, R, m0, P0_j, ys, dt, dt0, UKFParams(), state_order="first"
    )
)
ukf_res, ukf_time = time_call(ukf_jit)
ukf_est = stack_estimates(m0, ukf_res.means)

key = random.key(SEED)
enkf_jit = jit(
    lambda k: run_cd_enkf(
        drift_enkf, obs_fn, L_enkf, Qc, R, m0, P0_j, ys, dt, k,
        n_particles=ENKF_PARTICLES, noise_dim=2, dt0=dt0,
    )
)
enkf_res, enkf_time = time_call(enkf_jit, key)
enkf_est = stack_estimates(m0, enkf_res.means)

key, pf_key = random.split(key)
pf_jit = jit(
    lambda k: run_bootstrap_pf(
        drift_enkf, obs_fn, L_enkf, Qc, R, m0, P0_j, ys, dt, k,
        n_particles=PF_PARTICLES, noise_dim=2, dt0=dt0,
    )
)
pf_res, pf_time = time_call(pf_jit, pf_key)
pf_est = stack_estimates(m0, pf_res.means)


# ---------------------------------------------------------------------------
# Collect results (Nielsen MAPE metrics)
# ---------------------------------------------------------------------------
RESULTS = {
    "CD-EKF": dict(est=ekf_est, loglik=ekf_res.cum_loglik, time=ekf_time, color=COLORS["CD-EKF"]),
    "CD-UKF": dict(est=ukf_est, loglik=ukf_res.cum_loglik, time=ukf_time, color=COLORS["CD-UKF"]),
    "CD-EnKF": dict(est=enkf_est, loglik=enkf_res.cum_loglik, time=enkf_time, color=COLORS["CD-EnKF"]),
    "PF": dict(est=pf_est, loglik=pf_res.cum_loglik, time=pf_time, color=COLORS["PF"]),
    "CRF": dict(est=crf_est, loglik=crf_loglik, time=crf_time, color=COLORS["CRF"]),
}

for name, r in RESULTS.items():
    r["mapex"] = mape_pct(r["est"], true_traj, slice(0, 4))
    r["maped"] = mape_pct(r["est"], true_traj, slice(4, 6))


# ---------------------------------------------------------------------------
# Plot 1: tank heights [cm]
# ---------------------------------------------------------------------------
true_h = true_traj[:, :4] * beta
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for i, ax in enumerate(axes.ravel()):
    ax.plot(times_min, true_h[:, i], "k-", lw=1.2, label="True", alpha=0.8)
    for name, r in RESULTS.items():
        est_h = r["est"][:, :4] * beta
        ax.plot(times_min, est_h[:, i], color=r["color"], lw=1.0, label=name, alpha=0.9)
    ax.set_ylabel(f"$h_{i+1}$ [cm]")
    ax.grid(True, alpha=0.2)
axes[0, 0].legend(loc="upper right", fontsize=8, ncol=3)
axes[1, 0].set_xlabel("Time [min]")
axes[1, 1].set_xlabel("Time [min]")
fig.suptitle("MFTS tank heights — CRF vs cd_dynamax filters", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "heights.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Plot 2: disturbance flows F3, F4
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
for ax, idx, label in zip(axes, [4, 5], ["$F_3$", "$F_4$"]):
    ax.plot(times_min, true_traj[:, idx], "k-", lw=1.2, label="True")
    ax.axhline(p.F_nominal_est, color="0.55", ls="--", lw=1.0, label=r"Model $\bar F$=150")
    for name, r in RESULTS.items():
        ax.plot(times_min, r["est"][:, idx], color=r["color"], lw=1.0, label=name, alpha=0.9)
    ax.set_ylabel(f"{label} [cm³/s]")
    ax.grid(True, alpha=0.2)
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
axes[1].set_xlabel("Time [min]")
axes[0].set_xlabel("Time [min]")
fig.suptitle("Disturbance flows (unobserved states)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "disturbances.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Nielsen MAPE bars
# ---------------------------------------------------------------------------
names = list(RESULTS.keys())
xpos = np.arange(2)
width = 0.8 / len(names)
fig, ax = plt.subplots(figsize=(8, 4.5))
for j, name in enumerate(names):
    vals = [RESULTS[name]["mapex"], RESULTS[name]["maped"]]
    ax.bar(xpos + (j - (len(names) - 1) / 2) * width, vals, width, label=name, color=RESULTS[name]["color"])
ax.set_xticks(xpos)
ax.set_xticklabels(["MAPEx (tanks)", "MAPEd (disturbances)"], fontsize=11)
ax.set_ylabel("MAPE [%]")
ax.set_title("Nielsen MAPE metrics (eq. 51)")
ax.legend(fontsize=9, ncol=5)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "mape.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Plot 4: cumulative log-likelihood
# ---------------------------------------------------------------------------
obs_times_min = np.arange(1, N_OBS + 1) * dt_obs_min
fig, ax = plt.subplots(figsize=(10, 4))
for name, r in RESULTS.items():
    ax.plot(obs_times_min, r["loglik"], color=r["color"], lw=1.8, label=name)
ax.set_xlabel("Time [min]")
ax.set_ylabel("Cumulative log-likelihood")
ax.legend(fontsize=10, ncol=5)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "loglik.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Plot 5: runtime (log scale)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(names, [RESULTS[n]["time"] for n in names], color=[RESULTS[n]["color"] for n in names])
ax.set_yscale("log")
ax.set_ylabel("Wall-clock time (s, log scale)")
ax.set_title(f"Filter runtime (median of {N_TIMING}, excl. JIT)")
ax.grid(True, axis="y", which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "timing.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"{'Filter':10s} {'MAPEx%':>10s} {'MAPEd%':>10s} {'final LL':>12s} {'time (s)':>10s}")
print("-" * 72)
for name, r in RESULTS.items():
    print(
        f"{name:10s} {r['mapex']:>10.2f} {r['maped']:>10.2f} "
        f"{float(r['loglik'][-1]):>12.1f} {r['time']:>10.2f}"
    )
print("=" * 72)
print(f"CRF Wiener paths P={CRF_NUM_PATHS}, cubature order k={CUBATURE_ORDER}")
print(f"Plots written to {OUTPUT_DIR}")
