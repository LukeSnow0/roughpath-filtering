"""Compare cuthbert WSC-KF vs cd_dynamax-style filters on the Nielsen four-tank (MFTS) benchmark.

Mirrors ``wsckf_vs_dynamax/run_comparison.py`` but uses the modified four-tank system from
Nielsen et al. (2022), Sec. IV (Azam 2015 geometry).

Run:  python experiments/mfts_test/run_comparison.py
      python experiments/mfts_test/run_comparison.py --solver rk4
      python experiments/mfts_test/run_comparison.py --solver diffrax
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
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from experiments.cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)
from cuthbert import filter as run_filter  
from wsckf.library import default_recombined_num_paths  

from wsckf import build_filter, WSCKFState

from mfts_model import (
    MFTSParams,
    betas,
    init_estimate,
    make_mfts_dynamics,
    mape_pct,
    simulate_mfts,
)
from plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    METHOD_MARKERS,
    PALETTE,
    SINGLE_COLUMN_WIDTH,
    save_publication_figure,
    style_box_axis,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Nielsen MFTS benchmark settings (defaults from nielsen_mfts_filter_comparison.py)
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="WSC-KF vs cd_dynamax filters on Nielsen MFTS")
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
    help="WSC-KF Lyons macro-segment count k (Wiener cubature order)",
)
_parser.add_argument(
    "--solver",
    choices=("rk4", "diffrax"),
    default="rk4",
    help="Continuous-time integrator for CD filters (default: rk4, matched to WSC-KF)",
)
_args = _parser.parse_args()
SEED = _args.seed
CUBATURE_ORDER = _args.cubature_order
SOLVER = _args.solver


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
WSCKF_NUM_PATHS = default_recombined_num_paths(2, CUBATURE_ORDER)

F1, F2 = 5.0, 5.0
N_TIMING = 5

COLORS = {
    name: METHOD_COLORS[name]
    for name in ("CD-EKF", "CD-UKF", "CD-EnKF", "PF", "WSC-KF")
}
LINESTYLES = {
    "CD-EKF": (0, (3.0, 1.4)),
    "CD-UKF": (0, (1.0, 1.2)),
    "CD-EnKF": "-.",
    "PF": (0, (5.0, 1.4)),
    "WSC-KF": "-",
}


def plot_method(ax, x, y, name, *, markevery=15, label=None):
    """Draw one filter with a stable color/line/marker encoding."""

    return ax.plot(
        x,
        y,
        color=COLORS[name],
        linestyle=LINESTYLES[name],
        linewidth=1.7 if name == "WSC-KF" else 1.15,
        marker=METHOD_MARKERS[name],
        markersize=2.6,
        markerfacecolor="white",
        markeredgewidth=0.65,
        markevery=markevery,
        label=name if label is None else label,
        zorder=3 if name == "WSC-KF" else 2,
    )


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
# cuthbert WSC-KF (EnKF/PF/WSC-KF dynamics: lambda > 0 on disturbance states)
# ---------------------------------------------------------------------------
wsckf_drift, wsckf_L, _, _, _ = make_mfts_dynamics(
    p, lambda_dist=LAMBDA_ENKF_PF, sigma_dist=SIGMA_EKF_ENKF_PF, bar_F3=bar_est, bar_F4=bar_est, F1=F1, F2=F2
)

wsckf = build_filter(
    get_init_params=lambda mi: (m0, chol_P0),
    get_dynamics=lambda mi: (
        lambda t, x: wsckf_drift(t, x),
        lambda t, x: wsckf_L(t, x),
    ),
    get_observations=lambda mi: (obs_fn, chol_R, ys[mi - 1]),
    noise_dim=2,
    dt=dt,
    integration_dt=dt0,
    cubature_order=CUBATURE_ORDER,     # multidim_paths="recombined",
    num_paths=WSCKF_NUM_PATHS,
)
jitted_filter = jit(run_filter, static_argnames=["filter_obj"])
wsckf_states, wsckf_time = time_call(lambda: jitted_filter(wsckf, model_inputs))
wsckf_est = stack_estimates(m0, wsckf_states.mean[1:])
wsckf_loglik = wsckf_states.log_normalizing_constant[1:]


# ---------------------------------------------------------------------------
# cd_dynamax-style filters
# ---------------------------------------------------------------------------
# Moment-ODE predict (state_order="first") is stable on this plant benchmark.
# Default ``--solver rk4`` matches the WSC-KF integrator for fair wall-clock timing.
ekf_jit = jit(
    lambda: run_cd_ekf(
        drift_ekf, obs_fn, L_ekf, Qc, R, m0, P0_j, ys, dt, dt0,
        state_order="first", solver=SOLVER,
    )
)
ekf_res, ekf_time = time_call(ekf_jit)
ekf_est = stack_estimates(m0, ekf_res.means)

ukf_jit = jit(
    lambda: run_cd_ukf(
        drift_ukf, obs_fn, L_ukf, Qc, R, m0, P0_j, ys, dt, dt0, UKFParams(),
        state_order="first", solver=SOLVER,
    )
)
ukf_res, ukf_time = time_call(ukf_jit)
ukf_est = stack_estimates(m0, ukf_res.means)

key = random.key(SEED)
enkf_jit = jit(
    lambda k: run_cd_enkf(
        drift_enkf, obs_fn, L_enkf, Qc, R, m0, P0_j, ys, dt, k,
        n_particles=ENKF_PARTICLES, noise_dim=2, dt0=dt0, solver=SOLVER,
    )
)
enkf_res, enkf_time = time_call(enkf_jit, key)
enkf_est = stack_estimates(m0, enkf_res.means)

key, pf_key = random.split(key)
pf_jit = jit(
    lambda k: run_bootstrap_pf(
        drift_enkf, obs_fn, L_enkf, Qc, R, m0, P0_j, ys, dt, k,
        n_particles=PF_PARTICLES, noise_dim=2, dt0=dt0, solver=SOLVER,
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
    "WSC-KF": dict(est=wsckf_est, loglik=wsckf_loglik, time=wsckf_time, color=COLORS["WSC-KF"]),
}

for name, r in RESULTS.items():
    r["mapex"] = mape_pct(r["est"], true_traj, slice(0, 4))
    r["maped"] = mape_pct(r["est"], true_traj, slice(4, 6))


# ---------------------------------------------------------------------------
# Plot 1: tank heights [cm]
# ---------------------------------------------------------------------------
true_h = true_traj[:, :4] * beta
fig, axes = plt.subplots(
    2,
    2,
    figsize=(DOUBLE_COLUMN_WIDTH, 3.65),
    sharex=True,
)
for i, ax in enumerate(axes.ravel()):
    ax.plot(
        times_min,
        true_h[:, i],
        color=PALETTE["black"],
        linestyle="--",
        linewidth=1.35,
        label="True",
        zorder=4,
    )
    for name, r in RESULTS.items():
        est_h = r["est"][:, :4] * beta
        plot_method(ax, times_min, est_h[:, i], name)
    ax.set_ylabel(rf"$h_{i + 1}$ [cm]")
    style_curve_axis(ax, grid_axis="y")
axes[1, 0].set_xlabel("Time [min]")
axes[1, 1].set_xlabel("Time [min]")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=len(labels),
)
fig.subplots_adjust(left=0.082, right=0.992, bottom=0.12, top=0.89, wspace=0.18, hspace=0.12)
save_publication_figure(fig, os.path.join(OUTPUT_DIR, "heights.png"))
plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: disturbance flows F3, F4
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    2,
    figsize=(DOUBLE_COLUMN_WIDTH, 2.25),
    sharex=True,
)
for ax, idx, label in zip(axes, [4, 5], ["$F_3$", "$F_4$"]):
    ax.plot(
        times_min,
        true_traj[:, idx],
        color=PALETTE["black"],
        linestyle="--",
        linewidth=1.35,
        label="True",
        zorder=4,
    )
    ax.axhline(
        p.F_nominal_est,
        color=PALETTE["neutral_gray"],
        linestyle=":",
        linewidth=1.0,
        label=r"Nominal $\bar F$",
        zorder=1,
    )
    for name in RESULTS:
        plot_method(ax, times_min, RESULTS[name]["est"][:, idx], name)
    ax.set_ylabel(rf"{label} [cm$^3$ s$^{{-1}}$]")
    style_curve_axis(ax, grid_axis="y")
axes[0].set_xlabel("Time [min]")
axes[1].set_xlabel("Time [min]")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=len(labels),
)
fig.subplots_adjust(left=0.085, right=0.992, bottom=0.19, top=0.81, wspace=0.2)
save_publication_figure(fig, os.path.join(OUTPUT_DIR, "disturbances.png"))
plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Nielsen MAPE bars
# ---------------------------------------------------------------------------
names = list(RESULTS.keys())
xpos = np.arange(2)
width = 0.8 / len(names)
fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.25))
for j, name in enumerate(names):
    vals = [RESULTS[name]["mapex"], RESULTS[name]["maped"]]
    ax.bar(
        xpos + (j - (len(names) - 1) / 2) * width,
        vals,
        width,
        label=name,
        color=COLORS[name],
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
ax.set_xticks(xpos)
ax.set_xticklabels(["Tank states", "Disturbance states"])
ax.set_ylabel("MAPE [%]")
ax.set_ylim(bottom=0.0)
style_box_axis(ax, grid_axis="y")
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
)
fig.subplots_adjust(left=0.16, right=0.985, bottom=0.19, top=0.76)
save_publication_figure(fig, os.path.join(OUTPUT_DIR, "mape.png"))
plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: cumulative log-likelihood
# ---------------------------------------------------------------------------
obs_times_min = np.arange(1, N_OBS + 1) * dt_obs_min
fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.3))
for name in RESULTS:
    plot_method(
        ax,
        obs_times_min,
        RESULTS[name]["loglik"],
        name,
        markevery=max(1, N_OBS // 8),
    )
ax.set_xlabel("Time [min]")
ax.set_ylabel("Cumulative log likelihood [nats]")
style_curve_axis(ax, grid_axis="y")
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
)
fig.subplots_adjust(left=0.19, right=0.985, bottom=0.18, top=0.75)
save_publication_figure(fig, os.path.join(OUTPUT_DIR, "loglik.png"))
plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: runtime (log scale)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.05))
runtime_values = np.asarray([RESULTS[n]["time"] for n in names])
ypos = np.arange(len(names))
for y, name, value in zip(ypos, names, runtime_values):
    ax.plot(
        value,
        y,
        color=COLORS[name],
        marker=METHOD_MARKERS[name],
        linestyle="none",
        markersize=5.2,
        zorder=2,
    )
ax.set_yticks(ypos)
ax.set_yticklabels(names)
ax.set_xscale("log")
ax.set_xlabel("Wall-clock time [s]")
ax.invert_yaxis()
style_curve_axis(ax, grid_axis="x")
for y, value in zip(ypos, runtime_values):
    ax.text(
        value * 1.08,
        y,
        f"{value:.2g}",
        va="center",
        ha="left",
        fontsize=6.8,
    )
left, right = ax.get_xlim()
ax.set_xlim(left, right * 1.45)
fig.subplots_adjust(left=0.25, right=0.97, bottom=0.24, top=0.98)
save_publication_figure(fig, os.path.join(OUTPUT_DIR, "timing.png"))
plt.close(fig)


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
print(f"WSC-KF Wiener paths P={WSCKF_NUM_PATHS}, cubature order k={CUBATURE_ORDER}")
print(f"Plots written to {OUTPUT_DIR}")
