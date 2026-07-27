"""Sweep MFTS observation interval length and compare Nielsen MAPE across filters.

Simulates **one** shared fine-grid truth trajectory, then for each assimilation
interval ``Delta_t``:

1. Subsamples observations from that truth
2. Runs CD-EKF / CD-UKF / CD-EnKF / bootstrap PF / CRF
3. Reconstructs each filter's mean on the **same fine time grid** by integrating
   the filter drift between updates (filtered mean at observation times)
4. Scores MAPEx / MAPEd on that fine grid
5. Records predictive NLL (total and per observation) from each filter's cum. loglik
6. Times each filter (median wall-clock, excl. JIT; default CD solver=rk4)

Run:
  python experiments/mfts_test/run_mape_vs_interval.py
  python experiments/mfts_test/run_mape_vs_interval.py --solver rk4
  python experiments/mfts_test/run_mape_vs_interval.py --deltas 0.25,0.5,1,2,5,10
  python experiments/mfts_test/run_mape_vs_interval.py --eval-dt 0.05

Outputs:
  ``experiments/mfts_test/output/mape_vs_interval.png``
  ``experiments/mfts_test/output/nll_vs_interval.png``
  ``experiments/mfts_test/output/timing_vs_interval.png``
  ``experiments/mfts_test/output/mape_vs_interval.csv``
"""

from __future__ import annotations

import argparse
import csv
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
from crf.library import default_recombined_num_paths
from crf import build_filter

from mfts_model import (
    MFTSParams,
    init_estimate,
    make_mfts_dynamics,
    mape_pct,
    simulate_mfts,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(
    description="MAPE vs observation interval on Nielsen MFTS (fine-grid MAPE)"
)
_parser.add_argument(
    "--seed",
    type=int,
    default=100,
    help="RNG seed for truth simulation and stochastic filters",
)
_parser.add_argument(
    "--cubature-order",
    type=int,
    default=2,
    dest="cubature_order",
    help="CRF Lyons macro-segment count k (default 2; recombined)",
)
_parser.add_argument(
    "--num-paths",
    type=int,
    default=None,
    dest="num_paths",
    help="Post-recombination Wiener path budget (default: "
    "default_recombined_num_paths(2, k))",
)
_parser.add_argument(
    "--solver",
    choices=("rk4", "diffrax"),
    default="rk4",
    help="Continuous-time integrator for CD filters (default: rk4)",
)
_parser.add_argument(
    "--deltas",
    type=str,
    default="0.25,0.5,1,2,5,10",
    help="Comma-separated observation intervals Delta_t in minutes "
    "(each must be an integer multiple of --eval-dt)",
)
_parser.add_argument(
    "--eval-dt",
    type=float,
    default=0.05,
    dest="eval_dt",
    help="Fine evaluation / truth grid spacing in minutes (default 0.05)",
)
_parser.add_argument(
    "--n-timing",
    type=int,
    default=5,
    dest="n_timing",
    help="Timed repeats per filter after JIT warmup (median reported)",
)
_args = _parser.parse_args()

SEED = _args.seed
CUBATURE_ORDER = _args.cubature_order
SOLVER = _args.solver
DELTA_T_MIN_REQ = [float(x) for x in _args.deltas.split(",") if x.strip()]
EVAL_DT_MIN = float(_args.eval_dt)
N_TIMING = _args.n_timing

# ---------------------------------------------------------------------------
# Shared Nielsen MFTS settings (match run_comparison.py)
# ---------------------------------------------------------------------------

HORIZON_MIN = 30.0
# Absolute integration steps matching the default run (dt=15 s):
#   truth EM: 15/1000 = 0.015 s, filter RK4: 15/100 = 0.15 s
DT_TRUTH = 0.015
DT_FILTER = 0.15

LAMBDA_SIM = 0.1
SIGMA_SIM = 5.0
LAMBDA_EKF_UKF = 0.0
LAMBDA_ENKF_PF = 2.0e-3
SIGMA_EKF_ENKF_PF = 5.0
SIGMA_UKF = 1.0

ENKF_PARTICLES = 250
PF_PARTICLES = 1000
# Force recombined KLV (auto would pick product for small k, e.g. k=2 → 81 paths).
CRF_NUM_PATHS = (
    int(_args.num_paths)
    if _args.num_paths is not None
    else default_recombined_num_paths(2, CUBATURE_ORDER)
)
CRF_MULTIDIM_PATHS = "recombined"

F1, F2 = 5.0, 5.0

FILTER_NAMES = ("CD-EKF", "CD-UKF", "CD-EnKF", "PF", "CRF")
COLORS = {
    "CD-EKF": "#2E86AB",
    "CD-UKF": "#3DA35D",
    "CD-EnKF": "#F6AE2D",
    "PF": "#F24236",
    "CRF": "#6A4C93",
}


def stack_estimates(m0_arr, means):
    return np.vstack([np.asarray(m0_arr), np.asarray(means)])


def time_call(fn, *args):
    """Warmup + median wall-clock over ``N_TIMING`` runs (excl. compile)."""
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return out, float(np.median(samples))


def _as_numpy_drift(drift_fn):
    def drift_np(t_sec: float, x: np.ndarray) -> np.ndarray:
        return np.asarray(drift_fn(t_sec, jnp.asarray(x)), dtype=np.float64)

    return drift_np


def rk4_step(drift_np, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = drift_np(t, y)
    k2 = drift_np(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = drift_np(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = drift_np(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_to(drift_np, y: np.ndarray, t0: float, t1: float, dt0: float) -> np.ndarray:
    """Integrate drift from ``t0`` to ``t1`` with fixed-step RK4."""
    span = t1 - t0
    if span <= 0.0:
        return y.copy()
    n_steps = max(1, int(np.ceil(span / dt0)))
    h = span / n_steps
    t = t0
    for _ in range(n_steps):
        y = rk4_step(drift_np, t, y, h)
        t += h
    return y


def estimates_on_fine_grid(
    m_filt: np.ndarray,
    dt_obs_sec: float,
    times_fine_sec: np.ndarray,
    drift_fn,
    dt_integrate: float,
) -> np.ndarray:
    """Build mean trajectory on the fine grid from assimilation-time filtered means.

    Between observations the mean is propagated with the filter drift (ODE).
    At each observation time the filtered mean replaces the prediction.
    """
    drift_np = _as_numpy_drift(drift_fn)
    n_fine = times_fine_sec.shape[0]
    d = m_filt.shape[1]
    n_obs = m_filt.shape[0] - 1
    est = np.zeros((n_fine, d), dtype=np.float64)
    est[0] = m_filt[0]

    for k in range(n_obs):
        t0 = k * dt_obs_sec
        t1 = (k + 1) * dt_obs_sec
        js = np.where((times_fine_sec > t0 + 1e-12) & (times_fine_sec <= t1 + 1e-9))[0]
        m = m_filt[k].copy()
        t = t0
        for j in js:
            t_target = float(times_fine_sec[j])
            m = rk4_to(drift_np, m, t, t_target, dt_integrate)
            t = t_target
            if abs(t_target - t1) <= 1e-9:
                est[j] = m_filt[k + 1]
                m = m_filt[k + 1].copy()
            else:
                est[j] = m
    return est


def stride_for_delta(dt_min: float, eval_dt: float) -> int:
    stride = int(round(dt_min / eval_dt))
    if stride < 1 or not np.isclose(stride * eval_dt, dt_min, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Delta_t={dt_min} min must be an integer multiple of "
            f"--eval-dt={eval_dt} min (got stride={stride})"
        )
    return stride


# ---------------------------------------------------------------------------
# Shared fine-grid truth (fixed for the whole sweep)
# ---------------------------------------------------------------------------

if HORIZON_MIN / EVAL_DT_MIN != int(round(HORIZON_MIN / EVAL_DT_MIN)):
    raise SystemExit(
        f"Horizon {HORIZON_MIN} min must be an integer multiple of "
        f"--eval-dt={EVAL_DT_MIN} min"
    )
N_FINE = int(round(HORIZON_MIN / EVAL_DT_MIN))
DT_FINE_MIN = HORIZON_MIN / N_FINE
DT_FINE_SEC = DT_FINE_MIN * 60.0
N_SUBSTEPS_TRUTH_FINE = max(1, int(round(DT_FINE_SEC / DT_TRUTH)))

p = MFTSParams()
print(
    f"Simulating shared truth on fine grid: "
    f"eval_dt={DT_FINE_MIN:g} min, n_steps={N_FINE}, seed={SEED}"
)
true_fine, ys_fine_np, R = simulate_mfts(
    horizon_min=HORIZON_MIN,
    n_obs=N_FINE,
    n_substeps_truth=N_SUBSTEPS_TRUTH_FINE,
    p=p,
    lambda_sim=LAMBDA_SIM,
    sigma_sim=SIGMA_SIM,
    F1=F1,
    F2=F2,
    seed=SEED,
)
ys_fine = jnp.asarray(ys_fine_np)
times_fine_min = np.arange(N_FINE + 1) * DT_FINE_MIN
times_fine_sec = times_fine_min * 60.0

x0, P0 = init_estimate(p, F1, F2, true_state=true_fine[0], cov_scale=1.0)
m0 = jnp.asarray(x0, dtype=jnp.float64)
P0_j = jnp.asarray(P0, dtype=jnp.float64)
chol_P0 = jnp.linalg.cholesky(P0_j)
chol_R = jnp.linalg.cholesky(R)

bar_est = lambda t_min, nom=p.F_nominal_est: nom
drift_ekf, L_ekf, Qc, obs_fn, _ = make_mfts_dynamics(
    p,
    lambda_dist=LAMBDA_EKF_UKF,
    sigma_dist=SIGMA_EKF_ENKF_PF,
    bar_F3=bar_est,
    bar_F4=bar_est,
    F1=F1,
    F2=F2,
)
drift_ukf, L_ukf, _, _, _ = make_mfts_dynamics(
    p,
    lambda_dist=LAMBDA_EKF_UKF,
    sigma_dist=SIGMA_UKF,
    bar_F3=bar_est,
    bar_F4=bar_est,
    F1=F1,
    F2=F2,
)
drift_enkf, L_enkf, _, _, _ = make_mfts_dynamics(
    p,
    lambda_dist=LAMBDA_ENKF_PF,
    sigma_dist=SIGMA_EKF_ENKF_PF,
    bar_F3=bar_est,
    bar_F4=bar_est,
    F1=F1,
    F2=F2,
)
crf_drift, crf_L, _, _, _ = make_mfts_dynamics(
    p,
    lambda_dist=LAMBDA_ENKF_PF,
    sigma_dist=SIGMA_EKF_ENKF_PF,
    bar_F3=bar_est,
    bar_F4=bar_est,
    F1=F1,
    F2=F2,
)

DRIFT_FOR_MAPE = {
    "CD-EKF": drift_ekf,
    "CD-UKF": drift_ukf,
    "CD-EnKF": drift_enkf,
    "PF": drift_enkf,
    "CRF": crf_drift,
}


def run_filters_at_interval(
    dt_min_req: float,
) -> tuple[float, int, dict[str, dict[str, float]], dict[str, float]]:
    """Subsample fine truth, filter, score fine-grid MAPE, time each filter."""
    stride = stride_for_delta(dt_min_req, DT_FINE_MIN)
    n_obs = N_FINE // stride
    if n_obs * stride != N_FINE:
        raise ValueError(
            f"Delta_t={dt_min_req} min (stride={stride}) does not tile "
            f"the fine grid of {N_FINE} steps"
        )
    dt_min = stride * DT_FINE_MIN
    dt = dt_min * 60.0
    n_substeps_filter = max(1, int(round(dt / DT_FILTER)))
    dt0 = dt / n_substeps_filter

    # ys_fine[j] is the observation at fine time index j+1.
    # Coarse obs at fine indices stride, 2*stride, ... → ys indices stride-1, ...
    ys = ys_fine[stride - 1 :: stride]
    assert ys.shape[0] == n_obs

    model_inputs = jnp.arange(n_obs + 1)
    times: dict[str, float] = {}

    crf = build_filter(
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
        num_paths=CRF_NUM_PATHS,
        multidim_paths=CRF_MULTIDIM_PATHS,
    )
    jitted_crf = jit(run_filter, static_argnames=["filter_obj"])
    crf_states, times["CRF"] = time_call(lambda: jitted_crf(crf, model_inputs))
    estimates_obs = {
        "CRF": stack_estimates(m0, crf_states.mean[1:]),
    }
    final_ll: dict[str, float] = {
        "CRF": float(crf_states.log_normalizing_constant[-1]),
    }

    ekf_jit = jit(
        lambda: run_cd_ekf(
            drift_ekf,
            obs_fn,
            L_ekf,
            Qc,
            R,
            m0,
            P0_j,
            ys,
            dt,
            dt0,
            state_order="first",
            solver=SOLVER,
        )
    )
    ekf_res, times["CD-EKF"] = time_call(ekf_jit)
    estimates_obs["CD-EKF"] = stack_estimates(m0, ekf_res.means)
    final_ll["CD-EKF"] = float(ekf_res.cum_loglik[-1])

    ukf_jit = jit(
        lambda: run_cd_ukf(
            drift_ukf,
            obs_fn,
            L_ukf,
            Qc,
            R,
            m0,
            P0_j,
            ys,
            dt,
            dt0,
            UKFParams(),
            state_order="first",
            solver=SOLVER,
        )
    )
    ukf_res, times["CD-UKF"] = time_call(ukf_jit)
    estimates_obs["CD-UKF"] = stack_estimates(m0, ukf_res.means)
    final_ll["CD-UKF"] = float(ukf_res.cum_loglik[-1])

    key = random.key(SEED)
    enkf_jit = jit(
        lambda k: run_cd_enkf(
            drift_enkf,
            obs_fn,
            L_enkf,
            Qc,
            R,
            m0,
            P0_j,
            ys,
            dt,
            k,
            n_particles=ENKF_PARTICLES,
            noise_dim=2,
            dt0=dt0,
            solver=SOLVER,
        )
    )
    enkf_res, times["CD-EnKF"] = time_call(enkf_jit, key)
    estimates_obs["CD-EnKF"] = stack_estimates(m0, enkf_res.means)
    final_ll["CD-EnKF"] = float(enkf_res.cum_loglik[-1])

    key, pf_key = random.split(key)
    pf_jit = jit(
        lambda k: run_bootstrap_pf(
            drift_enkf,
            obs_fn,
            L_enkf,
            Qc,
            R,
            m0,
            P0_j,
            ys,
            dt,
            k,
            n_particles=PF_PARTICLES,
            noise_dim=2,
            dt0=dt0,
            solver=SOLVER,
        )
    )
    pf_res, times["PF"] = time_call(pf_jit, pf_key)
    estimates_obs["PF"] = stack_estimates(m0, pf_res.means)
    final_ll["PF"] = float(pf_res.cum_loglik[-1])

    mapes = {}
    for name, m_filt in estimates_obs.items():
        est_fine = estimates_on_fine_grid(
            m_filt, dt, times_fine_sec, DRIFT_FOR_MAPE[name], DT_FILTER
        )
        mapes[name] = {
            "mapex": mape_pct(est_fine, true_fine, slice(0, 4)),
            "maped": mape_pct(est_fine, true_fine, slice(4, 6)),
            "nll": -final_ll[name],
            "nll_per_obs": -final_ll[name] / n_obs,
        }
    return dt_min, n_obs, mapes, times


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

delta_t_eff: list[float] = []
n_obs_list: list[int] = []
series = {
    name: {"mapex": [], "maped": [], "nll": [], "nll_per_obs": [], "time": []}
    for name in FILTER_NAMES
}

print(f"Horizon={HORIZON_MIN:g} min | solver={SOLVER} | seed={SEED}")
print(
    f"CRF: multidim={CRF_MULTIDIM_PATHS} | k={CUBATURE_ORDER} | "
    f"num_paths budget={CRF_NUM_PATHS}"
)
print(f"Requested Delta_t [min]: {DELTA_T_MIN_REQ}")
print(f"Timing: median of {N_TIMING} runs after JIT warmup")
print("-" * 72)

for dt_req in DELTA_T_MIN_REQ:
    dt_eff, n_obs, mapes, times = run_filters_at_interval(dt_req)
    delta_t_eff.append(dt_eff)
    n_obs_list.append(n_obs)
    print(
        f"Delta_t={dt_eff:6.3f} min  (n_obs={n_obs:4d}, "
        f"dt={dt_eff * 60.0:7.2f} s)"
    )
    for name in FILTER_NAMES:
        series[name]["mapex"].append(mapes[name]["mapex"])
        series[name]["maped"].append(mapes[name]["maped"])
        series[name]["nll"].append(mapes[name]["nll"])
        series[name]["nll_per_obs"].append(mapes[name]["nll_per_obs"])
        series[name]["time"].append(times[name])
        print(
            f"  {name:8s}  MAPEx={mapes[name]['mapex']:8.2f}%  "
            f"MAPEd={mapes[name]['maped']:8.2f}%  "
            f"NLL/obs={mapes[name]['nll_per_obs']:8.2f}  "
            f"time={times[name] * 1e3:8.2f} ms"
        )

# ---------------------------------------------------------------------------
# Plot MAPE vs Delta_t
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
for ax, metric, title in zip(
    axes,
    ("mapex", "maped"),
    ("MAPEx (tanks)", "MAPEd (disturbances)"),
):
    for name in FILTER_NAMES:
        ax.plot(
            delta_t_eff,
            series[name][metric],
            color=COLORS[name],
            marker="o",
            lw=1.8,
            label=name,
        )
    ax.set_xlabel(r"Observation interval $\Delta t$ [min]")
    ax.set_ylabel("MAPE [%]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
axes[0].legend(fontsize=9, loc="best")
fig.suptitle("MFTS MAPE vs Delta_t", fontsize=12)
fig.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "mape_vs_interval.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Plot NLL/obs vs Delta_t
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))
for name in FILTER_NAMES:
    ax.plot(
        delta_t_eff,
        series[name]["nll_per_obs"],
        color=COLORS[name],
        marker="o",
        lw=1.8,
        label=name,
    )
ax.set_xlabel(r"Observation interval $\Delta t$ [min]")
ax.set_ylabel("Mean NLL per observation")
ax.set_title(r"MFTS NLL / obs vs $\Delta t$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9, loc="best")
fig.tight_layout()
nll_path = os.path.join(OUTPUT_DIR, "nll_vs_interval.png")
fig.savefig(nll_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Plot wall-clock time vs Delta_t
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

ax = axes[0]
for name in FILTER_NAMES:
    ax.plot(
        delta_t_eff,
        [1e3 * t for t in series[name]["time"]],
        color=COLORS[name],
        marker="o",
        lw=1.8,
        label=name,
    )
ax.set_xlabel(r"Observation interval $\Delta t$ [min]")
ax.set_ylabel("Wall-clock time [ms]")
ax.set_title("Total filter runtime")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9, loc="best")

ax = axes[1]
for name in FILTER_NAMES:
    per_obs = [
        1e3 * t / n if n > 0 else np.nan
        for t, n in zip(series[name]["time"], n_obs_list)
    ]
    ax.plot(
        delta_t_eff,
        per_obs,
        color=COLORS[name],
        marker="o",
        lw=1.8,
        label=name,
    )
ax.set_xlabel(r"Observation interval $\Delta t$ [min]")
ax.set_ylabel("Wall-clock time / observation [ms]")
ax.set_title("Cost per assimilation step")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)

fig.suptitle(
    r"MFTS filter compute time vs $\Delta t$",
    fontsize=12,
)
fig.tight_layout()
timing_path = os.path.join(OUTPUT_DIR, "timing_vs_interval.png")
fig.savefig(timing_path, dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "mape_vs_interval.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["delta_t_min", "n_obs", "eval_dt_min", "solver"]
    for name in FILTER_NAMES:
        header.extend(
            [
                f"{name}_mapex",
                f"{name}_maped",
                f"{name}_nll",
                f"{name}_nll_per_obs",
                f"{name}_time_s",
            ]
        )
    writer.writerow(header)
    for i, dt_eff in enumerate(delta_t_eff):
        row = [dt_eff, n_obs_list[i], DT_FINE_MIN, SOLVER]
        for name in FILTER_NAMES:
            row.extend(
                [
                    series[name]["mapex"][i],
                    series[name]["maped"][i],
                    series[name]["nll"][i],
                    series[name]["nll_per_obs"][i],
                    series[name]["time"][i],
                ]
            )
        writer.writerow(row)

print("-" * 72)
print(f"Wrote {plot_path}")
print(f"Wrote {nll_path}")
print(f"Wrote {timing_path}")
print(f"Wrote {csv_path}")
