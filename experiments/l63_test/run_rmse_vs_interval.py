"""Sweep Lorenz-63 observation interval length; fine-grid RMSE + timing.

Simulates **one** shared fine-grid truth trajectory, then for each assimilation
interval ``Delta_t``:

1. Subsamples observations from that truth
2. Runs CD-EKF / CD-UKF / CD-EnKF / bootstrap PF / WSC-KF (flow-map EKF/UKF)
3. Reconstructs each filter's mean on the **same fine time grid** by integrating
   the filter drift between updates (filtered mean at observation times)
4. Scores RMSE on that fine grid
5. Times each filter (median wall-clock, excl. JIT; default CD solver=rk4)

Run:
  python experiments/l63_test/run_rmse_vs_interval.py
  python experiments/l63_test/run_rmse_vs_interval.py --solver rk4
  python experiments/l63_test/run_rmse_vs_interval.py --deltas 0.05,0.1,0.35,0.7,1.4,3.5
  python experiments/l63_test/run_rmse_vs_interval.py --wsckf-paths 2,84 --lv-degrees 3

Compares library recombined KLV (1D-product expand + RMP) against explicit
multi-D Lyons–Victoir paths (``multidim_method="lv"``). With d = 3 only m = 3 is
available; the library implements LV m = 5 for d <= 2.

Outputs:
  ``experiments/l63_test/output/rmse_vs_interval.png``
  ``experiments/l63_test/output/timing_vs_interval.png``
  ``experiments/l63_test/output/rmse_vs_interval.csv``
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, EXPERIMENTS_DIR)

ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import jit, random

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)
from cuthbert import filter as run_filter
from wsckf import build_filter
from wsckf.library import (
    default_num_paths,
    default_recombined_num_paths,
    precompute_cubature_paths,
)
from mfts_test.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    METHOD_MARKERS,
    PALETTE,
    save_publication_figure,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(
    description="RMSE vs observation interval on stochastic Lorenz-63 (fine-grid RMSE)"
)
_parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="RNG seed for truth simulation and stochastic filters",
)
_parser.add_argument(
    "--cubature-order",
    type=int,
    default=2,
    dest="cubature_order",
    help="WSC-KF Lyons macro-segment count k for recombined KLV (default 2)",
)
_parser.add_argument(
    "--wsckf-paths",
    type=str,
    default="2,84",
    dest="wsckf_paths",
    help="Comma-separated recombined path budgets (default 2,84). "
    "Empty string skips recombined WSC-KF. "
    "Use 'default' to insert default_recombined_num_paths(noise_dim, k).",
)
_parser.add_argument(
    "--lv-degrees",
    type=str,
    default="3",
    dest="lv_degrees",
    help="Comma-separated explicit multi-D Lyons–Victoir degrees m "
    "(default 3; m=5 needs noise_dim <= 2). Empty string skips LV WSC-KF.",
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
    default="0.05,0.1,0.35,0.7,1.4,3.5",
    help="Comma-separated observation intervals Delta_t "
    "(each must be an integer multiple of --eval-dt)",
)
_parser.add_argument(
    "--eval-dt",
    type=float,
    default=0.01,
    dest="eval_dt",
    help="Fine evaluation / truth grid spacing (default 0.01, matches L63 dt_inner)",
)
_parser.add_argument(
    "--horizon",
    type=float,
    default=70.0,
    help="Total simulation horizon (default 70 = 200 * 0.35 from run_comparison)",
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
SOLVER = _args.solver
CUBATURE_ORDER = _args.cubature_order
DELTA_T_REQ = [float(x) for x in _args.deltas.split(",") if x.strip()]
EVAL_DT = float(_args.eval_dt)
HORIZON = float(_args.horizon)
N_TIMING = _args.n_timing

# ---------------------------------------------------------------------------
# Lorenz-63 settings (match run_comparison.py)
# ---------------------------------------------------------------------------

lorenz_sigma = 10.0
lorenz_rho = 28.0
lorenz_beta = 8.0 / 3.0

DT_INNER = 0.01  # filter / truth integration step
diff_std = 1.0
obs_std = 0.3

x_dim = 3
y_dim = 1
noise_dim = x_dim

ENKF_PARTICLES = 50
PF_PARTICLES = 200
WSCKF_MULTIDIM_PATHS = "recombined"
SPINUP_STEPS = 35_000  # ~1000 assimilation intervals of length 0.35

_DEFAULT_WSCKF_PATHS = default_recombined_num_paths(noise_dim, CUBATURE_ORDER)
WSCKF_PATH_BUDGETS: list[int] = []
for _tok in _args.wsckf_paths.split(","):
    _tok = _tok.strip()
    if not _tok:
        continue
    if _tok.lower() == "default":
        WSCKF_PATH_BUDGETS.append(_DEFAULT_WSCKF_PATHS)
    else:
        WSCKF_PATH_BUDGETS.append(int(_tok))
_seen_paths: set[int] = set()
WSCKF_PATH_BUDGETS = [
    n for n in WSCKF_PATH_BUDGETS if not (n in _seen_paths or _seen_paths.add(n))
]

LV_DEGREES: list[int] = []
for _tok in _args.lv_degrees.split(","):
    _tok = _tok.strip()
    if not _tok:
        continue
    m = int(_tok)
    if m not in (3, 5):
        raise ValueError(f"--lv-degrees entries must be 3 or 5, got {m}")
    LV_DEGREES.append(m)
_seen_lv: set[int] = set()
LV_DEGREES = [m for m in LV_DEGREES if not (m in _seen_lv or _seen_lv.add(m))]


def rec_label(n_paths: int) -> str:
    return f"WSC-KF-rec-P{n_paths}"


def lv_label(m: int) -> str:
    return f"WSC-KF-LV-m{m}"


BASE_FILTER_NAMES = ("CD-EKF", "CD-UKF", "CD-EnKF", "PF")
REC_FILTER_NAMES = tuple(rec_label(n) for n in WSCKF_PATH_BUDGETS)
LV_FILTER_NAMES = tuple(lv_label(m) for m in LV_DEGREES)
FILTER_NAMES = BASE_FILTER_NAMES + REC_FILTER_NAMES + LV_FILTER_NAMES

COLORS = {
    "CD-EKF": METHOD_COLORS["CD-EKF"],
    "CD-UKF": METHOD_COLORS["CD-UKF"],
    "CD-EnKF": METHOD_COLORS["CD-EnKF"],
    "PF": METHOD_COLORS["PF"],
}
for _name in REC_FILTER_NAMES:
    COLORS[_name] = METHOD_COLORS["WSC-KF"]
for _name in LV_FILTER_NAMES:
    COLORS[_name] = PALETTE["salmon"]

PLOT_MARKERS = {name: METHOD_MARKERS[name] for name in BASE_FILTER_NAMES}
for _i, _name in enumerate(REC_FILTER_NAMES):
    PLOT_MARKERS[_name] = ("o", "P", "X", "*", "h")[_i % 5]
for _i, _name in enumerate(LV_FILTER_NAMES):
    PLOT_MARKERS[_name] = ("D", "d")[_i % 2]

PLOT_LINESTYLES = {name: "-" for name in BASE_FILTER_NAMES}
for _i, _name in enumerate(REC_FILTER_NAMES):
    PLOT_LINESTYLES[_name] = ("-", "--", "-.", ":")[_i % 4]
for _i, _name in enumerate(LV_FILTER_NAMES):
    PLOT_LINESTYLES[_name] = ("--", ":")[_i % 2]


def lorenz_drift(x):
    return jnp.array(
        [
            lorenz_sigma * (x[1] - x[0]),
            x[0] * (lorenz_rho - x[2]) - x[1],
            x[0] * x[1] - lorenz_beta * x[2],
        ]
    )


def f_drift(t, x):
    return lorenz_drift(x)


def h_obs(x):
    return H @ x + d_obs


def L_fn(t, x):
    return diff_std * jnp.eye(x_dim)


def Qc_fn(t, x):
    return jnp.eye(x_dim)


H = jnp.array([[1.0, 0.0, 0.0]])
d_obs = jnp.zeros(y_dim)
R = (obs_std**2) * jnp.eye(y_dim)
chol_R = jnp.linalg.cholesky(R)


def heun_step_np(x: np.ndarray, h: float) -> np.ndarray:
    k1 = np.array(
        [
            lorenz_sigma * (x[1] - x[0]),
            x[0] * (lorenz_rho - x[2]) - x[1],
            x[0] * x[1] - lorenz_beta * x[2],
        ],
        dtype=np.float64,
    )
    x_pred = x + h * k1
    k2 = np.array(
        [
            lorenz_sigma * (x_pred[1] - x_pred[0]),
            x_pred[0] * (lorenz_rho - x_pred[2]) - x_pred[1],
            x_pred[0] * x_pred[1] - lorenz_beta * x_pred[2],
        ],
        dtype=np.float64,
    )
    return x + 0.5 * h * (k1 + k2)


def stack_estimates(m0_arr, means):
    return np.vstack([np.asarray(m0_arr), np.asarray(means)])


def time_call(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return out, float(np.median(samples))


def _as_numpy_drift(drift_fn):
    def drift_np(t: float, x: np.ndarray) -> np.ndarray:
        return np.asarray(drift_fn(t, jnp.asarray(x)), dtype=np.float64)

    return drift_np


def rk4_step(drift_np, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = drift_np(t, y)
    k2 = drift_np(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = drift_np(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = drift_np(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_to(drift_np, y: np.ndarray, t0: float, t1: float, dt0: float) -> np.ndarray:
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
    dt_obs: float,
    times_fine: np.ndarray,
    drift_fn,
    dt_integrate: float,
) -> np.ndarray:
    """Mean trajectory on the fine grid from assimilation-time filtered means."""
    drift_np = _as_numpy_drift(drift_fn)
    n_fine = times_fine.shape[0]
    d = m_filt.shape[1]
    n_obs = m_filt.shape[0] - 1
    est = np.zeros((n_fine, d), dtype=np.float64)
    est[0] = m_filt[0]

    for k in range(n_obs):
        t0 = k * dt_obs
        t1 = (k + 1) * dt_obs
        js = np.where((times_fine > t0 + 1e-12) & (times_fine <= t1 + 1e-9))[0]
        m = m_filt[k].copy()
        t = t0
        for j in js:
            t_target = float(times_fine[j])
            m = rk4_to(drift_np, m, t, t_target, dt_integrate)
            t = t_target
            if abs(t_target - t1) <= 1e-9:
                est[j] = m_filt[k + 1]
                m = m_filt[k + 1].copy()
            else:
                est[j] = m
    return est


def stride_for_delta(dt: float, eval_dt: float) -> int:
    stride = int(round(dt / eval_dt))
    if stride < 1 or not np.isclose(stride * eval_dt, dt, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Delta_t={dt} must be an integer multiple of --eval-dt={eval_dt} "
            f"(got stride={stride})"
        )
    return stride


def rmse_components(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((est - truth) ** 2, axis=0))


# ---------------------------------------------------------------------------
# Shared fine-grid truth
# ---------------------------------------------------------------------------

if abs(EVAL_DT - DT_INNER) > 1e-15 and EVAL_DT < DT_INNER:
    print(
        f"Warning: eval_dt={EVAL_DT} < filter dt_inner={DT_INNER}; "
        f"fine-grid mean forecasts still use dt_inner={DT_INNER}"
    )

if HORIZON / EVAL_DT != int(round(HORIZON / EVAL_DT)):
    raise SystemExit(
        f"Horizon {HORIZON} must be an integer multiple of --eval-dt={EVAL_DT}"
    )
N_FINE = int(round(HORIZON / EVAL_DT))
DT_FINE = HORIZON / N_FINE
times_fine = np.arange(N_FINE + 1, dtype=np.float64) * DT_FINE

print(
    f"Simulating shared L63 truth on fine grid: "
    f"eval_dt={DT_FINE:g}, n_steps={N_FINE}, horizon={HORIZON:g}, seed={SEED}"
)

rng = np.random.default_rng(SEED)
x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
chol_scale = diff_std * np.sqrt(DT_FINE)
for _ in range(SPINUP_STEPS):
    x = heun_step_np(x, DT_FINE)
    x = x + chol_scale * rng.standard_normal(x_dim)

m0_np = x.copy()
true_fine = [m0_np.copy()]
ys_fine_list = []
for _ in range(N_FINE):
    x = heun_step_np(x, DT_FINE)
    x = x + chol_scale * rng.standard_normal(x_dim)
    y = np.array([x[0]], dtype=np.float64) + obs_std * rng.standard_normal(y_dim)
    true_fine.append(x.copy())
    ys_fine_list.append(np.asarray(y, dtype=np.float64))

true_fine = np.asarray(true_fine, dtype=np.float64)
ys_fine = jnp.asarray(np.stack(ys_fine_list))
m0 = jnp.asarray(m0_np, dtype=jnp.float64)
P0 = 2.0 * jnp.eye(x_dim)
chol_P0 = jnp.linalg.cholesky(P0)


def run_filters_at_interval(
    dt_req: float,
) -> tuple[float, int, dict[str, dict[str, float]], dict[str, float]]:
    stride = stride_for_delta(dt_req, DT_FINE)
    n_obs = N_FINE // stride
    if n_obs * stride != N_FINE:
        raise ValueError(
            f"Delta_t={dt_req} (stride={stride}) does not tile the fine grid "
            f"of {N_FINE} steps"
        )
    dt = stride * DT_FINE
    dt0 = DT_INNER
    n_inner = max(1, int(round(dt / dt0)))
    if not np.isclose(n_inner * dt0, dt):
        # Still run; integration uses fixed dt0 with n_steps from filters' RK4 helper.
        pass

    ys = ys_fine[stride - 1 :: stride]
    assert ys.shape[0] == n_obs
    model_inputs = jnp.arange(n_obs + 1)
    times: dict[str, float] = {}
    estimates_obs: dict[str, np.ndarray] = {}

    def _run_wsckf(name: str, **build_kwargs):
        wsckf = build_filter(
            get_init_params=lambda mi: (m0, chol_P0),
            get_dynamics=lambda mi: (
                lambda t, x: lorenz_drift(x),
                lambda t, x: diff_std * jnp.eye(x_dim),
            ),
            get_observations=lambda mi: (
                lambda x: H @ x + d_obs,
                chol_R,
                ys[mi - 1],
            ),
            noise_dim=noise_dim,
            dt=dt,
            integration_dt=dt0,
            **build_kwargs,
        )
        jitted_wsckf = jit(run_filter, static_argnames=["filter_obj"])
        wsckf_states, times[name] = time_call(
            lambda f=jitted_wsckf, filt=wsckf: f(filt, model_inputs)
        )
        estimates_obs[name] = stack_estimates(m0, wsckf_states.mean[1:])

    for n_paths in WSCKF_PATH_BUDGETS:
        _run_wsckf(
            rec_label(n_paths),
            cubature_order=CUBATURE_ORDER,
            num_paths=n_paths,
            multidim_paths=WSCKF_MULTIDIM_PATHS,
        )

    for m in LV_DEGREES:
        ts_lv, pv_lv, pw_lv = precompute_cubature_paths(
            horizon=dt,
            noise_dim=noise_dim,
            integration_dt=dt0,
            cubature_order=1,
            multidim_method="lv",
            lyons_degree=m,
        )
        _run_wsckf(
            lv_label(m),
            wiener_paths=(
                jnp.asarray(ts_lv),
                jnp.asarray(pv_lv),
                jnp.asarray(pw_lv),
            ),
        )

    ekf_jit = jit(
        lambda: run_cd_ekf(
            f_drift,
            h_obs,
            L_fn,
            Qc_fn,
            R,
            m0,
            P0,
            ys,
            dt,
            dt0,
            state_order="flowmap",
            solver=SOLVER,
        )
    )
    ekf_res, times["CD-EKF"] = time_call(ekf_jit)
    estimates_obs["CD-EKF"] = stack_estimates(m0, ekf_res.means)

    ukf_jit = jit(
        lambda: run_cd_ukf(
            f_drift,
            h_obs,
            L_fn,
            Qc_fn,
            R,
            m0,
            P0,
            ys,
            dt,
            dt0,
            UKFParams(),
            state_order="flowmap",
            solver=SOLVER,
        )
    )
    ukf_res, times["CD-UKF"] = time_call(ukf_jit)
    estimates_obs["CD-UKF"] = stack_estimates(m0, ukf_res.means)

    key = random.key(SEED)
    enkf_jit = jit(
        lambda k: run_cd_enkf(
            f_drift,
            h_obs,
            L_fn,
            Qc_fn,
            R,
            m0,
            P0,
            ys,
            dt,
            k,
            n_particles=ENKF_PARTICLES,
            noise_dim=noise_dim,
            dt0=dt0,
            inflation_delta=0.02,
            solver=SOLVER,
        )
    )
    enkf_res, times["CD-EnKF"] = time_call(enkf_jit, key)
    estimates_obs["CD-EnKF"] = stack_estimates(m0, enkf_res.means)

    key, pf_key = random.split(key)
    pf_jit = jit(
        lambda k: run_bootstrap_pf(
            f_drift,
            h_obs,
            L_fn,
            Qc_fn,
            R,
            m0,
            P0,
            ys,
            dt,
            k,
            n_particles=PF_PARTICLES,
            noise_dim=noise_dim,
            dt0=dt0,
            solver=SOLVER,
        )
    )
    pf_res, times["PF"] = time_call(pf_jit, pf_key)
    estimates_obs["PF"] = stack_estimates(m0, pf_res.means)

    metrics = {}
    for name, m_filt in estimates_obs.items():
        est_fine = estimates_on_fine_grid(
            m_filt, dt, times_fine, f_drift, dt0
        )
        comps = rmse_components(est_fine, true_fine)
        metrics[name] = {
            "rmse_overall": float(np.mean(comps)),
            "rmse_x1": float(comps[0]),
            "rmse_unobs": float(np.mean(comps[1:])),
        }
    return dt, n_obs, metrics, times


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

delta_t_eff: list[float] = []
n_obs_list: list[int] = []
series = {
    name: {"rmse_overall": [], "rmse_x1": [], "rmse_unobs": [], "time": []}
    for name in FILTER_NAMES
}

print(f"Horizon={HORIZON:g} | solver={SOLVER} | seed={SEED}")
print(
    f"WSC-KF recombined: k={CUBATURE_ORDER} | path budgets={WSCKF_PATH_BUDGETS} "
    f"(moment-safe default={_DEFAULT_WSCKF_PATHS})"
)
print(
    "WSC-KF LV multi-D: "
    + ", ".join(f"m={m}: {default_num_paths(noise_dim, 1, 'lv', lyons_degree=m)} paths" for m in LV_DEGREES)
)
print(f"Requested Delta_t: {DELTA_T_REQ}")
print(f"Timing: median of {N_TIMING} runs after JIT warmup")
print("-" * 78)

for dt_req in DELTA_T_REQ:
    dt_eff, n_obs, metrics, times = run_filters_at_interval(dt_req)
    delta_t_eff.append(dt_eff)
    n_obs_list.append(n_obs)
    print(f"Delta_t={dt_eff:6.3f}  (n_obs={n_obs:4d})")
    for name in FILTER_NAMES:
        series[name]["rmse_overall"].append(metrics[name]["rmse_overall"])
        series[name]["rmse_x1"].append(metrics[name]["rmse_x1"])
        series[name]["rmse_unobs"].append(metrics[name]["rmse_unobs"])
        series[name]["time"].append(times[name])
        print(
            f"  {name:8s}  RMSE={metrics[name]['rmse_overall']:8.3f}  "
            f"x1={metrics[name]['rmse_x1']:7.3f}  "
            f"unobs={metrics[name]['rmse_unobs']:7.3f}  "
            f"time={times[name] * 1e3:8.2f} ms"
        )

# ---------------------------------------------------------------------------
# Plot RMSE vs Delta_t
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.65), sharex=True)
for ax, metric, title in zip(
    axes,
    ("rmse_overall", "rmse_unobs"),
    ("Overall RMSE", "Unobserved RMSE (mean of $x_2$, $x_3$)"),
):
    for name in FILTER_NAMES:
        ax.plot(
            delta_t_eff,
            series[name][metric],
            color=COLORS[name],
            marker=PLOT_MARKERS[name],
            linestyle=PLOT_LINESTYLES[name],
            markerfacecolor="white",
            markeredgewidth=0.7,
            markersize=3.8,
            lw=1.25,
            label=name,
        )
    ax.set_xlabel(r"Observation interval $\Delta t$")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    style_curve_axis(ax, grid_axis="both")
    ax.set_xscale("log")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.31, top=0.91, wspace=0.28)
plot_path = os.path.join(OUTPUT_DIR, "rmse_vs_interval.png")
rmse_png, rmse_pdf = save_publication_figure(fig, plot_path)
plt.close()

# ---------------------------------------------------------------------------
# Plot wall-clock time vs Delta_t
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.65), sharex=True)

ax = axes[0]
for name in FILTER_NAMES:
    ax.plot(
        delta_t_eff,
        [1e3 * t for t in series[name]["time"]],
        color=COLORS[name],
        marker=PLOT_MARKERS[name],
        linestyle=PLOT_LINESTYLES[name],
        markerfacecolor="white",
        markeredgewidth=0.7,
        markersize=3.8,
        lw=1.25,
        label=name,
    )
ax.set_xlabel(r"Observation interval $\Delta t$")
ax.set_ylabel("Wall-clock time [ms]")
ax.set_title("Total filter runtime")
ax.set_xscale("log")
ax.set_yscale("log")
style_curve_axis(ax, grid_axis="both")

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
        marker=PLOT_MARKERS[name],
        linestyle=PLOT_LINESTYLES[name],
        markerfacecolor="white",
        markeredgewidth=0.7,
        markersize=3.8,
        lw=1.25,
        label=name,
    )
ax.set_xlabel(r"Observation interval $\Delta t$")
ax.set_ylabel("Wall-clock time / observation [ms]")
ax.set_title("Cost per assimilation step")
ax.set_xscale("log")
ax.set_yscale("log")
style_curve_axis(ax, grid_axis="both")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.31, top=0.91, wspace=0.30)
timing_path = os.path.join(OUTPUT_DIR, "timing_vs_interval.png")
timing_png, timing_pdf = save_publication_figure(fig, timing_path)
plt.close()

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "rmse_vs_interval.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["delta_t", "n_obs", "eval_dt", "solver"]
    for name in FILTER_NAMES:
        header.extend(
            [
                f"{name}_rmse_overall",
                f"{name}_rmse_x1",
                f"{name}_rmse_unobs",
                f"{name}_time_s",
            ]
        )
    writer.writerow(header)
    for i, dt_eff in enumerate(delta_t_eff):
        row = [dt_eff, n_obs_list[i], DT_FINE, SOLVER]
        for name in FILTER_NAMES:
            row.extend(
                [
                    series[name]["rmse_overall"][i],
                    series[name]["rmse_x1"][i],
                    series[name]["rmse_unobs"][i],
                    series[name]["time"][i],
                ]
            )
        writer.writerow(row)

print("-" * 78)
print(f"Wrote {plot_path}")
print(f"Wrote {timing_path}")
print(f"Wrote {csv_path}")
