"""Sweep MFTS observation interval length and compare Nielsen MAPE across filters.

Simulates **one** shared fine-grid truth trajectory, then for each assimilation
interval ``Delta_t``:

1. Subsamples observations from that truth
2. Runs CD-EKF / CD-UKF / CD-EnKF / bootstrap PF / WSC-KF
3. Reconstructs each filter's mean on the **same fine time grid** by integrating
   the filter drift between updates (filtered mean at observation times)
4. Scores MAPEx / MAPEd on that fine grid
5. Records predictive NLL (total and per observation) from each filter's cum. loglik
6. Times each filter (median wall-clock, excl. JIT; default CD solver=rk4)

Optional ``--k-proportional`` runs a recombined WSC-KF with
``k = max(1, round(Δt / --k-ref-min))`` to test whether refining the Lyons
macro-segment partition with Δt closes the WSC-KF–EKF NLL gap at long intervals.

Run:
  python experiments/mfts_test/run_mape_vs_interval.py
  python experiments/mfts_test/run_mape_vs_interval.py --solver rk4
  python experiments/mfts_test/run_mape_vs_interval.py --deltas 0.25,0.5,1,2,5,10
  python experiments/mfts_test/run_mape_vs_interval.py --wsckf-paths 2,12,35
  python experiments/mfts_test/run_mape_vs_interval.py --k-proportional \\
      --k-ref-min 0.5 --wsckf-paths 2,35 --n-timing 2

Outputs:
  ``experiments/mfts_test/output/mape_vs_interval.png``
  ``experiments/mfts_test/output/nll_vs_interval.png``
  ``experiments/mfts_test/output/nll_gap_vs_ekf_interval.png``
  ``experiments/mfts_test/output/timing_vs_interval.png``
  ``experiments/mfts_test/output/pareto_time_vs_mape.png``
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
os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
from wsckf import build_filter

from mfts_model import (
    MFTSParams,
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
    style_curve_axis,
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
    help="Fixed WSC-KF Lyons macro-segment count k for WSC-KF-P* (default 2; recombined)",
)
_parser.add_argument(
    "--wsckf-paths",
    type=str,
    default="2,12,35",
    dest="wsckf_paths",
    help="Comma-separated recombined path budgets for fixed-k WSC-KF (default 2,12,35). "
    "Each plotted as WSC-KF-P{{n}}. Use 'default' for "
    "default_recombined_num_paths(2, k). Empty string skips fixed-k WSC-KF.",
)
_parser.add_argument(
    "--k-proportional",
    action="store_true",
    dest="k_proportional",
    help="Also run recombined WSC-KF with k=max(1, round(Δt / --k-ref-min)) "
    "and path budget default_recombined_num_paths(2, k) (tests cubature "
    "time-partition hypothesis).",
)
_parser.add_argument(
    "--k-ref-min",
    type=float,
    default=0.5,
    dest="k_ref_min",
    help="Target macro-segment length [min] for --k-proportional "
    "(default 0.5 → k≈Δt/0.5)",
)
_parser.add_argument(
    "--k-max",
    type=int,
    default=8,
    dest="k_max",
    help="Cap on k under --k-proportional (default 8; recombination cost grows with k)",
)
_parser.add_argument(
    "--k-prop-paths",
    type=str,
    default="default",
    dest="k_prop_paths",
    help="Path budget for k∝Δt WSC-KF: 'default' or an integer override",
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
K_PROPORTIONAL = bool(_args.k_proportional)
K_REF_MIN = float(_args.k_ref_min)
K_MAX = int(_args.k_max)
if K_PROPORTIONAL and K_REF_MIN <= 0.0:
    raise SystemExit("--k-ref-min must be > 0")
if K_PROPORTIONAL and K_MAX < 1:
    raise SystemExit("--k-max must be ≥ 1")


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
WSCKF_MULTIDIM_PATHS = "recombined"
WSCKF_KPROP_NAME = "WSC-KF-k∝Δt"

_DEFAULT_WSCKF_PATHS = default_recombined_num_paths(2, CUBATURE_ORDER)
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


def wsckf_label(n_paths: int) -> str:
    return f"WSC-KF-P{n_paths}"


def k_for_delta(dt_min: float) -> int:
    """Macro-segment count with target segment length ``K_REF_MIN`` minutes."""
    return min(K_MAX, max(1, int(round(float(dt_min) / K_REF_MIN))))


def kprop_num_paths(k: int) -> int:
    tok = _args.k_prop_paths.strip().lower()
    if tok == "default" or tok == "":
        return default_recombined_num_paths(2, k)
    return max(1, int(tok))


F1, F2 = 5.0, 5.0

BASE_FILTER_NAMES = ("CD-EKF", "CD-UKF", "CD-EnKF", "PF")
WSCKF_FILTER_NAMES = tuple(wsckf_label(n) for n in WSCKF_PATH_BUDGETS)
KPROP_FILTER_NAMES = (WSCKF_KPROP_NAME,) if K_PROPORTIONAL else ()
FILTER_NAMES = BASE_FILTER_NAMES + WSCKF_FILTER_NAMES + KPROP_FILTER_NAMES

COLORS = {
    name: METHOD_COLORS[name] for name in BASE_FILTER_NAMES
}
for _name in WSCKF_FILTER_NAMES + KPROP_FILTER_NAMES:
    COLORS[_name] = METHOD_COLORS["WSC-KF"]

_BASE_LINESTYLES = {
    "CD-EKF": "-",
    "CD-UKF": "--",
    "CD-EnKF": "-.",
    "PF": ":",
}
_WSCKF_MARKERS = (
    METHOD_MARKERS["WSC-KF"],
    "P",
    "X",
    "*",
    "h",
    "d",
    "<",
    ">",
)
_WSCKF_LINESTYLES = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
)
PLOT_STYLES = {
    name: {
        "color": COLORS[name],
        "marker": METHOD_MARKERS[name],
        "linestyle": _BASE_LINESTYLES[name],
    }
    for name in BASE_FILTER_NAMES
}
for _i, _name in enumerate(WSCKF_FILTER_NAMES + KPROP_FILTER_NAMES):
    PLOT_STYLES[_name] = {
        "color": METHOD_COLORS["WSC-KF"],
        "marker": _WSCKF_MARKERS[_i % len(_WSCKF_MARKERS)],
        "linestyle": _WSCKF_LINESTYLES[_i % len(_WSCKF_LINESTYLES)],
    }


def plot_method(ax, name, x, y, *, label=True, **kwargs):
    """Plot one filter with the manuscript-wide color/marker semantics."""
    style = PLOT_STYLES[name]
    return ax.plot(
        x,
        y,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        label=name if label else None,
        **kwargs,
    )


def add_shared_legend(fig, ax, *, ncol):
    """Place one compact method legend below a composite figure."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=ncol,
    )


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
wsckf_drift, wsckf_L, _, _, _ = make_mfts_dynamics(
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
    **{name: wsckf_drift for name in WSCKF_FILTER_NAMES},
    **({WSCKF_KPROP_NAME: wsckf_drift} if K_PROPORTIONAL else {}),
}


def run_filters_at_interval(
    dt_min_req: float,
) -> tuple[float, int, dict[str, dict[str, float]], dict[str, float], dict]:
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
    estimates_obs: dict[str, np.ndarray] = {}
    final_ll: dict[str, float] = {}
    meta: dict = {"k_prop": None, "k_prop_paths": None}

    def _run_wsckf(name: str, *, k: int, n_paths: int):
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
            cubature_order=k,
            num_paths=n_paths,
            multidim_paths=WSCKF_MULTIDIM_PATHS,
        )
        jitted_wsckf = jit(run_filter, static_argnames=["filter_obj"])
        wsckf_states, wall = time_call(
            lambda f=jitted_wsckf, filt=wsckf: f(filt, model_inputs)
        )
        times[name] = wall
        estimates_obs[name] = stack_estimates(m0, wsckf_states.mean[1:])
        final_ll[name] = float(wsckf_states.log_normalizing_constant[-1])

    for n_paths in WSCKF_PATH_BUDGETS:
        _run_wsckf(wsckf_label(n_paths), k=CUBATURE_ORDER, n_paths=n_paths)

    if K_PROPORTIONAL:
        k_prop = k_for_delta(dt_min)
        n_prop = kprop_num_paths(k_prop)
        meta["k_prop"] = k_prop
        meta["k_prop_paths"] = n_prop
        _run_wsckf(WSCKF_KPROP_NAME, k=k_prop, n_paths=n_prop)

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
    return dt_min, n_obs, mapes, times, meta


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

delta_t_eff: list[float] = []
n_obs_list: list[int] = []
k_prop_list: list[int | None] = []
k_prop_paths_list: list[int | None] = []
series = {
    name: {"mapex": [], "maped": [], "nll": [], "nll_per_obs": [], "time": []}
    for name in FILTER_NAMES
}

print(f"Horizon={HORIZON_MIN:g} min | solver={SOLVER} | seed={SEED}")
print(
    f"WSC-KF fixed-k: multidim={WSCKF_MULTIDIM_PATHS} | k={CUBATURE_ORDER} | "
    f"path budgets={WSCKF_PATH_BUDGETS} "
    f"(moment-safe default={_DEFAULT_WSCKF_PATHS})"
)
if K_PROPORTIONAL:
    print(
        f"WSC-KF k∝Δt: k=min({K_MAX}, max(1, round(Δt/{K_REF_MIN:g}))), "
        f"paths={_args.k_prop_paths!r} → "
        + ", ".join(
            f"Δt={d:g}→k={k_for_delta(d)}/P={kprop_num_paths(k_for_delta(d))}"
            for d in DELTA_T_MIN_REQ
        )
    )
print(f"Requested Delta_t [min]: {DELTA_T_MIN_REQ}")
print(f"Timing: median of {N_TIMING} runs after JIT warmup")
print("-" * 72)

for dt_req in DELTA_T_MIN_REQ:
    dt_eff, n_obs, mapes, times, meta = run_filters_at_interval(dt_req)
    delta_t_eff.append(dt_eff)
    n_obs_list.append(n_obs)
    k_prop_list.append(meta.get("k_prop"))
    k_prop_paths_list.append(meta.get("k_prop_paths"))
    k_info = ""
    if meta.get("k_prop") is not None:
        k_info = f", k∝Δt={meta['k_prop']} P={meta['k_prop_paths']}"
    print(
        f"Delta_t={dt_eff:6.3f} min  (n_obs={n_obs:4d}, "
        f"dt={dt_eff * 60.0:7.2f} s{k_info})"
    )
    for name in FILTER_NAMES:
        series[name]["mapex"].append(mapes[name]["mapex"])
        series[name]["maped"].append(mapes[name]["maped"])
        series[name]["nll"].append(mapes[name]["nll"])
        series[name]["nll_per_obs"].append(mapes[name]["nll_per_obs"])
        series[name]["time"].append(times[name])
        print(
            f"  {name:10s}  MAPEx={mapes[name]['mapex']:8.2f}%  "
            f"MAPEd={mapes[name]['maped']:8.2f}%  "
            f"NLL/obs={mapes[name]['nll_per_obs']:8.2f}  "
            f"time={times[name] * 1e3:8.2f} ms"
        )

# ---------------------------------------------------------------------------
# Plot MAPE vs Delta_t
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(DOUBLE_COLUMN_WIDTH, 2.65),
    sharex=True,
)
for ax, metric, title in zip(
    axes,
    ("mapex", "maped"),
    ("(a) Tank states", "(b) Disturbance states"),
):
    for name in FILTER_NAMES:
        plot_method(
            ax,
            name,
            delta_t_eff,
            series[name][metric],
            linewidth=1.35,
            markersize=3.6,
        )
    ax.set_xlabel(r"Observation interval $\Delta t$ [min]")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_curve_axis(ax, grid_axis="both")
axes[0].set_ylabel("MAPE [%]")
add_shared_legend(fig, axes[0], ncol=min(4, len(FILTER_NAMES)))
fig.subplots_adjust(left=0.115, right=0.99, top=0.91, bottom=0.29, wspace=0.30)
plot_path = os.path.join(OUTPUT_DIR, "mape_vs_interval.png")
save_publication_figure(fig, plot_path)
plt.close()

# ---------------------------------------------------------------------------
# Plot NLL/obs vs Delta_t
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.45))
for name in FILTER_NAMES:
    plot_method(
        ax,
        name,
        delta_t_eff,
        series[name]["nll_per_obs"],
        linewidth=1.35,
        markersize=3.5,
    )
ax.set_xlabel(r"Observation interval $\Delta t$ [min]")
ax.set_ylabel("NLL\n[nats/observation]")
ax.set_xscale("log")
ax.set_yscale("log")
style_curve_axis(ax, grid_axis="both")
add_shared_legend(fig, ax, ncol=min(3, len(FILTER_NAMES)))
fig.subplots_adjust(left=0.245, right=0.97, top=0.98, bottom=0.39)
nll_path = os.path.join(OUTPUT_DIR, "nll_vs_interval.png")
save_publication_figure(fig, nll_path)
plt.close()

# ---------------------------------------------------------------------------
# Hypothesis test: NLL gap vs EKF grows with Δt for WSC-KF (cubature truncation)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.45))
ekf_nll = np.asarray(series["CD-EKF"]["nll_per_obs"], dtype=np.float64)
for name in FILTER_NAMES:
    if name == "CD-EKF":
        continue
    gap = np.asarray(series[name]["nll_per_obs"], dtype=np.float64) - ekf_nll
    plot_method(
        ax,
        name,
        delta_t_eff,
        gap,
        linewidth=1.35,
        markersize=3.5,
    )
ax.axhline(0.0, color=PALETTE["neutral_gray"], ls="--", lw=0.8)
ax.set_xlabel(r"Observation interval $\Delta t$ [min]")
ax.set_ylabel("$\\Delta$NLL vs. EKF\n[nats/observation]")
ax.set_xscale("log")
style_curve_axis(ax, grid_axis="both")
add_shared_legend(fig, ax, ncol=min(3, max(1, len(FILTER_NAMES) - 1)))
fig.subplots_adjust(left=0.20, right=0.97, top=0.98, bottom=0.39)
gap_path = os.path.join(OUTPUT_DIR, "nll_gap_vs_ekf_interval.png")
save_publication_figure(fig, gap_path)
plt.close()
print(f"Wrote {gap_path}")

# ---------------------------------------------------------------------------
# Plot wall-clock time vs Delta_t
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(DOUBLE_COLUMN_WIDTH, 2.65),
    sharex=True,
)

ax = axes[0]
for name in FILTER_NAMES:
    plot_method(
        ax,
        name,
        delta_t_eff,
        [1e3 * t for t in series[name]["time"]],
        linewidth=1.35,
        markersize=3.6,
    )
ax.set_xlabel(r"Observation interval $\Delta t$ [min]")
ax.set_ylabel("Wall-clock time [ms]")
ax.set_title("(a) Full record")
ax.set_xscale("log")
ax.set_yscale("log")
style_curve_axis(ax, grid_axis="both")

ax = axes[1]
for name in FILTER_NAMES:
    per_obs = [
        1e3 * t / n if n > 0 else np.nan
        for t, n in zip(series[name]["time"], n_obs_list)
    ]
    plot_method(
        ax,
        name,
        delta_t_eff,
        per_obs,
        linewidth=1.35,
        markersize=3.6,
    )
ax.set_xlabel(r"Observation interval $\Delta t$ [min]")
ax.set_ylabel("Time / observation [ms]")
ax.set_title("(b) Per assimilation")
ax.set_xscale("log")
ax.set_yscale("log")
style_curve_axis(ax, grid_axis="both")

add_shared_legend(fig, axes[0], ncol=min(4, len(FILTER_NAMES)))
fig.subplots_adjust(left=0.115, right=0.99, top=0.91, bottom=0.29, wspace=0.31)
timing_path = os.path.join(OUTPUT_DIR, "timing_vs_interval.png")
save_publication_figure(fig, timing_path)
plt.close()

# ---------------------------------------------------------------------------
# Pareto: wall-clock vs accuracy (lower-left is better)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(DOUBLE_COLUMN_WIDTH, 2.65),
    sharex=True,
)
for ax, metric, ylabel, title in zip(
    axes,
    ("mapex", "maped"),
    ("MAPEx [%]", "MAPEd [%]"),
    ("(a) Tank states", "(b) Disturbance states"),
):
    for name in FILTER_NAMES:
        xs = [1e3 * t for t in series[name]["time"]]
        ys = series[name][metric]
        plot_method(
            ax,
            name,
            xs,
            ys,
            linewidth=1.15,
            markersize=3.5,
            alpha=0.9,
        )
    ax.set_xlabel("Wall-clock time [ms]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_curve_axis(ax, grid_axis="both")
add_shared_legend(fig, axes[0], ncol=min(4, len(FILTER_NAMES)))
fig.subplots_adjust(left=0.115, right=0.99, top=0.91, bottom=0.29, wspace=0.31)
pareto_path = os.path.join(OUTPUT_DIR, "pareto_time_vs_mape.png")
save_publication_figure(fig, pareto_path)
plt.close()

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "mape_vs_interval.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = [
        "delta_t_min",
        "n_obs",
        "eval_dt_min",
        "solver",
        "k_prop",
        "k_prop_paths",
    ]
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
        row = [
            dt_eff,
            n_obs_list[i],
            DT_FINE_MIN,
            SOLVER,
            k_prop_list[i],
            k_prop_paths_list[i],
        ]
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
print(f"Wrote {gap_path}")
print(f"Wrote {timing_path}")
print(f"Wrote {pareto_path}")
print(f"Wrote {csv_path}")
