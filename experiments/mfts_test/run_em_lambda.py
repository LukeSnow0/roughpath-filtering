"""Gradient filter-MLE for MFTS ``λ``, compared apples-to-apples across filters.

Shared setup for WSC-KF / CD-EKF / CD-UKF / CD-EnKF / bootstrap PF:
  same truth, matched F̄ schedule, same ``dt``/``dt0``, ``m0,P0,R``, ``λ₀``,
  same outer MLE budget.

Gradients
---------
* All filters — plain pathwise ``jax.value_and_grad`` (no WSC-KF checkpoint / ODE adjoint).
  WSC-KF uses uncheckpointed scan VJP; EnKF/PF use a fixed RNG key (reparameterized noise).

Panels
------
1. ``L(λ)`` (WSC-KF once; EnKF/PF: mean ± std over seeds)
2. MLE ``λ̂`` traces (all methods via plain AD)
3. Plain-AD VJP cost vs series length ``N_obs``
4. Peak RSS of one grad vs inner RK4 steps (fixed ``Δt``, ``N_obs``):
   WSC-KF ckpt ODE adjoint vs CD-EKF / CD-UKF pathwise VJP (memory probe only)

Run:
  python experiments/mfts_test/run_em_lambda.py
  python experiments/mfts_test/run_em_lambda.py --n-obs 60 --em-iters 25
  python experiments/mfts_test/run_em_lambda.py --skip-stochastic
  python experiments/mfts_test/run_em_lambda.py --only-timing
  python experiments/mfts_test/run_em_lambda.py --skip-adjoint-scaling
  python experiments/mfts_test/run_em_lambda.py --skip-memory-scaling
  python experiments/mfts_test/run_em_lambda.py --only-memory

Outputs under ``experiments/mfts_test/output/``:
  mle_lambda_sensitivity.{png,pdf}, mle_lambda_trace.{png,pdf},
  mle_lambda_timing.{png,pdf}, mle_lambda_memory.{png,pdf},
  mle_lambda_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import resource
import subprocess
import sys
import time
from typing import Callable, NamedTuple

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
from jax import jit, random, value_and_grad

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cuthbert import filter as run_filter
from wsckf import build_filter
from wsckf.library import default_recombined_num_paths
from experiments.cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)

from mfts_model import (
    MFTSParams,
    bar_F_schedule_jax,
    init_estimate,
    make_mfts_dynamics,
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
    description="Filter MLE / sensitivity for MFTS λ (WSC-KF vs EKF/UKF/EnKF/PF)"
)
_parser.add_argument("--seed", type=int, default=100)
_parser.add_argument(
    "--cubature-order", type=int, default=3, dest="cubature_order",
)
_parser.add_argument("--n-obs", type=int, default=60, dest="n_obs")
_parser.add_argument("--em-iters", type=int, default=30, dest="em_iters")
_parser.add_argument("--lambda0", type=float, default=2.0e-3)
_parser.add_argument(
    "--lr", type=float, default=0.05,
    help="Step size on log(λ) in the MLE update",
)
_parser.add_argument("--n-timing", type=int, default=3, dest="n_timing")
_parser.add_argument(
    "--enkf-particles", type=int, default=250, dest="enkf_particles",
)
_parser.add_argument(
    "--pf-particles", type=int, default=500, dest="pf_particles",
)
_parser.add_argument(
    "--stoch-seeds", type=int, default=3, dest="stoch_seeds",
    help="RNG replicates for EnKF/PF sensitivity bands",
)
_parser.add_argument(
    "--skip-stochastic",
    action="store_true",
    dest="skip_stochastic",
    help="Skip EnKF/PF (faster; keeps WSC-KF-AD / WSC-KF-FD / EKF / UKF)",
)
_parser.add_argument(
    "--skip-adjoint-scaling",
    action="store_true",
    dest="skip_adjoint_scaling",
)
_parser.add_argument(
    "--skip-memory-scaling",
    action="store_true",
    dest="skip_memory_scaling",
    help="Skip peak-RSS vs N_steps probe (subprocess per point)",
)
_parser.add_argument(
    "--only-memory",
    action="store_true",
    dest="only_memory",
    help="Only run the WSC-KF-ckpt vs EKF/UKF peak-RSS vs N_steps plot",
)
_parser.add_argument(
    "--only-timing",
    action="store_true",
    dest="only_timing",
    help="Skip sensitivity/MLE/memory; only regenerate mle_lambda_timing.png",
)
_args = _parser.parse_args()

SEED = _args.seed
CUBATURE_ORDER = _args.cubature_order
N_OBS = _args.n_obs
EM_ITERS = _args.em_iters
LAMBDA0 = _args.lambda0
LR = _args.lr
N_TIMING = _args.n_timing
ENKF_PARTICLES = _args.enkf_particles
PF_PARTICLES = _args.pf_particles
STOCH_SEEDS = _args.stoch_seeds
SKIP_STOCHASTIC = _args.skip_stochastic
SKIP_ADJOINT_SCALING = _args.skip_adjoint_scaling
SKIP_MEMORY_SCALING = _args.skip_memory_scaling or _args.only_timing
ONLY_MEMORY = _args.only_memory
ONLY_TIMING = _args.only_timing
if ONLY_TIMING:
    SKIP_ADJOINT_SCALING = False

# ---------------------------------------------------------------------------
# Shared MFTS settings
# ---------------------------------------------------------------------------

HORIZON_MIN = 30.0
N_SUBSTEPS_TRUTH = 1000
N_SUBSTEPS_FILTER = 100

LAMBDA_TRUE = 0.1
SIGMA = 5.0
F1, F2 = 5.0, 5.0
FD_EPS_ALPHA = 1e-3  # central FD step on log(λ) for EnKF
FD_EPS_ALPHA_PF = 5e-2  # larger: PF evidence is discontinuous under resampling
GRAD_CLIP_ALPHA = 50.0  # clip |∂L/∂α| for stochastic FD M-steps
ALPHA_MIN = float(np.log(1e-6))
ALPHA_MAX = float(np.log(5.0))

WSCKF_NUM_PATHS = default_recombined_num_paths(2, CUBATURE_ORDER)

COLORS = {
    "WSC-KF-AD": METHOD_COLORS["WSC-KF-AD"],
    "WSC-KF-FD": METHOD_COLORS["WSC-KF-FD"],
    "CD-EKF": METHOD_COLORS["CD-EKF"],
    "CD-UKF": METHOD_COLORS["CD-UKF"],
    "CD-EnKF": METHOD_COLORS["CD-EnKF"],
    "PF": METHOD_COLORS["PF"],
    "WSC-KF": METHOD_COLORS["WSC-KF"],
}
MARKERS = {
    "WSC-KF-AD": METHOD_MARKERS["WSC-KF-AD"],
    "WSC-KF-FD": METHOD_MARKERS["WSC-KF-FD"],
    "CD-EKF": METHOD_MARKERS["CD-EKF"],
    "CD-UKF": METHOD_MARKERS["CD-UKF"],
    "CD-EnKF": METHOD_MARKERS["CD-EnKF"],
    "PF": METHOD_MARKERS["PF"],
    "WSC-KF": METHOD_MARKERS["WSC-KF"],
}
AD_METHODS = ("WSC-KF-AD", "CD-EKF", "CD-UKF", "CD-EnKF", "PF")
FD_METHODS = ()  # plain-AD run: no FD baselines in the MLE loop
STOCH_METHODS = ("CD-EnKF", "PF")
ALL_METHODS = AD_METHODS


class StepInputs(NamedTuple):
    t_idx: jnp.ndarray
    lambda_dist: jnp.ndarray


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def median_time(fn, *args, n: int = N_TIMING) -> float:
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def run_memory_scaling() -> list[dict]:
    """Peak RSS vs N_steps: WSC-KF ckpt ODE adjoint vs EKF/UKF pathwise VJP."""
    print("\n=== Peak grad RSS vs inner RK4 steps (WSC-KF ckpt vs EKF/UKF) ===")
    # Hold assimilation interval and N_obs fixed; only refine dt0 = Δt / N_steps.
    probe = os.path.join(HERE, "grad_memory_probe.py")
    dt_mem = 30.0
    n_obs_mem = 32
    n_steps_grid = (50, 100, 200, 400, 800, 1600, 3200)
    mem_methods = ("WSC-KF-ckpt", "CD-EKF", "CD-UKF")
    style = {
        "WSC-KF-ckpt": (
            COLORS["WSC-KF"],
            MARKERS["WSC-KF"],
            "WSC-KF (checkpointed)",
        ),
        "CD-EKF": (COLORS["CD-EKF"], MARKERS["CD-EKF"], "CD-EKF"),
        "CD-UKF": (COLORS["CD-UKF"], MARKERS["CD-UKF"], "CD-UKF"),
    }
    rows: list[dict] = []

    for n_steps in n_steps_grid:
        dt0_s = dt_mem / n_steps
        row: dict = {
            "delta_t_s": dt_mem,
            "n_steps": n_steps,
            "dt0_s": dt0_s,
            "n_obs": n_obs_mem,
        }
        for method in mem_methods:
            cmd = [
                sys.executable,
                probe,
                "--method",
                method,
                "--delta-t",
                str(dt_mem),
                "--dt0",
                str(dt0_s),
                "--n-obs",
                str(n_obs_mem),
                "--seed",
                str(SEED),
                "--cubature-order",
                str(CUBATURE_ORDER),
            ]
            try:
                out = subprocess.check_output(
                    cmd, text=True, stderr=subprocess.STDOUT, timeout=900
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print(f"  {method} N_steps={n_steps}: probe failed ({exc})")
                row[method] = float("nan")
                continue
            line = [ln for ln in out.strip().splitlines() if ln.strip()][-1]
            rss = float(line.split()[-1])
            row[method] = rss
            print(
                f"  {method:10s}  N_steps={n_steps:4d}  dt0={dt0_s:.4g}s  "
                f"peak RSS={rss:8.1f} MiB"
            )
        rows.append(row)

    if not rows:
        return rows

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.6),
    )
    ns = np.asarray([r["n_steps"] for r in rows], dtype=float)
    series = {
        m: np.asarray([r[m] for r in rows], dtype=float) for m in mem_methods
    }

    for method, (color, marker, label) in style.items():
        axes[0].plot(
            ns,
            series[method],
            color=color,
            marker=marker,
            label=label,
        )
    axes[0].set_xlabel(r"RK4 steps / interval, $N_{\mathrm{steps}}$")
    axes[0].set_ylabel("Peak RSS [MiB]")
    axes[0].set_xscale("log")
    axes[0].set_title("Peak memory")
    style_curve_axis(axes[0])

    for method, (color, marker, label) in style.items():
        axes[1].plot(
            ns,
            series[method] - series[method][0],
            color=color,
            marker=marker,
            label=label,
        )
    axes[1].set_xlabel(r"RK4 steps / interval, $N_{\mathrm{steps}}$")
    axes[1].set_ylabel(r"$\Delta$RSS [MiB]")
    axes[1].set_xscale("log")
    axes[1].set_title("Memory growth")
    style_curve_axis(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.28, top=0.91, wspace=0.37)
    mem_path = os.path.join(OUTPUT_DIR, "mle_lambda_memory.png")
    mem_png, mem_pdf = save_publication_figure(fig, mem_path)
    plt.close()
    print(f"Wrote {mem_png}")
    print(f"Wrote {mem_pdf}")
    return rows


if ONLY_MEMORY:
    memory_rows = run_memory_scaling()
    summary_path = os.path.join(OUTPUT_DIR, "mle_lambda_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        for r in memory_rows:
            for k, v in r.items():
                w.writerow(["memory", f"nsteps{r['n_steps']}_{k}", v])
    print(f"Wrote {summary_path}")
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Simulate truth
# ---------------------------------------------------------------------------

p = MFTSParams()
dt_obs_min = HORIZON_MIN / N_OBS
dt = dt_obs_min * 60.0
dt0 = dt / N_SUBSTEPS_FILTER

true_traj, ys_np, R = simulate_mfts(
    horizon_min=HORIZON_MIN,
    n_obs=N_OBS,
    n_substeps_truth=N_SUBSTEPS_TRUTH,
    p=p,
    lambda_sim=LAMBDA_TRUE,
    sigma_sim=SIGMA,
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

bar_true = lambda t_min: bar_F_schedule_jax(t_min)
_, _, _, obs_fn, _ = make_mfts_dynamics(
    p,
    lambda_dist=LAMBDA_TRUE,
    sigma_dist=SIGMA,
    bar_F3=bar_true,
    bar_F4=bar_true,
    F1=F1,
    F2=F2,
)

key_root = random.key(SEED)
key_root, key_enkf_em, key_pf_em = random.split(key_root, 3)
stoch_keys = random.split(key_root, STOCH_SEEDS)


def dynamics_for(lam: jnp.ndarray):
    """Ito drift/diffusion/Qc for CD filters (absolute time in ``t``)."""
    return make_mfts_dynamics(
        p,
        lambda_dist=lam,
        sigma_dist=SIGMA,
        bar_F3=bar_true,
        bar_F4=bar_true,
        F1=F1,
        F2=F2,
    )


def _shift_dynamics(drift, diffusion, t0_abs: jnp.ndarray):
    def drift_abs(t, x):
        return drift(t0_abs + t, x)

    def diffusion_abs(t, x):
        return diffusion(t0_abs + t, x)

    return drift_abs, diffusion_abs


# ---------------------------------------------------------------------------
# Filter log-likelihoods (shared objective family)
# ---------------------------------------------------------------------------

def build_wsckf(*, checkpoint_steps: bool, dt_=None, dt0_=None, ys_=None, n_obs_=None):
    dt_ = dt if dt_ is None else dt_
    dt0_ = dt0 if dt0_ is None else dt0_
    ys_ = ys if ys_ is None else ys_
    n_obs_ = N_OBS if n_obs_ is None else n_obs_

    def get_init_params(mi: StepInputs):
        return m0, chol_P0

    def get_dynamics(mi: StepInputs):
        t0_abs = (mi.t_idx - 1) * dt_
        drift, L, _, _, _ = dynamics_for(mi.lambda_dist)
        return _shift_dynamics(drift, L, t0_abs)

    def get_observations(mi: StepInputs):
        return obs_fn, chol_R, ys_[mi.t_idx - 1]

    return build_filter(
        get_init_params=get_init_params,
        get_dynamics=get_dynamics,
        get_observations=get_observations,
        noise_dim=2,
        dt=dt_,
        integration_dt=dt0_,
        cubature_order=CUBATURE_ORDER,
        num_paths=WSCKF_NUM_PATHS,
        checkpoint_steps=checkpoint_steps,
        sde_form="ito",
    )


def model_inputs_for(lam: jnp.ndarray, n_obs_: int = N_OBS) -> StepInputs:
    return StepInputs(
        t_idx=jnp.arange(n_obs_ + 1),
        lambda_dist=jnp.full((n_obs_ + 1,), lam),
    )


wsckf_scan = build_wsckf(checkpoint_steps=False)
wsckf_ckpt = build_wsckf(checkpoint_steps=True)
_run_wsckf_scan = jit(run_filter, static_argnames=["filter_obj"])
_run_wsckf_ckpt = jit(run_filter, static_argnames=["filter_obj"])


def wsckf_loglik(lam: jnp.ndarray, *, checkpoint: bool = True) -> jnp.ndarray:
    filt = wsckf_ckpt if checkpoint else wsckf_scan
    run = _run_wsckf_ckpt if checkpoint else _run_wsckf_scan
    return run(filt, model_inputs_for(lam)).log_normalizing_constant[-1]


def wsckf_states(lam: jnp.ndarray):
    return _run_wsckf_ckpt(wsckf_ckpt, model_inputs_for(lam))


def ekf_loglik(lam: jnp.ndarray, *, dt_=None, dt0_=None, ys_=None) -> jnp.ndarray:
    dt_ = dt if dt_ is None else dt_
    dt0_ = dt0 if dt0_ is None else dt0_
    ys_ = ys if ys_ is None else ys_
    drift, L, Qc, _, _ = dynamics_for(lam)
    res = run_cd_ekf(
        drift, obs_fn, L, Qc, R, m0, P0_j, ys_, dt_, dt0_,
        state_order="first", solver="rk4",
    )
    return res.cum_loglik[-1]


def ukf_loglik(lam: jnp.ndarray, *, dt_=None, dt0_=None, ys_=None) -> jnp.ndarray:
    dt_ = dt if dt_ is None else dt_
    dt0_ = dt0 if dt0_ is None else dt0_
    ys_ = ys if ys_ is None else ys_
    drift, L, Qc, _, _ = dynamics_for(lam)
    res = run_cd_ukf(
        drift, obs_fn, L, Qc, R, m0, P0_j, ys_, dt_, dt0_, UKFParams(),
        state_order="first", solver="rk4",
    )
    return res.cum_loglik[-1]


def enkf_loglik(lam: jnp.ndarray, key, *, dt_=None, dt0_=None, ys_=None) -> jnp.ndarray:
    dt_ = dt if dt_ is None else dt_
    dt0_ = dt0 if dt0_ is None else dt0_
    ys_ = ys if ys_ is None else ys_
    drift, L, Qc, _, _ = dynamics_for(lam)
    res = run_cd_enkf(
        drift, obs_fn, L, Qc, R, m0, P0_j, ys_, dt_, key,
        n_particles=ENKF_PARTICLES, noise_dim=2, dt0=dt0_, solver="rk4",
    )
    return res.cum_loglik[-1]


def pf_loglik(lam: jnp.ndarray, key, *, dt_=None, dt0_=None, ys_=None) -> jnp.ndarray:
    dt_ = dt if dt_ is None else dt_
    dt0_ = dt0 if dt0_ is None else dt0_
    ys_ = ys if ys_ is None else ys_
    drift, L, Qc, _, _ = dynamics_for(lam)
    res = run_bootstrap_pf(
        drift, obs_fn, L, Qc, R, m0, P0_j, ys_, dt_, key,
        n_particles=PF_PARTICLES, noise_dim=2, dt0=dt0_, solver="rk4",
    )
    return res.cum_loglik[-1]


# Jitted pathwise value_and_grad on α = log(λ)
# WSC-KF-AD uses plain scan VJP (no checkpointed ODE adjoint).
wsckf_vg_scan = jit(value_and_grad(lambda a: wsckf_loglik(jnp.exp(a), checkpoint=False)))
wsckf_vg = wsckf_vg_scan  # alias: MLE / checks use plain AD
wsckf_vg_ckpt = jit(value_and_grad(lambda a: wsckf_loglik(jnp.exp(a), checkpoint=True)))
ekf_vg = jit(value_and_grad(lambda a: ekf_loglik(jnp.exp(a))))
ukf_vg = jit(value_and_grad(lambda a: ukf_loglik(jnp.exp(a))))
enkf_vg = jit(value_and_grad(lambda a: enkf_loglik(jnp.exp(a), key_enkf_em)))
pf_vg = jit(value_and_grad(lambda a: pf_loglik(jnp.exp(a), key_pf_em)))

_enkf_ll_jit = jit(enkf_loglik)
_pf_ll_jit = jit(pf_loglik)


def fd_value_and_grad_alpha(
    ll_fn: Callable[[jnp.ndarray], jnp.ndarray],
    alpha: jnp.ndarray,
    eps: float = FD_EPS_ALPHA,
    *,
    clip: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Central FD of L(exp(α)) w.r.t. α (two forward passes, fixed randomness)."""
    ap = alpha + eps
    am = alpha - eps
    lp = ll_fn(jnp.exp(ap))
    lm = ll_fn(jnp.exp(am))
    L = 0.5 * (lp + lm)
    g = (lp - lm) / (2.0 * eps)
    if clip is not None:
        g = jnp.clip(g, -clip, clip)
    return L, g


def ad_value_and_grad(name: str, alpha: jnp.ndarray):
    if name == "WSC-KF-AD":
        return wsckf_vg(alpha)
    if name == "CD-EKF":
        return ekf_vg(alpha)
    if name == "CD-UKF":
        return ukf_vg(alpha)
    if name == "CD-EnKF":
        return enkf_vg(alpha)
    if name == "PF":
        return pf_vg(alpha)
    raise KeyError(name)


def fd_ll_fn(name: str, key=None):
    if name == "WSC-KF-FD":
        return lambda lam: wsckf_loglik(lam)
    if name == "CD-EnKF":
        return lambda lam: _enkf_ll_jit(lam, key)
    if name == "PF":
        return lambda lam: _pf_ll_jit(lam, key)
    raise KeyError(name)


def loglik_scalar(name: str, lam: float, key=None) -> float:
    lam_j = jnp.asarray(lam, dtype=jnp.float64)
    if name in ("WSC-KF-AD", "WSC-KF-FD", "WSC-KF"):
        return float(wsckf_loglik(lam_j))
    if name == "CD-EKF":
        return float(ekf_loglik(lam_j))
    if name == "CD-UKF":
        return float(ukf_loglik(lam_j))
    if name == "CD-EnKF":
        return float(_enkf_ll_jit(lam_j, key))
    if name == "PF":
        return float(_pf_ll_jit(lam_j, key))
    raise KeyError(name)


# ---------------------------------------------------------------------------
# 1–2. Sensitivity + MLE (skipped under --only-timing)
# ---------------------------------------------------------------------------

if ONLY_TIMING:
    print("\n=== Skipping sensitivity / MLE (--only-timing) ===")
    lam_final = float("nan")
    L_final = float("nan")
    lam_star_wsckf = float("nan")
    lam_moment0 = float("nan")
    em_traces = {"WSC-KF-AD": [{"lambda_after": float("nan"), "loglik": float("nan")}]}
    rss_before_em = peak_rss_mb()
    L_by_method = {}
    methods_plot: list[str] = []
else:
    # 1. Sensitivity + 2. MLE


    print("\n=== Sensitivity L(λ) across filters ===")

    # Coarse geomspace for the wings + denser coverage around λ_true so the

    # discrete argmax is not an artifact of a sparse log-grid.

    lam_grid = np.unique(

        np.concatenate(

            [

                np.geomspace(1e-3, 1.0, 15),

                np.geomspace(0.04, 0.25, 25),

            ]

        )

    )

    L_by_method: dict[str, np.ndarray] = {}

    L_stoch_std: dict[str, np.ndarray] = {}



    # Warmup AD methods

    alpha_warm = jnp.log(jnp.asarray(0.05, dtype=jnp.float64))

    for vg in (wsckf_vg, ekf_vg, ukf_vg):

        jax.block_until_ready(vg(alpha_warm))

    if not SKIP_STOCHASTIC:

        jax.block_until_ready(enkf_vg(alpha_warm))

        jax.block_until_ready(pf_vg(alpha_warm))



    for name in ("WSC-KF-AD", "CD-EKF", "CD-UKF"):

        vals = []

        for lam in lam_grid:

            vals.append(loglik_scalar(name, float(lam)))

        L_by_method[name] = np.asarray(vals)

        star = float(lam_grid[int(np.argmax(vals))])

        print(f"  {name:8s}  grid-argmax λ≈{star:.4f}  L∈[{min(vals):.1f},{max(vals):.1f}]")






    if not SKIP_STOCHASTIC:

        # Warmup stochastic

        jax.block_until_ready(_enkf_ll_jit(jnp.asarray(0.05), stoch_keys[0]))

        jax.block_until_ready(_pf_ll_jit(jnp.asarray(0.05), stoch_keys[0]))

        for name in STOCH_METHODS:

            reps = []

            for k in stoch_keys:

                vals = [loglik_scalar(name, float(lam), key=k) for lam in lam_grid]

                reps.append(vals)

            reps = np.asarray(reps)

            L_by_method[name] = reps.mean(axis=0)

            L_stoch_std[name] = reps.std(axis=0)

            star = float(lam_grid[int(np.argmax(L_by_method[name]))])

            print(

                f"  {name:8s}  grid-argmax λ≈{star:.4f}  "

                f"L∈[{L_by_method[name].min():.1f},{L_by_method[name].max():.1f}]  "

                f"(mean±std over {STOCH_SEEDS} seeds)"

            )



    # WSC-KF gradient check (pathwise vs FD)

    print("  WSC-KF ∂L/∂λ AD vs FD (spot checks):")

    for lam in (1e-3, 0.05, 0.1, 1.0):

        a = jnp.log(jnp.asarray(lam, dtype=jnp.float64))

        _, g_a = wsckf_vg(a)

        g_ad = float(g_a) / lam

        _, g_fd_a = fd_value_and_grad_alpha(lambda x: wsckf_loglik(x), a, eps=1e-4)

        g_fd = float(g_fd_a) / lam

        print(f"    λ={lam:6.3f}  AD={g_ad:10.2f}  FD={g_fd:10.2f}")



    methods_plot = [m for m in ALL_METHODS if m in L_by_method]

    sens_plot = list(methods_plot)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.65))

    for name in sens_plot:

        label = "WSC-KF" if name == "WSC-KF-AD" else name

        ax.plot(
            lam_grid,
            L_by_method[name],
            color=COLORS[name],
            marker=MARKERS[name],
            markevery=max(1, len(lam_grid) // 7),
            label=label,
        )

        if name in L_stoch_std:

            mu, sd = L_by_method[name], L_stoch_std[name]

            ax.fill_between(
                lam_grid,
                mu - sd,
                mu + sd,
                color=COLORS[name],
                alpha=0.14,
                linewidth=0,
            )

    ax.axvline(
        LAMBDA_TRUE,
        color=PALETTE["black"],
        ls="--",
        lw=1.1,
        label="True",
    )

    ax.set_xscale("log")

    ax.set_xlabel(r"Rate, $\lambda$ [s$^{-1}$]")

    ax.set_ylabel("Log-likelihood [nats]")

    ax.set_title(r"Likelihood sensitivity")
    if L_stoch_std:
        ax.text(
            0.02,
            0.96,
            r"shading: $\pm 1$ SD",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color=PALETTE["neutral_gray"],
        )
    style_curve_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.31, top=0.91)
    save_publication_figure(
        fig,
        os.path.join(OUTPUT_DIR, "mle_lambda_sensitivity.png"),
    )

    plt.close()



    lam_star_wsckf = float(lam_grid[int(np.argmax(L_by_method["WSC-KF-AD"]))])





    # ---------------------------------------------------------------------------

    # 2. Gradient filter-MLE for every method

    # ---------------------------------------------------------------------------



    print("\n=== Gradient filter-MLE (shared outer loop) ===")

    em_traces: dict[str, list[dict]] = {m: [] for m in methods_plot}

    rss_before_em = peak_rss_mb()



    for name in methods_plot:

        alpha = jnp.log(jnp.asarray(LAMBDA0, dtype=jnp.float64))

        # Fixed key for stochastic FD path (reproducible common-RNG MLE)

        if name == "CD-EnKF":

            em_key = key_enkf_em

        elif name == "PF":

            em_key = key_pf_em

        else:

            em_key = None



        print(f"\n  -- {name} --")

        for it in range(EM_ITERS):

            lam = float(jnp.exp(alpha))

            t0 = time.perf_counter()

            if name in AD_METHODS:

                L, g_alpha = ad_value_and_grad(name, alpha)

                jax.block_until_ready((L, g_alpha))

                mode = "AD"

            else:

                eps = FD_EPS_ALPHA_PF if name == "PF" else FD_EPS_ALPHA

                clip = None if name == "WSC-KF-FD" else GRAD_CLIP_ALPHA

                L, g_alpha = fd_value_and_grad_alpha(

                    fd_ll_fn(name, em_key),

                    alpha,

                    eps=eps,

                    clip=clip,

                )

                jax.block_until_ready((L, g_alpha))

                mode = "FD"

            step_t = time.perf_counter() - t0

            alpha = jnp.clip(alpha + LR * g_alpha, ALPHA_MIN, ALPHA_MAX)

            lam_new = float(jnp.exp(alpha))

            em_traces[name].append(

                dict(

                    iter=it,

                    lambda_=lam,

                    loglik=float(L),

                    grad_alpha=float(g_alpha),

                    lambda_after=lam_new,

                    step_time_s=step_t,

                    mode=mode,

                )

            )

            if it % max(1, EM_ITERS // 6) == 0 or it == EM_ITERS - 1:

                print(

                    f"    iter {it:02d}: λ={lam:.5f}  L={float(L):10.2f}  "

                    f"∂L/∂α={float(g_alpha):+.4f}  → {lam_new:.5f}  "

                    f"({mode}, {step_t:.3f}s)"

                )



        lam_hat = float(jnp.exp(alpha))

        print(f"    λ̂={lam_hat:.5f}  |err|={abs(lam_hat - LAMBDA_TRUE):.5f}")



    # Moment diagnostic from WSC-KF filtered means at λ0

    wsckf_means = np.asarray(wsckf_states(jnp.asarray(LAMBDA0, dtype=jnp.float64)).mean)

    times_min = np.arange(wsckf_means.shape[0]) * (HORIZON_MIN / N_OBS)

    F = wsckf_means[:, 4:6]

    bar = np.stack([np.asarray(bar_F_schedule_jax(t)) * np.ones(2) for t in times_min])

    dF = F[1:] - F[:-1]

    resid = bar[:-1] - F[:-1]

    den = np.sum(resid * resid) * dt

    lam_moment0 = (

        float(np.clip(np.sum(dF * resid) / den, 1e-6, 5.0)) if den > 0 else float("nan")

    )



    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.6),
    )

    for name in methods_plot:

        rows = em_traces[name]

        ls = "-" if name in AD_METHODS else "--"
        label = "WSC-KF" if name == "WSC-KF-AD" else name

        axes[0].plot(

            [r["iter"] for r in rows],

            [r["lambda_"] for r in rows],

            color=COLORS[name],

            ls=ls,

            marker=MARKERS[name],

            markevery=max(1, len(rows) // 6),

            label=label,

        )

        axes[1].plot(

            [r["iter"] for r in rows],

            [r["loglik"] for r in rows],

            color=COLORS[name],

            ls=ls,

            marker=MARKERS[name],

            markevery=max(1, len(rows) // 6),

            label=label,

        )

    axes[0].axhline(
        LAMBDA_TRUE,
        color=PALETTE["black"],
        ls="--",
        lw=1.1,
        label="True",
    )

    axes[0].set_xlabel("MLE iteration")

    axes[0].set_ylabel(r"Rate, $\lambda$ [s$^{-1}$]")

    axes[0].set_title("Rate estimate")
    style_curve_axis(axes[0])



    axes[1].set_xlabel("MLE iteration")

    axes[1].set_ylabel("Log-likelihood [nats]")

    axes[1].set_title("Objective")
    style_curve_axis(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(6, len(labels)),
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.28, top=0.91, wspace=0.36)
    save_publication_figure(
        fig,
        os.path.join(OUTPUT_DIR, "mle_lambda_trace.png"),
    )

    plt.close()



    lam_final = em_traces["WSC-KF-AD"][-1]["lambda_after"]

    L_final = em_traces["WSC-KF-AD"][-1]["loglik"]



# ---------------------------------------------------------------------------
# 3. Gradient cost: plain pathwise AD (no WSC-KF checkpoint/adjoint)
# ---------------------------------------------------------------------------

print("\n=== Gradient cost (plain pathwise AD; WSC-KF scan, no ckpt/adjoint) ===")
alpha_bench = jnp.log(jnp.asarray(0.05, dtype=jnp.float64))

jax.block_until_ready(wsckf_vg_scan(alpha_bench))
jax.block_until_ready(wsckf_vg_ckpt(alpha_bench))
jax.block_until_ready(ekf_vg(alpha_bench))
jax.block_until_ready(ukf_vg(alpha_bench))

fwd_t = median_time(lambda: wsckf_loglik(jnp.exp(alpha_bench), checkpoint=True))
scan_t = median_time(lambda: wsckf_vg_scan(alpha_bench))
ckpt_t = median_time(lambda: wsckf_vg_ckpt(alpha_bench))
ekf_t = median_time(lambda: ekf_vg(alpha_bench))
ukf_t = median_time(lambda: ukf_vg(alpha_bench))

(_, g_scan), (_, g_ckpt) = wsckf_vg_scan(alpha_bench), wsckf_vg_ckpt(alpha_bench)
_, g_ekf = ekf_vg(alpha_bench)
_, g_ukf = ukf_vg(alpha_bench)

wsckf_fd_t = median_time(
    lambda: fd_value_and_grad_alpha(fd_ll_fn("WSC-KF-FD"), alpha_bench, eps=FD_EPS_ALPHA)
)

print(f"  Forward WSC-KF:                  {fwd_t:.4f}s")
print(f"  WSC-KF-AD VJP (scan):            {scan_t:.4f}s  (∂α={float(g_scan):+.4f})")
print(f"  WSC-KF-AD VJP (ckpt):            {ckpt_t:.4f}s  (∂α={float(g_ckpt):+.4f})")
print(f"  WSC-KF-FD (2× forward):          {wsckf_fd_t:.4f}s")
print(f"  CD-EKF VJP:                   {ekf_t:.4f}s  (∂α={float(g_ekf):+.4f})")
print(f"  CD-UKF VJP:                   {ukf_t:.4f}s  (∂α={float(g_ukf):+.4f})")

enkf_fd_t = pf_fd_t = float("nan")
if not SKIP_STOCHASTIC:
    enkf_fd_t = median_time(
        lambda: fd_value_and_grad_alpha(fd_ll_fn("CD-EnKF", key_enkf_em), alpha_bench)
    )
    pf_fd_t = median_time(
        lambda: fd_value_and_grad_alpha(fd_ll_fn("PF", key_pf_em), alpha_bench)
    )
    print(f"  CD-EnKF FD (2× forward):      {enkf_fd_t:.4f}s")
    print(f"  PF FD (2× forward):           {pf_fd_t:.4f}s")
print(
    "  Scaling plot: plain VJP for WSC-KF/EKF/UKF/EnKF/PF (fixed key for stochastic)."
)

adjoint_scale_rows: list[dict] = []
if not SKIP_ADJOINT_SCALING:
    # Plain pathwise AD on every filter (WSC-KF scan — no checkpointed ODE adjoint).
    print("  Grad cost vs N_obs (plain pathwise AD, fixed Δt, dt0)...")
    dt_s = dt  # same assimilation interval as the main MLE run
    dt0_s = dt0
    n_sub = max(2, int(round(dt_s / dt0_s)))
    n_obs_grid = (20, 40, 60, 80, 120)

    for n_obs_s in n_obs_grid:
        horizon_min_s = (n_obs_s * dt_s) / 60.0
        _, ys_np_s, _ = simulate_mfts(
            horizon_min=horizon_min_s,
            n_obs=n_obs_s,
            n_substeps_truth=max(N_SUBSTEPS_TRUTH, n_sub * 4),
            p=p,
            lambda_sim=LAMBDA_TRUE,
            sigma_sim=SIGMA,
            F1=F1,
            F2=F2,
            seed=SEED,
        )
        ys_s = jnp.asarray(ys_np_s)

        filt_s = build_wsckf(
            checkpoint_steps=False, dt_=dt_s, dt0_=dt0_s, ys_=ys_s, n_obs_=n_obs_s,
        )
        run_s = jit(run_filter, static_argnames=["filter_obj"])

        def ll_wsckf_scan(alpha_, _fs=filt_s, _rs=run_s, _n=n_obs_s):
            st = _rs(_fs, model_inputs_for(jnp.exp(alpha_), _n))
            return st.log_normalizing_constant[-1]

        def ll_ekf(alpha_, _ys=ys_s):
            return ekf_loglik(jnp.exp(alpha_), dt_=dt_s, dt0_=dt0_s, ys_=_ys)

        def ll_ukf(alpha_, _ys=ys_s):
            return ukf_loglik(jnp.exp(alpha_), dt_=dt_s, dt0_=dt0_s, ys_=_ys)

        vg_wsckf_s = jit(value_and_grad(ll_wsckf_scan))
        vg_ekf_s = jit(value_and_grad(ll_ekf))
        vg_ukf_s = jit(value_and_grad(ll_ukf))
        for vg in (vg_wsckf_s, vg_ekf_s, vg_ukf_s):
            jax.block_until_ready(vg(alpha_bench))

        t_wsckf = median_time(lambda: vg_wsckf_s(alpha_bench), n=max(2, N_TIMING))
        t_ekf = median_time(lambda: vg_ekf_s(alpha_bench), n=max(2, N_TIMING))
        t_ukf = median_time(lambda: vg_ukf_s(alpha_bench), n=max(2, N_TIMING))

        row = dict(
            n_obs=n_obs_s,
            n_sub=n_sub,
            horizon_min=horizon_min_s,
            t_wsckf_scan=t_wsckf,
            t_ekf=t_ekf,
            t_ukf=t_ukf,
        )

        if not SKIP_STOCHASTIC:
            def ll_enkf(alpha_, _ys=ys_s):
                return enkf_loglik(
                    jnp.exp(alpha_), key_enkf_em, dt_=dt_s, dt0_=dt0_s, ys_=_ys
                )

            def ll_pf(alpha_, _ys=ys_s):
                return pf_loglik(
                    jnp.exp(alpha_), key_pf_em, dt_=dt_s, dt0_=dt0_s, ys_=_ys
                )

            vg_enkf_s = jit(value_and_grad(ll_enkf))
            vg_pf_s = jit(value_and_grad(ll_pf))
            jax.block_until_ready(vg_enkf_s(alpha_bench))
            jax.block_until_ready(vg_pf_s(alpha_bench))
            t_enkf = median_time(lambda: vg_enkf_s(alpha_bench), n=max(2, N_TIMING))
            t_pf = median_time(lambda: vg_pf_s(alpha_bench), n=max(2, N_TIMING))
            row["t_enkf_ad"] = t_enkf
            row["t_pf_ad"] = t_pf
        else:
            row["t_enkf_ad"] = float("nan")
            row["t_pf_ad"] = float("nan")

        adjoint_scale_rows.append(row)
        extra = ""
        if not SKIP_STOCHASTIC:
            extra = f"  EnKF-AD {row['t_enkf_ad']:.3f}s  PF-AD {row['t_pf_ad']:.3f}s"
        print(
            f"    N_obs={n_obs_s:3d}  (T={horizon_min_s:.1f} min):  "
            f"WSC-KF-AD {t_wsckf:.3f}s  EKF {t_ekf:.3f}s  UKF {t_ukf:.3f}s{extra}"
        )

    if adjoint_scale_rows:
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.6))
        ns = [r["n_obs"] for r in adjoint_scale_rows]
        ax.plot(
            ns,
            [r["t_wsckf_scan"] for r in adjoint_scale_rows],
            color=COLORS["WSC-KF-AD"],
            marker=MARKERS["WSC-KF-AD"],
            label="WSC-KF",
        )
        ax.plot(
            ns,
            [r["t_ekf"] for r in adjoint_scale_rows],
            color=COLORS["CD-EKF"],
            marker=MARKERS["CD-EKF"],
            label="CD-EKF",
        )
        ax.plot(
            ns,
            [r["t_ukf"] for r in adjoint_scale_rows],
            color=COLORS["CD-UKF"],
            marker=MARKERS["CD-UKF"],
            label="CD-UKF",
        )
        if not SKIP_STOCHASTIC:
            ax.plot(
                ns,
                [r["t_enkf_ad"] for r in adjoint_scale_rows],
                color=COLORS["CD-EnKF"],
                marker=MARKERS["CD-EnKF"],
                label="CD-EnKF",
            )
            ax.plot(
                ns,
                [r["t_pf_ad"] for r in adjoint_scale_rows],
                color=COLORS["PF"],
                marker=MARKERS["PF"],
                label="PF",
            )
        ax.set_xlabel(r"Observations, $N_{\mathrm{obs}}$")
        ax.set_ylabel("Gradient time [s]")
        ax.set_yscale("log")
        ax.set_title("Gradient cost")
        style_curve_axis(ax)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=3,
        )
        fig.subplots_adjust(left=0.19, right=0.985, bottom=0.31, top=0.91)
        timing_png, timing_pdf = save_publication_figure(
            fig,
            os.path.join(OUTPUT_DIR, "mle_lambda_timing.png"),
        )
        plt.close()
        print(f"Wrote {timing_png}")
        print(f"Wrote {timing_pdf}")

if ONLY_TIMING:
    summary_path = os.path.join(OUTPUT_DIR, "mle_lambda_timing_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        w.writerow(["setup", "mode", "plain_pathwise_ad"])
        w.writerow(["timing", "wsckf_forward_s", fwd_t])
        w.writerow(["timing", "wsckf_vjp_scan_s", scan_t])
        w.writerow(["timing", "wsckf_vjp_ckpt_s", ckpt_t])
        w.writerow(["timing", "ekf_vjp_s", ekf_t])
        w.writerow(["timing", "ukf_vjp_s", ukf_t])
        for r in adjoint_scale_rows:
            for k, v in r.items():
                if k.startswith("t_") or k in ("n_obs", "n_sub", "horizon_min"):
                    w.writerow(["scale", f"nobs{r['n_obs']}_{k}", v])
    print(f"Wrote {summary_path}")
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# 4. Peak RSS vs N_steps (WSC-KF ckpt ODE adjoint vs EKF/UKF VJP)
# ---------------------------------------------------------------------------

memory_rows: list[dict] = []
if not SKIP_MEMORY_SCALING:
    memory_rows = run_memory_scaling()


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

summary_path = os.path.join(OUTPUT_DIR, "mle_lambda_summary.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["section", "key", "value"])
    w.writerow(["setup", "lambda_true", LAMBDA_TRUE])
    w.writerow(["setup", "lambda0", LAMBDA0])
    w.writerow(["setup", "n_obs", N_OBS])
    w.writerow(["setup", "cubature_order", CUBATURE_ORDER])
    w.writerow(["setup", "num_paths", WSCKF_NUM_PATHS])
    w.writerow(["setup", "enkf_particles", ENKF_PARTICLES])
    w.writerow(["setup", "pf_particles", PF_PARTICLES])
    w.writerow(["setup", "skip_stochastic", SKIP_STOCHASTIC])
    w.writerow(["result", "wsckf_lambda_hat", lam_final])
    w.writerow(["result", "wsckf_loglik_final", L_final])
    w.writerow(["result", "wsckf_lambda_grid_argmax", lam_star_wsckf])
    w.writerow(["result", "moment_mstep_lambda0", lam_moment0])
    w.writerow(["timing", "wsckf_forward_s", fwd_t])
    w.writerow(["timing", "wsckf_vjp_scan_s", scan_t])
    w.writerow(["timing", "wsckf_vjp_ckpt_s", ckpt_t])
    w.writerow(["timing", "wsckf_fd_s", wsckf_fd_t])
    w.writerow(["timing", "ekf_vjp_s", ekf_t])
    w.writerow(["timing", "ukf_vjp_s", ukf_t])
    w.writerow(["timing", "enkf_fd_s", enkf_fd_t])
    w.writerow(["timing", "pf_fd_s", pf_fd_t])
    w.writerow(["timing", "rss_before_mle_mib", rss_before_em])
    w.writerow(["timing", "rss_peak_mib", peak_rss_mb()])
    for name, rows in em_traces.items():
        hat = rows[-1]["lambda_after"]
        w.writerow(["mle", f"{name}_lambda_hat", hat])
        w.writerow(["mle", f"{name}_abs_err", abs(hat - LAMBDA_TRUE)])
        w.writerow(["mle", f"{name}_mode", rows[0]["mode"]])
        for r in rows:
            w.writerow(["mle", f"{name}_iter_{r['iter']}_lambda", r["lambda_"]])
            w.writerow(["mle", f"{name}_iter_{r['iter']}_loglik", r["loglik"]])
    for name, arr in L_by_method.items():
        for lam, L in zip(lam_grid, arr):
            w.writerow(["sensitivity", f"{name}_lam_{lam:.6g}", L])
    for r in adjoint_scale_rows:
        for k, v in r.items():
            if k.startswith("t_") or k in ("n_obs", "n_sub", "horizon_min"):
                w.writerow(["adjoint", f"nobs{r['n_obs']}_{k}", v])
    for r in memory_rows:
        for k, v in r.items():
            w.writerow(["memory", f"nsteps{r['n_steps']}_{k}", v])

print("\n" + "=" * 72)
print(f"{'Method':10s} {'λ̂':>10s} {'|err|':>10s} {'final L':>12s} {'grad':>6s}")
print("-" * 72)
for name in methods_plot:
    r = em_traces[name][-1]
    hat = r["lambda_after"]
    print(
        f"{name:10s} {hat:10.5f} {abs(hat - LAMBDA_TRUE):10.5f} "
        f"{r['loglik']:12.1f} {r['mode']:>6s}"
    )
print("=" * 72)
print(
    f"λ_true={LAMBDA_TRUE}  λ0={LAMBDA0}  WSC-KF P={WSCKF_NUM_PATHS} k={CUBATURE_ORDER}  "
    f"N_obs={N_OBS}"
)
print(f"Wrote plots + {summary_path}")
print("=" * 72)
