"""Filter-EM / MLE for MFTS ``λ``, compared apples-to-apples across filters.

Shared setup for CRF / CD-EKF / CD-UKF / CD-EnKF / bootstrap PF:
  same truth, matched F̄ schedule, same ``dt``/``dt0``, ``m0,P0,R``, ``λ₀``,
  same outer EM budget.

Gradients
---------
* CRF / EKF / UKF — pathwise ``jax.value_and_grad`` (CRF: checkpointed ODE adjoint)
* EnKF / PF — central finite differences with a **fixed** RNG key (no pathwise
  ODE adjoint through the stochastic forecast)

Panels
------
1. ``L(λ)`` for all five (EnKF/PF: mean ± std over seeds)
2. EM ``λ̂`` traces (AD vs FD M-steps)
3. VJP / FD cost vs series length ``N_obs``
4. Peak RSS of one grad vs inner RK4 steps (fixed ``Δt``, ``N_obs``):
   CRF ckpt ODE adjoint vs CD-EKF / CD-UKF pathwise VJP

Run:
  python experiments/mfts_test/run_em_lambda.py
  python experiments/mfts_test/run_em_lambda.py --n-obs 60 --em-iters 25
  python experiments/mfts_test/run_em_lambda.py --skip-stochastic
  python experiments/mfts_test/run_em_lambda.py --skip-adjoint-scaling
  python experiments/mfts_test/run_em_lambda.py --skip-memory-scaling
  python experiments/mfts_test/run_em_lambda.py --only-memory

Outputs under ``experiments/mfts_test/output/``:
  em_lambda_sensitivity.png, em_lambda_trace.png, em_lambda_adjoint.png,
  em_lambda_memory.png, em_lambda_summary.csv
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
from crf import build_filter
from crf.library import default_recombined_num_paths
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

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(
    description="Filter-EM / adjoint sensitivity for MFTS λ (CRF vs EKF/UKF/EnKF/PF)"
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
    help="Step size on log(λ) in the M-step",
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
    help="Skip EnKF/PF (faster; AD methods only)",
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
    help="Only run the CRF-ckpt vs EKF/UKF peak-RSS vs N_steps plot",
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
SKIP_MEMORY_SCALING = _args.skip_memory_scaling
ONLY_MEMORY = _args.only_memory

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

CRF_NUM_PATHS = default_recombined_num_paths(2, CUBATURE_ORDER)

COLORS = {
    "CD-EKF": "#2E86AB",
    "CD-UKF": "#3DA35D",
    "CD-EnKF": "#F6AE2D",
    "PF": "#F24236",
    "CRF": "#6A4C93",
}
AD_METHODS = ("CRF", "CD-EKF", "CD-UKF")
STOCH_METHODS = ("CD-EnKF", "PF")
ALL_METHODS = AD_METHODS + STOCH_METHODS


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
    """Peak RSS vs N_steps: CRF ckpt ODE adjoint vs EKF/UKF pathwise VJP."""
    print("\n=== Peak grad RSS vs inner RK4 steps (CRF ckpt vs EKF/UKF) ===")
    # Hold assimilation interval and N_obs fixed; only refine dt0 = Δt / N_steps.
    probe = os.path.join(HERE, "grad_memory_probe.py")
    dt_mem = 30.0
    n_obs_mem = 32
    n_steps_grid = (50, 100, 200, 400, 800, 1600, 3200)
    mem_methods = ("CRF-ckpt", "CD-EKF", "CD-UKF")
    style = {
        "CRF-ckpt": (COLORS["CRF"], "s-", "CRF ckpt ODE adjoint"),
        "CD-EKF": (COLORS["CD-EKF"], "^-", "CD-EKF VJP"),
        "CD-UKF": (COLORS["CD-UKF"], "v-", "CD-UKF VJP"),
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

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ns = np.asarray([r["n_steps"] for r in rows], dtype=float)
    series = {
        m: np.asarray([r[m] for r in rows], dtype=float) for m in mem_methods
    }

    for method, (color, fmt, label) in style.items():
        axes[0].plot(ns, series[method], fmt, color=color, lw=1.8, label=label)
    axes[0].set_xlabel(r"RK4 steps per assimilation interval $N_{\mathrm{steps}}$")
    axes[0].set_ylabel("Peak process RSS [MiB]")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].set_title("Absolute peak RSS")

    for method, (color, fmt, label) in style.items():
        axes[1].plot(
            ns,
            series[method] - series[method][0],
            fmt,
            color=color,
            lw=1.8,
            label=label,
        )
    axes[1].set_xlabel(r"RK4 steps per assimilation interval $N_{\mathrm{steps}}$")
    axes[1].set_ylabel(r"$\Delta$RSS vs smallest $N_{\mathrm{steps}}$ [MiB]")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].set_title(r"Growth with ODE resolution")

    fig.suptitle(
        f"Grad memory vs ODE resolution "
        f"(fixed Δt={dt_mem:g}s, N_obs={n_obs_mem})",
        fontsize=11,
    )
    fig.tight_layout()
    mem_path = os.path.join(OUTPUT_DIR, "em_lambda_memory.png")
    fig.savefig(mem_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {mem_path}")
    return rows


if ONLY_MEMORY:
    memory_rows = run_memory_scaling()
    summary_path = os.path.join(OUTPUT_DIR, "em_lambda_summary.csv")
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

def build_crf(*, checkpoint_steps: bool, dt_=None, dt0_=None, ys_=None, n_obs_=None):
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
        num_paths=CRF_NUM_PATHS,
        checkpoint_steps=checkpoint_steps,
        sde_form="ito",
    )


def model_inputs_for(lam: jnp.ndarray, n_obs_: int = N_OBS) -> StepInputs:
    return StepInputs(
        t_idx=jnp.arange(n_obs_ + 1),
        lambda_dist=jnp.full((n_obs_ + 1,), lam),
    )


crf_scan = build_crf(checkpoint_steps=False)
crf_ckpt = build_crf(checkpoint_steps=True)
_run_crf_scan = jit(run_filter, static_argnames=["filter_obj"])
_run_crf_ckpt = jit(run_filter, static_argnames=["filter_obj"])


def crf_loglik(lam: jnp.ndarray, *, checkpoint: bool = True) -> jnp.ndarray:
    filt = crf_ckpt if checkpoint else crf_scan
    run = _run_crf_ckpt if checkpoint else _run_crf_scan
    return run(filt, model_inputs_for(lam)).log_normalizing_constant[-1]


def crf_states(lam: jnp.ndarray):
    return _run_crf_ckpt(crf_ckpt, model_inputs_for(lam))


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
crf_vg = jit(value_and_grad(lambda a: crf_loglik(jnp.exp(a), checkpoint=True)))
crf_vg_scan = jit(value_and_grad(lambda a: crf_loglik(jnp.exp(a), checkpoint=False)))
ekf_vg = jit(value_and_grad(lambda a: ekf_loglik(jnp.exp(a))))
ukf_vg = jit(value_and_grad(lambda a: ukf_loglik(jnp.exp(a))))

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
    if name == "CRF":
        return crf_vg(alpha)
    if name == "CD-EKF":
        return ekf_vg(alpha)
    if name == "CD-UKF":
        return ukf_vg(alpha)
    raise KeyError(name)


def stoch_ll_fn(name: str, key):
    if name == "CD-EnKF":
        return lambda lam: _enkf_ll_jit(lam, key)
    if name == "PF":
        return lambda lam: _pf_ll_jit(lam, key)
    raise KeyError(name)


def loglik_scalar(name: str, lam: float, key=None) -> float:
    lam_j = jnp.asarray(lam, dtype=jnp.float64)
    if name == "CRF":
        return float(crf_loglik(lam_j))
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
# 1. Sensitivity: L(λ) for all methods
# ---------------------------------------------------------------------------

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
for vg in (crf_vg, ekf_vg, ukf_vg):
    jax.block_until_ready(vg(alpha_warm))

for name in AD_METHODS:
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

# CRF gradient check (pathwise vs FD)
print("  CRF ∂L/∂λ AD vs FD (spot checks):")
for lam in (1e-3, 0.05, 0.1, 1.0):
    a = jnp.log(jnp.asarray(lam, dtype=jnp.float64))
    _, g_a = crf_vg(a)
    g_ad = float(g_a) / lam
    _, g_fd_a = fd_value_and_grad_alpha(lambda x: crf_loglik(x), a, eps=1e-4)
    g_fd = float(g_fd_a) / lam
    print(f"    λ={lam:6.3f}  AD={g_ad:10.2f}  FD={g_fd:10.2f}")

methods_plot = [m for m in ALL_METHODS if m in L_by_method]
fig, ax = plt.subplots(figsize=(8, 4.5))
for name in methods_plot:
    ax.plot(lam_grid, L_by_method[name], color=COLORS[name], lw=2.0, label=name)
    if name in L_stoch_std:
        mu, sd = L_by_method[name], L_stoch_std[name]
        ax.fill_between(lam_grid, mu - sd, mu + sd, color=COLORS[name], alpha=0.2)
ax.axvline(LAMBDA_TRUE, color="k", ls="--", lw=1.2, label=r"$\lambda_{\mathrm{true}}$")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("Filter log-likelihood")
ax.legend(fontsize=9, ncol=3)
ax.grid(True, alpha=0.25)
ax.set_title("Shared-objective sensitivity (matched F̄, same data)")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "em_lambda_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.close()

lam_star_crf = float(lam_grid[int(np.argmax(L_by_method["CRF"]))])


# ---------------------------------------------------------------------------
# 2. Filter-EM for every method
# ---------------------------------------------------------------------------

print("\n=== Filter-EM / gradient MLE (shared outer loop) ===")
em_traces: dict[str, list[dict]] = {m: [] for m in methods_plot}
rss_before_em = peak_rss_mb()

for name in methods_plot:
    alpha = jnp.log(jnp.asarray(LAMBDA0, dtype=jnp.float64))
    # Fixed key for stochastic FD path (reproducible common-RNG EM)
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
            L, g_alpha = fd_value_and_grad_alpha(
                stoch_ll_fn(name, em_key),
                alpha,
                eps=eps,
                clip=GRAD_CLIP_ALPHA,
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

# Moment M-step reference from CRF filtered means at λ0 (diagnostic)
crf_means = np.asarray(crf_states(jnp.asarray(LAMBDA0, dtype=jnp.float64)).mean)
times_min = np.arange(crf_means.shape[0]) * (HORIZON_MIN / N_OBS)
F = crf_means[:, 4:6]
bar = np.stack([np.asarray(bar_F_schedule_jax(t)) * np.ones(2) for t in times_min])
dF = F[1:] - F[:-1]
resid = bar[:-1] - F[:-1]
den = np.sum(resid * resid) * dt
lam_moment0 = (
    float(np.clip(np.sum(dF * resid) / den, 1e-6, 5.0)) if den > 0 else float("nan")
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for name in methods_plot:
    rows = em_traces[name]
    ls = "-" if name in AD_METHODS else "--"
    axes[0].plot(
        [r["iter"] for r in rows],
        [r["lambda_"] for r in rows],
        color=COLORS[name],
        lw=2.0,
        ls=ls,
        label=f"{name} ({rows[0]['mode']})",
    )
    axes[1].plot(
        [r["iter"] for r in rows],
        [r["loglik"] for r in rows],
        color=COLORS[name],
        lw=1.8,
        ls=ls,
        label=name,
    )
axes[0].axhline(LAMBDA_TRUE, color="k", ls="--", lw=1.2, label=r"$\lambda_{\mathrm{true}}$")
axes[0].set_xlabel("EM iteration")
axes[0].set_ylabel(r"$\lambda$")
axes[0].legend(fontsize=7, ncol=2)
axes[0].grid(True, alpha=0.25)
axes[0].set_title(r"Parameter recovery (AD solid, FD dashed)")

axes[1].set_xlabel("EM iteration")
axes[1].set_ylabel("Filter log-likelihood")
axes[1].legend(fontsize=8, ncol=3)
axes[1].grid(True, alpha=0.25)
axes[1].set_title("E-step objective")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "em_lambda_trace.png"), dpi=150, bbox_inches="tight")
plt.close()

lam_final = em_traces["CRF"][-1]["lambda_after"]
L_final = em_traces["CRF"][-1]["loglik"]


# ---------------------------------------------------------------------------
# 3. Adjoint / gradient cost: pathwise methods vs FD baselines
# ---------------------------------------------------------------------------

print("\n=== Gradient cost (pathwise AD vs FD) ===")
alpha_bench = jnp.log(jnp.asarray(0.05, dtype=jnp.float64))

jax.block_until_ready(crf_vg_scan(alpha_bench))
jax.block_until_ready(crf_vg(alpha_bench))
jax.block_until_ready(ekf_vg(alpha_bench))
jax.block_until_ready(ukf_vg(alpha_bench))

fwd_t = median_time(lambda: crf_loglik(jnp.exp(alpha_bench), checkpoint=True))
scan_t = median_time(lambda: crf_vg_scan(alpha_bench))
ckpt_t = median_time(lambda: crf_vg(alpha_bench))
ekf_t = median_time(lambda: ekf_vg(alpha_bench))
ukf_t = median_time(lambda: ukf_vg(alpha_bench))

(_, g_scan), (_, g_ckpt) = crf_vg_scan(alpha_bench), crf_vg(alpha_bench)
_, g_ekf = ekf_vg(alpha_bench)
_, g_ukf = ukf_vg(alpha_bench)

print(f"  Forward CRF:                  {fwd_t:.4f}s")
print(f"  CRF VJP (scan AD):            {scan_t:.4f}s  (∂α={float(g_scan):+.4f})")
print(f"  CRF VJP (checkpoint ODE adj): {ckpt_t:.4f}s  (∂α={float(g_ckpt):+.4f})")
print(f"  CD-EKF VJP:                   {ekf_t:.4f}s  (∂α={float(g_ekf):+.4f})")
print(f"  CD-UKF VJP:                   {ukf_t:.4f}s  (∂α={float(g_ukf):+.4f})")

enkf_fd_t = pf_fd_t = float("nan")
if not SKIP_STOCHASTIC:
    enkf_fd_t = median_time(
        lambda: fd_value_and_grad_alpha(stoch_ll_fn("CD-EnKF", key_enkf_em), alpha_bench)
    )
    pf_fd_t = median_time(
        lambda: fd_value_and_grad_alpha(stoch_ll_fn("PF", key_pf_em), alpha_bench)
    )
    print(f"  CD-EnKF FD (2× forward):      {enkf_fd_t:.4f}s")
    print(f"  PF FD (2× forward):           {pf_fd_t:.4f}s")
print(
    "  EnKF/PF have no pathwise ODE adjoint through the Brownian forecast; "
    "FD (fixed key) is the fair gradient baseline."
)

adjoint_scale_rows: list[dict] = []
if not SKIP_ADJOINT_SCALING:
    # Scale with series length at a fixed practical integrator setting — closer
    # to EM cost than refining RK4 within a fixed interval (all methods ∝ steps).
    print("  Grad cost vs N_obs (fixed Δt, dt0)...")
    dt_s = dt  # same assimilation interval as the main EM run
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

        filt_c = build_crf(
            checkpoint_steps=True, dt_=dt_s, dt0_=dt0_s, ys_=ys_s, n_obs_=n_obs_s,
        )
        run_c = jit(run_filter, static_argnames=["filter_obj"])

        def ll_crf_ckpt(alpha_, _fc=filt_c, _rc=run_c, _n=n_obs_s):
            st = _rc(_fc, model_inputs_for(jnp.exp(alpha_), _n))
            return st.log_normalizing_constant[-1]

        def ll_ekf(alpha_, _ys=ys_s):
            return ekf_loglik(jnp.exp(alpha_), dt_=dt_s, dt0_=dt0_s, ys_=_ys)

        def ll_ukf(alpha_, _ys=ys_s):
            return ukf_loglik(jnp.exp(alpha_), dt_=dt_s, dt0_=dt0_s, ys_=_ys)

        vg_crf_c = jit(value_and_grad(ll_crf_ckpt))
        vg_ekf_s = jit(value_and_grad(ll_ekf))
        vg_ukf_s = jit(value_and_grad(ll_ukf))
        for vg in (vg_crf_c, vg_ekf_s, vg_ukf_s):
            jax.block_until_ready(vg(alpha_bench))

        t_crf_c = median_time(lambda: vg_crf_c(alpha_bench), n=max(2, N_TIMING))
        t_ekf = median_time(lambda: vg_ekf_s(alpha_bench), n=max(2, N_TIMING))
        t_ukf = median_time(lambda: vg_ukf_s(alpha_bench), n=max(2, N_TIMING))

        row = dict(
            n_obs=n_obs_s,
            n_sub=n_sub,
            horizon_min=horizon_min_s,
            t_crf_ckpt=t_crf_c,
            t_ekf=t_ekf,
            t_ukf=t_ukf,
        )

        if not SKIP_STOCHASTIC:
            def enkf_ll_fixed(lam, _ys=ys_s):
                return enkf_loglik(lam, key_enkf_em, dt_=dt_s, dt0_=dt0_s, ys_=_ys)

            def pf_ll_fixed(lam, _ys=ys_s):
                return pf_loglik(lam, key_pf_em, dt_=dt_s, dt0_=dt0_s, ys_=_ys)

            t_enkf = median_time(
                lambda: fd_value_and_grad_alpha(enkf_ll_fixed, alpha_bench),
                n=max(2, N_TIMING),
            )
            t_pf = median_time(
                lambda: fd_value_and_grad_alpha(pf_ll_fixed, alpha_bench),
                n=max(2, N_TIMING),
            )
            row["t_enkf_fd"] = t_enkf
            row["t_pf_fd"] = t_pf
        else:
            row["t_enkf_fd"] = float("nan")
            row["t_pf_fd"] = float("nan")

        adjoint_scale_rows.append(row)
        extra = ""
        if not SKIP_STOCHASTIC:
            extra = f"  |  EnKF-FD {row['t_enkf_fd']:.3f}s  PF-FD {row['t_pf_fd']:.3f}s"
        print(
            f"    N_obs={n_obs_s:3d}  (T={horizon_min_s:.1f} min):  "
            f"CRF-ckpt {t_crf_c:.3f}s  EKF {t_ekf:.3f}s  UKF {t_ukf:.3f}s{extra}"
        )

    if adjoint_scale_rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ns = [r["n_obs"] for r in adjoint_scale_rows]
        ax.plot(ns, [r["t_crf_ckpt"] for r in adjoint_scale_rows], "s-",
                color=COLORS["CRF"], label="CRF ckpt ODE adjoint")
        ax.plot(ns, [r["t_ekf"] for r in adjoint_scale_rows], "^-",
                color=COLORS["CD-EKF"], label="CD-EKF VJP")
        ax.plot(ns, [r["t_ukf"] for r in adjoint_scale_rows], "v-",
                color=COLORS["CD-UKF"], label="CD-UKF VJP")
        if not SKIP_STOCHASTIC:
            ax.plot(ns, [r["t_enkf_fd"] for r in adjoint_scale_rows], "x--",
                    color=COLORS["CD-EnKF"], label="EnKF FD (2×fwd)")
            ax.plot(ns, [r["t_pf_fd"] for r in adjoint_scale_rows], "+--",
                    color=COLORS["PF"], label="PF FD (2×fwd)")
        ax.set_xlabel(r"Number of observations $N_{\mathrm{obs}}$")
        ax.set_ylabel("Median grad wall time [s]")
        ax.set_yscale("log")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.25, which="both")
        ax.set_title(
            f"Grad cost vs series length "
            f"(Δt={dt_s:g}s, dt0={dt0_s:g}s, {n_sub} RK4 steps/Δt)\n"
            "AD for CRF/EKF/UKF; FD for EnKF/PF"
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(OUTPUT_DIR, "em_lambda_adjoint.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()


# ---------------------------------------------------------------------------
# 4. Peak RSS vs N_steps (CRF ckpt ODE adjoint vs EKF/UKF VJP)
# ---------------------------------------------------------------------------

memory_rows: list[dict] = []
if not SKIP_MEMORY_SCALING:
    memory_rows = run_memory_scaling()


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

summary_path = os.path.join(OUTPUT_DIR, "em_lambda_summary.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["section", "key", "value"])
    w.writerow(["setup", "lambda_true", LAMBDA_TRUE])
    w.writerow(["setup", "lambda0", LAMBDA0])
    w.writerow(["setup", "n_obs", N_OBS])
    w.writerow(["setup", "cubature_order", CUBATURE_ORDER])
    w.writerow(["setup", "num_paths", CRF_NUM_PATHS])
    w.writerow(["setup", "enkf_particles", ENKF_PARTICLES])
    w.writerow(["setup", "pf_particles", PF_PARTICLES])
    w.writerow(["setup", "skip_stochastic", SKIP_STOCHASTIC])
    w.writerow(["result", "crf_lambda_hat", lam_final])
    w.writerow(["result", "crf_loglik_final", L_final])
    w.writerow(["result", "crf_lambda_grid_argmax", lam_star_crf])
    w.writerow(["result", "moment_mstep_lambda0", lam_moment0])
    w.writerow(["timing", "crf_forward_s", fwd_t])
    w.writerow(["timing", "crf_vjp_scan_s", scan_t])
    w.writerow(["timing", "crf_vjp_ckpt_s", ckpt_t])
    w.writerow(["timing", "ekf_vjp_s", ekf_t])
    w.writerow(["timing", "ukf_vjp_s", ukf_t])
    w.writerow(["timing", "enkf_fd_s", enkf_fd_t])
    w.writerow(["timing", "pf_fd_s", pf_fd_t])
    w.writerow(["timing", "rss_before_em_mib", rss_before_em])
    w.writerow(["timing", "rss_peak_mib", peak_rss_mb()])
    for name, rows in em_traces.items():
        hat = rows[-1]["lambda_after"]
        w.writerow(["em", f"{name}_lambda_hat", hat])
        w.writerow(["em", f"{name}_abs_err", abs(hat - LAMBDA_TRUE)])
        w.writerow(["em", f"{name}_mode", rows[0]["mode"]])
        for r in rows:
            w.writerow(["em", f"{name}_iter_{r['iter']}_lambda", r["lambda_"]])
            w.writerow(["em", f"{name}_iter_{r['iter']}_loglik", r["loglik"]])
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
    f"λ_true={LAMBDA_TRUE}  λ0={LAMBDA0}  CRF P={CRF_NUM_PATHS} k={CUBATURE_ORDER}  "
    f"N_obs={N_OBS}"
)
print(f"Wrote plots + {summary_path}")
print("=" * 72)
