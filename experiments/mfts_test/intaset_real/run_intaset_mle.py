"""Filter-MLE of InTaSet Johansson parameters via WSC-KF / CD-EKF / CD-EnKF.

Maximises the continuous–discrete filter predictive log-likelihood over

    θ = (log k_pump, logit γ₁, logit γ₂, log a₁…a₄)

with tank areas ``A``, sensor bias, ``meas_std``, and ``sigma_u`` held at the
open-loop calibration values (``calibrated_params.json``).

Train / holdout (default)
-------------------------
* Fit on the first ``--fit-min`` minutes of ``4T-M2-train`` (observations every
  ``--delta`` minutes).
* After fitting, score each ``θ̂`` by a **shared reference EnKF** on the full
  train+holdout window (train as warmup; held-out NLL/obs from the suffix).

Compares WSC-KF-AD, CD-EKF-AD, and CD-EnKF-AD (fixed RNG) under a shared Adam
outer loop.

Run:
  uv run --extra experiments python \\
      experiments/mfts_test/intaset_real/run_intaset_mle.py
  uv run --extra experiments python \\
      experiments/mfts_test/intaset_real/run_intaset_mle.py \\
      --mle-iters 20 --fit-min 10 --holdout-min 10 --enkf-particles 50

Outputs under ``experiments/mfts_test/intaset_real/output/``:
  intaset_mle_trace.png, intaset_mle_heldout.png, intaset_mle_timing.png,
  intaset_mle_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import NamedTuple

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for pth in (HERE, EXPERIMENTS_DIR, ROOT_DIR):
    sys.path.insert(0, pth)

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
from jax import jit, value_and_grad

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from cuthbert import filter as run_filter
from wsckf import build_filter
from wsckf.library import product_path_count
from experiments.cd_dynamax_mirror.filters import run_cd_ekf, run_cd_enkf

from intaset_io import load_named, window_trajectory
from intaset_model import (
    IntaSetParams,
    init_estimate,
    load_params,
    mape_pct,
    prepare_filter_grids,
    save_params,
    simulate_open_loop,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="InTaSet filter-MLE of (k_pump, γ, a_out) via WSC-KF / EKF / EnKF"
)
_parser.add_argument("--source", type=str, default="4T-M2-train")
_parser.add_argument(
    "--params",
    type=str,
    default=os.path.join(OUTPUT_DIR, "calibrated_params.json"),
    help="Open-loop warm start + fixed A / noise / bias",
)
_parser.add_argument("--fit-min", type=float, default=15.0, dest="fit_min")
_parser.add_argument("--holdout-min", type=float, default=10.0, dest="holdout_min")
_parser.add_argument("--delta", type=float, default=1.0, help="Observation Δt [min]")
_parser.add_argument("--mle-iters", type=int, default=30, dest="mle_iters")
_parser.add_argument("--lr", type=float, default=0.02, help="Adam step on unconstrained θ")
_parser.add_argument(
    "--grad-clip",
    type=float,
    default=50.0,
    dest="grad_clip",
    help="Clip per-coordinate Adam grads (0 disables)",
)
_parser.add_argument("--dt0", type=float, default=0.15, help="Inner RK4 step [s]")
_parser.add_argument("--lyons-degree", type=int, default=5, dest="lyons_degree")
_parser.add_argument("--cubature-order", type=int, default=1, dest="cubature_order")
_parser.add_argument("--enkf-particles", type=int, default=100, dest="enkf_particles")
_parser.add_argument(
    "--ref-enkf-particles",
    type=int,
    default=250,
    dest="ref_enkf_particles",
)
_parser.add_argument(
    "--ref-seeds",
    type=str,
    default="0,1,2",
    dest="ref_seeds",
)
_parser.add_argument("--n-timing", type=int, default=2, dest="n_timing")
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument(
    "--skip-stochastic",
    action="store_true",
    help="Skip EnKF (WSC-KF + EKF only)",
)
_parser.add_argument(
    "--cold-start",
    action="store_true",
    help="Ignore open-loop θ; start from IntaSetParams defaults (noise still from --params)",
)
_args = _parser.parse_args()

ENKF_PARTICLES = int(_args.enkf_particles)
REF_ENKF_PARTICLES = int(_args.ref_enkf_particles)
N_TIMING = int(_args.n_timing)
DT0 = float(_args.dt0)
SOLVER = "rk4"
WSCKF_K = int(_args.cubature_order)
WSCKF_M = int(_args.lyons_degree)
WSCKF_N_PATHS = product_path_count(2, WSCKF_K, lyons_degree=WSCKF_M)


def _parse_ints(s: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


REF_SEEDS = _parse_ints(_args.ref_seeds)

# ---------------------------------------------------------------------------
# Params + data windows
# ---------------------------------------------------------------------------

if os.path.exists(_args.params):
    P0_CAL, META = load_params(_args.params)
    print(f"Loaded warm-start / fixed noise from {_args.params}")
else:
    P0_CAL, META = IntaSetParams(), {}
    print(f"WARNING: {_args.params} missing; using IntaSetParams defaults")

P_FIXED = P0_CAL  # A, bias, meas_std, sigma_u, g, h_eps
if _args.cold_start:
    THETA_INIT_SRC = IntaSetParams(
        A=P_FIXED.A,
        meas_std=P_FIXED.meas_std,
        height_bias=P_FIXED.height_bias,
        sigma_u=P_FIXED.sigma_u,
    )
else:
    THETA_INIT_SRC = P0_CAL


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def pack_theta(p: IntaSetParams) -> np.ndarray:
    return np.array(
        [
            np.log(max(p.k_pump, 1e-3)),
            _logit(p.gamma1),
            _logit(p.gamma2),
            *np.log(np.maximum(np.asarray(p.a_out), 1e-6)),
        ],
        dtype=np.float64,
    )


def unpack_theta_np(theta: np.ndarray) -> IntaSetParams:
    return IntaSetParams(
        gamma1=float(1.0 / (1.0 + np.exp(-theta[1]))),
        gamma2=float(1.0 / (1.0 + np.exp(-theta[2]))),
        A=P_FIXED.A,
        a_out=tuple(float(x) for x in np.exp(theta[3:7])),
        k_pump=float(np.exp(theta[0])),
        meas_std=P_FIXED.meas_std,
        height_bias=P_FIXED.height_bias,
        sigma_u=P_FIXED.sigma_u,
        g=P_FIXED.g,
        h_eps=P_FIXED.h_eps,
    )


def unpack_theta_jax(theta: jnp.ndarray):
    k = jnp.exp(theta[0])
    g1 = jax.nn.sigmoid(theta[1])
    g2 = jax.nn.sigmoid(theta[2])
    a = jnp.exp(theta[3:7])
    return k, g1, g2, a


THETA0_NP = pack_theta(THETA_INIT_SRC)
THETA0 = jnp.asarray(THETA0_NP)

traj_full = load_named(_args.source)
fit_sec = float(_args.fit_min) * 60.0
hold_sec = float(_args.holdout_min) * 60.0
total_sec = fit_sec + hold_sec
traj_full_win = window_trajectory(
    traj_full, start_sec=0.0, duration_sec=total_sec, window_id=0
)
traj_train = window_trajectory(
    traj_full, start_sec=0.0, duration_sec=fit_sec, window_id=0
)

grids_full = prepare_filter_grids(
    traj_full_win,
    delta_t_min=float(_args.delta),
    horizon_min=float(_args.fit_min) + float(_args.holdout_min),
    height_bias=P_FIXED.height_bias,
)
grids_train = prepare_filter_grids(
    traj_train,
    delta_t_min=float(_args.delta),
    horizon_min=float(_args.fit_min),
    height_bias=P_FIXED.height_bias,
)

T_FINE_FULL = grids_full["t_fine_sec"]
U_FINE_FULL = grids_full["u_fine"]
H_FINE_FULL = grids_full["h_fine"]
YS_FULL = jnp.asarray(grids_full["ys"])
N_OBS = int(grids_full["n_obs"])

T_FINE_TRAIN = grids_train["t_fine_sec"]
U_FINE_TRAIN = grids_train["u_fine"]
H_FINE_TRAIN = grids_train["h_fine"]
YS_TRAIN = jnp.asarray(grids_train["ys"])
N_TRAIN = int(grids_train["n_obs"])
N_HELD = N_OBS - N_TRAIN
DT = float(grids_train["dt_obs_sec"])
assert abs(DT - float(grids_full["dt_obs_sec"])) < 1e-9

x0_np, P0_np = init_estimate(H_FINE_TRAIN[0], P_FIXED)
M0 = jnp.asarray(x0_np, dtype=jnp.float64)
P0 = jnp.asarray(P0_np, dtype=jnp.float64)
CHOL_P0 = jnp.linalg.cholesky(P0)
R = jnp.diag(jnp.full(4, P_FIXED.meas_std**2))
CHOL_R = jnp.linalg.cholesky(R)
SIGMA_U = float(P_FIXED.sigma_u)
A_FIXED = jnp.asarray(P_FIXED.A, dtype=jnp.float64)
G_FIXED = float(P_FIXED.g)
H_EPS = float(P_FIXED.h_eps)


def _obs_fn(x):
    return x


print(
    f"InTaSet filter-MLE: source={_args.source}  "
    f"train={_args.fit_min:g} min ({N_TRAIN} obs)  "
    f"hold={_args.holdout_min:g} min ({N_HELD} obs)  "
    f"Δt={_args.delta:g} min ({DT:g}s)  dt0={DT0:g}s"
)
print(
    f"  WSC-KF product m={WSCKF_M} k={WSCKF_K} → {WSCKF_N_PATHS} paths;  "
    f"EnKF fit={ENKF_PARTICLES}p  ref={REF_ENKF_PARTICLES}p"
)
print(
    f"  warm θ: k={THETA_INIT_SRC.k_pump:.3g}  "
    f"γ=({THETA_INIT_SRC.gamma1:.3f},{THETA_INIT_SRC.gamma2:.3f})  "
    f"meas_std={P_FIXED.meas_std:.3f}  sigma_u={SIGMA_U:.3g}"
)


# ---------------------------------------------------------------------------
# Dynamics from unconstrained θ (JAX)
# ---------------------------------------------------------------------------

def make_dynamics_theta(
    theta: jnp.ndarray,
    *,
    t_sec: jnp.ndarray,
    pumps: jnp.ndarray,
    sigma_u: float = SIGMA_U,
):
    k, g1, g2, a = unpack_theta_jax(theta)
    A = A_FIXED
    t_grid = jnp.asarray(t_sec, dtype=jnp.float64)
    u_grid = jnp.asarray(pumps, dtype=jnp.float64)
    t_lo, t_hi = t_grid[0], t_grid[-1]

    def _pump_frac(t):
        tt = jnp.clip(t, t_lo, t_hi)
        u1 = jnp.interp(tt, t_grid, u_grid[:, 0])
        u2 = jnp.interp(tt, t_grid, u_grid[:, 1])
        return jnp.array([u1, u2]) / 100.0

    def drift_fn(t, x):
        h = x
        q = a * jnp.sqrt(2.0 * G_FIXED * jnp.maximum(h, H_EPS))
        u1, u2 = _pump_frac(t)
        f1 = k * u1
        f2 = k * u2
        dh1 = (-q[0] + q[2] + g1 * f1) / A[0]
        dh2 = (-q[1] + q[3] + g2 * f2) / A[1]
        dh3 = (-q[2] + (1.0 - g2) * f2) / A[2]
        dh4 = (-q[3] + (1.0 - g1) * f1) / A[3]
        return jnp.array([dh1, dh2, dh3, dh4])

    def L_fn(t, x):
        s = sigma_u
        return jnp.array(
            [
                [g1 * s / A[0], 0.0],
                [0.0, g2 * s / A[1]],
                [0.0, (1.0 - g2) * s / A[2]],
                [(1.0 - g1) * s / A[3], 0.0],
            ]
        )

    def Qc_fn(t, x):
        return jnp.eye(2)

    return drift_fn, L_fn, Qc_fn


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class MInputs(NamedTuple):
    t_idx: jnp.ndarray
    theta: jnp.ndarray  # (n_steps, 7)


def _mi(theta: jnp.ndarray, n_obs: int) -> MInputs:
    th = jnp.asarray(theta, dtype=jnp.float64)
    return MInputs(
        t_idx=jnp.arange(n_obs + 1),
        theta=jnp.broadcast_to(th, (n_obs + 1, th.shape[0])),
    )


def build_wsckf(ys, t_fine, u_fine, *, checkpoint_steps: bool):
    t_j = jnp.asarray(t_fine, dtype=jnp.float64)
    u_j = jnp.asarray(u_fine, dtype=jnp.float64)

    def get_init_params(mi):
        return M0, CHOL_P0

    def get_dynamics(mi):
        drift, L, _ = make_dynamics_theta(mi.theta, t_sec=t_j, pumps=u_j)
        return drift, L

    def get_observations(mi):
        return _obs_fn, CHOL_R, ys[mi.t_idx - 1]

    return build_filter(
        get_init_params=get_init_params,
        get_dynamics=get_dynamics,
        get_observations=get_observations,
        noise_dim=2,
        dt=DT,
        integration_dt=DT0,
        cubature_order=WSCKF_K,
        lyons_degree=WSCKF_M,
        multidim_paths="product",
        sde_form="ito",
        checkpoint_steps=checkpoint_steps,
    )


_run = jit(run_filter, static_argnames=["filter_obj"])

WSCKF_TRAIN = build_wsckf(YS_TRAIN, T_FINE_TRAIN, U_FINE_TRAIN, checkpoint_steps=False)
WSCKF_TRAIN_CKPT = build_wsckf(YS_TRAIN, T_FINE_TRAIN, U_FINE_TRAIN, checkpoint_steps=True)
WSCKF_FULL = build_wsckf(YS_FULL, T_FINE_FULL, U_FINE_FULL, checkpoint_steps=False)


def wsckf_train_loglik(theta, *, checkpoint=True):
    filt = WSCKF_TRAIN_CKPT if checkpoint else WSCKF_TRAIN
    return _run(filt, _mi(theta, N_TRAIN)).log_normalizing_constant[-1]


def wsckf_full_cum(theta):
    return _run(WSCKF_FULL, _mi(theta, N_OBS)).log_normalizing_constant


def ekf_loglik_on(ys, t_fine, u_fine, theta):
    drift, L, Qc = make_dynamics_theta(
        theta, t_sec=jnp.asarray(t_fine), pumps=jnp.asarray(u_fine)
    )
    return run_cd_ekf(
        drift,
        _obs_fn,
        L,
        Qc,
        R,
        M0,
        P0,
        ys,
        DT,
        DT0,
        state_order="first",
        solver=SOLVER,
    ).cum_loglik


def enkf_loglik_on(ys, t_fine, u_fine, theta, key, *, n_particles: int):
    drift, L, Qc = make_dynamics_theta(
        theta, t_sec=jnp.asarray(t_fine), pumps=jnp.asarray(u_fine)
    )
    return run_cd_enkf(
        drift,
        _obs_fn,
        L,
        Qc,
        R,
        M0,
        P0,
        ys,
        DT,
        key,
        n_particles=n_particles,
        noise_dim=2,
        dt0=DT0,
        solver=SOLVER,
    ).cum_loglik


def ekf_train_loglik(theta):
    return ekf_loglik_on(YS_TRAIN, T_FINE_TRAIN, U_FINE_TRAIN, theta)[-1]


def enkf_train_loglik(theta, key):
    return enkf_loglik_on(
        YS_TRAIN,
        T_FINE_TRAIN,
        U_FINE_TRAIN,
        theta,
        key,
        n_particles=ENKF_PARTICLES,
    )[-1]


def heldout_nll_per_obs(cum_ll, n_train: int, n_obs: int) -> float:
    if n_train >= n_obs:
        return float("nan")
    ll = np.asarray(cum_ll, dtype=np.float64)
    if ll.shape[0] == n_obs + 1:
        held = float(ll[-1] - ll[n_train])
    elif ll.shape[0] == n_obs:
        base = float(ll[n_train - 1]) if n_train > 0 else 0.0
        held = float(ll[-1] - base)
    else:
        raise ValueError(f"unexpected cum_ll length {ll.shape[0]} for n_obs={n_obs}")
    return -held / float(n_obs - n_train)


def ref_enkf_heldout_nll(theta, *, n_particles: int, seeds: list[int]):
    vals = []
    th = jnp.asarray(theta, dtype=jnp.float64)
    for s in seeds:
        key = jr.PRNGKey(s + 1009 + 17 * n_particles)
        cum = enkf_loglik_on(
            YS_FULL,
            T_FINE_FULL,
            U_FINE_FULL,
            th,
            key,
            n_particles=n_particles,
        )
        vals.append(heldout_nll_per_obs(cum, N_TRAIN, N_OBS))
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# Gradients on train LL only.
wsckf_vg = jit(
    value_and_grad(lambda th: wsckf_train_loglik(th, checkpoint=True))
)
ekf_vg = jit(value_and_grad(ekf_train_loglik))
_enkf_key_ad = jr.PRNGKey(_args.seed + 17)
enkf_vg = jit(
    value_and_grad(lambda th: enkf_train_loglik(th, _enkf_key_ad))
)
_wsckf_train_ll = jit(lambda th: wsckf_train_loglik(th, checkpoint=False))
_ekf_train_ll = jit(ekf_train_loglik)
_enkf_train_ll = jit(enkf_train_loglik)


def median_time(fn, *, n: int = N_TIMING) -> float:
    out = fn()
    jax.block_until_ready(out)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def adam_mle(vg_fn, theta0: np.ndarray):
    th = np.asarray(theta0, dtype=np.float64).copy()
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(th)
    v = np.zeros_like(th)
    th_tr, ll_tr = [], []
    clip = float(_args.grad_clip)
    for t in range(1, _args.mle_iters + 1):
        val, g = vg_fn(jnp.asarray(th))
        ll = float(val)
        g = np.asarray(g, dtype=np.float64)
        if clip > 0.0:
            g = np.clip(g, -clip, clip)
        th_tr.append(th.copy())
        ll_tr.append(ll)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        th = th + _args.lr * mhat / (np.sqrt(vhat) + eps)
    # Final evaluation after last update.
    ll_final = float(vg_fn(jnp.asarray(th))[0])
    th_tr.append(th.copy())
    ll_tr.append(ll_final)
    return np.asarray(th_tr), np.asarray(ll_tr), th


def open_loop_mape(theta: np.ndarray, *, train: bool) -> float:
    p = unpack_theta_np(theta)
    if train:
        H = H_FINE_TRAIN
        t = T_FINE_TRAIN
        U = U_FINE_TRAIN
    else:
        # Holdout segment only (relative times restart at 0 in window).
        seg = window_trajectory(
            traj_full,
            start_sec=fit_sec,
            duration_sec=hold_sec,
            window_id=1,
        )
        H = seg.heights - np.asarray(P_FIXED.height_bias)
        t = seg.t_sec - seg.t_sec[0]
        U = seg.pumps
    hat = simulate_open_loop(p, t, U, np.maximum(H[0], H_EPS))
    return mape_pct(hat, H)


# ---------------------------------------------------------------------------
def main():
    stochastic = not _args.skip_stochastic
    methods = ["WSC-KF-AD", "CD-EKF"]
    if stochastic:
        methods.append("CD-EnKF")

    # ---- timing ----
    print(f"\n=== Median wall time (n={N_TIMING}, N_train={N_TRAIN}) ===")
    t_wsckf_fwd = median_time(lambda: _wsckf_train_ll(THETA0))
    t_wsckf_ad = median_time(lambda: wsckf_vg(THETA0))
    t_ekf_fwd = median_time(lambda: _ekf_train_ll(THETA0))
    t_ekf_ad = median_time(lambda: ekf_vg(THETA0))
    print(f"  WSC-KF forward:     {t_wsckf_fwd:.4f}s")
    print(f"  WSC-KF-AD:          {t_wsckf_ad:.4f}s")
    print(f"  CD-EKF forward:  {t_ekf_fwd:.4f}s")
    print(f"  CD-EKF-AD:       {t_ekf_ad:.4f}s")
    t_enkf_fwd = t_enkf_ad = float("nan")
    if stochastic:
        key_t = jr.PRNGKey(_args.seed + 1)
        t_enkf_fwd = median_time(lambda: _enkf_train_ll(THETA0, key_t))
        t_enkf_ad = median_time(lambda: enkf_vg(THETA0))
        print(f"  CD-EnKF forward: {t_enkf_fwd:.4f}s  ({ENKF_PARTICLES}p)")
        print(f"  CD-EnKF-AD:      {t_enkf_ad:.4f}s")

    # ---- Adam MLE ----
    print(f"\n=== Adam filter-MLE ({_args.mle_iters} iters, train only) ===")
    traces: dict[str, dict] = {}
    walls: dict[str, float] = {}
    hats: dict[str, np.ndarray] = {}

    for name, vg in (("WSC-KF-AD", wsckf_vg), ("CD-EKF", ekf_vg)):
        t0 = time.perf_counter()
        th_tr, ll_tr, th_hat = adam_mle(vg, THETA0_NP)
        walls[name] = time.perf_counter() - t0
        traces[name] = {"theta": th_tr, "ll": ll_tr}
        hats[name] = th_hat
        p = unpack_theta_np(th_hat)
        print(
            f"  {name:8s}  wall={walls[name]:6.1f}s  "
            f"L0={ll_tr[0]:.1f} → L*={ll_tr[-1]:.1f}  "
            f"k={p.k_pump:.3g} γ=({p.gamma1:.3f},{p.gamma2:.3f})"
        )

    if stochastic:
        name = "CD-EnKF"
        t0 = time.perf_counter()
        th_tr, ll_tr, th_hat = adam_mle(enkf_vg, THETA0_NP)
        walls[name] = time.perf_counter() - t0
        traces[name] = {"theta": th_tr, "ll": ll_tr}
        hats[name] = th_hat
        p = unpack_theta_np(th_hat)
        print(
            f"  {name:8s}  wall={walls[name]:6.1f}s  "
            f"L0={ll_tr[0]:.1f} → L*={ll_tr[-1]:.1f}  "
            f"k={p.k_pump:.3g} γ=({p.gamma1:.3f},{p.gamma2:.3f})"
        )

    # Open-loop LS baseline (θ from calibration JSON, or defaults).
    th_ol = pack_theta(P0_CAL)
    ll_ol = {
        "WSC-KF-AD": float(_wsckf_train_ll(jnp.asarray(th_ol))),
        "CD-EKF": float(_ekf_train_ll(jnp.asarray(th_ol))),
    }
    if stochastic:
        ll_ol["CD-EnKF"] = float(
            _enkf_train_ll(jnp.asarray(th_ol), jr.PRNGKey(_args.seed + 99))
        )

    # ---- Held-out reference score ----
    print(
        f"\n=== Reference EnKF held-out NLL/obs "
        f"({REF_ENKF_PARTICLES}p, seeds={REF_SEEDS}) ==="
    )
    held: dict[str, tuple[float, float]] = {}
    held["open-loop"] = ref_enkf_heldout_nll(
        th_ol, n_particles=REF_ENKF_PARTICLES, seeds=REF_SEEDS
    )
    print(
        f"  {'open-loop':8s}  NLL/obs={held['open-loop'][0]:.4f} "
        f"± {held['open-loop'][1]:.4f}"
    )
    for name in methods:
        held[name] = ref_enkf_heldout_nll(
            hats[name], n_particles=REF_ENKF_PARTICLES, seeds=REF_SEEDS
        )
        print(
            f"  {name:8s}  NLL/obs={held[name][0]:.4f} ± {held[name][1]:.4f}"
        )

    # Open-loop MAPE at each θ̂ (secondary diagnostic).
    mape_train = {"open-loop": open_loop_mape(th_ol, train=True)}
    mape_hold = {"open-loop": open_loop_mape(th_ol, train=False)}
    for name in methods:
        mape_train[name] = open_loop_mape(hats[name], train=True)
        mape_hold[name] = open_loop_mape(hats[name], train=False)

    print("\n=== Open-loop MAPE at θ̂ (diagnostic; not the MLE objective) ===")
    for name in ["open-loop", *methods]:
        print(
            f"  {name:8s}  train MAPE={mape_train[name]:6.2f}%  "
            f"hold MAPE={mape_hold[name]:6.2f}%"
        )

    # ---- Plots ----
    colors = {"WSC-KF-AD": "#6A4C93", "CD-EKF": "#2E86AB", "CD-EnKF": "#D97706"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for name in methods:
        ax.plot(
            np.arange(1, len(traces[name]["ll"]) + 1),
            traces[name]["ll"],
            color=colors[name],
            lw=1.6,
            label=name,
        )
        ax.axhline(ll_ol[name], color=colors[name], ls="--", lw=0.9, alpha=0.7)
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("train log-likelihood")
    ax.set_title("Filter-MLE traces (dashed = open-loop θ)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    # k_pump and γ1 traces
    for name in methods:
        ths = traces[name]["theta"]
        k = np.exp(ths[:, 0])
        ax.plot(
            np.arange(1, len(k) + 1),
            k,
            color=colors[name],
            lw=1.4,
            label=name,
        )
    ax.axhline(float(np.exp(th_ol[0])), color="0.4", ls="--", lw=1.0, label="open-loop")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel(r"$k_{\mathrm{pump}}$")
    ax.set_title("Pump-gain trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    trace_path = os.path.join(OUTPUT_DIR, "intaset_mle_trace.png")
    fig.savefig(trace_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {trace_path}")

    # Held-out bars
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = ["open-loop", *methods]
    means = [held[n][0] for n in labels]
    stds = [held[n][1] for n in labels]
    bar_colors = ["0.55", *[colors[n] for n in methods]]
    xpos = np.arange(len(labels))
    ax.bar(
        xpos,
        means,
        yerr=stds,
        color=bar_colors,
        edgecolor="0.25",
        lw=0.6,
        capsize=4,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("held-out NLL / obs (ref EnKF)")
    ax.set_title(
        f"InTaSet held-out NLL ({_args.source}, "
        f"train {_args.fit_min:g}+hold {_args.holdout_min:g} min)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    held_path = os.path.join(OUTPUT_DIR, "intaset_mle_heldout.png")
    fig.savefig(held_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {held_path}")

    # Timing
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    timing_rows = [
        ("WSC-KF fwd", t_wsckf_fwd, colors["WSC-KF-AD"]),
        ("WSC-KF-AD", t_wsckf_ad, colors["WSC-KF-AD"]),
        ("EKF fwd", t_ekf_fwd, colors["CD-EKF"]),
        ("EKF-AD", t_ekf_ad, colors["CD-EKF"]),
    ]
    if stochastic:
        timing_rows += [
            (f"EnKF fwd\n({ENKF_PARTICLES}p)", t_enkf_fwd, colors["CD-EnKF"]),
            ("EnKF-AD", t_enkf_ad, colors["CD-EnKF"]),
        ]
    labs = [r[0] for r in timing_rows]
    times = np.array([r[1] for r in timing_rows], dtype=float)
    cols = [r[2] for r in timing_rows]
    xpos = np.arange(len(labs))
    bars = ax.bar(xpos, times, color=cols, edgecolor="0.3", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("median wall [s]")
    ax.set_yscale("log")
    ax.set_title(f"Per-eval on train (median of {N_TIMING})")
    ax.grid(axis="y", alpha=0.3, which="both")
    for b, t in zip(bars, times):
        if np.isfinite(t) and t > 0:
            ax.text(
                b.get_x() + b.get_width() / 2,
                t * 1.2,
                f"{t:.3f}s",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax = axes[1]
    mle_labs = list(methods)
    mle_t = [walls[n] for n in methods]
    mle_c = [colors[n] for n in methods]
    xpos = np.arange(len(mle_labs))
    bars = ax.bar(xpos, mle_t, color=mle_c, edgecolor="0.3", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(mle_labs, fontsize=9)
    ax.set_ylabel("wall [s]")
    ax.set_yscale("log")
    ax.set_title(f"Adam MLE ({_args.mle_iters} iters)")
    ax.grid(axis="y", alpha=0.3, which="both")
    for b, t in zip(bars, mle_t):
        if np.isfinite(t) and t > 0:
            ax.text(
                b.get_x() + b.get_width() / 2,
                t * 1.15,
                f"{t:.1f}s",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    timing_path = os.path.join(OUTPUT_DIR, "intaset_mle_timing.png")
    fig.savefig(timing_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {timing_path}")

    # ---- CSV + best params ----
    summary_path = os.path.join(OUTPUT_DIR, "intaset_mle_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "train_ll_init",
                "train_ll_final",
                "mle_wall_s",
                "heldout_nll_mean",
                "heldout_nll_std",
                "mape_train_pct",
                "mape_hold_pct",
                "k_pump",
                "gamma1",
                "gamma2",
                "a1",
                "a2",
                "a3",
                "a4",
                "fwd_s",
                "ad_s",
            ]
        )
        # open-loop row
        p_ol = unpack_theta_np(th_ol)
        w.writerow(
            [
                "open-loop",
                "",
                "",
                "",
                held["open-loop"][0],
                held["open-loop"][1],
                mape_train["open-loop"],
                mape_hold["open-loop"],
                p_ol.k_pump,
                p_ol.gamma1,
                p_ol.gamma2,
                *p_ol.a_out,
                "",
                "",
            ]
        )
        fwd = {
            "WSC-KF-AD": t_wsckf_fwd,
            "CD-EKF": t_ekf_fwd,
            "CD-EnKF": t_enkf_fwd,
        }
        ads = {
            "WSC-KF-AD": t_wsckf_ad,
            "CD-EKF": t_ekf_ad,
            "CD-EnKF": t_enkf_ad,
        }
        for name in methods:
            p = unpack_theta_np(hats[name])
            w.writerow(
                [
                    name,
                    traces[name]["ll"][0],
                    traces[name]["ll"][-1],
                    walls[name],
                    held[name][0],
                    held[name][1],
                    mape_train[name],
                    mape_hold[name],
                    p.k_pump,
                    p.gamma1,
                    p.gamma2,
                    *p.a_out,
                    fwd[name],
                    ads[name],
                ]
            )
    print(f"Wrote {summary_path}")

    # Save WSC-KF MLE params for downstream use.
    best_name = min(methods, key=lambda n: held[n][0])
    best_p = unpack_theta_np(hats[best_name])
    out_params = os.path.join(OUTPUT_DIR, "mle_params_best.json")
    save_params(
        out_params,
        best_p,
        meta={
            "method": best_name,
            "source": _args.source,
            "fit_min": float(_args.fit_min),
            "holdout_min": float(_args.holdout_min),
            "delta_min": float(_args.delta),
            "heldout_nll_mean": held[best_name][0],
            "heldout_nll_std": held[best_name][1],
            "train_ll_final": float(traces[best_name]["ll"][-1]),
            "warm_start": not _args.cold_start,
            "open_loop_params": _args.params,
        },
    )
    print(f"Wrote {out_params} (best by held-out NLL: {best_name})")


if __name__ == "__main__":
    main()
