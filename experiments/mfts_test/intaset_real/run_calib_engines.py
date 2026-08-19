"""Compare WSC-KF / EKF / UKF / EnKF / PF as InTaSet calibration engines.

Pipeline (train prefix only; no holdout leakage into θ̂)
-------------------------------------------------------
1. Warm-start from open-loop LS (``calibrated_params.json``), but replace the
   pathological ``meas_std≈0.05`` with ``0.5 × open-loop RMSE``.
2. Each filter maximises its own predictive log-likelihood over the joint
   unconstrained parameter

       θ = (log k, logit γ₁, logit γ₂, log a₁…₄, log meas_std, log σ_u)

   via Adam + pathwise AD (fixed RNG for EnKF/PF).
3. Score every ``θ̂`` with a **shared reference EnKF** on train+holdout
   (held-out NLL/obs). Also report self-likelihood, open-loop MAPE, and
   MLE wall time.

Run:
  uv run --extra experiments python \\
      experiments/mfts_test/intaset_real/run_calib_engines.py
  uv run --extra experiments python \\
      experiments/mfts_test/intaset_real/run_calib_engines.py \\
      --mle-iters 20 --enkf-particles 50 --pf-particles 50

Outputs under ``experiments/mfts_test/intaset_real/output/``:
  calib_engines_summary.csv, calib_engines_trace.png,
  calib_engines_heldout.png, calib_engines_timing.png,
  calibrated_params_<engine_slug>.json  (e.g. wsckf, ekf; hyphens stripped)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Callable, NamedTuple

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
MFTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for pth in (HERE, MFTS_DIR, EXPERIMENTS_DIR, ROOT_DIR):
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
from wsckf.library import default_recombined_num_paths
from experiments.cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)

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
from plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    METHOD_MARKERS,
    PALETTE,
    save_publication_figure,
    style_box_axis,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_parser = argparse.ArgumentParser(
    description="InTaSet: redeploy each filter as MLE calibration engine"
)
_parser.add_argument("--source", type=str, default="4T-M2-train")
_parser.add_argument(
    "--params",
    type=str,
    default=os.path.join(OUTPUT_DIR, "calibrated_params.json"),
)
_parser.add_argument("--fit-min", type=float, default=15.0, dest="fit_min")
_parser.add_argument("--holdout-min", type=float, default=10.0, dest="holdout_min")
_parser.add_argument("--delta", type=float, default=1.0)
_parser.add_argument("--mle-iters", type=int, default=25, dest="mle_iters")
_parser.add_argument("--lr", type=float, default=0.02)
_parser.add_argument("--grad-clip", type=float, default=30.0, dest="grad_clip")
_parser.add_argument("--dt0", type=float, default=0.15)
_parser.add_argument(
    "--k-ref-min",
    type=float,
    default=0.5,
    dest="k_ref_min",
    help="Target WSC-KF macro-segment length [min]; k=min(--k-max, round(Δt/k_ref))",
)
_parser.add_argument("--k-max", type=int, default=8, dest="k_max")
_parser.add_argument(
    "--wsckf-paths",
    type=str,
    default="default",
    dest="wsckf_paths",
    help="Recombined WSC-KF path budget: 'default' or an integer",
)
_parser.add_argument("--enkf-particles", type=int, default=100, dest="enkf_particles")
_parser.add_argument("--pf-particles", type=int, default=100, dest="pf_particles")
_parser.add_argument(
    "--ref-enkf-particles", type=int, default=250, dest="ref_enkf_particles"
)
_parser.add_argument("--ref-seeds", type=str, default="0,1,2", dest="ref_seeds")
_parser.add_argument("--n-timing", type=int, default=1, dest="n_timing")
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument(
    "--only",
    type=str,
    default="",
    help="Comma subset of engines: WSC-KF,EKF,UKF,EnKF,PF",
)
_args = _parser.parse_args()

DT0 = float(_args.dt0)
SOLVER = "rk4"
WSCKF_K = min(int(_args.k_max), max(1, int(round(float(_args.delta) / float(_args.k_ref_min)))))
_tok = _args.wsckf_paths.strip().lower()
WSCKF_N = (
    default_recombined_num_paths(2, WSCKF_K)
    if _tok in ("", "default")
    else max(1, int(_tok))
)
ENKF_P = int(_args.enkf_particles)
PF_P = int(_args.pf_particles)
REF_P = int(_args.ref_enkf_particles)
UKF_P = UKFParams()

def _parse_ints(s: str) -> list[int]:
    out, seen = [], set()
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
ONLY = {s.strip() for s in _args.only.split(",") if s.strip()} or None

# ---------------------------------------------------------------------------
# Data + warm start
# ---------------------------------------------------------------------------

p_ol, meta_ol = load_params(_args.params) if os.path.exists(_args.params) else (
    IntaSetParams(),
    {},
)
rmse_ol = float(meta_ol.get("rmse_fit_cm", 1.0))
meas0 = float(np.clip(0.5 * rmse_ol, 0.2, 2.0))

traj_full = load_named(_args.source)
fit_sec = float(_args.fit_min) * 60.0
hold_sec = float(_args.holdout_min) * 60.0
traj_train = window_trajectory(
    traj_full, start_sec=0.0, duration_sec=fit_sec, window_id=0
)
traj_full_win = window_trajectory(
    traj_full, start_sec=0.0, duration_sec=fit_sec + hold_sec, window_id=0
)

grids_train = prepare_filter_grids(
    traj_train,
    delta_t_min=float(_args.delta),
    horizon_min=float(_args.fit_min),
    height_bias=p_ol.height_bias,
)
grids_full = prepare_filter_grids(
    traj_full_win,
    delta_t_min=float(_args.delta),
    horizon_min=float(_args.fit_min) + float(_args.holdout_min),
    height_bias=p_ol.height_bias,
)

YS_TRAIN = jnp.asarray(grids_train["ys"])
N_TRAIN = int(grids_train["n_obs"])
DT = float(grids_train["dt_obs_sec"])
T_TRAIN, U_TRAIN = grids_train["t_fine_sec"], grids_train["u_fine"]
H0 = grids_train["h_fine"][0]

YS_FULL = jnp.asarray(grids_full["ys"])
N_OBS = int(grids_full["n_obs"])
N_HELD = N_OBS - N_TRAIN
T_FULL, U_FULL = grids_full["t_fine_sec"], grids_full["u_fine"]
H_FINE_TRAIN = grids_train["h_fine"]

A_FIXED = jnp.asarray(p_ol.A, dtype=jnp.float64)
G_FIXED = float(p_ol.g)
H_EPS = float(p_ol.h_eps)
BIAS = p_ol.height_bias


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def pack_theta(p: IntaSetParams, *, meas_std: float | None = None) -> np.ndarray:
    ms = p.meas_std if meas_std is None else meas_std
    return np.array(
        [
            np.log(max(p.k_pump, 1e-3)),
            _logit(p.gamma1),
            _logit(p.gamma2),
            *np.log(np.maximum(np.asarray(p.a_out), 1e-6)),
            np.log(max(ms, 1e-3)),
            np.log(max(p.sigma_u, 1e-3)),
        ],
        dtype=np.float64,
    )


def unpack_theta(theta: np.ndarray | jnp.ndarray) -> IntaSetParams:
    th = np.asarray(theta, dtype=np.float64)
    return IntaSetParams(
        gamma1=float(1.0 / (1.0 + np.exp(-th[1]))),
        gamma2=float(1.0 / (1.0 + np.exp(-th[2]))),
        A=tuple(float(x) for x in np.asarray(A_FIXED)),
        a_out=tuple(float(x) for x in np.exp(th[3:7])),
        k_pump=float(np.exp(th[0])),
        meas_std=float(np.exp(th[7])),
        height_bias=BIAS,
        sigma_u=float(np.exp(th[8])),
        g=G_FIXED,
        h_eps=H_EPS,
    )


def unpack_jax(theta: jnp.ndarray):
    k = jnp.exp(theta[0])
    g1 = jax.nn.sigmoid(theta[1])
    g2 = jax.nn.sigmoid(theta[2])
    a = jnp.exp(theta[3:7])
    meas = jnp.exp(theta[7])
    su = jnp.exp(theta[8])
    return k, g1, g2, a, meas, su


THETA0_NP = pack_theta(p_ol, meas_std=meas0)
THETA0 = jnp.asarray(THETA0_NP)

print(
    f"InTaSet calib-engines: source={_args.source}  "
    f"train={_args.fit_min:g}min/{N_TRAIN}obs  hold={_args.holdout_min:g}min/{N_HELD}obs  "
    f"Δt={_args.delta:g}min"
)
print(
    f"  warm: k={p_ol.k_pump:.3g} γ=({p_ol.gamma1:.3f},{p_ol.gamma2:.3f})  "
    f"meas_std0={meas0:.3f} (was {p_ol.meas_std:.3f})  sigma_u={p_ol.sigma_u:.3g}"
)
print(
    f"  WSC-KF recombined k={WSCKF_K} (Δt/{_args.k_ref_min:g}, max={_args.k_max}) "
    f"→ {WSCKF_N}p  EnKF={ENKF_P}  PF={PF_P}  refEnKF={REF_P}"
)


# ---------------------------------------------------------------------------
# Dynamics from θ
# ---------------------------------------------------------------------------

def make_dyn(theta: jnp.ndarray, t_sec, pumps):
    k, g1, g2, a, meas, su = unpack_jax(theta)
    A = A_FIXED
    t_grid = jnp.asarray(t_sec, dtype=jnp.float64)
    u_grid = jnp.asarray(pumps, dtype=jnp.float64)
    t_lo, t_hi = t_grid[0], t_grid[-1]

    def _pump_frac(t):
        tt = jnp.clip(t, t_lo, t_hi)
        return jnp.array(
            [
                jnp.interp(tt, t_grid, u_grid[:, 0]),
                jnp.interp(tt, t_grid, u_grid[:, 1]),
            ]
        ) / 100.0

    def drift_fn(t, x):
        q = a * jnp.sqrt(2.0 * G_FIXED * jnp.maximum(x, H_EPS))
        u1, u2 = _pump_frac(t)
        f1, f2 = k * u1, k * u2
        return jnp.array(
            [
                (-q[0] + q[2] + g1 * f1) / A[0],
                (-q[1] + q[3] + g2 * f2) / A[1],
                (-q[2] + (1.0 - g2) * f2) / A[2],
                (-q[3] + (1.0 - g1) * f1) / A[3],
            ]
        )

    def L_fn(t, x):
        return jnp.array(
            [
                [g1 * su / A[0], 0.0],
                [0.0, g2 * su / A[1]],
                [0.0, (1.0 - g2) * su / A[2]],
                [(1.0 - g1) * su / A[3], 0.0],
            ]
        )

    def Qc_fn(t, x):
        return jnp.eye(2)

    def obs_fn(x):
        return x

    R = jnp.diag(jnp.full(4, meas**2))
    m0 = jnp.maximum(jnp.asarray(H0, dtype=jnp.float64), H_EPS)
    P0 = jnp.eye(4) * (meas**2 + 0.25)
    return drift_fn, L_fn, Qc_fn, obs_fn, R, m0, P0


# ---------------------------------------------------------------------------
# Likelihood engines
# ---------------------------------------------------------------------------

class MInputs(NamedTuple):
    t_idx: jnp.ndarray
    theta: jnp.ndarray


def _mi(theta: jnp.ndarray, n_obs: int) -> MInputs:
    th = jnp.asarray(theta, dtype=jnp.float64)
    return MInputs(
        t_idx=jnp.arange(n_obs + 1),
        theta=jnp.broadcast_to(th, (n_obs + 1, th.shape[0])),
    )


def build_wsckf(ys, t_fine, u_fine, *, checkpoint_steps: bool):
    t_j = jnp.asarray(t_fine)
    u_j = jnp.asarray(u_fine)

    def get_init_params(mi):
        _, _, _, _, _, m0, P0 = make_dyn(mi.theta, t_j, u_j)
        return m0, jnp.linalg.cholesky(P0)

    def get_dynamics(mi):
        drift, L, _, _, _, _, _ = make_dyn(mi.theta, t_j, u_j)
        return drift, L

    def get_observations(mi):
        _, _, _, obs, R, _, _ = make_dyn(mi.theta, t_j, u_j)
        return obs, jnp.linalg.cholesky(R), ys[mi.t_idx - 1]

    return build_filter(
        get_init_params=get_init_params,
        get_dynamics=get_dynamics,
        get_observations=get_observations,
        noise_dim=2,
        dt=DT,
        integration_dt=DT0,
        cubature_order=WSCKF_K,
        num_paths=WSCKF_N,
        multidim_paths="recombined",
        sde_form="ito",
        checkpoint_steps=checkpoint_steps,
    )


_run = jit(run_filter, static_argnames=["filter_obj"])
WSCKF_TRAIN_CKPT = build_wsckf(YS_TRAIN, T_TRAIN, U_TRAIN, checkpoint_steps=True)
WSCKF_TRAIN = build_wsckf(YS_TRAIN, T_TRAIN, U_TRAIN, checkpoint_steps=False)
WSCKF_FULL = build_wsckf(YS_FULL, T_FULL, U_FULL, checkpoint_steps=False)

_key_enkf = jr.PRNGKey(_args.seed + 17)
_key_pf = jr.PRNGKey(_args.seed + 31)


def wsckf_train_ll(theta):
    return _run(WSCKF_TRAIN_CKPT, _mi(theta, N_TRAIN)).log_normalizing_constant[-1]


def ekf_ll_on(ys, t_fine, u_fine, theta):
    drift, L, Qc, obs, R, m0, P0 = make_dyn(theta, t_fine, u_fine)
    return run_cd_ekf(
        drift, obs, L, Qc, R, m0, P0, ys, DT, DT0,
        state_order="first", solver=SOLVER,
    ).cum_loglik


def ukf_ll_on(ys, t_fine, u_fine, theta):
    drift, L, Qc, obs, R, m0, P0 = make_dyn(theta, t_fine, u_fine)
    return run_cd_ukf(
        drift, obs, L, Qc, R, m0, P0, ys, DT, DT0, UKF_P,
        state_order="first", solver=SOLVER,
    ).cum_loglik


def enkf_ll_on(ys, t_fine, u_fine, theta, key, *, n_particles: int):
    drift, L, Qc, obs, R, m0, P0 = make_dyn(theta, t_fine, u_fine)
    return run_cd_enkf(
        drift, obs, L, Qc, R, m0, P0, ys, DT, key,
        n_particles=n_particles, noise_dim=2, dt0=DT0, solver=SOLVER,
    ).cum_loglik


def pf_ll_on(ys, t_fine, u_fine, theta, key, *, n_particles: int):
    drift, L, Qc, obs, R, m0, P0 = make_dyn(theta, t_fine, u_fine)
    return run_bootstrap_pf(
        drift, obs, L, Qc, R, m0, P0, ys, DT, key,
        n_particles=n_particles, noise_dim=2, dt0=DT0, solver=SOLVER,
    ).cum_loglik


def ekf_train_ll(theta):
    return ekf_ll_on(YS_TRAIN, T_TRAIN, U_TRAIN, theta)[-1]


def ukf_train_ll(theta):
    return ukf_ll_on(YS_TRAIN, T_TRAIN, U_TRAIN, theta)[-1]


def enkf_train_ll(theta):
    return enkf_ll_on(
        YS_TRAIN, T_TRAIN, U_TRAIN, theta, _key_enkf, n_particles=ENKF_P
    )[-1]


def pf_train_ll(theta):
    return pf_ll_on(YS_TRAIN, T_TRAIN, U_TRAIN, theta, _key_pf, n_particles=PF_P)[-1]


ENGINES: dict[str, dict] = {
    "WSC-KF": {
        "color": METHOD_COLORS["WSC-KF"],
        "vg": jit(value_and_grad(wsckf_train_ll)),
        "fwd": jit(lambda th: _run(WSCKF_TRAIN, _mi(th, N_TRAIN)).log_normalizing_constant[-1]),
    },
    "EKF": {
        "color": METHOD_COLORS["EKF"],
        "vg": jit(value_and_grad(ekf_train_ll)),
        "fwd": jit(ekf_train_ll),
    },
    "UKF": {
        "color": METHOD_COLORS["UKF"],
        "vg": jit(value_and_grad(ukf_train_ll)),
        "fwd": jit(ukf_train_ll),
    },
    "EnKF": {
        "color": METHOD_COLORS["EnKF"],
        "vg": jit(value_and_grad(enkf_train_ll)),
        "fwd": jit(enkf_train_ll),
    },
    "PF": {
        "color": METHOD_COLORS["PF"],
        "vg": jit(value_and_grad(pf_train_ll)),
        "fwd": jit(pf_train_ll),
    },
}
if ONLY is not None:
    missing = ONLY - set(ENGINES)
    if missing:
        raise SystemExit(f"--only unknown engines: {missing}")
    ENGINES = {k: v for k, v in ENGINES.items() if k in ONLY}


def heldout_nll_per_obs(cum_ll, n_train: int, n_obs: int) -> float:
    ll = np.asarray(cum_ll, dtype=np.float64)
    if ll.shape[0] == n_obs + 1:
        held = float(ll[-1] - ll[n_train])
    elif ll.shape[0] == n_obs:
        base = float(ll[n_train - 1]) if n_train > 0 else 0.0
        held = float(ll[-1] - base)
    else:
        raise ValueError(f"cum_ll length {ll.shape[0]} vs n_obs={n_obs}")
    return -held / float(n_obs - n_train)


def ref_enkf_heldout(theta: np.ndarray) -> tuple[float, float]:
    th = jnp.asarray(theta)
    vals = []
    for s in REF_SEEDS:
        key = jr.PRNGKey(s + 1009 + 17 * REF_P)
        cum = enkf_ll_on(YS_FULL, T_FULL, U_FULL, th, key, n_particles=REF_P)
        vals.append(heldout_nll_per_obs(cum, N_TRAIN, N_OBS))
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def self_heldout(name: str, theta: np.ndarray) -> float:
    th = jnp.asarray(theta)
    if name == "WSC-KF":
        cum = _run(WSCKF_FULL, _mi(th, N_OBS)).log_normalizing_constant
        return heldout_nll_per_obs(cum, N_TRAIN, N_OBS)
    if name == "EKF":
        return heldout_nll_per_obs(
            ekf_ll_on(YS_FULL, T_FULL, U_FULL, th), N_TRAIN, N_OBS
        )
    if name == "UKF":
        return heldout_nll_per_obs(
            ukf_ll_on(YS_FULL, T_FULL, U_FULL, th), N_TRAIN, N_OBS
        )
    if name == "EnKF":
        return heldout_nll_per_obs(
            enkf_ll_on(
                YS_FULL, T_FULL, U_FULL, th, jr.PRNGKey(999), n_particles=ENKF_P
            ),
            N_TRAIN,
            N_OBS,
        )
    if name == "PF":
        return heldout_nll_per_obs(
            pf_ll_on(YS_FULL, T_FULL, U_FULL, th, jr.PRNGKey(1001), n_particles=PF_P),
            N_TRAIN,
            N_OBS,
        )
    raise KeyError(name)


def open_loop_mape(theta: np.ndarray, *, train: bool) -> float:
    p = unpack_theta(theta)
    if train:
        H, t, U = H_FINE_TRAIN, T_TRAIN, U_TRAIN
    else:
        seg = window_trajectory(
            traj_full, start_sec=fit_sec, duration_sec=hold_sec, window_id=1
        )
        H = seg.heights - np.asarray(BIAS)
        t = seg.t_sec - seg.t_sec[0]
        U = seg.pumps
    hat = simulate_open_loop(p, t, U, np.maximum(H[0], H_EPS))
    return mape_pct(hat, H)


def adam_mle(vg_fn: Callable, theta0: np.ndarray):
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
        if np.any(~np.isfinite(g)) or not np.isfinite(ll):
            print(f"    WARNING non-finite at iter {t}; stopping early")
            break
        if clip > 0:
            g = np.clip(g, -clip, clip)
        th_tr.append(th.copy())
        ll_tr.append(ll)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        th = th + _args.lr * mhat / (np.sqrt(vhat) + eps)
    ll_f = float(vg_fn(jnp.asarray(th))[0])
    th_tr.append(th.copy())
    ll_tr.append(ll_f)
    return np.asarray(th_tr), np.asarray(ll_tr), th


def median_time(fn, *, n: int = _args.n_timing) -> float:
    out = fn()
    jax.block_until_ready(out)
    samples = []
    for _ in range(max(1, n)):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Warmup + per-eval timing
    print(f"\n=== Per-eval timing on train (n={_args.n_timing}) ===")
    timings = {}
    for name, eng in ENGINES.items():
        try:
            t_fwd = median_time(lambda n=name: ENGINES[n]["fwd"](THETA0))
            t_ad = median_time(lambda n=name: ENGINES[n]["vg"](THETA0))
        except Exception as exc:
            print(f"  {name}: FAILED warmup ({exc})")
            continue
        timings[name] = (t_fwd, t_ad)
        print(f"  {name:5s}  fwd={t_fwd:.4f}s  AD={t_ad:.4f}s")

    # Open-loop baseline (θ0, not optimised)
    print("\n=== Shared ref-EnKF held-out @ warm start ===")
    held0 = ref_enkf_heldout(THETA0_NP)
    print(f"  warm-start  NLL/obs={held0[0]:.4f} ± {held0[1]:.4f}")

    traces: dict[str, dict] = {}
    hats: dict[str, np.ndarray] = {}
    walls: dict[str, float] = {}
    results: list[dict] = []

    print(f"\n=== Adam MLE ({_args.mle_iters} iters, joint θ∈R^9) ===")
    for name, eng in ENGINES.items():
        if name not in timings:
            continue
        t0 = time.perf_counter()
        th_tr, ll_tr, th_hat = adam_mle(eng["vg"], THETA0_NP)
        walls[name] = time.perf_counter() - t0
        traces[name] = {"theta": th_tr, "ll": ll_tr}
        hats[name] = th_hat
        p = unpack_theta(th_hat)
        print(
            f"  {name:5s}  wall={walls[name]:6.1f}s  "
            f"L0={ll_tr[0]:.1f}→L*={ll_tr[-1]:.1f}  "
            f"k={p.k_pump:.3g} meas={p.meas_std:.3f} σu={p.sigma_u:.3g}"
        )

    # Score
    print(
        f"\n=== Scoring (ref EnKF {REF_P}p, seeds={REF_SEEDS}) ==="
    )
    # warm-start row
    results.append(
        {
            "engine": "warm-start",
            "train_ll_init": "",
            "train_ll_final": "",
            "mle_wall_s": "",
            "fwd_s": "",
            "ad_s": "",
            "heldout_ref_nll": held0[0],
            "heldout_ref_nll_std": held0[1],
            "heldout_self_nll": "",
            "mape_train": open_loop_mape(THETA0_NP, train=True),
            "mape_hold": open_loop_mape(THETA0_NP, train=False),
            "k_pump": float(np.exp(THETA0_NP[0])),
            "gamma1": float(1 / (1 + np.exp(-THETA0_NP[1]))),
            "gamma2": float(1 / (1 + np.exp(-THETA0_NP[2]))),
            "meas_std": float(np.exp(THETA0_NP[7])),
            "sigma_u": float(np.exp(THETA0_NP[8])),
        }
    )

    for name in hats:
        th = hats[name]
        p = unpack_theta(th)
        href, href_s = ref_enkf_heldout(th)
        try:
            hself = self_heldout(name, th)
        except Exception as exc:
            hself = float("nan")
            print(f"  {name} self-heldout failed: {exc}")
        mt = open_loop_mape(th, train=True)
        mh = open_loop_mape(th, train=False)
        print(
            f"  {name:5s}  refNLL={href:.3f}±{href_s:.3f}  "
            f"selfNLL={hself:.3f}  MAPE tr/hold={mt:.1f}/{mh:.1f}%"
        )
        results.append(
            {
                "engine": name,
                "train_ll_init": traces[name]["ll"][0],
                "train_ll_final": traces[name]["ll"][-1],
                "mle_wall_s": walls[name],
                "fwd_s": timings[name][0],
                "ad_s": timings[name][1],
                "heldout_ref_nll": href,
                "heldout_ref_nll_std": href_s,
                "heldout_self_nll": hself,
                "mape_train": mt,
                "mape_hold": mh,
                "k_pump": p.k_pump,
                "gamma1": p.gamma1,
                "gamma2": p.gamma2,
                "meas_std": p.meas_std,
                "sigma_u": p.sigma_u,
            }
        )
        # Sanitize engine name for filenames (e.g. "WSC-KF" -> "wsckf").
        engine_slug = "".join(ch for ch in name.lower() if ch.isalnum())
        save_params(
            os.path.join(OUTPUT_DIR, f"calibrated_params_{engine_slug}.json"),
            p,
            meta={
                "engine": name,
                "source": _args.source,
                "fit_min": float(_args.fit_min),
                "holdout_min": float(_args.holdout_min),
                "heldout_ref_nll": href,
                "heldout_ref_nll_std": href_s,
                "train_ll_final": float(traces[name]["ll"][-1]),
                "mle_wall_s": walls[name],
            },
        )

    # CSV
    sum_path = os.path.join(OUTPUT_DIR, "calib_engines_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {sum_path}")

    # Plots
    names = [n for n in ENGINES if n in hats]
    colors = [ENGINES[n]["color"] for n in names]
    line_styles = {
        "WSC-KF": "-",
        "EKF": "--",
        "UKF": "-.",
        "EnKF": ":",
        "PF": (0, (4, 1, 1, 1)),
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.65),
    )
    method_handles = []
    ax = axes[0]
    for name in names:
        (line,) = ax.plot(
            np.arange(len(traces[name]["ll"])),
            traces[name]["ll"],
            color=ENGINES[name]["color"],
            marker=METHOD_MARKERS[name],
            linestyle=line_styles[name],
            linewidth=1.35,
            markersize=3.3,
            markevery=max(1, len(traces[name]["ll"]) // 6),
            label=name,
        )
        method_handles.append(line)
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("Training log likelihood [nats]")
    ax.set_title("(a) Self objective")
    style_curve_axis(ax, grid_axis="y")

    ax = axes[1]
    for name in names:
        meas = np.exp(traces[name]["theta"][:, 7])
        ax.plot(
            np.arange(len(meas)),
            meas,
            color=ENGINES[name]["color"],
            marker=METHOD_MARKERS[name],
            linestyle=line_styles[name],
            linewidth=1.35,
            markersize=3.3,
            markevery=max(1, len(meas) // 6),
        )
    warm_handle = ax.axhline(
        meas0,
        color=PALETTE["neutral_gray"],
        linestyle="--",
        linewidth=0.9,
        label="Warm start",
    )
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel(r"$\sigma_y$ [cm]")
    ax.set_title("(b) Sensor noise")
    style_curve_axis(ax, grid_axis="y")
    fig.legend(
        [*method_handles, warm_handle],
        [*names, "Warm start"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(6, len(names) + 1),
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.90, bottom=0.28, wspace=0.29)
    trace_path = os.path.join(OUTPUT_DIR, "calib_engines_trace.png")
    save_publication_figure(fig, trace_path)
    plt.close()
    print(f"Wrote {trace_path}")

    # Held-out bars
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.55),
    )
    ax = axes[0]
    labels = ["warm-start", *names]
    hrefs = [held0[0]] + [
        next(r["heldout_ref_nll"] for r in results if r["engine"] == n) for n in names
    ]
    hstds = [held0[1]] + [
        next(r["heldout_ref_nll_std"] for r in results if r["engine"] == n)
        for n in names
    ]
    cols = [PALETTE["neutral_gray"], *colors]
    xpos = np.arange(len(labels))
    ax.bar(
        xpos,
        hrefs,
        yerr=hstds,
        color=cols,
        edgecolor=PALETTE["black"],
        linewidth=0.55,
        capsize=2.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7},
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("NLL [nats/observation]")
    ax.set_title("(a) Reference-EnKF NLL")
    style_box_axis(ax, grid_axis="y")

    ax = axes[1]
    selfs = [
        next(r["heldout_self_nll"] for r in results if r["engine"] == n) for n in names
    ]
    ax.bar(
        np.arange(len(names)),
        selfs,
        color=colors,
        edgecolor=PALETTE["black"],
        linewidth=0.55,
    )
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("NLL [nats/observation]")
    ax.set_title("(b) Self likelihood")
    ax.text(
        0.98,
        0.96,
        "not cross-filter comparable",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        color=PALETTE["neutral_gray"],
    )
    style_box_axis(ax, grid_axis="y")
    fig.subplots_adjust(left=0.09, right=0.995, top=0.94, bottom=0.19, wspace=0.29)
    held_path = os.path.join(OUTPUT_DIR, "calib_engines_heldout.png")
    save_publication_figure(fig, held_path)
    plt.close()
    print(f"Wrote {held_path}")

    # Timing
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.65),
    )
    ax = axes[0]
    fwd = [timings[n][0] for n in names]
    ads = [timings[n][1] for n in names]
    y = np.arange(len(names))
    bar_height = 0.34
    ax.barh(
        y - bar_height / 2,
        fwd,
        bar_height,
        label="Forward",
        color=PALETTE["light_gray"],
        edgecolor=PALETTE["black"],
        linewidth=0.5,
    )
    ax.barh(
        y + bar_height / 2,
        ads,
        bar_height,
        label="Gradient (method color)",
        color=colors,
        edgecolor=PALETTE["black"],
        linewidth=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_title("(a) Per evaluation")
    ax.invert_yaxis()
    style_box_axis(ax, grid_axis="x")

    ax = axes[1]
    mle_t = [walls[n] for n in names]
    ax.barh(
        y,
        mle_t,
        0.56,
        color=colors,
        edgecolor=PALETTE["black"],
        linewidth=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_title("(b) Full optimization")
    ax.invert_yaxis()
    style_box_axis(ax, grid_axis="x")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
    )
    fig.supxlabel("Wall time [s]", y=0.105)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.94, bottom=0.25, wspace=0.30)
    timing_path = os.path.join(OUTPUT_DIR, "calib_engines_timing.png")
    save_publication_figure(fig, timing_path)
    plt.close()
    print(f"Wrote {timing_path}")

    # Rank by ref held-out
    ranked = sorted(
        [r for r in results if r["engine"] != "warm-start"],
        key=lambda r: r["heldout_ref_nll"],
    )
    print("\n=== Ranking by shared ref-EnKF held-out NLL (lower better) ===")
    for i, r in enumerate(ranked, 1):
        print(
            f"  {i}. {r['engine']:5s}  refNLL={r['heldout_ref_nll']:.3f}  "
            f"meas={r['meas_std']:.3f}  σu={r['sigma_u']:.3g}  "
            f"wall={r['mle_wall_s']:.1f}s"
        )


if __name__ == "__main__":
    main()
