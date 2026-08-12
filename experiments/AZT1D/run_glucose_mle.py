"""Filter-MLE of (p₃, σ_G) on AZT1D CGM with shared reference held-out scoring.

Fits the Bergman-style glucose–insulin SDE by maximising each filter's
**train-prefix** marginal log-likelihood over the interpretable pair:

    p₃     insulin sensitivity  [1/min per (U/min)]
    σ_G    √G diffusion scale   [√(mg/dL)/√min]

Fixed: p₁, p₂, G_b, k_c, σ_X, CGM R. Insulin/carb inputs from pump logs.

Protocol (movement-test style)
-----------------------------
* Adam MLE + likelihood grids use **only** ``y[:n_train]`` (no held-out leakage).
* Each ``θ̂`` is scored by a **shared reference EnKF** on the held-out suffix
  (full-track run; train is warmup). Self-likelihoods are diagnostics only.

Compares CRF, CD-EKF, CD-UKF, CD-EnKF, PF (``--skip-stochastic`` drops EnKF/PF).

Run:
  uv run --extra experiments python experiments/AZT1D/run_glucose_mle.py
  uv run --extra experiments python experiments/AZT1D/run_glucose_mle.py \\
      --subject 1 --n-obs 2016 --mle-iters 30 --grid 8

Outputs under ``experiments/AZT1D/output/``:
  ``glucose_loglik.png``, ``glucose_mle_trace.png``, ``glucose_track.png``,
  ``glucose_heldout_compare.png``, ``glucose_summary.csv``
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
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
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
from crf import build_filter
from cd_dynamax_mirror.filters import (
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)

from azt1d_data import load_azt1d_series
from glucose_model import GlucoseParams, crf_glucose_dynamics, init_estimate, make_glucose_dynamics, mape_glucose

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_parser = argparse.ArgumentParser(
    description="AZT1D glucose filter-MLE of (p3, sigma_G) with ref-EnKF held-out"
)
_parser.add_argument("--subject", type=int, default=1)
_parser.add_argument(
    "--n-obs",
    type=int,
    default=2016,
    dest="n_obs",
    help="CGM samples (~5 min); default 2016 ≈ 7 days",
)
_parser.add_argument("--holdout-frac", type=float, default=0.3, dest="holdout_frac")
_parser.add_argument("--mle-iters", type=int, default=30, dest="mle_iters")
_parser.add_argument("--lr", type=float, default=0.1, help="Adam step on (log p3, log σ_G)")
_parser.add_argument("--p3-init", type=float, default=5e-4, dest="p3_init")
_parser.add_argument("--sigma-g-init", type=float, default=1.5, dest="sigma_g_init")
_parser.add_argument("--grid", type=int, default=8, help="per-axis train-likelihood grid")
_parser.add_argument("--dt0", type=float, default=1.0, help="inner SDE step [min]")
_parser.add_argument("--cubature-order", type=int, default=3, dest="cubature_order")
_parser.add_argument("--num-paths", type=int, default=35, dest="num_paths")
_parser.add_argument("--enkf-particles", type=int, default=100, dest="enkf_particles")
_parser.add_argument("--pf-particles", type=int, default=100, dest="pf_particles")
_parser.add_argument(
    "--ref-enkf-particles", type=int, default=200, dest="ref_enkf_particles"
)
_parser.add_argument("--ref-seeds", type=str, default="0,1,2", dest="ref_seeds")
_parser.add_argument("--n-timing", type=int, default=2, dest="n_timing")
_parser.add_argument("--solver", type=str, default="rk4", choices=("rk4", "diffrax"))
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--skip-stochastic", action="store_true", help="skip EnKF and PF")
_parser.add_argument("--skip-ukf", action="store_true")
_parser.add_argument("--meas-std", type=float, default=8.0, dest="meas_std")
_args = _parser.parse_args()


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


ENKF_PARTICLES = int(_args.enkf_particles)
PF_PARTICLES = int(_args.pf_particles)
REF_ENKF_PARTICLES = int(_args.ref_enkf_particles)
REF_SEEDS = _parse_ints(_args.ref_seeds)
N_TIMING = max(1, int(_args.n_timing))
SOLVER = _args.solver

# ---------------------------------------------------------------------------
series = load_azt1d_series(subject=int(_args.subject), max_obs=int(_args.n_obs))
YS = jnp.asarray(series.cgm.reshape(-1, 1))
N_OBS = int(YS.shape[0])
DT = float(series.dt_min)
DT0 = float(_args.dt0)

if _args.holdout_frac < 0.0 or _args.holdout_frac >= 1.0:
    raise SystemExit("--holdout-frac must be in [0, 1)")
if _args.holdout_frac == 0.0:
    N_HELD = 0
    N_TRAIN = N_OBS
else:
    if not (0.05 <= _args.holdout_frac <= 0.5):
        raise SystemExit("non-zero --holdout-frac must be in [0.05, 0.5]")
    N_HELD = max(1, int(round(N_OBS * _args.holdout_frac)))
    N_TRAIN = N_OBS - N_HELD
    if N_TRAIN < 20:
        raise SystemExit(f"train split too short: n_train={N_TRAIN}")

YS_TRAIN = YS[:N_TRAIN]
DO_HELDOUT = N_HELD > 0

P = GlucoseParams(meas_std=float(_args.meas_std))
m0_np, P0_np = init_estimate(np.asarray(YS_TRAIN).ravel(), P)
M0 = jnp.asarray(m0_np)
P0 = jnp.asarray(P0_np)
CHOL_P0 = jnp.linalg.cholesky(P0)
_, _, _, OBS_FN, R = make_glucose_dynamics(
    P,
    t_min=series.t_min,
    insulin_u_per_min=series.insulin_u_per_min,
    carb_g_per_min=series.carb_g_per_min,
)
CHOL_R = jnp.linalg.cholesky(R)

print(
    f"subject={series.subject}  N={N_OBS} (~{series.span_days:.1f}d)  "
    f"train={N_TRAIN} held={N_HELD}  dt={DT:g} min  solver={SOLVER}"
)
print(
    f"fit (p3, σ_G); fixed p1={P.p1}, p2={P.p2}, Gb={P.Gb}, kc={P.k_c}, "
    f"R^0.5={P.meas_std}"
)
print(
    f"CRF recombined paths={_args.num_paths} k={_args.cubature_order}  "
    f"EnKF fit={ENKF_PARTICLES}p  PF fit={PF_PARTICLES}p  ref={REF_ENKF_PARTICLES}p"
)


class MInputs(NamedTuple):
    t_idx: jnp.ndarray
    p3: jnp.ndarray
    sigma_g: jnp.ndarray


def build_crf(ys, *, checkpoint_steps: bool):
    def get_init_params(mi):
        return M0, CHOL_P0

    def get_dynamics(mi):
        drift, L, _, _, _ = make_glucose_dynamics(
            P,
            t_min=series.t_min,
            insulin_u_per_min=series.insulin_u_per_min,
            carb_g_per_min=series.carb_g_per_min,
            p3=mi.p3,
            sigma_g=mi.sigma_g,
        )
        return crf_glucose_dynamics(drift, L, mi, DT)

    def get_observations(mi):
        return OBS_FN, CHOL_R, ys[mi.t_idx - 1]

    return build_filter(
        get_init_params=get_init_params,
        get_dynamics=get_dynamics,
        get_observations=get_observations,
        noise_dim=2,
        dt=DT,
        integration_dt=DT0,
        cubature_order=_args.cubature_order,
        num_paths=_args.num_paths,
        multidim_paths="recombined",
        sde_form="ito",
        checkpoint_steps=checkpoint_steps,
    )


def _mi(p3, sigma_g, n_obs: int):
    return MInputs(
        jnp.arange(n_obs + 1),
        jnp.full(n_obs + 1, p3),
        jnp.full(n_obs + 1, sigma_g),
    )


_run = jit(run_filter, static_argnames=["filter_obj"])

CRF_TRAIN = build_crf(YS_TRAIN, checkpoint_steps=False)
CRF_TRAIN_CKPT = build_crf(YS_TRAIN, checkpoint_steps=True)
CRF_FULL = build_crf(YS, checkpoint_steps=False)


def crf_train_loglik(p3, sigma_g, *, checkpoint=True):
    filt = CRF_TRAIN_CKPT if checkpoint else CRF_TRAIN
    return _run(filt, _mi(p3, sigma_g, N_TRAIN)).log_normalizing_constant[-1]


def crf_full_states(p3, sigma_g):
    return _run(CRF_FULL, _mi(p3, sigma_g, N_OBS))


def _dyn(p3, sigma_g):
    return make_glucose_dynamics(
        P,
        t_min=series.t_min,
        insulin_u_per_min=series.insulin_u_per_min,
        carb_g_per_min=series.carb_g_per_min,
        p3=p3,
        sigma_g=sigma_g,
    )


def ekf_loglik_on(ys, p3, sigma_g):
    drift, L, Qc, _, _ = _dyn(p3, sigma_g)
    return run_cd_ekf(
        drift, OBS_FN, L, Qc, R, M0, P0, ys, DT, DT0,
        state_order="first", solver=SOLVER,
    ).cum_loglik


def ukf_loglik_on(ys, p3, sigma_g):
    drift, L, Qc, _, _ = _dyn(p3, sigma_g)
    return run_cd_ukf(
        drift, OBS_FN, L, Qc, R, M0, P0, ys, DT, DT0,
        state_order="first", solver=SOLVER,
    ).cum_loglik


def enkf_loglik_on(ys, p3, sigma_g, key, *, n_particles: int = ENKF_PARTICLES):
    drift, L, Qc, _, _ = _dyn(p3, sigma_g)
    return run_cd_enkf(
        drift, OBS_FN, L, Qc, R, M0, P0, ys, DT, key,
        n_particles=n_particles, noise_dim=2, dt0=DT0, solver=SOLVER,
    ).cum_loglik


def pf_loglik_on(ys, p3, sigma_g, key, *, n_particles: int = PF_PARTICLES):
    drift, L, Qc, _, _ = _dyn(p3, sigma_g)
    return run_bootstrap_pf(
        drift, OBS_FN, L, Qc, R, M0, P0, ys, DT, key,
        n_particles=n_particles, noise_dim=2, dt0=DT0, solver=SOLVER,
    ).cum_loglik


def ekf_train_loglik(p3, sigma_g):
    return ekf_loglik_on(YS_TRAIN, p3, sigma_g)[-1]


def ukf_train_loglik(p3, sigma_g):
    return ukf_loglik_on(YS_TRAIN, p3, sigma_g)[-1]


def enkf_train_loglik(p3, sigma_g, key):
    return enkf_loglik_on(YS_TRAIN, p3, sigma_g, key)[-1]


def pf_train_loglik(p3, sigma_g, key):
    return pf_loglik_on(YS_TRAIN, p3, sigma_g, key)[-1]


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
        raise ValueError(f"unexpected cum_ll length {ll.shape[0]}")
    return -held / float(n_obs - n_train)


def ref_enkf_heldout_nll(p3, sigma_g, *, n_particles: int, seeds: list[int]):
    vals = []
    for s in seeds:
        key = jr.PRNGKey(s + 1009 + 17 * n_particles)
        cum = enkf_loglik_on(YS, p3, sigma_g, key, n_particles=n_particles)
        vals.append(heldout_nll_per_obs(cum, N_TRAIN, N_OBS))
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _ll_from_theta(ll2, theta):
    return ll2(jnp.exp(theta[0]), jnp.exp(theta[1]))


crf_vg = jit(
    value_and_grad(
        lambda th: _ll_from_theta(lambda a, b: crf_train_loglik(a, b, checkpoint=True), th)
    )
)
ekf_vg = jit(value_and_grad(lambda th: _ll_from_theta(ekf_train_loglik, th)))
ukf_vg = jit(value_and_grad(lambda th: _ll_from_theta(ukf_train_loglik, th)))
_enkf_key_ad = jr.PRNGKey(_args.seed + 17)
_pf_key_ad = jr.PRNGKey(_args.seed + 31)
enkf_vg = jit(
    value_and_grad(
        lambda th: _ll_from_theta(lambda a, b: enkf_train_loglik(a, b, _enkf_key_ad), th)
    )
)
pf_vg = jit(
    value_and_grad(
        lambda th: _ll_from_theta(lambda a, b: pf_train_loglik(a, b, _pf_key_ad), th)
    )
)
_crf_train_ll = jit(lambda a, b: crf_train_loglik(a, b, checkpoint=False))
_ekf_train_ll = jit(ekf_train_loglik)
_ukf_train_ll = jit(ukf_train_loglik)
_enkf_train_ll = jit(enkf_train_loglik)
_pf_train_ll = jit(pf_train_loglik)


def adam_mle_2d(vg_fn):
    th = np.array([np.log(_args.p3_init), np.log(_args.sigma_g_init)], dtype=float)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(2)
    v = np.zeros(2)
    p3_tr, sg_tr, ll_tr = [], [], []
    for t in range(1, _args.mle_iters + 1):
        val, g = vg_fn(jnp.asarray(th))
        ll = float(val)
        g = np.asarray(g, dtype=float)
        if not np.isfinite(ll) or not np.all(np.isfinite(g)):
            # Keep last finite point and stop early.
            if not p3_tr:
                p3_tr.append(float(np.exp(th[0])))
                sg_tr.append(float(np.exp(th[1])))
                ll_tr.append(ll)
            break
        p3_tr.append(float(np.exp(th[0])))
        sg_tr.append(float(np.exp(th[1])))
        ll_tr.append(ll)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        th = th + _args.lr * mhat / (np.sqrt(vhat) + eps)
    return np.array(p3_tr), np.array(sg_tr), np.array(ll_tr)


def _grid_argmax(p3_axis, sg_axis, Z):
    finite = np.isfinite(Z)
    if not np.any(finite):
        return float("nan"), float("nan"), float("nan")
    flat = np.nanargmax(Z)
    i, j = np.unravel_index(flat, Z.shape)
    return float(p3_axis[i]), float(sg_axis[j]), float(Z[i, j])


def main():
    do_ukf = not _args.skip_ukf
    stochastic = not _args.skip_stochastic
    theta0 = jnp.array([np.log(_args.p3_init), np.log(_args.sigma_g_init)])

    print(f"\n=== Timing (n={N_TIMING}, N_train={N_TRAIN}) ===")

    def _med(fn):
        out = fn()
        jax.block_until_ready(out)
        samples = []
        for _ in range(N_TIMING):
            t0 = time.perf_counter()
            jax.block_until_ready(fn())
            samples.append(time.perf_counter() - t0)
        return float(np.median(samples))

    print(f"  CRF fwd: {_med(lambda: _crf_train_ll(_args.p3_init, _args.sigma_g_init)):.3f}s")
    print(f"  CRF-AD:  {_med(lambda: crf_vg(theta0)):.3f}s")
    print(f"  EKF fwd: {_med(lambda: _ekf_train_ll(_args.p3_init, _args.sigma_g_init)):.3f}s")
    print(f"  EKF-AD:  {_med(lambda: ekf_vg(theta0)):.3f}s")
    if do_ukf:
        print(f"  UKF fwd: {_med(lambda: _ukf_train_ll(_args.p3_init, _args.sigma_g_init)):.3f}s")
        print(f"  UKF-AD:  {_med(lambda: ukf_vg(theta0)):.3f}s")
    if stochastic:
        key_t = jr.PRNGKey(_args.seed + 1)
        print(
            f"  EnKF fwd: {_med(lambda: _enkf_train_ll(_args.p3_init, _args.sigma_g_init, key_t)):.3f}s"
        )
        print(f"  EnKF-AD:  {_med(lambda: enkf_vg(theta0)):.3f}s")
        print(
            f"  PF fwd:   {_med(lambda: _pf_train_ll(_args.p3_init, _args.sigma_g_init, key_t)):.3f}s"
        )
        print(f"  PF-AD:    {_med(lambda: pf_vg(theta0)):.3f}s")

    n_g = int(_args.grid)
    p3_axis = np.geomspace(5e-5, 5e-3, n_g)
    sg_axis = np.geomspace(0.3, 8.0, n_g)
    print(f"\n=== Train loglik grid ({n_g}×{n_g}) ===")
    crf_Z = np.full((n_g, n_g), np.nan)
    ekf_Z = np.full((n_g, n_g), np.nan)
    ukf_Z = np.full((n_g, n_g), np.nan) if do_ukf else None
    enkf_Z = np.full((n_g, n_g), np.nan) if stochastic else None
    pf_Z = np.full((n_g, n_g), np.nan) if stochastic else None
    key_grid = jr.PRNGKey(_args.seed)
    for i, p3 in enumerate(p3_axis):
        for j, sg in enumerate(sg_axis):
            crf_Z[i, j] = float(_crf_train_ll(p3, sg))
            ekf_Z[i, j] = float(_ekf_train_ll(p3, sg))
            if do_ukf:
                ukf_Z[i, j] = float(_ukf_train_ll(p3, sg))
            if stochastic:
                enkf_Z[i, j] = float(_enkf_train_ll(p3, sg, key_grid))
                pf_Z[i, j] = float(_pf_train_ll(p3, sg, key_grid))
        if (i + 1) % max(1, n_g // 4) == 0:
            print(f"  grid row {i + 1}/{n_g}")

    crf_p3_g, crf_sg_g, crf_L = _grid_argmax(p3_axis, sg_axis, crf_Z)
    ekf_p3_g, ekf_sg_g, ekf_L = _grid_argmax(p3_axis, sg_axis, ekf_Z)
    print(f"  CRF  grid-argmax: p3={crf_p3_g:.3e}  σ_G={crf_sg_g:.3f}  L={crf_L:.1f}")
    print(f"  EKF  grid-argmax: p3={ekf_p3_g:.3e}  σ_G={ekf_sg_g:.3f}  L={ekf_L:.1f}")
    if do_ukf:
        ukf_p3_g, ukf_sg_g, ukf_L = _grid_argmax(p3_axis, sg_axis, ukf_Z)
        print(f"  UKF  grid-argmax: p3={ukf_p3_g:.3e}  σ_G={ukf_sg_g:.3f}  L={ukf_L:.1f}")
    if stochastic:
        enkf_p3_g, enkf_sg_g, enkf_L = _grid_argmax(p3_axis, sg_axis, enkf_Z)
        pf_p3_g, pf_sg_g, pf_L = _grid_argmax(p3_axis, sg_axis, pf_Z)
        print(f"  EnKF grid-argmax: p3={enkf_p3_g:.3e}  σ_G={enkf_sg_g:.3f}  L={enkf_L:.1f}")
        print(f"  PF   grid-argmax: p3={pf_p3_g:.3e}  σ_G={pf_sg_g:.3f}  L={pf_L:.1f}")

    print(f"\n=== Adam MLE on train ({_args.mle_iters} iters) ===")
    t0 = time.perf_counter()
    crf_vg(theta0)
    crf_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    crf_p3_tr, crf_sg_tr, crf_ll_tr = adam_mle_2d(crf_vg)
    crf_mle_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    ekf_p3_tr, ekf_sg_tr, ekf_ll_tr = adam_mle_2d(ekf_vg)
    ekf_mle_t = time.perf_counter() - t0
    print(
        f"  CRF-AD:  p̂3={crf_p3_tr[-1]:.3e}  σ̂_G={crf_sg_tr[-1]:.3f}  "
        f"L={crf_ll_tr[-1]:.1f}  (compile={crf_compile:.1f}s, {crf_mle_t:.1f}s)"
    )
    print(
        f"  EKF-AD:  p̂3={ekf_p3_tr[-1]:.3e}  σ̂_G={ekf_sg_tr[-1]:.3f}  "
        f"L={ekf_ll_tr[-1]:.1f}  ({ekf_mle_t:.1f}s)"
    )

    ukf_p3_tr = ukf_sg_tr = ukf_ll_tr = None
    if do_ukf:
        t0 = time.perf_counter()
        ukf_p3_tr, ukf_sg_tr, ukf_ll_tr = adam_mle_2d(ukf_vg)
        ukf_mle_t = time.perf_counter() - t0
        print(
            f"  UKF-AD:  p̂3={ukf_p3_tr[-1]:.3e}  σ̂_G={ukf_sg_tr[-1]:.3f}  "
            f"L={ukf_ll_tr[-1]:.1f}  ({ukf_mle_t:.1f}s)"
        )

    enkf_p3_tr = enkf_sg_tr = enkf_ll_tr = None
    pf_p3_tr = pf_sg_tr = pf_ll_tr = None
    if stochastic:
        t0 = time.perf_counter()
        enkf_p3_tr, enkf_sg_tr, enkf_ll_tr = adam_mle_2d(enkf_vg)
        enkf_mle_t = time.perf_counter() - t0
        print(
            f"  EnKF-AD: p̂3={enkf_p3_tr[-1]:.3e}  σ̂_G={enkf_sg_tr[-1]:.3f}  "
            f"L={enkf_ll_tr[-1]:.1f}  ({enkf_mle_t:.1f}s)"
        )
        t0 = time.perf_counter()
        pf_p3_tr, pf_sg_tr, pf_ll_tr = adam_mle_2d(pf_vg)
        pf_mle_t = time.perf_counter() - t0
        # Fall back to grid argmax if AD path collapsed.
        if len(pf_p3_tr) < 2 or not np.isfinite(pf_ll_tr[-1]):
            pf_p3_tr = np.array([pf_p3_g])
            pf_sg_tr = np.array([pf_sg_g])
            pf_ll_tr = np.array([pf_L])
            print(
                f"  PF-AD:   collapsed → using grid-argmax "
                f"p̂3={pf_p3_tr[-1]:.3e}  σ̂_G={pf_sg_tr[-1]:.3f}  L={pf_ll_tr[-1]:.1f}"
            )
        else:
            print(
                f"  PF-AD:   p̂3={pf_p3_tr[-1]:.3e}  σ̂_G={pf_sg_tr[-1]:.3f}  "
                f"L={pf_ll_tr[-1]:.1f}  ({pf_mle_t:.1f}s)"
            )

    enkf_ref = {}
    enkf_ref_std = {}
    self_held = {}
    st = None
    theta = {
        "CRF": (float(crf_p3_tr[-1]), float(crf_sg_tr[-1])),
        "EKF": (float(ekf_p3_tr[-1]), float(ekf_sg_tr[-1])),
    }
    if do_ukf:
        theta["UKF"] = (float(ukf_p3_tr[-1]), float(ukf_sg_tr[-1]))
    if stochastic:
        theta["EnKF"] = (float(enkf_p3_tr[-1]), float(enkf_sg_tr[-1]))
        theta["PF"] = (float(pf_p3_tr[-1]), float(pf_sg_tr[-1]))

    if DO_HELDOUT:
        print(
            f"\n=== Ref-EnKF held-out NLL/obs "
            f"(EnKF-{REF_ENKF_PARTICLES}p @ each train θ̂) ==="
        )
        for name, th in theta.items():
            mean, std = ref_enkf_heldout_nll(
                *th, n_particles=REF_ENKF_PARTICLES, seeds=REF_SEEDS
            )
            enkf_ref[name] = mean
            enkf_ref_std[name] = std
            print(f"  ref-EnKF @ θ̂_{name:4s} = {mean:.4f} ± {std:.4f}")

        print("\n=== Self-score diagnostics (not for ranking) ===")
        st = crf_full_states(*theta["CRF"])
        self_held["CRF"] = heldout_nll_per_obs(st.log_normalizing_constant, N_TRAIN, N_OBS)
        self_held["EKF"] = heldout_nll_per_obs(ekf_loglik_on(YS, *theta["EKF"]), N_TRAIN, N_OBS)
        print(f"  CRF  self = {self_held['CRF']:.4f}")
        print(f"  EKF  self = {self_held['EKF']:.4f}")
        if do_ukf:
            self_held["UKF"] = heldout_nll_per_obs(
                ukf_loglik_on(YS, *theta["UKF"]), N_TRAIN, N_OBS
            )
            print(f"  UKF  self = {self_held['UKF']:.4f}")
        if stochastic:
            enkf_vals = [
                heldout_nll_per_obs(
                    enkf_loglik_on(YS, *theta["EnKF"], jr.PRNGKey(s + 101)),
                    N_TRAIN,
                    N_OBS,
                )
                for s in REF_SEEDS
            ]
            self_held["EnKF"] = float(np.mean(enkf_vals))
            pf_vals = [
                heldout_nll_per_obs(
                    pf_loglik_on(YS, *theta["PF"], jr.PRNGKey(s + 201)),
                    N_TRAIN,
                    N_OBS,
                )
                for s in REF_SEEDS
            ]
            self_held["PF"] = float(np.mean(pf_vals))
            print(f"  EnKF self = {self_held['EnKF']:.4f}")
            print(f"  PF   self = {self_held['PF']:.4f}")
    else:
        st = crf_full_states(*theta["CRF"])

    g_hat = np.asarray(st.mean)[:, 0]
    cgm = series.cgm
    mape_crf = mape_glucose(g_hat, cgm)

    # ---- plots ----
    panel_specs = [
        ("CRF", crf_Z, "C2", (crf_p3_g, crf_sg_g), theta["CRF"]),
        ("CD-EKF", ekf_Z, "C3", (ekf_p3_g, ekf_sg_g), theta["EKF"]),
    ]
    if do_ukf:
        panel_specs.append(("CD-UKF", ukf_Z, "C4", (ukf_p3_g, ukf_sg_g), theta["UKF"]))
    if stochastic:
        panel_specs.append(("CD-EnKF", enkf_Z, "C1", (enkf_p3_g, enkf_sg_g), theta["EnKF"]))
        panel_specs.append(("PF", pf_Z, "C5", (pf_p3_g, pf_sg_g), theta["PF"]))

    n_panels = len(panel_specs)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig1, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows), sharey=True, squeeze=False
    )
    finite_all = np.concatenate([Z[np.isfinite(Z)] for _, Z, *_ in panel_specs])
    vmax = float(np.nanmax(finite_all))
    vmin = vmax - max(200.0, 0.15 * abs(vmax) if vmax != 0 else 200.0)
    P3, SG = np.meshgrid(p3_axis, sg_axis, indexing="ij")
    for idx, (name, Z, color, grid_star, mle_star) in enumerate(panel_specs):
        ax = axes[idx // ncols, idx % ncols]
        pcm = ax.pcolormesh(P3, SG, Z, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.plot(*grid_star, "*", color="white", ms=12, markeredgecolor="k", label="grid argmax")
        ax.plot(*mle_star, "o", color=color, ms=7, markeredgecolor="k", label="Adam MLE")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$p_3$")
        ax.set_title(name)
        ax.legend(fontsize=6, loc="lower left")
        fig1.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        if idx % ncols == 0:
            ax.set_ylabel(r"$\sigma_G$")
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")
    fig1.suptitle(
        f"Train $L(p_3,\\sigma_G)$ — AZT1D subject {series.subject}, "
        f"N_train={N_TRAIN}/{N_OBS}",
        fontsize=11,
    )
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "glucose_loglik.png"), dpi=150)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes2[0]
    styles = [
        ("CRF", crf_p3_tr, crf_sg_tr, "C2", "o-"),
        ("EKF", ekf_p3_tr, ekf_sg_tr, "C3", "s-"),
    ]
    if do_ukf:
        styles.append(("UKF", ukf_p3_tr, ukf_sg_tr, "C4", "^-"))
    if stochastic:
        styles.append(("EnKF", enkf_p3_tr, enkf_sg_tr, "C1", "D-"))
        styles.append(("PF", pf_p3_tr, pf_sg_tr, "C5", "v-"))
    for name, p3_tr, sg_tr, color, mk in styles:
        ax.plot(p3_tr, sg_tr, mk, color=color, ms=4, label=f"{name}-AD")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_3$")
    ax.set_ylabel(r"$\sigma_G$")
    ax.set_title("Adam path (train likelihood)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes2[1]
    ll_styles = [("CRF", crf_ll_tr, "C2", "o-"), ("EKF", ekf_ll_tr, "C3", "s-")]
    if do_ukf:
        ll_styles.append(("UKF", ukf_ll_tr, "C4", "^-"))
    if stochastic:
        ll_styles.append(("EnKF", enkf_ll_tr, "C1", "D-"))
        ll_styles.append(("PF", pf_ll_tr, "C5", "v-"))
    for name, ll_tr, color, mk in ll_styles:
        ax.plot(np.arange(len(ll_tr)), ll_tr, mk, color=color, ms=3, label=name)
    ax.set_xlabel("MLE iteration")
    ax.set_ylabel("train log-likelihood")
    ax.set_title("Train L traces")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig2.suptitle(
        f"Train-only filter-MLE of $(p_3,\\sigma_G)$ (N_train={N_TRAIN})",
        fontsize=11,
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "glucose_mle_trace.png"), dpi=150)

    t_h = series.t_min / 60.0
    fig3, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(t_h[:N_TRAIN], cgm[:N_TRAIN], ".", ms=2, color="0.55", label="CGM train")
    if DO_HELDOUT:
        ax.plot(t_h[N_TRAIN:], cgm[N_TRAIN:], ".", ms=2, color="0.8", label="CGM held-out")
    ax.plot(
        t_h,
        g_hat[1:] if g_hat.shape[0] == N_OBS + 1 else g_hat[:N_OBS],
        "-",
        lw=1.0,
        color="C2",
        label=f"CRF @ train MLE (MAPE={mape_crf:.2f}%)",
    )
    if DO_HELDOUT:
        ax.axvline(t_h[N_TRAIN], color="C0", ls="--", lw=0.8, label="holdout start")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("glucose [mg/dL]")
    ax.set_title(f"AZT1D subject {series.subject}: CGM + CRF @ train MLE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "glucose_track.png"), dpi=150)

    if DO_HELDOUT:
        fig4, axes4 = plt.subplots(1, 2, figsize=(11.0, 4.2))
        ax = axes4[0]
        compare = [
            ("CRF", theta["CRF"], enkf_ref["CRF"], enkf_ref_std["CRF"], "#2ca02c", "o", crf_p3_tr, crf_sg_tr),
            ("EKF", theta["EKF"], enkf_ref["EKF"], enkf_ref_std["EKF"], "#d62728", "s", ekf_p3_tr, ekf_sg_tr),
        ]
        if do_ukf:
            compare.append(
                ("UKF", theta["UKF"], enkf_ref["UKF"], enkf_ref_std["UKF"], "#9467bd", "^", ukf_p3_tr, ukf_sg_tr)
            )
        if stochastic:
            compare.append(
                ("EnKF", theta["EnKF"], enkf_ref["EnKF"], enkf_ref_std["EnKF"], "#ff7f0e", "D", enkf_p3_tr, enkf_sg_tr)
            )
            compare.append(
                ("PF", theta["PF"], enkf_ref["PF"], enkf_ref_std["PF"], "#17becf", "v", pf_p3_tr, pf_sg_tr)
            )
        for name, th, held, err, color, mk, p3_tr, sg_tr in compare:
            ax.plot(th[0], th[1], mk, color=color, ms=10, markeredgecolor="k", label=name)
            ax.plot(p3_tr, sg_tr, "-", color=color, alpha=0.35, lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$p_3$")
        ax.set_ylabel(r"$\sigma_G$")
        ax.set_title(r"Train $\hat\theta$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        ax = axes4[1]
        names = [c[0] for c in compare]
        helds = [c[2] for c in compare]
        errs = [c[3] for c in compare]
        colors = [c[4] for c in compare]
        xpos = np.arange(len(names))
        ax.bar(xpos, helds, yerr=errs, color=colors, edgecolor="0.3", lw=0.6, capsize=4)
        ax.set_xticks(xpos)
        ax.set_xticklabels(names)
        ax.set_ylabel("ref-EnKF held-out NLL / obs")
        ax.set_title(f"Shared reference score (EnKF-{REF_ENKF_PARTICLES})")
        ax.grid(axis="y", alpha=0.3)
        fig4.suptitle(
            f"AZT1D subject {series.subject}: filter-MLE held-out comparison",
            fontsize=11,
        )
        fig4.tight_layout()
        fig4.savefig(os.path.join(OUTPUT_DIR, "glucose_heldout_compare.png"), dpi=150)

    train_ll = {
        "CRF": float(crf_ll_tr[-1]),
        "EKF": float(ekf_ll_tr[-1]),
    }
    if do_ukf:
        train_ll["UKF"] = float(ukf_ll_tr[-1])
    if stochastic:
        train_ll["EnKF"] = float(enkf_ll_tr[-1])
        train_ll["PF"] = float(pf_ll_tr[-1])

    rows = []
    for name in theta:
        rows.append(
            {
                "filter": name,
                "p3_hat": theta[name][0],
                "sigma_g_hat": theta[name][1],
                "train_ll": train_ll[name],
                "ref_enkf_held_nll": enkf_ref.get(name, float("nan")),
                "ref_enkf_held_std": enkf_ref_std.get(name, float("nan")),
                "self_held_nll": self_held.get(name, float("nan")),
                "mape_pct": mape_crf if name == "CRF" else float("nan"),
            }
        )

    sum_path = os.path.join(OUTPUT_DIR, "glucose_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "n_obs",
                "n_train",
                "n_held",
                "filter",
                "p3_hat",
                "sigma_g_hat",
                "train_ll",
                "ref_enkf_held_nll",
                "ref_enkf_held_std",
                "self_held_nll",
                "mape_pct",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "subject": series.subject,
                    "n_obs": N_OBS,
                    "n_train": N_TRAIN,
                    "n_held": N_HELD,
                    **r,
                }
            )

    print(f"\nWrote {sum_path}")
    print(f"Wrote {OUTPUT_DIR}/glucose_*.png")


if __name__ == "__main__":
    main()
