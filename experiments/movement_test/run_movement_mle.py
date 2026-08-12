"""Joint filter-MLE of (τ_v, σ_v) on REAL duck GPS telemetry.

Fits an OU-velocity movement model with speed-dependent diffusion to a single
individual's hourly GPS track (Dryad doi:10.5061/dryad.1ns1rn93p) by maximising
the Bayesian-filter marginal log-likelihood over the interpretable pair:

    τ_v = 1/β_v     velocity autocorrelation time (hours)
    σ_v             asymptotic per-axis velocity std under frozen g≡σ₀
                    (σ_v = σ₀/√(2β_v)  ⇒  σ₀ = σ_v √(2/τ_v))

Home-range spring ω², speed scale v_c, and GPS R are held fixed.

Train / held-out protocol (default --holdout-frac 0.3)
------------------------------------------------------
* Adam MLE and likelihood grids use **only** the train prefix ``y[:n_train]``
  (no held-out leakage into ``θ̂``).
* After fitting, each ``θ̂`` is scored by a **shared reference likelihood**
  (default: large RK4 EnKF) on the held-out suffix: run the reference filter on
  the **full** track at that ``θ̂`` (train is warmup; no cold start). This avoids
  comparing incomparable self-likelihoods (EKF closure vs WSC-KF/EnKF).
* Heatmaps show train ``L(θ)``; the accuracy column is reference held-out NLL/obs.

Compares:
* WSC-KF-AD -- pathwise jax.value_and_grad through checkpointed ODE adjoint
* CD-EKF-AD -- moment-ODE EKF pathwise VJP
* CD-EnKF-AD -- ensemble filter, pathwise VJP with fixed RNG key
  (--skip-stochastic to drop EnKF)

Outputs (experiments/movement_test/output/):
  movement_loglik.png       2D train L(τ_v, σ_v) heatmaps
  movement_mle_trace.png    joint Adam traces in (τ_v, σ_v)
  movement_track.png        GPS + WSC-KF filtered mean at train MLE
  movement_heldout.png      held-out NLL/obs bar chart
  movement_timing.png       per-eval wall times
  movement_summary.csv

Run:
  uv run --extra experiments python experiments/movement_test/run_movement_mle.py
  uv run --extra experiments python experiments/movement_test/run_movement_mle.py \\
      --n-obs 600 --holdout-frac 0.3 --mle-iters 40 --grid 10
  uv run --extra experiments python experiments/movement_test/run_movement_mle.py \\
      --holdout-frac 0   # full-track fit (no held-out score)
  uv run --extra experiments python experiments/movement_test/run_movement_mle.py --timing-only
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

EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENTS_DIR, ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
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
from cd_dynamax_mirror.filters import run_cd_ekf, run_cd_enkf

from movement_data import load_duck_track
from movement_model import (
    MovementParams,
    init_estimate,
    make_movement_dynamics,
    tau_sigma_to_beta_sigma0,
    beta_sigma0_to_tau_sigma,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="Joint movement-SDE filter-MLE of (tau_v, sigma_v) on real duck GPS"
)
_parser.add_argument("--band", type=str, default="MALL 825 M")
_parser.add_argument("--n-obs", type=int, default=600, dest="n_obs")
_parser.add_argument(
    "--holdout-frac",
    type=float,
    default=0.3,
    dest="holdout_frac",
    help="Suffix fraction reserved for held-out NLL (0 = fit/score on full track)",
)
_parser.add_argument("--mle-iters", type=int, default=40, dest="mle_iters")
_parser.add_argument("--lr", type=float, default=0.1, help="Adam step on (log τ_v, log σ_v)")
_parser.add_argument("--tau-init", type=float, default=2.0, dest="tau_init",
                     help="initial velocity autocorr time (hours)")
_parser.add_argument("--sigma-v-init", type=float, default=2.0, dest="sigma_v_init",
                     help="initial asymptotic velocity std (km/h)")
_parser.add_argument("--grid", type=int, default=10,
                     help="per-axis points for 2D train-likelihood heatmap")
_parser.add_argument("--dt0", type=float, default=0.1, help="inner RK4 step (hours)")
_parser.add_argument("--cubature-order", type=int, default=3, dest="cubature_order")
_parser.add_argument(
    "--num-paths",
    type=int,
    default=35,
    dest="num_paths",
    help="Recombined Wiener cubature path count (default 35; library default at k=3 is 84)",
)
_parser.add_argument("--enkf-particles", type=int, default=200, dest="enkf_particles")
_parser.add_argument(
    "--ref-enkf-particles",
    type=int,
    default=500,
    dest="ref_enkf_particles",
    help="EnKF particle count for the shared reference held-out score",
)
_parser.add_argument(
    "--ref-seeds",
    type=str,
    default="0,1,2",
    dest="ref_seeds",
    help="RNG seeds for reference-EnKF held-out NLL mean±std",
)
_parser.add_argument(
    "--heldout-enkf-seeds",
    type=str,
    default="0,1,2",
    dest="heldout_enkf_seeds",
    help="Seeds for optional self-score EnKF diagnostics (fit still uses --seed)",
)
_parser.add_argument("--n-timing", type=int, default=3, dest="n_timing")
_parser.add_argument(
    "--solver", type=str, default="rk4", choices=("rk4", "diffrax"),
    help="CD-filter forecast integrator (WSC-KF uses its own cubature RK4)",
)
_parser.add_argument("--timing-only", action="store_true")
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--skip-stochastic", action="store_true",
                     help="skip EnKF (WSC-KF + EKF only)")
_args = _parser.parse_args()
ENKF_PARTICLES = _args.enkf_particles
REF_ENKF_PARTICLES = _args.ref_enkf_particles
N_TIMING = _args.n_timing
SOLVER = _args.solver


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


HELDOUT_ENKF_SEEDS = _parse_ints(_args.heldout_enkf_seeds)
REF_SEEDS = _parse_ints(_args.ref_seeds)

# ---------------------------------------------------------------------------
track = load_duck_track(band=_args.band, max_obs=_args.n_obs)
YS = jnp.asarray(track.xy_km)
N_OBS = int(YS.shape[0])
DT = 1.0
DT0 = _args.dt0

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
        raise SystemExit(f"train split too short: n_train={N_TRAIN}, n_obs={N_OBS}")

YS_TRAIN = YS[:N_TRAIN]
DO_HELDOUT = N_HELD > 0

# Init from the first GPS fix (start of train / full track) — never from held-out.
P = MovementParams(center=(0.0, 0.0))
m0_np, P0_np = init_estimate(np.asarray(YS_TRAIN), P)
M0 = jnp.asarray(m0_np)
P0 = jnp.asarray(P0_np)
CHOL_P0 = jnp.linalg.cholesky(P0)
_, _, _, OBS_FN, R = make_movement_dynamics(P, beta_v=P.beta_v, sigma0=P.sigma0)
CHOL_R = jnp.linalg.cholesky(R)

_tau0_default, _sv0_default = beta_sigma0_to_tau_sigma(P.beta_v, P.sigma0)
print(
    f"track: band={track.band} spp={track.species} sex={track.sex}  "
    f"N={N_OBS} hourly (~{N_OBS/24:.1f} days)  extent(km)="
    f"{np.round(np.asarray(YS).max(0)-np.asarray(YS).min(0),1)}  "
    f"cd_solver={SOLVER}"
)
print(
    f"split: train={N_TRAIN}  held={N_HELD}  "
    f"(holdout_frac={_args.holdout_frac:g}; MLE/grid use train only)"
)
print(
    f"WSC-KF: recombined paths={_args.num_paths}  k={_args.cubature_order}  "
    f"EnKF fit={ENKF_PARTICLES}p  ref={REF_ENKF_PARTICLES}p"
)
print(
    f"parametrization: (τ_v, σ_v) with σ₀=σ_v√(2/τ_v), β_v=1/τ_v;  "
    f"defaults from MovementParams → τ_v={float(_tau0_default):.2f}h, "
    f"σ_v={float(_sv0_default):.2f} km/h"
)


# ---------------------------------------------------------------------------
# Filter likelihoods threaded by (τ_v, σ_v) and observation series
# ---------------------------------------------------------------------------
class MInputs(NamedTuple):
    t_idx: jnp.ndarray
    tau_v: jnp.ndarray
    sigma_v: jnp.ndarray


def build_wsckf(ys, *, checkpoint_steps: bool):
    """WSC-KF closed over a fixed observation array ``ys`` (train or full)."""

    def get_init_params(mi):
        return M0, CHOL_P0

    def get_dynamics(mi):
        beta_v, sigma0 = tau_sigma_to_beta_sigma0(mi.tau_v, mi.sigma_v)
        drift, L, _, _, _ = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
        return drift, L

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


def _mi(tau_v, sigma_v, n_obs: int):
    return MInputs(
        jnp.arange(n_obs + 1),
        jnp.full(n_obs + 1, tau_v),
        jnp.full(n_obs + 1, sigma_v),
    )


_run = jit(run_filter, static_argnames=["filter_obj"])

# Train filters (fit + grid) — closed over YS_TRAIN only.
WSCKF_TRAIN = build_wsckf(YS_TRAIN, checkpoint_steps=False)
WSCKF_TRAIN_CKPT = build_wsckf(YS_TRAIN, checkpoint_steps=True)
# Full-track filters (held-out score + track plot) — never used for Adam/grid.
WSCKF_FULL = build_wsckf(YS, checkpoint_steps=False)
WSCKF_FULL_CKPT = build_wsckf(YS, checkpoint_steps=True)


def wsckf_train_loglik(tau_v, sigma_v, *, checkpoint=True):
    filt = WSCKF_TRAIN_CKPT if checkpoint else WSCKF_TRAIN
    return _run(filt, _mi(tau_v, sigma_v, N_TRAIN)).log_normalizing_constant[-1]


def wsckf_full_states(tau_v, sigma_v, *, checkpoint=True):
    filt = WSCKF_FULL_CKPT if checkpoint else WSCKF_FULL
    return _run(filt, _mi(tau_v, sigma_v, N_OBS))


def ekf_loglik_on(ys, tau_v, sigma_v):
    beta_v, sigma0 = tau_sigma_to_beta_sigma0(tau_v, sigma_v)
    drift, L, Qc, _, _ = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
    return run_cd_ekf(
        drift, OBS_FN, L, Qc, R, M0, P0, ys, DT, DT0,
        state_order="first", solver=SOLVER,
    ).cum_loglik


def enkf_loglik_on(ys, tau_v, sigma_v, key, *, n_particles: int = ENKF_PARTICLES):
    beta_v, sigma0 = tau_sigma_to_beta_sigma0(tau_v, sigma_v)
    drift, L, Qc, _, _ = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
    return run_cd_enkf(
        drift, OBS_FN, L, Qc, R, M0, P0, ys, DT, key,
        n_particles=n_particles, noise_dim=2, dt0=DT0, solver=SOLVER,
    ).cum_loglik


def ekf_train_loglik(tau_v, sigma_v):
    return ekf_loglik_on(YS_TRAIN, tau_v, sigma_v)[-1]


def enkf_train_loglik(tau_v, sigma_v, key):
    return enkf_loglik_on(YS_TRAIN, tau_v, sigma_v, key)[-1]


def heldout_nll_per_obs(cum_ll, n_train: int, n_obs: int) -> float:
    """Incremental NLL on obs[n_train:] from a cumulative loglik array.

    WSC-KF ``log_normalizing_constant`` has length ``n_obs+1`` (index 0 = prior).
    CD filters ``cum_loglik`` has length ``n_obs``.
    """
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


def ref_enkf_heldout_nll(tau_v, sigma_v, *, n_particles: int, seeds: list[int]):
    """Held-out NLL/obs under a shared EnKF reference at fixed ``(τ_v, σ_v)``."""
    vals = []
    for s in seeds:
        key = jr.PRNGKey(s + 1009 + 17 * n_particles)
        cum = enkf_loglik_on(
            YS, tau_v, sigma_v, key, n_particles=n_particles
        )
        vals.append(heldout_nll_per_obs(cum, N_TRAIN, N_OBS))
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def wsckf_ref_heldout_nll(tau_v, sigma_v):
    """Held-out NLL/obs under the WSC-KF likelihood at fixed ``(τ_v, σ_v)``."""
    st = wsckf_full_states(tau_v, sigma_v, checkpoint=False)
    return heldout_nll_per_obs(st.log_normalizing_constant, N_TRAIN, N_OBS), st



def _ll_from_theta(ll2, theta):
    """ll2(tau, sigma_v) evaluated at theta = (log τ_v, log σ_v)."""
    return ll2(jnp.exp(theta[0]), jnp.exp(theta[1]))


# Gradients / forwards for **train** likelihood only (no held-out leakage).
wsckf_vg = jit(value_and_grad(
    lambda th: _ll_from_theta(lambda t, s: wsckf_train_loglik(t, s, checkpoint=True), th)
))
ekf_vg = jit(value_and_grad(lambda th: _ll_from_theta(ekf_train_loglik, th)))
_enkf_key_ad = jr.PRNGKey(_args.seed + 17)
enkf_vg = jit(value_and_grad(
    lambda th: _ll_from_theta(lambda t, s: enkf_train_loglik(t, s, _enkf_key_ad), th)
))
_wsckf_train_ll = jit(lambda t, s: wsckf_train_loglik(t, s, checkpoint=False))
_ekf_train_ll = jit(ekf_train_loglik)
_enkf_train_ll = jit(enkf_train_loglik)
_ekf_full_cum = jit(lambda t, s: ekf_loglik_on(YS, t, s))
_enkf_full_cum = jit(lambda t, s, k: enkf_loglik_on(YS, t, s, k))


def median_time(fn, *, n: int = N_TIMING) -> float:
    out = fn()
    jax.block_until_ready(out)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def adam_mle_2d(vg_fn):
    """Adam ascent on theta=(log τ_v, log σ_v); returns traces + train loglik."""
    th = np.array([np.log(_args.tau_init), np.log(_args.sigma_v_init)], dtype=float)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(2)
    v = np.zeros(2)
    tau_tr, sv_tr, ll_tr = [], [], []
    for t in range(1, _args.mle_iters + 1):
        val, g = vg_fn(jnp.asarray(th))
        ll = float(val)
        g = np.asarray(g, dtype=float)
        tau_tr.append(float(np.exp(th[0])))
        sv_tr.append(float(np.exp(th[1])))
        ll_tr.append(ll)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        th = th + _args.lr * mhat / (np.sqrt(vhat) + eps)
    return np.array(tau_tr), np.array(sv_tr), np.array(ll_tr)


def _grid_argmax(tau_axis, sv_axis, Z):
    flat = np.nanargmax(Z)
    i, j = np.unravel_index(flat, Z.shape)
    return float(tau_axis[i]), float(sv_axis[j]), float(Z[i, j])


def main():
    stochastic = not _args.skip_stochastic
    timing_only = _args.timing_only
    theta0 = jnp.array([np.log(_args.tau_init), np.log(_args.sigma_v_init)])

    # ---- timing on train likelihood (matches the fit objective) ----
    print(
        f"\n=== Median wall time (n={N_TIMING}, N_train={N_TRAIN}, "
        f"cd_solver={SOLVER}) ==="
    )
    t_wsckf_fwd = median_time(lambda: _wsckf_train_ll(_args.tau_init, _args.sigma_v_init))
    t_wsckf_ad = median_time(lambda: wsckf_vg(theta0))
    t_ekf_fwd = median_time(lambda: _ekf_train_ll(_args.tau_init, _args.sigma_v_init))
    t_ekf_ad = median_time(lambda: ekf_vg(theta0))
    print(f"  WSC-KF forward:        {t_wsckf_fwd:.4f}s")
    print(f"  WSC-KF-AD (2-param):   {t_wsckf_ad:.4f}s")
    print(f"  CD-EKF forward:     {t_ekf_fwd:.4f}s")
    print(f"  CD-EKF-AD:          {t_ekf_ad:.4f}s")
    t_enkf_fwd = t_enkf_ad = float("nan")
    if stochastic:
        key_t = jr.PRNGKey(_args.seed + 1)
        t_enkf_fwd = median_time(
            lambda: _enkf_train_ll(_args.tau_init, _args.sigma_v_init, key_t)
        )
        t_enkf_ad = median_time(lambda: enkf_vg(theta0))
        print(f"  CD-EnKF forward:    {t_enkf_fwd:.4f}s  ({ENKF_PARTICLES}p)")
        print(f"  CD-EnKF-AD:         {t_enkf_ad:.4f}s")

    fig_t, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    timing_rows = [
        ("WSC-KF fwd", t_wsckf_fwd, "C2"),
        ("WSC-KF-AD", t_wsckf_ad, "C2"),
        ("EKF fwd", t_ekf_fwd, "C3"),
        ("EKF-AD", t_ekf_ad, "C3"),
    ]
    if stochastic:
        timing_rows += [
            (f"EnKF fwd\n({ENKF_PARTICLES}p)", t_enkf_fwd, "C1"),
            ("EnKF-AD", t_enkf_ad, "C1"),
        ]
    labels = [r[0] for r in timing_rows]
    times = np.array([r[1] for r in timing_rows], dtype=float)
    colors = [r[2] for r in timing_rows]
    xpos = np.arange(len(labels))
    bars = ax.bar(xpos, times, color=colors, edgecolor="0.3", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("median wall time [s]")
    ax.set_yscale("log")
    ax.set_title(f"Per-eval on train (median of {N_TIMING})")
    ax.grid(axis="y", alpha=0.3, which="both")
    for b, t in zip(bars, times):
        if np.isfinite(t) and t > 0:
            ax.text(
                b.get_x() + b.get_width() / 2, t * 1.15, f"{t:.3f}s",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    print(f"\n=== Joint Adam MLE wall ({_args.mle_iters} iters, train only) ===")
    t0 = time.perf_counter()
    adam_mle_2d(wsckf_vg)
    wsckf_mle_wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    adam_mle_2d(ekf_vg)
    ekf_mle_wall = time.perf_counter() - t0
    enkf_mle_wall = float("nan")
    print(f"  WSC-KF-AD:  {wsckf_mle_wall:.2f}s")
    print(f"  EKF-AD:  {ekf_mle_wall:.2f}s")
    if stochastic:
        t0 = time.perf_counter()
        adam_mle_2d(enkf_vg)
        enkf_mle_wall = time.perf_counter() - t0
        print(f"  EnKF-AD: {enkf_mle_wall:.2f}s")

    ax = axes[1]
    mle_labels = ["WSC-KF-AD", "EKF-AD"]
    mle_times = [wsckf_mle_wall, ekf_mle_wall]
    mle_colors = ["C2", "C3"]
    if stochastic:
        mle_labels.append("EnKF-AD")
        mle_times.append(enkf_mle_wall)
        mle_colors.append("C1")
    xpos = np.arange(len(mle_labels))
    bars = ax.bar(xpos, mle_times, color=mle_colors, edgecolor="0.3", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(mle_labels, fontsize=9)
    ax.set_ylabel("wall time [s]")
    ax.set_yscale("log")
    ax.set_title(f"Train Adam MLE ({_args.mle_iters} iters)")
    ax.grid(axis="y", alpha=0.3, which="both")
    for b, t in zip(bars, mle_times):
        if np.isfinite(t) and t > 0:
            ax.text(
                b.get_x() + b.get_width() / 2, t * 1.15, f"{t:.2f}s",
                ha="center", va="bottom", fontsize=8,
            )

    fig_t.suptitle(
        f"Joint $(\\tau_v,\\sigma_v)$ timing — {track.band}, "
        f"N_train={N_TRAIN}/{N_OBS}, dt0={DT0}, cd_solver={SOLVER}",
        fontsize=11,
    )
    fig_t.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"movement_timing_{SOLVER}.png")
    fig_t.savefig(out_png, dpi=150)
    fig_t.savefig(os.path.join(OUTPUT_DIR, "movement_timing.png"), dpi=150)
    out_csv = os.path.join(OUTPUT_DIR, f"movement_timing_{SOLVER}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "kind", "method", "seconds", "solver", "n_obs", "n_train",
            "n_timing", "mle_iters",
        ])
        for lab, t, _ in timing_rows:
            w.writerow([
                "per_eval", lab.replace("\n", " "), t, SOLVER, N_OBS, N_TRAIN,
                N_TIMING, "",
            ])
        w.writerow([
            "mle_wall", "WSC-KF-AD", wsckf_mle_wall, SOLVER, N_OBS, N_TRAIN, "",
            _args.mle_iters,
        ])
        w.writerow([
            "mle_wall", "EKF-AD", ekf_mle_wall, SOLVER, N_OBS, N_TRAIN, "",
            _args.mle_iters,
        ])
        if stochastic:
            w.writerow([
                "mle_wall", "EnKF-AD", enkf_mle_wall, SOLVER, N_OBS, N_TRAIN, "",
                _args.mle_iters,
            ])
    print(f"Saved: {out_png}, movement_timing.png, {out_csv}")

    if timing_only:
        return

    # ---- 2D train-likelihood grids (no held-out leakage) ----
    n_g = _args.grid
    tau_axis = np.geomspace(0.4, 8.0, n_g)
    sv_axis = np.geomspace(0.5, 8.0, n_g)
    print(f"\n=== 2D train loglik grid ({n_g}×{n_g}, N_train={N_TRAIN}) ===")
    wsckf_Z = np.full((n_g, n_g), np.nan)
    ekf_Z = np.full((n_g, n_g), np.nan)
    enkf_Z = np.full((n_g, n_g), np.nan) if stochastic else None
    key_grid = jr.PRNGKey(_args.seed)
    for i, tau in enumerate(tau_axis):
        for j, sv in enumerate(sv_axis):
            wsckf_Z[i, j] = float(_wsckf_train_ll(tau, sv))
            ekf_Z[i, j] = float(_ekf_train_ll(tau, sv))
            if stochastic:
                enkf_Z[i, j] = float(_enkf_train_ll(tau, sv, key_grid))
        if (i + 1) % max(1, n_g // 4) == 0:
            print(f"  grid row {i+1}/{n_g} done")

    wsckf_tau, wsckf_sv, wsckf_L = _grid_argmax(tau_axis, sv_axis, wsckf_Z)
    ekf_tau, ekf_sv, ekf_L = _grid_argmax(tau_axis, sv_axis, ekf_Z)
    print(f"  WSC-KF  grid-argmax:  τ_v={wsckf_tau:.2f}h  σ_v={wsckf_sv:.2f}  L={wsckf_L:.1f}")
    print(f"  EKF  grid-argmax:  τ_v={ekf_tau:.2f}h  σ_v={ekf_sv:.2f}  L={ekf_L:.1f}")
    if stochastic:
        enkf_tau, enkf_sv, enkf_L = _grid_argmax(tau_axis, sv_axis, enkf_Z)
        print(f"  EnKF grid-argmax:  τ_v={enkf_tau:.2f}h  σ_v={enkf_sv:.2f}  L={enkf_L:.1f}")

    # ---- joint Adam MLE on train only ----
    print(f"\n=== Joint Adam MLE on train ({_args.mle_iters} iters, N_train={N_TRAIN}) ===")
    t0 = time.perf_counter()
    wsckf_vg(theta0)
    wsckf_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    wsckf_tau_tr, wsckf_sv_tr, wsckf_ll_tr = adam_mle_2d(wsckf_vg)
    wsckf_mle_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    ekf_tau_tr, ekf_sv_tr, ekf_ll_tr = adam_mle_2d(ekf_vg)
    ekf_mle_t = time.perf_counter() - t0
    print(
        f"  WSC-KF-AD:  τ̂={wsckf_tau_tr[-1]:.3f}h  σ̂_v={wsckf_sv_tr[-1]:.3f}  "
        f"(compile={wsckf_compile:.2f}s, {wsckf_mle_t:.2f}s)"
    )
    print(
        f"  EKF-AD:  τ̂={ekf_tau_tr[-1]:.3f}h  σ̂_v={ekf_sv_tr[-1]:.3f}  "
        f"({ekf_mle_t:.2f}s)"
    )
    enkf_tau_tr = enkf_sv_tr = enkf_ll_tr = None
    enkf_mle_t = float("nan")
    if stochastic:
        t0 = time.perf_counter()
        enkf_tau_tr, enkf_sv_tr, enkf_ll_tr = adam_mle_2d(enkf_vg)
        enkf_mle_t = time.perf_counter() - t0
        print(
            f"  EnKF-AD: τ̂={enkf_tau_tr[-1]:.3f}h  σ̂_v={enkf_sv_tr[-1]:.3f}  "
            f"({enkf_mle_t:.2f}s)"
        )

    _, sig0_wsckf = tau_sigma_to_beta_sigma0(wsckf_tau_tr[-1], wsckf_sv_tr[-1])
    _, sig0_ekf = tau_sigma_to_beta_sigma0(ekf_tau_tr[-1], ekf_sv_tr[-1])
    print(f"  implied σ₀:  WSC-KF={float(sig0_wsckf):.3f}  EKF={float(sig0_ekf):.3f}")

    # ---- held-out scores ----
    # Primary: shared reference EnKF likelihood at each train θ̂ (fair ranking).
    # Secondary: self-likelihood diagnostics + WSC-KF-reference cross-scores (CSV).
    wsckf_held = ekf_held = enkf_held = float("nan")
    enkf_held_std = float("nan")
    wsckf_ref_at_wsckf = wsckf_ref_at_ekf = wsckf_ref_at_enkf = float("nan")
    enkf_ref_at_wsckf = enkf_ref_at_ekf = enkf_ref_at_enkf = float("nan")
    enkf_ref_at_wsckf_std = enkf_ref_at_ekf_std = enkf_ref_at_enkf_std = float("nan")
    st = None
    if DO_HELDOUT:
        print(
            f"\n=== Reference held-out NLL/obs "
            f"(EnKF-{REF_ENKF_PARTICLES}p @ each train θ̂; "
            f"N_held={N_HELD}; state carried through train) ==="
        )
        theta_wsckf = (float(wsckf_tau_tr[-1]), float(wsckf_sv_tr[-1]))
        theta_ekf = (float(ekf_tau_tr[-1]), float(ekf_sv_tr[-1]))
        theta_enkf = (
            (float(enkf_tau_tr[-1]), float(enkf_sv_tr[-1]))
            if stochastic else None
        )

        enkf_ref_at_wsckf, enkf_ref_at_wsckf_std = ref_enkf_heldout_nll(
            *theta_wsckf, n_particles=REF_ENKF_PARTICLES, seeds=REF_SEEDS
        )
        enkf_ref_at_ekf, enkf_ref_at_ekf_std = ref_enkf_heldout_nll(
            *theta_ekf, n_particles=REF_ENKF_PARTICLES, seeds=REF_SEEDS
        )
        print(
            f"  ref-EnKF @ θ̂_WSCKF  = {enkf_ref_at_wsckf:.4f} ± {enkf_ref_at_wsckf_std:.4f}"
        )
        print(
            f"  ref-EnKF @ θ̂_EKF  = {enkf_ref_at_ekf:.4f} ± {enkf_ref_at_ekf_std:.4f}"
        )
        if theta_enkf is not None:
            enkf_ref_at_enkf, enkf_ref_at_enkf_std = ref_enkf_heldout_nll(
                *theta_enkf, n_particles=REF_ENKF_PARTICLES, seeds=REF_SEEDS
            )
            print(
                f"  ref-EnKF @ θ̂_EnKF = {enkf_ref_at_enkf:.4f} "
                f"± {enkf_ref_at_enkf_std:.4f}"
            )

        # WSC-KF-reference cross-score (deterministic) + track states at θ̂_WSCKF.
        wsckf_ref_at_wsckf, st = wsckf_ref_heldout_nll(*theta_wsckf)
        wsckf_ref_at_ekf, _ = wsckf_ref_heldout_nll(*theta_ekf)
        print(f"  ref-WSC-KF  @ θ̂_WSCKF  = {wsckf_ref_at_wsckf:.4f}")
        print(f"  ref-WSC-KF  @ θ̂_EKF  = {wsckf_ref_at_ekf:.4f}")
        if theta_enkf is not None:
            wsckf_ref_at_enkf, _ = wsckf_ref_heldout_nll(*theta_enkf)
            print(f"  ref-WSC-KF  @ θ̂_EnKF = {wsckf_ref_at_enkf:.4f}")

        # Self-score diagnostics (not used for ranking — incomparable scorers).
        print("\n=== Self-score diagnostics (own filter @ own θ̂; not for ranking) ===")
        wsckf_held = wsckf_ref_at_wsckf
        ekf_cum = _ekf_full_cum(jnp.asarray(theta_ekf[0]), jnp.asarray(theta_ekf[1]))
        ekf_held = heldout_nll_per_obs(ekf_cum, N_TRAIN, N_OBS)
        print(f"  WSC-KF  self held-out NLL/obs = {wsckf_held:.4f}")
        print(f"  EKF  self held-out NLL/obs = {ekf_held:.4f}")
        if stochastic:
            enkf_vals = []
            for s in HELDOUT_ENKF_SEEDS:
                key = jr.PRNGKey(s + 101)
                cum = _enkf_full_cum(
                    jnp.asarray(theta_enkf[0]), jnp.asarray(theta_enkf[1]), key
                )
                enkf_vals.append(heldout_nll_per_obs(cum, N_TRAIN, N_OBS))
            enkf_held = float(np.mean(enkf_vals))
            enkf_held_std = (
                float(np.std(enkf_vals, ddof=1)) if len(enkf_vals) > 1 else 0.0
            )
            print(
                f"  EnKF self held-out NLL/obs = {enkf_held:.4f} "
                f"± {enkf_held_std:.4f}"
            )
    else:
        st = wsckf_full_states(
            jnp.asarray(wsckf_tau_tr[-1]), jnp.asarray(wsckf_sv_tr[-1]), checkpoint=False
        )

    filt_xy = np.asarray(st.mean)[:, :2]
    obs_xy = np.asarray(YS)

    # ===== plots =====
    n_panels = 3 if stochastic else 2
    fig1, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.4), sharey=True)
    panels = [
        ("WSC-KF", wsckf_Z, "C2", (wsckf_tau, wsckf_sv), (wsckf_tau_tr[-1], wsckf_sv_tr[-1])),
        ("CD-EKF", ekf_Z, "C3", (ekf_tau, ekf_sv), (ekf_tau_tr[-1], ekf_sv_tr[-1])),
    ]
    if stochastic:
        panels.append(
            ("CD-EnKF", enkf_Z, "C1", (enkf_tau, enkf_sv),
             (enkf_tau_tr[-1], enkf_sv_tr[-1]))
        )
    finite_all = np.concatenate([Z[np.isfinite(Z)] for _, Z, *_ in panels])
    vmax = float(np.nanmax(finite_all))
    vmin = vmax - 6000.0

    Tau, Sv = np.meshgrid(tau_axis, sv_axis, indexing="ij")
    for ax, (name, Z, color, grid_star, mle_star) in zip(axes, panels):
        pcm = ax.pcolormesh(
            Tau, Sv, Z, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        ax.plot(*grid_star, "*", color="white", ms=14, markeredgecolor="k",
                label="grid argmax")
        ax.plot(*mle_star, "o", color=color, ms=8, markeredgecolor="k",
                label="Adam MLE")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"velocity autocorr $\tau_v$ [h]")
        ax.set_title(name)
        ax.legend(fontsize=7, loc="lower left")
        fig1.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(r"velocity scale $\sigma_v$ [km/h]")
    fig1.suptitle(
        f"Train filter log-likelihood $L(\\tau_v,\\sigma_v)$ — {track.band}, "
        f"N_train={N_TRAIN}/{N_OBS}",
        fontsize=11,
    )
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "movement_loglik.png"), dpi=150)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes2[0]
    ax.plot(wsckf_tau_tr, wsckf_sv_tr, "o-", color="C2", ms=4, label="WSC-KF-AD")
    ax.plot(ekf_tau_tr, ekf_sv_tr, "s-", color="C3", ms=4, label="EKF-AD")
    if stochastic:
        ax.plot(enkf_tau_tr, enkf_sv_tr, "D-", color="C1", ms=4, label="EnKF-AD")
    ax.plot(wsckf_tau, wsckf_sv, "*", color="C2", ms=14, markeredgecolor="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_v$ [h]")
    ax.set_ylabel(r"$\sigma_v$ [km/h]")
    ax.set_title("Adam path (train likelihood)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes2[1]
    it = np.arange(_args.mle_iters)
    ax.plot(it, wsckf_tau_tr, "o-", color="C2", ms=3, label=r"WSC-KF $\tau_v$")
    ax.plot(it, wsckf_sv_tr, "o--", color="C2", ms=3, alpha=0.7, label=r"WSC-KF $\sigma_v$")
    ax.plot(it, ekf_tau_tr, "s-", color="C3", ms=3, label=r"EKF $\tau_v$")
    ax.plot(it, ekf_sv_tr, "s--", color="C3", ms=3, alpha=0.7, label=r"EKF $\sigma_v$")
    if stochastic:
        ax.plot(it, enkf_tau_tr, "D-", color="C1", ms=3, label=r"EnKF $\tau_v$")
        ax.plot(it, enkf_sv_tr, "D--", color="C1", ms=3, alpha=0.7, label=r"EnKF $\sigma_v$")
    ax.set_xlabel("MLE iteration")
    ax.set_ylabel("parameter value")
    ax.set_yscale("log")
    ax.set_title("Per-parameter traces")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig2.suptitle(
        f"Train-only gradient filter-MLE of $(\\tau_v,\\sigma_v)$ "
        f"(N_train={N_TRAIN})",
        fontsize=11,
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "movement_mle_trace.png"), dpi=150)

    # track: train vs held-out GPS
    fig3, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        obs_xy[:N_TRAIN, 0], obs_xy[:N_TRAIN, 1], ".", ms=3, color="0.55",
        label=f"GPS train (n={N_TRAIN})",
    )
    if DO_HELDOUT:
        ax.plot(
            obs_xy[N_TRAIN:, 0], obs_xy[N_TRAIN:, 1], ".", ms=3, color="0.85",
            label=f"GPS held-out (n={N_HELD})",
        )
    ax.plot(
        filt_xy[:, 0], filt_xy[:, 1], "-", lw=1.0, color="C2",
        label=(
            f"WSC-KF filtered mean @ train MLE\n"
            f"$\\hat\\tau_v$={wsckf_tau_tr[-1]:.2f}h, $\\hat\\sigma_v$={wsckf_sv_tr[-1]:.2f}"
        ),
    )
    ax.plot(obs_xy[0, 0], obs_xy[0, 1], "o", color="green", label="start")
    if DO_HELDOUT:
        ax.plot(
            obs_xy[N_TRAIN, 0], obs_xy[N_TRAIN, 1], "s", color="C0", ms=6,
            label="holdout start",
        )
    ax.set_xlabel("east (km, centred)")
    ax.set_ylabel("north (km, centred)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{track.species} {track.band}: GPS + WSC-KF @ train MLE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "movement_track.png"), dpi=150)

    if DO_HELDOUT:
        # ---- comparison figure: θ̂ plane | reference held-out NLL | track ----
        compare = []
        compare.append(
            {
                "name": "WSC-KF",
                "color": "#2ca02c",
                "marker": "o",
                "tau": float(wsckf_tau_tr[-1]),
                "sv": float(wsckf_sv_tr[-1]),
                "held": float(enkf_ref_at_wsckf),
                "held_err": float(enkf_ref_at_wsckf_std),
                "path_tau": wsckf_tau_tr,
                "path_sv": wsckf_sv_tr,
            }
        )
        compare.append(
            {
                "name": "EKF",
                "color": "#d62728",
                "marker": "s",
                "tau": float(ekf_tau_tr[-1]),
                "sv": float(ekf_sv_tr[-1]),
                "held": float(enkf_ref_at_ekf),
                "held_err": float(enkf_ref_at_ekf_std),
                "path_tau": ekf_tau_tr,
                "path_sv": ekf_sv_tr,
            }
        )
        if stochastic:
            compare.append(
                {
                    "name": "EnKF",
                    "color": "#1f77b4",
                    "marker": "D",
                    "tau": float(enkf_tau_tr[-1]),
                    "sv": float(enkf_sv_tr[-1]),
                    "held": float(enkf_ref_at_enkf),
                    "held_err": float(enkf_ref_at_enkf_std),
                    "path_tau": enkf_tau_tr,
                    "path_sv": enkf_sv_tr,
                }
            )

        fig_c = plt.figure(figsize=(12.8, 4.4))
        gs = fig_c.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 1.15], wspace=0.32)

        # A — parameter estimates from train MLE
        ax_a = fig_c.add_subplot(gs[0, 0])
        finite = wsckf_Z[np.isfinite(wsckf_Z)]
        if finite.size:
            c_vmax = float(np.max(finite))
            c_vmin = c_vmax - 4000.0
            ax_a.contourf(
                Tau, Sv, wsckf_Z, levels=18, cmap="Greys",
                vmin=c_vmin, vmax=c_vmax, alpha=0.55, zorder=0,
            )
        for rec in compare:
            ax_a.plot(
                rec["path_tau"], rec["path_sv"], "-", color=rec["color"],
                lw=1.0, alpha=0.45, zorder=2,
            )
            ax_a.plot(
                rec["tau"], rec["sv"], rec["marker"], color=rec["color"],
                ms=10, markeredgecolor="k", markeredgewidth=0.8, zorder=3,
                label=rf"{rec['name']}  $(\hat\tau_v,\hat\sigma_v)$",
            )
        ax_a.set_xscale("log")
        ax_a.set_yscale("log")
        ax_a.set_xlabel(r"velocity autocorr $\tau_v$ [h]")
        ax_a.set_ylabel(r"velocity scale $\sigma_v$ [km/h]")
        ax_a.set_title("A  Train MLE in parameter plane")
        ax_a.legend(fontsize=7.5, loc="best", framealpha=0.92)
        ax_a.grid(True, which="both", alpha=0.25)

        # B — shared reference held-out score at each θ̂
        ax_b = fig_c.add_subplot(gs[0, 1])
        names = [r["name"] for r in compare]
        vals = np.asarray([r["held"] for r in compare], dtype=float)
        errs = np.asarray([r["held_err"] for r in compare], dtype=float)
        cols = [r["color"] for r in compare]
        xpos = np.arange(len(names))
        ax_b.bar(
            xpos, vals,
            yerr=errs if np.any(errs > 0) else None,
            color=cols, edgecolor="0.2", lw=0.7, width=0.62,
            capsize=4, ecolor="0.15", zorder=3,
        )
        y0 = float(np.nanmin(vals - np.nan_to_num(errs)))
        y1 = float(np.nanmax(vals + np.nan_to_num(errs)))
        pad = max(0.08 * (y1 - y0), 0.15)
        ax_b.set_ylim(y0 - pad, y1 + 1.6 * pad)
        ax_b.set_xticks(xpos)
        ax_b.set_xticklabels(
            [rf"$\hat\theta_{{\mathrm{{{n}}}}}$" for n in names], fontsize=10
        )
        ax_b.set_ylabel(
            f"held-out NLL / obs\n(ref EnKF-{REF_ENKF_PARTICLES}p)"
        )
        ax_b.set_title("B  Reference score at each $\\hat\\theta$")
        ax_b.grid(axis="y", alpha=0.3, zorder=0)
        best = float(np.nanmin(vals))
        best_name = names[int(np.nanargmin(vals))]
        for x, v, e in zip(xpos, vals, errs):
            dlt = v - best
            label = f"{v:.3f}" if dlt < 1e-9 else f"{v:.3f}\n(+{dlt:.3f})"
            ax_b.text(
                x, v + (e if np.isfinite(e) else 0.0) + 0.02 * pad,
                label, ha="center", va="bottom", fontsize=8, color="0.15",
            )
        ax_b.axhline(best, color="0.45", ls="--", lw=0.9, zorder=1, alpha=0.8)
        ax_b.text(
            0.98, 0.02,
            f"best: $\\hat\\theta_{{\\mathrm{{{best_name}}}}}$",
            transform=ax_b.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="0.35",
        )

        # C — spatial split
        ax_c = fig_c.add_subplot(gs[0, 2])
        ax_c.plot(
            obs_xy[:N_TRAIN, 0], obs_xy[:N_TRAIN, 1], ".", ms=2.8,
            color="0.45", label="GPS train", zorder=1,
        )
        ax_c.plot(
            obs_xy[N_TRAIN:, 0], obs_xy[N_TRAIN:, 1], ".", ms=2.8,
            color="#c9b458", label="GPS held-out", zorder=1,
        )
        ax_c.plot(
            filt_xy[:, 0], filt_xy[:, 1], "-", lw=1.1, color="#2ca02c",
            label="WSC-KF mean @ train MLE", zorder=2,
        )
        ax_c.plot(obs_xy[0, 0], obs_xy[0, 1], "o", color="#2ca02c", ms=6, zorder=3)
        ax_c.plot(
            obs_xy[N_TRAIN, 0], obs_xy[N_TRAIN, 1], "s", color="#1f77b4",
            ms=6, markeredgecolor="k", markeredgewidth=0.5, zorder=3,
            label="holdout start",
        )
        ax_c.set_xlabel("east (km, centred)")
        ax_c.set_ylabel("north (km, centred)")
        ax_c.set_aspect("equal", adjustable="datalim")
        ax_c.set_title("C  Train / held-out split")
        ax_c.legend(fontsize=7, loc="best", framealpha=0.9)
        ax_c.grid(alpha=0.25)

        fig_c.suptitle(
            f"{track.band}: train-fit $\\hat\\theta$ → held-out NLL under "
            f"ref EnKF-{REF_ENKF_PARTICLES}p  "
            f"(N_train={N_TRAIN}, N_held={N_HELD})",
            fontsize=11.5,
            y=1.02,
        )
        fig_c.savefig(
            os.path.join(OUTPUT_DIR, "movement_heldout_compare.png"),
            dpi=160, bbox_inches="tight",
        )
        fig_h, ax = plt.subplots(figsize=(4.8, 3.8))
        ax.bar(
            xpos, vals,
            yerr=errs if np.any(errs > 0) else None,
            color=cols, edgecolor="0.2", lw=0.6, width=0.65,
            capsize=4, ecolor="0.2",
        )
        ax.set_ylim(y0 - pad, y1 + 1.6 * pad)
        ax.set_xticks(xpos)
        ax.set_xticklabels(
            [rf"$\hat\theta_{{\mathrm{{{n}}}}}$" for n in names]
        )
        ax.set_ylabel(f"held-out NLL / obs (ref EnKF-{REF_ENKF_PARTICLES}p)")
        ax.set_title("Reference held-out NLL at each $\\hat\\theta$")
        ax.grid(axis="y", alpha=0.3)
        for x, v in zip(xpos, vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        fig_h.tight_layout()
        fig_h.savefig(os.path.join(OUTPUT_DIR, "movement_heldout.png"), dpi=150)
        plt.close(fig_c)
        plt.close(fig_h)

    # ===== CSV =====
    with open(os.path.join(OUTPUT_DIR, "movement_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        w.writerow(["setup", "band", track.band])
        w.writerow(["setup", "n_obs", N_OBS])
        w.writerow(["setup", "n_train", N_TRAIN])
        w.writerow(["setup", "n_held", N_HELD])
        w.writerow(["setup", "holdout_frac", _args.holdout_frac])
        w.writerow(["setup", "parametrization", "tau_v, sigma_v"])
        w.writerow(["setup", "cd_solver", SOLVER])
        w.writerow(["setup", "grid", n_g])
        w.writerow(["setup", "mle_iters", _args.mle_iters])
        w.writerow(["setup", "enkf_particles", ENKF_PARTICLES])
        w.writerow(["setup", "wsckf_num_paths", _args.num_paths])
        w.writerow(["setup", "wsckf_cubature_order", _args.cubature_order])
        w.writerow(["setup", "mle_uses", "train_prefix_only"])
        w.writerow(["setup", "heldout_score", "ref_enkf_at_each_train_theta"])
        w.writerow(["setup", "ref_enkf_particles", REF_ENKF_PARTICLES])
        w.writerow(["setup", "ref_seeds", ",".join(str(s) for s in REF_SEEDS)])
        w.writerow(["mle", "wsckf_tau_v", wsckf_tau_tr[-1]])
        w.writerow(["mle", "wsckf_sigma_v", wsckf_sv_tr[-1]])
        w.writerow(["mle", "wsckf_sigma0_implied", float(sig0_wsckf)])
        w.writerow(["mle", "ekf_tau_v", ekf_tau_tr[-1]])
        w.writerow(["mle", "ekf_sigma_v", ekf_sv_tr[-1]])
        w.writerow(["mle", "ekf_sigma0_implied", float(sig0_ekf)])
        if stochastic:
            _, sig0_enkf = tau_sigma_to_beta_sigma0(enkf_tau_tr[-1], enkf_sv_tr[-1])
            w.writerow(["mle", "enkf_tau_v", enkf_tau_tr[-1]])
            w.writerow(["mle", "enkf_sigma_v", enkf_sv_tr[-1]])
            w.writerow(["mle", "enkf_sigma0_implied", float(sig0_enkf)])
        w.writerow(["mle", "wsckf_grid_tau_v", wsckf_tau])
        w.writerow(["mle", "wsckf_grid_sigma_v", wsckf_sv])
        w.writerow(["mle", "ekf_grid_tau_v", ekf_tau])
        w.writerow(["mle", "ekf_grid_sigma_v", ekf_sv])
        if DO_HELDOUT:
            w.writerow(["heldout_ref_enkf", "at_theta_wsckf_mean", enkf_ref_at_wsckf])
            w.writerow(["heldout_ref_enkf", "at_theta_wsckf_std", enkf_ref_at_wsckf_std])
            w.writerow(["heldout_ref_enkf", "at_theta_ekf_mean", enkf_ref_at_ekf])
            w.writerow(["heldout_ref_enkf", "at_theta_ekf_std", enkf_ref_at_ekf_std])
            if stochastic:
                w.writerow(["heldout_ref_enkf", "at_theta_enkf_mean", enkf_ref_at_enkf])
                w.writerow(["heldout_ref_enkf", "at_theta_enkf_std", enkf_ref_at_enkf_std])
            w.writerow(["heldout_ref_wsckf", "at_theta_wsckf", wsckf_ref_at_wsckf])
            w.writerow(["heldout_ref_wsckf", "at_theta_ekf", wsckf_ref_at_ekf])
            if stochastic:
                w.writerow(["heldout_ref_wsckf", "at_theta_enkf", wsckf_ref_at_enkf])
            w.writerow(["heldout_self", "wsckf_nll_per_obs", wsckf_held])
            w.writerow(["heldout_self", "ekf_nll_per_obs", ekf_held])
            if stochastic:
                w.writerow(["heldout_self", "enkf_nll_per_obs_mean", enkf_held])
                w.writerow(["heldout_self", "enkf_nll_per_obs_std", enkf_held_std])
                w.writerow([
                    "heldout_self", "enkf_seeds",
                    ",".join(str(s) for s in HELDOUT_ENKF_SEEDS),
                ])
        w.writerow(["timing", "wsckf_forward_s", t_wsckf_fwd])
        w.writerow(["timing", "wsckf_ad_vjp_s", t_wsckf_ad])
        w.writerow(["timing", "ekf_forward_s", t_ekf_fwd])
        w.writerow(["timing", "ekf_ad_vjp_s", t_ekf_ad])
        w.writerow(["timing", "wsckf_ad_mle_s", wsckf_mle_t])
        w.writerow(["timing", "ekf_ad_mle_s", ekf_mle_t])
        if stochastic:
            w.writerow(["timing", "enkf_forward_s", t_enkf_fwd])
            w.writerow(["timing", "enkf_ad_vjp_s", t_enkf_ad])
            w.writerow(["timing", "enkf_ad_mle_s", enkf_mle_t])
        w.writerow([])
        w.writerow(
            ["tau_v", "sigma_v", "wsckf_train_loglik", "ekf_train_loglik"]
            + (["enkf_train_loglik"] if stochastic else [])
        )
        for i, tau in enumerate(tau_axis):
            for j, sv in enumerate(sv_axis):
                row = [tau, sv, wsckf_Z[i, j], ekf_Z[i, j]]
                if stochastic:
                    row.append(enkf_Z[i, j])
                w.writerow(row)

    saved = [
        "movement_loglik.png",
        "movement_mle_trace.png",
        "movement_track.png",
        "movement_timing.png",
        "movement_summary.csv",
    ]
    if DO_HELDOUT:
        saved.insert(3, "movement_heldout_compare.png")
        saved.insert(4, "movement_heldout.png")
    print("Saved: " + ", ".join(saved))


if __name__ == "__main__":
    main()
