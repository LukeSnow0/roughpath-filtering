"""Joint filter-MLE of (τ_v, σ_v) on REAL duck GPS telemetry.

Fits an OU-velocity movement model with speed-dependent diffusion to a single
individual's hourly GPS track (Dryad doi:10.5061/dryad.1ns1rn93p) by maximising
the Bayesian-filter marginal log-likelihood over the interpretable pair:

    τ_v = 1/β_v     velocity autocorrelation time (hours)
    σ_v             asymptotic per-axis velocity std under frozen g≡σ₀
                    (σ_v = σ₀/√(2β_v)  ⇒  σ₀ = σ_v √(2/τ_v))

Home-range spring ω², speed scale v_c, and GPS R are held fixed.

Compares:
* CRF-AD -- pathwise jax.value_and_grad through checkpointed ODE adjoint
* CD-EKF-AD -- moment-ODE EKF pathwise VJP
* CD-EnKF-AD -- ensemble filter, pathwise VJP with fixed RNG key
  (--skip-stochastic to drop EnKF)

Outputs (experiments/movement_test/output/):
  movement_loglik.png       2D L(τ_v, σ_v) heatmaps
  movement_mle_trace.png    joint Adam traces in (τ_v, σ_v)
  movement_track.png        GPS + CRF filtered mean at MLE
  movement_timing.png       per-eval wall times
  movement_summary.csv

Run:
  uv run --extra experiments python experiments/movement_test/run_movement_mle.py
  uv run --extra experiments python experiments/movement_test/run_movement_mle.py \\
      --n-obs 600 --mle-iters 40 --grid 10
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
from crf import build_filter
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
_parser.add_argument("--mle-iters", type=int, default=40, dest="mle_iters")
_parser.add_argument("--lr", type=float, default=0.1, help="Adam step on (log τ_v, log σ_v)")
_parser.add_argument("--tau-init", type=float, default=2.0, dest="tau_init",
                     help="initial velocity autocorr time (hours)")
_parser.add_argument("--sigma-v-init", type=float, default=2.0, dest="sigma_v_init",
                     help="initial asymptotic velocity std (km/h)")
_parser.add_argument("--grid", type=int, default=10,
                     help="per-axis points for 2D likelihood heatmap")
_parser.add_argument("--dt0", type=float, default=0.1, help="inner RK4 step (hours)")
_parser.add_argument("--cubature-order", type=int, default=3, dest="cubature_order")
_parser.add_argument("--enkf-particles", type=int, default=200, dest="enkf_particles")
_parser.add_argument("--n-timing", type=int, default=3, dest="n_timing")
_parser.add_argument(
    "--solver", type=str, default="rk4", choices=("rk4", "diffrax"),
    help="CD-filter forecast integrator (CRF uses its own cubature RK4)",
)
_parser.add_argument("--timing-only", action="store_true")
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--skip-stochastic", action="store_true",
                     help="skip EnKF (CRF + EKF only)")
_args = _parser.parse_args()
ENKF_PARTICLES = _args.enkf_particles
N_TIMING = _args.n_timing
SOLVER = _args.solver

# ---------------------------------------------------------------------------
track = load_duck_track(band=_args.band, max_obs=_args.n_obs)
YS = jnp.asarray(track.xy_km)
N_OBS = YS.shape[0]
DT = 1.0
DT0 = _args.dt0

P = MovementParams(center=(0.0, 0.0))
m0_np, P0_np = init_estimate(np.asarray(YS), P)
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
    f"parametrization: (τ_v, σ_v) with σ₀=σ_v√(2/τ_v), β_v=1/τ_v;  "
    f"defaults from MovementParams → τ_v={float(_tau0_default):.2f}h, "
    f"σ_v={float(_sv0_default):.2f} km/h"
)


# ---------------------------------------------------------------------------
# Filter likelihoods threaded by (τ_v, σ_v)
# ---------------------------------------------------------------------------
class MInputs(NamedTuple):
    t_idx: jnp.ndarray
    tau_v: jnp.ndarray
    sigma_v: jnp.ndarray


def build_crf(*, checkpoint_steps: bool):
    def get_init_params(mi):
        return M0, CHOL_P0

    def get_dynamics(mi):
        beta_v, sigma0 = tau_sigma_to_beta_sigma0(mi.tau_v, mi.sigma_v)
        drift, L, _, _, _ = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
        return drift, L

    def get_observations(mi):
        return OBS_FN, CHOL_R, YS[mi.t_idx - 1]

    return build_filter(
        get_init_params=get_init_params,
        get_dynamics=get_dynamics,
        get_observations=get_observations,
        noise_dim=2,
        dt=DT,
        integration_dt=DT0,
        cubature_order=_args.cubature_order,
        multidim_paths="recombined",
        sde_form="ito",
        checkpoint_steps=checkpoint_steps,
    )


CRF = build_crf(checkpoint_steps=False)
CRF_CKPT = build_crf(checkpoint_steps=True)
_run = jit(run_filter, static_argnames=["filter_obj"])


def _mi(tau_v, sigma_v):
    return MInputs(
        jnp.arange(N_OBS + 1),
        jnp.full(N_OBS + 1, tau_v),
        jnp.full(N_OBS + 1, sigma_v),
    )


def crf_loglik(tau_v, sigma_v, *, checkpoint=True):
    filt = CRF_CKPT if checkpoint else CRF
    return _run(filt, _mi(tau_v, sigma_v)).log_normalizing_constant[-1]


def crf_states(tau_v, sigma_v):
    return _run(CRF_CKPT, _mi(tau_v, sigma_v))


def ekf_loglik(tau_v, sigma_v):
    beta_v, sigma0 = tau_sigma_to_beta_sigma0(tau_v, sigma_v)
    drift, L, Qc, _, _ = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
    return run_cd_ekf(
        drift, OBS_FN, L, Qc, R, M0, P0, YS, DT, DT0,
        state_order="first", solver=SOLVER,
    ).cum_loglik[-1]


def enkf_loglik(tau_v, sigma_v, key):
    beta_v, sigma0 = tau_sigma_to_beta_sigma0(tau_v, sigma_v)
    drift, L, Qc, _, _ = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
    return run_cd_enkf(
        drift, OBS_FN, L, Qc, R, M0, P0, YS, DT, key,
        n_particles=ENKF_PARTICLES, noise_dim=2, dt0=DT0, solver=SOLVER,
    ).cum_loglik[-1]


def _ll_from_theta(ll2, theta):
    """ll2(tau, sigma_v) evaluated at theta = (log τ_v, log σ_v)."""
    return ll2(jnp.exp(theta[0]), jnp.exp(theta[1]))


crf_vg = jit(value_and_grad(
    lambda th: _ll_from_theta(lambda t, s: crf_loglik(t, s, checkpoint=True), th)
))
ekf_vg = jit(value_and_grad(lambda th: _ll_from_theta(ekf_loglik, th)))
_enkf_key_ad = jr.PRNGKey(_args.seed + 17)
enkf_vg = jit(value_and_grad(
    lambda th: _ll_from_theta(lambda t, s: enkf_loglik(t, s, _enkf_key_ad), th)
))
_crf_ll = jit(lambda t, s: crf_loglik(t, s, checkpoint=False))
_ekf_ll = jit(ekf_loglik)
_enkf_ll = jit(enkf_loglik)


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
    """Adam ascent on theta=(log τ_v, log σ_v); returns traces + loglik."""
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

    # ---- timing (always; cheap relative to 2D grid) ----
    print(f"\n=== Median wall time (n={N_TIMING}, N={N_OBS}, cd_solver={SOLVER}) ===")
    t_crf_fwd = median_time(lambda: _crf_ll(_args.tau_init, _args.sigma_v_init))
    t_crf_ad = median_time(lambda: crf_vg(theta0))
    t_ekf_fwd = median_time(lambda: _ekf_ll(_args.tau_init, _args.sigma_v_init))
    t_ekf_ad = median_time(lambda: ekf_vg(theta0))
    print(f"  CRF forward:        {t_crf_fwd:.4f}s")
    print(f"  CRF-AD (2-param):   {t_crf_ad:.4f}s")
    print(f"  CD-EKF forward:     {t_ekf_fwd:.4f}s")
    print(f"  CD-EKF-AD:          {t_ekf_ad:.4f}s")
    t_enkf_fwd = t_enkf_ad = float("nan")
    if stochastic:
        key_t = jr.PRNGKey(_args.seed + 1)
        t_enkf_fwd = median_time(
            lambda: _enkf_ll(_args.tau_init, _args.sigma_v_init, key_t)
        )
        t_enkf_ad = median_time(lambda: enkf_vg(theta0))
        print(f"  CD-EnKF forward:    {t_enkf_fwd:.4f}s  ({ENKF_PARTICLES}p)")
        print(f"  CD-EnKF-AD:         {t_enkf_ad:.4f}s")

    fig_t, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    timing_rows = [
        ("CRF fwd", t_crf_fwd, "C2"),
        ("CRF-AD", t_crf_ad, "C2"),
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
    ax.set_title(f"Per-eval (median of {N_TIMING})")
    ax.grid(axis="y", alpha=0.3, which="both")
    for b, t in zip(bars, times):
        if np.isfinite(t) and t > 0:
            ax.text(
                b.get_x() + b.get_width() / 2, t * 1.15, f"{t:.3f}s",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    # Full joint Adam MLE wall time (same iteration budget as a real fit).
    print(f"\n=== Joint Adam MLE wall ({_args.mle_iters} iters) ===")
    t0 = time.perf_counter()
    adam_mle_2d(crf_vg)
    crf_mle_wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    adam_mle_2d(ekf_vg)
    ekf_mle_wall = time.perf_counter() - t0
    enkf_mle_wall = float("nan")
    print(f"  CRF-AD:  {crf_mle_wall:.2f}s")
    print(f"  EKF-AD:  {ekf_mle_wall:.2f}s")
    if stochastic:
        t0 = time.perf_counter()
        adam_mle_2d(enkf_vg)
        enkf_mle_wall = time.perf_counter() - t0
        print(f"  EnKF-AD: {enkf_mle_wall:.2f}s")

    ax = axes[1]
    mle_labels = ["CRF-AD", "EKF-AD"]
    mle_times = [crf_mle_wall, ekf_mle_wall]
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
    ax.set_title(f"Full joint Adam MLE ({_args.mle_iters} iters)")
    ax.grid(axis="y", alpha=0.3, which="both")
    for b, t in zip(bars, mle_times):
        if np.isfinite(t) and t > 0:
            ax.text(
                b.get_x() + b.get_width() / 2, t * 1.15, f"{t:.2f}s",
                ha="center", va="bottom", fontsize=8,
            )

    fig_t.suptitle(
        f"Joint $(\\tau_v,\\sigma_v)$ timing — {track.band}, N={N_OBS}, "
        f"dt0={DT0}, cd_solver={SOLVER}",
        fontsize=11,
    )
    fig_t.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"movement_timing_{SOLVER}.png")
    fig_t.savefig(out_png, dpi=150)
    fig_t.savefig(os.path.join(OUTPUT_DIR, "movement_timing.png"), dpi=150)
    out_csv = os.path.join(OUTPUT_DIR, f"movement_timing_{SOLVER}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "method", "seconds", "solver", "n_obs", "n_timing", "mle_iters"])
        for lab, t, _ in timing_rows:
            w.writerow(["per_eval", lab.replace("\n", " "), t, SOLVER, N_OBS, N_TIMING, ""])
        w.writerow(["mle_wall", "CRF-AD", crf_mle_wall, SOLVER, N_OBS, "", _args.mle_iters])
        w.writerow(["mle_wall", "EKF-AD", ekf_mle_wall, SOLVER, N_OBS, "", _args.mle_iters])
        if stochastic:
            w.writerow(["mle_wall", "EnKF-AD", enkf_mle_wall, SOLVER, N_OBS, "", _args.mle_iters])
    print(f"Saved: {out_png}, movement_timing.png, {out_csv}")

    if timing_only:
        return

    # Rebuild a simple single-panel timing plot is already done; continue to MLE/grid.
    # (per-eval + MLE wall already written above.)
    # ---- 2D likelihood grids ----
    n_g = _args.grid
    tau_axis = np.geomspace(0.4, 8.0, n_g)       # hours
    sv_axis = np.geomspace(0.5, 8.0, n_g)        # km/h
    print(f"\n=== 2D loglik grid ({n_g}×{n_g}) ===")
    crf_Z = np.full((n_g, n_g), np.nan)
    ekf_Z = np.full((n_g, n_g), np.nan)
    enkf_Z = np.full((n_g, n_g), np.nan) if stochastic else None
    key_grid = jr.PRNGKey(_args.seed)
    for i, tau in enumerate(tau_axis):
        for j, sv in enumerate(sv_axis):
            crf_Z[i, j] = float(_crf_ll(tau, sv))
            ekf_Z[i, j] = float(_ekf_ll(tau, sv))
            if stochastic:
                enkf_Z[i, j] = float(_enkf_ll(tau, sv, key_grid))
        if (i + 1) % max(1, n_g // 4) == 0:
            print(f"  grid row {i+1}/{n_g} done")

    crf_tau, crf_sv, crf_L = _grid_argmax(tau_axis, sv_axis, crf_Z)
    ekf_tau, ekf_sv, ekf_L = _grid_argmax(tau_axis, sv_axis, ekf_Z)
    print(f"  CRF  grid-argmax:  τ_v={crf_tau:.2f}h  σ_v={crf_sv:.2f}  L={crf_L:.1f}")
    print(f"  EKF  grid-argmax:  τ_v={ekf_tau:.2f}h  σ_v={ekf_sv:.2f}  L={ekf_L:.1f}")
    if stochastic:
        enkf_tau, enkf_sv, enkf_L = _grid_argmax(tau_axis, sv_axis, enkf_Z)
        print(f"  EnKF grid-argmax:  τ_v={enkf_tau:.2f}h  σ_v={enkf_sv:.2f}  L={enkf_L:.1f}")

    # ---- joint Adam MLE ----
    print(f"\n=== Joint Adam MLE ({_args.mle_iters} iters) ===")
    t0 = time.perf_counter()
    crf_vg(theta0)
    crf_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    crf_tau_tr, crf_sv_tr, crf_ll_tr = adam_mle_2d(crf_vg)
    crf_mle_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    ekf_tau_tr, ekf_sv_tr, ekf_ll_tr = adam_mle_2d(ekf_vg)
    ekf_mle_t = time.perf_counter() - t0
    line = (
        f"  CRF-AD:  τ̂={crf_tau_tr[-1]:.3f}h  σ̂_v={crf_sv_tr[-1]:.3f}  "
        f"(compile={crf_compile:.2f}s, {crf_mle_t:.2f}s)"
    )
    print(line)
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

    # implied σ₀ at CRF MLE
    _, sig0_crf = tau_sigma_to_beta_sigma0(crf_tau_tr[-1], crf_sv_tr[-1])
    _, sig0_ekf = tau_sigma_to_beta_sigma0(ekf_tau_tr[-1], ekf_sv_tr[-1])
    print(f"  implied σ₀:  CRF={float(sig0_crf):.3f}  EKF={float(sig0_ekf):.3f}")

    st = crf_states(jnp.asarray(crf_tau_tr[-1]), jnp.asarray(crf_sv_tr[-1]))
    filt_xy = np.asarray(st.mean)[:, :2]
    obs_xy = np.asarray(YS)

    # ===== plots =====
    # 1. 2D likelihood heatmaps
    n_panels = 3 if stochastic else 2
    fig1, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.4), sharey=True)
    panels = [
        ("CRF", crf_Z, "C2", (crf_tau, crf_sv), (crf_tau_tr[-1], crf_sv_tr[-1])),
        ("CD-EKF", ekf_Z, "C3", (ekf_tau, ekf_sv), (ekf_tau_tr[-1], ekf_sv_tr[-1])),
    ]
    if stochastic:
        panels.append(
            ("CD-EnKF", enkf_Z, "C1", (enkf_tau, enkf_sv),
             (enkf_tau_tr[-1], enkf_sv_tr[-1]))
        )
    # Shared color scale near the peaks (ignore catastrophic tails).
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
        f"Joint filter log-likelihood $L(\\tau_v,\\sigma_v)$ — {track.band}, N={N_OBS}",
        fontsize=11,
    )
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "movement_loglik.png"), dpi=150)

    # 2. MLE traces in the (τ_v, σ_v) plane + per-parameter traces
    fig2, axes2 = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes2[0]
    ax.plot(crf_tau_tr, crf_sv_tr, "o-", color="C2", ms=4, label="CRF-AD")
    ax.plot(ekf_tau_tr, ekf_sv_tr, "s-", color="C3", ms=4, label="EKF-AD")
    if stochastic:
        ax.plot(enkf_tau_tr, enkf_sv_tr, "D-", color="C1", ms=4, label="EnKF-AD")
    ax.plot(crf_tau, crf_sv, "*", color="C2", ms=14, markeredgecolor="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_v$ [h]")
    ax.set_ylabel(r"$\sigma_v$ [km/h]")
    ax.set_title("Adam path in parameter plane")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes2[1]
    it = np.arange(_args.mle_iters)
    ax.plot(it, crf_tau_tr, "o-", color="C2", ms=3, label=r"CRF $\tau_v$")
    ax.plot(it, crf_sv_tr, "o--", color="C2", ms=3, alpha=0.7, label=r"CRF $\sigma_v$")
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
    fig2.suptitle("Joint gradient filter-MLE of $(\\tau_v,\\sigma_v)$", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "movement_mle_trace.png"), dpi=150)

    # 3. track
    fig3, ax = plt.subplots(figsize=(7, 7))
    ax.plot(obs_xy[:, 0], obs_xy[:, 1], ".", ms=3, color="0.6", label="GPS fixes")
    ax.plot(
        filt_xy[:, 0], filt_xy[:, 1], "-", lw=1.0, color="C2",
        label=(
            f"CRF filtered mean\n"
            f"$\\hat\\tau_v$={crf_tau_tr[-1]:.2f}h, $\\hat\\sigma_v$={crf_sv_tr[-1]:.2f}"
        ),
    )
    ax.plot(obs_xy[0, 0], obs_xy[0, 1], "o", color="green", label="start")
    ax.set_xlabel("east (km, centred)")
    ax.set_ylabel("north (km, centred)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{track.species} {track.band}: GPS + CRF @ joint MLE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "movement_track.png"), dpi=150)

    # ===== CSV =====
    with open(os.path.join(OUTPUT_DIR, "movement_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        w.writerow(["setup", "band", track.band])
        w.writerow(["setup", "n_obs", N_OBS])
        w.writerow(["setup", "parametrization", "tau_v, sigma_v"])
        w.writerow(["setup", "cd_solver", SOLVER])
        w.writerow(["setup", "grid", n_g])
        w.writerow(["setup", "mle_iters", _args.mle_iters])
        w.writerow(["setup", "enkf_particles", ENKF_PARTICLES])
        w.writerow(["mle", "crf_tau_v", crf_tau_tr[-1]])
        w.writerow(["mle", "crf_sigma_v", crf_sv_tr[-1]])
        w.writerow(["mle", "crf_sigma0_implied", float(sig0_crf)])
        w.writerow(["mle", "ekf_tau_v", ekf_tau_tr[-1]])
        w.writerow(["mle", "ekf_sigma_v", ekf_sv_tr[-1]])
        w.writerow(["mle", "ekf_sigma0_implied", float(sig0_ekf)])
        if stochastic:
            _, sig0_enkf = tau_sigma_to_beta_sigma0(enkf_tau_tr[-1], enkf_sv_tr[-1])
            w.writerow(["mle", "enkf_tau_v", enkf_tau_tr[-1]])
            w.writerow(["mle", "enkf_sigma_v", enkf_sv_tr[-1]])
            w.writerow(["mle", "enkf_sigma0_implied", float(sig0_enkf)])
        w.writerow(["mle", "crf_grid_tau_v", crf_tau])
        w.writerow(["mle", "crf_grid_sigma_v", crf_sv])
        w.writerow(["mle", "ekf_grid_tau_v", ekf_tau])
        w.writerow(["mle", "ekf_grid_sigma_v", ekf_sv])
        w.writerow(["timing", "crf_forward_s", t_crf_fwd])
        w.writerow(["timing", "crf_ad_vjp_s", t_crf_ad])
        w.writerow(["timing", "ekf_forward_s", t_ekf_fwd])
        w.writerow(["timing", "ekf_ad_vjp_s", t_ekf_ad])
        w.writerow(["timing", "crf_ad_mle_s", crf_mle_t])
        w.writerow(["timing", "ekf_ad_mle_s", ekf_mle_t])
        if stochastic:
            w.writerow(["timing", "enkf_forward_s", t_enkf_fwd])
            w.writerow(["timing", "enkf_ad_vjp_s", t_enkf_ad])
            w.writerow(["timing", "enkf_ad_mle_s", enkf_mle_t])
        w.writerow([])
        w.writerow(["tau_v", "sigma_v", "crf_loglik", "ekf_loglik"]
                   + (["enkf_loglik"] if stochastic else []))
        for i, tau in enumerate(tau_axis):
            for j, sv in enumerate(sv_axis):
                row = [tau, sv, crf_Z[i, j], ekf_Z[i, j]]
                if stochastic:
                    row.append(enkf_Z[i, j])
                w.writerow(row)

    print(
        "Saved: movement_loglik.png, movement_mle_trace.png, "
        "movement_track.png, movement_timing.png, movement_summary.csv"
    )


if __name__ == "__main__":
    main()
