"""Pooled multi-bird filter-MLE + leave-one-bird-out (LOBO) on duck GPS.

Design
------
* Select several same-species tracks (default: longest mallards).
* **Pooled MLE:** shared ``(τ_v, σ_v)`` maximising
  ``Σ_i L_filter(θ; y^{(i)})`` over in-fold birds (full selected prefix).
* **LOBO:** for each held-out bird ``b``, fit on all birds except ``b``, then
  score bird ``b`` under a **shared reference EnKF** (full-track NLL/obs).
  The held-out bird never enters the fit (no leakage).
* Compares WSC-KF / CD-EKF / CD-EnKF.

This targets cross-individual prediction, not one-track suffix holdout.

Run:
  uv run --extra experiments python experiments/movement_test/run_movement_pooled_lobo.py
  uv run --extra experiments python experiments/movement_test/run_movement_pooled_lobo.py \\
      --n-birds 4 --n-obs 400 --mle-iters 25 --num-paths 35

Outputs under ``experiments/movement_test/output/``:
  ``movement_lobo_compare.png``, ``movement_lobo_summary.csv``,
  ``movement_lobo_folds.csv``
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
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

from movement_data import load_duck_track, select_bands
from movement_model import (
    MovementParams,
    init_estimate,
    make_movement_dynamics,
    tau_sigma_to_beta_sigma0,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="Pooled multi-bird movement filter-MLE + leave-one-bird-out"
)
_parser.add_argument("--species", type=str, default="MALL")
_parser.add_argument("--n-birds", type=int, default=4, dest="n_birds")
_parser.add_argument("--min-n", type=int, default=500, dest="min_n",
                    help="Min longest-run length to be eligible")
_parser.add_argument(
    "--bands",
    type=str,
    default="",
    help="Comma-separated bands (default: longest --n-birds of --species)",
)
_parser.add_argument("--n-obs", type=int, default=400, dest="n_obs")
_parser.add_argument("--mle-iters", type=int, default=25, dest="mle_iters")
_parser.add_argument("--lr", type=float, default=0.1)
_parser.add_argument("--tau-init", type=float, default=2.0, dest="tau_init")
_parser.add_argument("--sigma-v-init", type=float, default=2.0, dest="sigma_v_init")
_parser.add_argument("--dt0", type=float, default=0.1)
_parser.add_argument("--cubature-order", type=int, default=3, dest="cubature_order")
_parser.add_argument("--num-paths", type=int, default=35, dest="num_paths")
_parser.add_argument("--enkf-particles", type=int, default=100, dest="enkf_particles")
_parser.add_argument(
    "--ref-enkf-particles", type=int, default=300, dest="ref_enkf_particles"
)
_parser.add_argument("--ref-seeds", type=str, default="0,1")
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument(
    "--solver", type=str, default="rk4", choices=("rk4", "diffrax")
)
_parser.add_argument(
    "--skip-stochastic",
    action="store_true",
    help="Skip EnKF MLE / LOBO (WSC-KF + EKF only)",
)
_parser.add_argument(
    "--pooled-only",
    action="store_true",
    help="Skip LOBO folds; only fit pooled θ̂ on all birds",
)
_args = _parser.parse_args()

DT = 1.0
DT0 = float(_args.dt0)
SOLVER = _args.solver
ENKF_PARTICLES = _args.enkf_particles
REF_ENKF_PARTICLES = _args.ref_enkf_particles
P = MovementParams(center=(0.0, 0.0))
_, _, _, OBS_FN, R = make_movement_dynamics(P, beta_v=P.beta_v, sigma0=P.sigma0)
CHOL_R = jnp.linalg.cholesky(R)


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


@dataclass
class TrackPack:
    band: str
    species: str
    sex: str
    ys: jnp.ndarray
    m0: jnp.ndarray
    p0: jnp.ndarray
    chol_p0: jnp.ndarray
    n_obs: int


class MInputs(NamedTuple):
    t_idx: jnp.ndarray
    tau_v: jnp.ndarray
    sigma_v: jnp.ndarray


def _mi(tau_v, sigma_v, n_obs: int):
    return MInputs(
        jnp.arange(n_obs + 1),
        jnp.full(n_obs + 1, tau_v),
        jnp.full(n_obs + 1, sigma_v),
    )


def load_packs(bands: list[str], n_obs: int) -> list[TrackPack]:
    packs: list[TrackPack] = []
    for band in bands:
        tr = load_duck_track(band=band, max_obs=n_obs)
        ys = jnp.asarray(tr.xy_km)
        m0_np, p0_np = init_estimate(np.asarray(ys), P)
        m0 = jnp.asarray(m0_np)
        p0 = jnp.asarray(p0_np)
        packs.append(
            TrackPack(
                band=tr.band,
                species=tr.species,
                sex=tr.sex,
                ys=ys,
                m0=m0,
                p0=p0,
                chol_p0=jnp.linalg.cholesky(p0),
                n_obs=int(ys.shape[0]),
            )
        )
    return packs


def build_wsckf_for(pack: TrackPack, *, checkpoint_steps: bool):
    def get_init_params(_mi):
        return pack.m0, pack.chol_p0

    def get_dynamics(mi):
        beta_v, sigma0 = tau_sigma_to_beta_sigma0(mi.tau_v, mi.sigma_v)
        drift, L, _, _, _ = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
        return drift, L

    def get_observations(mi):
        return OBS_FN, CHOL_R, pack.ys[mi.t_idx - 1]

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


_run = jit(run_filter, static_argnames=["filter_obj"])


def make_track_logliks(pack: TrackPack):
    """Per-track train (full selected prefix) loglik callables + ref scorer."""
    wsckf = build_wsckf_for(pack, checkpoint_steps=True)
    wsckf_fwd = build_wsckf_for(pack, checkpoint_steps=False)

    def wsckf_ll(tau_v, sigma_v):
        return _run(wsckf, _mi(tau_v, sigma_v, pack.n_obs)).log_normalizing_constant[-1]

    def wsckf_ll_fwd(tau_v, sigma_v):
        return _run(
            wsckf_fwd, _mi(tau_v, sigma_v, pack.n_obs)
        ).log_normalizing_constant[-1]

    def ekf_cum(tau_v, sigma_v):
        beta_v, sigma0 = tau_sigma_to_beta_sigma0(tau_v, sigma_v)
        drift, L, Qc, _, _ = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
        return run_cd_ekf(
            drift, OBS_FN, L, Qc, R, pack.m0, pack.p0, pack.ys, DT, DT0,
            state_order="first", solver=SOLVER,
        ).cum_loglik

    def enkf_cum(tau_v, sigma_v, key, *, n_particles: int):
        beta_v, sigma0 = tau_sigma_to_beta_sigma0(tau_v, sigma_v)
        drift, L, Qc, _, _ = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
        return run_cd_enkf(
            drift, OBS_FN, L, Qc, R, pack.m0, pack.p0, pack.ys, DT, key,
            n_particles=n_particles, noise_dim=2, dt0=DT0, solver=SOLVER,
        ).cum_loglik

    return {
        "band": pack.band,
        "n_obs": pack.n_obs,
        "wsckf_ll": jit(wsckf_ll),
        "wsckf_ll_fwd": jit(wsckf_ll_fwd),
        "ekf_ll": jit(lambda t, s: ekf_cum(t, s)[-1]),
        "ekf_cum": jit(ekf_cum),
        "enkf_cum": enkf_cum,  # n_particles varies; jit inside callers
    }


def nll_per_obs_from_total(ll: float, n_obs: int) -> float:
    return -float(ll) / float(n_obs)


def ref_enkf_nll_per_obs(fns, tau_v, sigma_v) -> tuple[float, float]:
    vals = []
    for s in REF_SEEDS:
        key = jr.PRNGKey(s + 1009 + 17 * REF_ENKF_PARTICLES)
        cum = fns["enkf_cum"](
            tau_v, sigma_v, key, n_particles=REF_ENKF_PARTICLES
        )
        vals.append(nll_per_obs_from_total(float(cum[-1]), fns["n_obs"]))
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def adam_pooled(vg_fn, label: str):
    th = np.array([np.log(_args.tau_init), np.log(_args.sigma_v_init)], dtype=float)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(2)
    v = np.zeros(2)
    tau_tr, sv_tr, ll_tr = [], [], []
    t0 = time.perf_counter()
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
        if t == 1 or t == _args.mle_iters or t % max(1, _args.mle_iters // 5) == 0:
            print(
                f"    {label} iter {t}/{_args.mle_iters}: "
                f"τ={tau_tr[-1]:.3f} σ_v={sv_tr[-1]:.3f} L={ll:.1f}"
            )
    wall = time.perf_counter() - t0
    return {
        "tau": float(tau_tr[-1]),
        "sigma_v": float(sv_tr[-1]),
        "ll_trace": np.asarray(ll_tr),
        "tau_trace": np.asarray(tau_tr),
        "sv_trace": np.asarray(sv_tr),
        "wall_s": wall,
    }


def make_pooled_vg(track_fns: list[dict], method: str, enkf_key=None):
    """value_and_grad of Σ_i L_i(θ) for method in {wsckf, ekf, enkf}."""

    if method == "wsckf":
        def total(th):
            tau, sv = jnp.exp(th[0]), jnp.exp(th[1])
            s = 0.0
            for fns in track_fns:
                s = s + fns["wsckf_ll"](tau, sv)
            return s

    elif method == "ekf":
        def total(th):
            tau, sv = jnp.exp(th[0]), jnp.exp(th[1])
            s = 0.0
            for fns in track_fns:
                s = s + fns["ekf_ll"](tau, sv)
            return s

    elif method == "enkf":
        assert enkf_key is not None

        def total(th):
            tau, sv = jnp.exp(th[0]), jnp.exp(th[1])
            s = 0.0
            for i, fns in enumerate(track_fns):
                key = jr.fold_in(enkf_key, i)
                cum = fns["enkf_cum"](
                    tau, sv, key, n_particles=ENKF_PARTICLES
                )
                s = s + cum[-1]
            return s

    else:
        raise ValueError(method)

    return jit(value_and_grad(total))


# ---------------------------------------------------------------------------
# Select / load birds
# ---------------------------------------------------------------------------
if _args.bands.strip():
    BANDS = [b.strip() for b in _args.bands.split(",") if b.strip()]
else:
    BANDS = select_bands(
        n_birds=_args.n_birds, min_n=_args.min_n, species=_args.species
    )

packs = load_packs(BANDS, _args.n_obs)
all_fns = [make_track_logliks(p) for p in packs]
stochastic = not _args.skip_stochastic
methods = ["wsckf", "ekf"] + (["enkf"] if stochastic else [])
method_labels = {"wsckf": "WSC-KF", "ekf": "EKF", "enkf": "EnKF"}
method_colors = {"wsckf": "#2ca02c", "ekf": "#d62728", "enkf": "#1f77b4"}

print(
    f"birds ({len(packs)}): "
    + ", ".join(f"{p.band}[N={p.n_obs}]" for p in packs)
)
print(
    f"WSC-KF paths={_args.num_paths}  EnKF fit={ENKF_PARTICLES}p  "
    f"ref={REF_ENKF_PARTICLES}p  mle_iters={_args.mle_iters}  solver={SOLVER}"
)

# ---------------------------------------------------------------------------
# Pooled fit on all birds
# ---------------------------------------------------------------------------
print("\n=== Pooled MLE on all birds ===")
pooled = {}
enkf_key = jr.PRNGKey(_args.seed + 17)
for method in methods:
    key = enkf_key if method == "enkf" else None
    vg = make_pooled_vg(all_fns, method, enkf_key=key)
    # warmup
    vg(jnp.array([np.log(_args.tau_init), np.log(_args.sigma_v_init)]))
    pooled[method] = adam_pooled(vg, method_labels[method])
    print(
        f"  {method_labels[method]}: τ̂={pooled[method]['tau']:.3f}  "
        f"σ̂_v={pooled[method]['sigma_v']:.3f}  ({pooled[method]['wall_s']:.1f}s)"
    )

# ---------------------------------------------------------------------------
# LOBO
# ---------------------------------------------------------------------------
fold_rows: list[dict] = []
lobo_summary: dict[str, list[float]] = {m: [] for m in methods}

if not _args.pooled_only:
    print("\n=== Leave-one-bird-out ===")
    for hold_i, hold_pack in enumerate(packs):
        fit_fns = [f for j, f in enumerate(all_fns) if j != hold_i]
        hold_fns = all_fns[hold_i]
        print(f"\n-- hold out {hold_pack.band} (fit on {len(fit_fns)} birds) --")
        fold_theta = {}
        for method in methods:
            key = jr.PRNGKey(_args.seed + 100 * (hold_i + 1)) if method == "enkf" else None
            vg = make_pooled_vg(fit_fns, method, enkf_key=key)
            vg(jnp.array([np.log(_args.tau_init), np.log(_args.sigma_v_init)]))
            fit = adam_pooled(vg, f"{method_labels[method]}|hold={hold_pack.band}")
            fold_theta[method] = fit
            ref_mean, ref_std = ref_enkf_nll_per_obs(
                hold_fns, fit["tau"], fit["sigma_v"]
            )
            lobo_summary[method].append(ref_mean)
            fold_rows.append(
                {
                    "hold_band": hold_pack.band,
                    "method": method_labels[method],
                    "tau_v": fit["tau"],
                    "sigma_v": fit["sigma_v"],
                    "ref_nll_per_obs_mean": ref_mean,
                    "ref_nll_per_obs_std": ref_std,
                    "n_fit_birds": len(fit_fns),
                    "n_obs_held": hold_pack.n_obs,
                    "mle_wall_s": fit["wall_s"],
                }
            )
            print(
                f"  {method_labels[method]}: τ̂={fit['tau']:.3f} σ̂_v={fit['sigma_v']:.3f}  "
                f"ref-NLL/obs={ref_mean:.4f}±{ref_std:.4f}"
            )

# ---------------------------------------------------------------------------
# Plots: trajectories (top) | pooled θ̂ + LOBO bars (bottom)
# ---------------------------------------------------------------------------
n_birds_plot = len(packs)
fig = plt.figure(figsize=(12.0, 7.0))
gs = fig.add_gridspec(2, n_birds_plot, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.32)

track_colors = plt.cm.tab10(np.linspace(0, 0.85, max(n_birds_plot, 1)))
for i, pack in enumerate(packs):
    ax_t = fig.add_subplot(gs[0, i])
    xy = np.asarray(pack.ys)
    ax_t.plot(xy[:, 0], xy[:, 1], "-", color=track_colors[i], lw=0.95, alpha=0.95)
    ax_t.plot(
        xy[0, 0], xy[0, 1], "o", color=track_colors[i], ms=5.5,
        markeredgecolor="k", markeredgewidth=0.45, label="start",
    )
    ax_t.plot(
        xy[-1, 0], xy[-1, 1], "s", color="0.2", ms=4,
        markeredgecolor="k", markeredgewidth=0.35, label="end",
    )
    ax_t.set_aspect("equal", adjustable="datalim")
    ax_t.set_title(f"{pack.band}  (N={pack.n_obs})", fontsize=9)
    ax_t.tick_params(labelsize=7)
    ax_t.grid(alpha=0.25)
    if i == 0:
        ax_t.set_ylabel("north (km, centred)", fontsize=8)
    ax_t.set_xlabel("east (km, centred)", fontsize=8)

# Bottom panels span half the bird columns each (or full width if pooled-only).
if _args.pooled_only:
    ax = fig.add_subplot(gs[1, :])
    ax_b = None
else:
    mid = max(1, n_birds_plot // 2)
    ax = fig.add_subplot(gs[1, 0:mid])
    ax_b = fig.add_subplot(gs[1, mid:n_birds_plot])

for method in methods:
    rec = pooled[method]
    ax.plot(
        rec["tau_trace"], rec["sv_trace"], "-", color=method_colors[method],
        lw=1.0, alpha=0.5,
    )
    ax.plot(
        rec["tau"], rec["sigma_v"], "o", color=method_colors[method],
        ms=10, markeredgecolor="k", markeredgewidth=0.7,
        label=rf"{method_labels[method]} pooled $(\hat\tau_v,\hat\sigma_v)$",
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\tau_v$ [h]")
ax.set_ylabel(r"$\sigma_v$ [km/h]")
ax.set_title("A  Pooled train MLE (all birds)")
ax.legend(fontsize=7.5, framealpha=0.92)
ax.grid(True, which="both", alpha=0.25)

if ax_b is not None:
    names = [method_labels[m] for m in methods]
    means = [float(np.mean(lobo_summary[m])) for m in methods]
    stds = [
        float(np.std(lobo_summary[m], ddof=1)) if len(lobo_summary[m]) > 1 else 0.0
        for m in methods
    ]
    cols = [method_colors[m] for m in methods]
    xpos = np.arange(len(names))
    ax_b.bar(
        xpos, means, yerr=stds if any(s > 0 for s in stds) else None,
        color=cols, edgecolor="0.2", lw=0.7, width=0.65,
        capsize=4, ecolor="0.2",
    )
    pad = 0.12
    finite = [m for m in means if np.isfinite(m)]
    if finite:
        smax = max(stds) if stds else 0.0
        ax_b.set_ylim(min(finite) - smax - pad, max(finite) + smax + 1.5 * pad)
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels(names)
    ax_b.set_ylabel(
        f"LOBO held-bird NLL / obs\n(ref EnKF-{REF_ENKF_PARTICLES}p)"
    )
    ax_b.set_title("B  Leave-one-bird-out (lower better)")
    ax_b.grid(axis="y", alpha=0.3)
    best = min(finite) if finite else float("nan")
    for x, m, s in zip(xpos, means, stds):
        if not np.isfinite(m):
            ax_b.text(x, ax_b.get_ylim()[1] * 0.9, "NaN", ha="center", fontsize=9)
            continue
        dlt = m - best
        lab = f"{m:.3f}" if dlt < 1e-9 else f"{m:.3f}\n(+{dlt:.3f})"
        ax_b.text(x, m + s + 0.02 * pad, lab, ha="center", va="bottom", fontsize=8)

fig.suptitle(
    f"{_args.species} pooled MLE + LOBO  "
    f"(n_birds={len(packs)}, n_obs≤{_args.n_obs}, WSC-KF-P{_args.num_paths})",
    fontsize=11,
    y=0.995,
)
fig.savefig(
    os.path.join(OUTPUT_DIR, "movement_lobo_compare.png"),
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "movement_lobo_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["section", "key", "value"])
    w.writerow(["setup", "species", _args.species])
    w.writerow(["setup", "bands", ",".join(p.band for p in packs)])
    w.writerow(["setup", "n_birds", len(packs)])
    w.writerow(["setup", "n_obs", _args.n_obs])
    w.writerow(["setup", "mle_iters", _args.mle_iters])
    w.writerow(["setup", "wsckf_num_paths", _args.num_paths])
    w.writerow(["setup", "enkf_particles", ENKF_PARTICLES])
    w.writerow(["setup", "ref_enkf_particles", REF_ENKF_PARTICLES])
    w.writerow(["setup", "protocol", "pooled_mle_plus_lobo"])
    for method in methods:
        w.writerow(["pooled_mle", f"{method}_tau_v", pooled[method]["tau"]])
        w.writerow(["pooled_mle", f"{method}_sigma_v", pooled[method]["sigma_v"]])
        w.writerow(["pooled_mle", f"{method}_wall_s", pooled[method]["wall_s"]])
    if not _args.pooled_only:
        for method in methods:
            xs = lobo_summary[method]
            w.writerow(
                ["lobo", f"{method}_ref_nll_mean", float(np.mean(xs))]
            )
            w.writerow(
                [
                    "lobo",
                    f"{method}_ref_nll_std",
                    float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
                ]
            )

with open(os.path.join(OUTPUT_DIR, "movement_lobo_folds.csv"), "w", newline="") as f:
    fields = [
        "hold_band", "method", "tau_v", "sigma_v",
        "ref_nll_per_obs_mean", "ref_nll_per_obs_std",
        "n_fit_birds", "n_obs_held", "mle_wall_s",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in fold_rows:
        w.writerow(row)

print(
    "\nSaved: movement_lobo_compare.png, movement_lobo_summary.csv, "
    "movement_lobo_folds.csv"
)
if not _args.pooled_only:
    print("LOBO mean ref-NLL/obs:")
    for method in methods:
        xs = lobo_summary[method]
        print(
            f"  {method_labels[method]:4s}: {np.mean(xs):.4f} "
            f"± {(np.std(xs, ddof=1) if len(xs) > 1 else 0.0):.4f}"
        )
