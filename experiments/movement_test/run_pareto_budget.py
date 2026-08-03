"""Fixed-θ* cost–accuracy Pareto: cubature paths vs EnKF particles (duck GPS).

Protocol
--------
* Shared parameters ``(τ_v, σ_v)`` fixed at the CRF/EnKF consensus MLE from
  ``run_movement_mle.py`` (defaults ``τ_v=0.75``, ``σ_v=2.6``). No per-budget
  refit — isolates SDE-propagator efficiency.
* Filter the full track; score **held-out NLL / obs** on the suffix after a
  train warmup prefix (default 70% / 30%).
* Fair wall-clock: CRF product Lyons paths vs **RK4** EnKF (match integrators).
* EnKF averaged over RNG seeds (stochastic likelihood); CRF/EKF are deterministic.

Primary plot: wall-clock (ms) vs held-out NLL/obs.

Run:
  uv run --extra experiments python experiments/movement_test/run_pareto_budget.py
  uv run --extra experiments python experiments/movement_test/run_pareto_budget.py \\
      --n-obs 600 --holdout-frac 0.3 --enkf 25,50,100,200,500 \\
      --crf-max-paths 81 --n-timing 3

Outputs under ``experiments/movement_test/output/``:
  ``pareto_budget_vs_nll.png``
  ``pareto_budget_vs_nll.csv`` / ``pareto_budget_vs_nll_raw.csv``
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass

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
from jax import jit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from cuthbert import filter as run_filter
from crf import build_filter
from crf.library import product_path_count
from cd_dynamax_mirror.filters import run_cd_ekf, run_cd_enkf

from movement_data import load_duck_track
from movement_model import (
    MovementParams,
    init_estimate,
    make_movement_dynamics,
    tau_sigma_to_beta_sigma0,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


@dataclass(frozen=True)
class CrfConfig:
    family: str
    label: str
    n_paths: int
    k: int = 1
    lyons_degree: int = 5
    method: str = "product"  # product | recombined


def default_product_ladder(noise_dim: int = 2, max_paths: int = 81) -> list[CrfConfig]:
    specs = [(3, 1), (5, 1), (3, 2), (3, 3), (5, 2), (3, 4), (5, 3)]
    configs: list[CrfConfig] = []
    seen: set[int] = set()
    for m, k in specs:
        n = product_path_count(noise_dim, k, lyons_degree=m)
        if n > max_paths or n in seen:
            continue
        seen.add(n)
        configs.append(
            CrfConfig(
                family="CRF-product",
                label=f"prod-m{m}-k{k}",
                n_paths=n,
                k=k,
                lyons_degree=m,
                method="product",
            )
        )
    return sorted(configs, key=lambda c: c.n_paths)


def recombined_configs(budgets: list[int], k: int = 3) -> list[CrfConfig]:
    return [
        CrfConfig(
            family="CRF-recombined",
            label=f"rec-P{n}",
            n_paths=n,
            k=k,
            lyons_degree=5,
            method="recombined",
        )
        for n in budgets
    ]


_parser = argparse.ArgumentParser(
    description="Movement GPS Pareto: cubature paths vs EnKF particles (held-out NLL)"
)
_parser.add_argument("--band", type=str, default="MALL 825 M")
_parser.add_argument("--n-obs", type=int, default=600, dest="n_obs")
_parser.add_argument(
    "--holdout-frac",
    type=float,
    default=0.3,
    dest="holdout_frac",
    help="Fraction of the track reserved for held-out NLL (suffix)",
)
_parser.add_argument(
    "--tau-v",
    type=float,
    default=0.75,
    dest="tau_v",
    help="Fixed τ_v (h); default ≈ CRF/EnKF consensus MLE",
)
_parser.add_argument(
    "--sigma-v",
    type=float,
    default=2.6,
    dest="sigma_v",
    help="Fixed σ_v (km/h); default ≈ CRF/EnKF consensus MLE",
)
_parser.add_argument("--dt0", type=float, default=0.1, help="inner RK4 step (hours)")
_parser.add_argument(
    "--crf-max-paths",
    type=int,
    default=81,
    dest="crf_max_paths",
    help="Include product (m,k) configs with exact path count ≤ this",
)
_parser.add_argument(
    "--recombined",
    type=str,
    default="35,84",
    help="Comma-separated recombined path budgets (empty to skip; "
    "avoid tiny budgets — recombination can collapse, e.g. P16)",
)
_parser.add_argument(
    "--enkf",
    type=str,
    default="25,50,100,200,500",
    help="Comma-separated EnKF particle counts",
)
_parser.add_argument(
    "--enkf-seeds",
    type=str,
    default="0,1,2",
    dest="enkf_seeds",
    help="RNG seeds for EnKF likelihood averaging",
)
_parser.add_argument("--n-timing", type=int, default=3, dest="n_timing")
_parser.add_argument(
    "--solver",
    type=str,
    default="rk4",
    choices=("rk4", "diffrax"),
    help="CD-filter forecast integrator (fair baseline: rk4)",
)
_parser.add_argument(
    "--skip-ekf",
    action="store_true",
    help="Omit CD-EKF reference point",
)
_parser.add_argument(
    "--skip-recombined",
    action="store_true",
    help="Omit CRF-recombined ladder",
)
_args = _parser.parse_args()

ENKF_PARTICLE_LIST = _parse_ints(_args.enkf)
ENKF_SEEDS = _parse_ints(_args.enkf_seeds)
N_TIMING = max(1, int(_args.n_timing))
SOLVER = _args.solver
DT = 1.0
DT0 = float(_args.dt0)
TAU_V = float(_args.tau_v)
SIGMA_V = float(_args.sigma_v)

CRF_CONFIGS = default_product_ladder(2, max_paths=_args.crf_max_paths)
if not _args.skip_recombined and _args.recombined.strip():
    CRF_CONFIGS = sorted(
        CRF_CONFIGS + recombined_configs(_parse_ints(_args.recombined), k=3),
        key=lambda c: (c.family, c.n_paths),
    )

# ---------------------------------------------------------------------------
# Data + fixed dynamics at θ*
# ---------------------------------------------------------------------------
track = load_duck_track(band=_args.band, max_obs=_args.n_obs)
YS = jnp.asarray(track.xy_km)
N_OBS = int(YS.shape[0])
if not (0.05 <= _args.holdout_frac <= 0.5):
    raise SystemExit("--holdout-frac must be in [0.05, 0.5]")
N_HELD = max(1, int(round(N_OBS * _args.holdout_frac)))
N_TRAIN = N_OBS - N_HELD
if N_TRAIN < 10:
    raise SystemExit(f"train split too short: n_train={N_TRAIN}, n_obs={N_OBS}")

P = MovementParams(center=(0.0, 0.0))
m0_np, P0_np = init_estimate(np.asarray(YS), P)
M0 = jnp.asarray(m0_np)
P0 = jnp.asarray(P0_np)
CHOL_P0 = jnp.linalg.cholesky(P0)

beta_v, sigma0 = tau_sigma_to_beta_sigma0(TAU_V, SIGMA_V)
DRIFT, L_FN, QC, OBS_FN, R = make_movement_dynamics(P, beta_v=beta_v, sigma0=sigma0)
CHOL_R = jnp.linalg.cholesky(R)

print(
    f"track={track.band}  N={N_OBS}  train={N_TRAIN} held={N_HELD}  "
    f"θ*=(τ_v={TAU_V:g}, σ_v={SIGMA_V:g})  solver={SOLVER}"
)
print(
    f"CRF configs={[c.label + f'({c.n_paths})' for c in CRF_CONFIGS]}  "
    f"EnKF particles={ENKF_PARTICLE_LIST}  enkf_seeds={ENKF_SEEDS}"
)


def heldout_nll_per_obs(cum_ll, n_train: int, n_obs: int) -> float:
    """Incremental NLL on obs[n_train:] from a cumulative loglik array.

    CRF ``log_normalizing_constant`` has length ``n_obs+1`` (index 0 = prior).
    CD filters ``cum_loglik`` has length ``n_obs`` (index ``i`` = after obs ``i``).
    """
    ll = np.asarray(cum_ll, dtype=np.float64)
    if ll.shape[0] == n_obs + 1:
        held = float(ll[-1] - ll[n_train])
    elif ll.shape[0] == n_obs:
        base = float(ll[n_train - 1]) if n_train > 0 else 0.0
        held = float(ll[-1] - base)
    else:
        raise ValueError(f"unexpected cum_ll length {ll.shape[0]} for n_obs={n_obs}")
    return -held / float(n_obs - n_train)


def time_call(fn, *args, n_timing: int = N_TIMING):
    """JIT warmup + mean wall time over ``n_timing`` repeats."""
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(n_timing):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return out, float(np.mean(times))


raw_rows: list[dict] = []


def score(
    family: str,
    label: str,
    budget: int,
    seed: int,
    cum_ll,
    wall_s: float,
):
    nll_h = heldout_nll_per_obs(cum_ll, N_TRAIN, N_OBS)
    ll = np.asarray(cum_ll, dtype=np.float64)
    nll_full = -float(ll[-1]) / float(N_OBS)
    rec = {
        "seed": seed,
        "family": family,
        "label": label,
        "budget": budget,
        "nll_held_per_obs": nll_h,
        "nll_full_per_obs": nll_full,
        "time_s": wall_s,
    }
    raw_rows.append(rec)
    print(
        f"  seed={seed}  {family:16s} {label:14s} n={budget:4d}  "
        f"NLL_held/obs={nll_h:8.3f}  time={1e3 * wall_s:7.1f} ms"
    )


def build_crf(cfg: CrfConfig):
    model_inputs = jnp.arange(N_OBS + 1)

    def get_init_params(_mi):
        return M0, CHOL_P0

    def get_dynamics(_mi):
        return DRIFT, L_FN

    def get_observations(mi):
        return OBS_FN, CHOL_R, YS[mi - 1]

    return (
        build_filter(
            get_init_params=get_init_params,
            get_dynamics=get_dynamics,
            get_observations=get_observations,
            noise_dim=2,
            dt=DT,
            integration_dt=DT0,
            cubature_order=cfg.k,
            lyons_degree=cfg.lyons_degree,
            num_paths=cfg.n_paths if cfg.method == "recombined" else None,
            multidim_paths=cfg.method,  # type: ignore[arg-type]
            sde_form="ito",
            checkpoint_steps=False,
        ),
        model_inputs,
    )


# ---------------------------------------------------------------------------
# Run ladder
# ---------------------------------------------------------------------------
_run_crf = jit(run_filter, static_argnames=["filter_obj"])

for cfg in CRF_CONFIGS:
    if cfg.method == "product":
        n_expect = product_path_count(2, cfg.k, lyons_degree=cfg.lyons_degree)
        if n_expect != cfg.n_paths:
            raise RuntimeError(f"config mismatch: {cfg} vs product count {n_expect}")
    filt, mi = build_crf(cfg)
    states, wall = time_call(lambda f=filt, m=mi: _run_crf(f, m))
    score(
        cfg.family,
        cfg.label,
        cfg.n_paths,
        seed=0,
        cum_ll=states.log_normalizing_constant,
        wall_s=wall,
    )

if not _args.skip_ekf:
    ekf_jit = jit(
        lambda: run_cd_ekf(
            DRIFT,
            OBS_FN,
            L_FN,
            QC,
            R,
            M0,
            P0,
            YS,
            DT,
            DT0,
            state_order="first",
            solver=SOLVER,
        )
    )
    ekf_res, wall = time_call(ekf_jit)
    score("CD-EKF", "EKF", 0, seed=0, cum_ll=ekf_res.cum_loglik, wall_s=wall)

for n_part in ENKF_PARTICLE_LIST:
    for seed in ENKF_SEEDS:
        key = jr.key(seed + 17 * n_part)

        def _enkf(k=key, n=n_part):
            return run_cd_enkf(
                DRIFT,
                OBS_FN,
                L_FN,
                QC,
                R,
                M0,
                P0,
                YS,
                DT,
                k,
                n_particles=n,
                noise_dim=2,
                dt0=DT0,
                solver=SOLVER,
            )

        enkf_jit = jit(_enkf)
        enkf_res, wall = time_call(enkf_jit)
        score(
            "CD-EnKF",
            f"EnKF-{n_part}",
            n_part,
            seed=seed,
            cum_ll=enkf_res.cum_loglik,
            wall_s=wall,
        )

# ---------------------------------------------------------------------------
# Aggregate + write CSV
# ---------------------------------------------------------------------------
RAW_PATH = os.path.join(OUTPUT_DIR, "pareto_budget_vs_nll_raw.csv")
CSV_PATH = os.path.join(OUTPUT_DIR, "pareto_budget_vs_nll.csv")

with open(RAW_PATH, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "band",
            "n_obs",
            "n_train",
            "n_held",
            "tau_v",
            "sigma_v",
            "solver",
            "seed",
            "family",
            "label",
            "budget",
            "nll_held_per_obs",
            "nll_full_per_obs",
            "time_s",
        ],
    )
    w.writeheader()
    for r in raw_rows:
        w.writerow(
            {
                "band": track.band,
                "n_obs": N_OBS,
                "n_train": N_TRAIN,
                "n_held": N_HELD,
                "tau_v": TAU_V,
                "sigma_v": SIGMA_V,
                "solver": SOLVER,
                **r,
            }
        )


def _mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    xs = np.asarray([r[key] for r in rows], dtype=np.float64)
    if xs.size == 0:
        return float("nan"), float("nan")
    if xs.size == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1))


summary: list[dict] = []
keys = sorted({(r["family"], r["label"], int(r["budget"])) for r in raw_rows})
for fam, lab, bud in keys:
    group = [r for r in raw_rows if r["family"] == fam and r["label"] == lab and int(r["budget"]) == bud]
    nll_m, nll_s = _mean_std(group, "nll_held_per_obs")
    nll_f_m, nll_f_s = _mean_std(group, "nll_full_per_obs")
    t_m, t_s = _mean_std(group, "time_s")
    summary.append(
        {
            "family": fam,
            "label": lab,
            "budget": bud,
            "n_seeds": len(group),
            "nll_held_per_obs_mean": nll_m,
            "nll_held_per_obs_std": nll_s,
            "nll_full_per_obs_mean": nll_f_m,
            "nll_full_per_obs_std": nll_f_s,
            "time_s_mean": t_m,
            "time_s_std": t_s,
        }
    )

with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "band",
            "n_obs",
            "n_train",
            "n_held",
            "tau_v",
            "sigma_v",
            "solver",
            "family",
            "label",
            "budget",
            "n_seeds",
            "nll_held_per_obs_mean",
            "nll_held_per_obs_std",
            "nll_full_per_obs_mean",
            "nll_full_per_obs_std",
            "time_s_mean",
            "time_s_std",
        ],
    )
    w.writeheader()
    for rec in sorted(summary, key=lambda r: (r["family"], r["budget"], r["label"])):
        w.writerow(
            {
                "band": track.band,
                "n_obs": N_OBS,
                "n_train": N_TRAIN,
                "n_held": N_HELD,
                "tau_v": TAU_V,
                "sigma_v": SIGMA_V,
                "solver": SOLVER,
                **rec,
            }
        )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = {
    "CRF-product": "#2ca02c",
    "CRF-recombined": "#98df8a",
    "CD-EnKF": "#1f77b4",
    "CD-EKF": "#d62728",
}
MARKERS = {
    "CRF-product": "*",
    "CRF-recombined": "P",
    "CD-EnKF": "o",
    "CD-EKF": "s",
}
DISPLAY = {
    "CRF-product": "CRF (product)",
    "CRF-recombined": "CRF (recombined)",
    "CD-EnKF": "CD-EnKF",
    "CD-EKF": "CD-EKF",
}
PLOT_ORDER = ["CRF-product", "CRF-recombined", "CD-EnKF", "CD-EKF"]
BAND_SCALE = 0.5


def _annotate_budgets(ax, fam: str, pts: list[dict], xs, ys):
    if fam == "CD-EKF":
        return
    color = COLORS.get(fam, "#333")
    for i, (pt, x, y) in enumerate(zip(pts, xs, ys)):
        dy = 7 if i % 2 == 0 else -9
        ax.annotate(
            str(int(pt["budget"])),
            (x, y),
            textcoords="offset points",
            xytext=(0, dy),
            fontsize=7,
            color=color,
            ha="center",
            va="bottom" if dy > 0 else "top",
            alpha=0.9,
        )


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
metrics = [
    ("nll_held_per_obs", "held-out NLL / observation", axes[0]),
    ("nll_full_per_obs", "full-track NLL / observation", axes[1]),
]
# Drop exploded recombinations (moment collapse) from the figure only.
NLL_PLOT_CAP = 50.0

for metric, ylabel, ax in metrics:
    for fam in PLOT_ORDER:
        pts = [
            r
            for r in summary
            if r["family"] == fam and r[f"{metric}_mean"] < NLL_PLOT_CAP
        ]
        if not pts:
            continue
        pts = sorted(pts, key=lambda r: r["budget"])
        xs = np.asarray([1e3 * r["time_s_mean"] for r in pts])
        ys = np.asarray([r[f"{metric}_mean"] for r in pts])
        ystd = np.asarray([r[f"{metric}_std"] for r in pts]) * BAND_SCALE
        color = COLORS[fam]
        marker = MARKERS[fam]
        if len(pts) == 1:
            x0 = float(xs[0])
            ax.fill_between(
                [x0 / 1.05, x0 * 1.05],
                [ys[0] - ystd[0]] * 2,
                [ys[0] + ystd[0]] * 2,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
            ax.plot(
                xs,
                ys,
                marker,
                color=color,
                ms=11 if marker == "*" else 8,
                label=DISPLAY[fam],
                zorder=3,
            )
        else:
            ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=0.18, linewidth=0)
            ax.plot(
                xs,
                ys,
                "-" + marker,
                color=color,
                ms=8 if marker != "*" else 11,
                lw=1.4,
                label=DISPLAY[fam],
                zorder=3,
            )
        _annotate_budgets(ax, fam, pts, xs, ys)

    ax.set_xscale("log")
    ax.set_xlabel("wall-clock per filter pass (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)

axes[0].legend(frameon=False, fontsize=8)
fig.suptitle(
    f"{track.band}: paths vs particles @ "
    f"$(\\tau_v,\\sigma_v)=({TAU_V:g},{SIGMA_V:g})$, "
    f"held={N_HELD}/{N_OBS}, {SOLVER}",
    fontsize=11,
)
fig.tight_layout()
PLOT_PATH = os.path.join(OUTPUT_DIR, "pareto_budget_vs_nll.png")
fig.savefig(PLOT_PATH, dpi=150)
plt.close(fig)

print(f"\nWrote {RAW_PATH}")
print(f"Wrote {CSV_PATH}")
print(f"Wrote {PLOT_PATH}")
