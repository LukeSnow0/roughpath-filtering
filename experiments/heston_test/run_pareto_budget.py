"""Heston Pareto: wall-clock vs fine-grid RMSE.

Simulated twin with noisy log-price observations and latent variance.
Mirrors ``l63_test/run_pareto_budget_smoke.py`` with Brownian dim d=2.

Run:
  uv run --extra experiments python experiments/heston_test/run_pareto_budget.py
  uv run --extra experiments python experiments/heston_test/run_pareto_budget.py \\
      --seeds 0,1,2,3,4,5,6,7,8 --wsckf-max-paths 64

Outputs:
  ``experiments/heston_test/output/pareto_budget_vs_rmse.png``
  ``experiments/heston_test/output/pareto_budget_vs_rmse.csv``
  ``experiments/heston_test/output/pareto_budget_vs_rmse_raw.csv``
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import jit, random

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)
from cuthbert import filter as run_filter
from wsckf import build_filter
from wsckf.library import product_path_count
from heston_model import (
    HestonParams,
    init_estimate,
    make_heston_dynamics,
    rmse,
    rmse_components,
    simulate_heston,
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
class WSCKFConfig:
    label: str
    lyons_degree: int
    k: int
    n_paths: int
    family: str = "WSC-KF-product"


def default_wsckf_configs(noise_dim: int = 2, max_paths: int = 16) -> list[WSCKFConfig]:
    """Exact product path counts up to ``max_paths`` (sorted by n_paths).

    For d=2: (m,k) → P = (2^k)^2 for m=3, (3^k)^2 for m=5, so
    4, 9, 16, 64, 81, 256, ...
    """
    specs = [(3, 1), (5, 1), (3, 2), (3, 3), (5, 2), (3, 4), (5, 3)]
    configs: list[WSCKFConfig] = []
    seen: set[int] = set()
    for m, k in specs:
        n = product_path_count(noise_dim, k, lyons_degree=m)
        if n > max_paths or n in seen:
            continue
        seen.add(n)
        configs.append(
            WSCKFConfig(
                label=f"prod-m{m}-k{k}",
                lyons_degree=m,
                k=k,
                n_paths=n,
            )
        )
    return sorted(configs, key=lambda c: c.n_paths)


_parser = argparse.ArgumentParser(description="Heston Pareto: budget vs RMSE")
_parser.add_argument(
    "--seeds",
    type=str,
    default="0,1,2,3,4,5,6,7,8",
    help="Comma-separated truth/filter seeds (default: 9 seeds)",
)
_parser.add_argument(
    "--delta",
    type=float,
    default=0.05,
    help="Observation interval (years); default 0.05",
)
_parser.add_argument("--horizon", type=float, default=2.0)
_parser.add_argument("--eval-dt", type=float, default=0.005, dest="eval_dt")
_parser.add_argument("--enkf", type=str, default="25,50,100")
_parser.add_argument("--pf", type=str, default="50,100,200")
_parser.add_argument(
    "--wsckf-max-paths",
    type=int,
    default=16,
    dest="wsckf_max_paths",
    help="Include exact product WSC-KF configs with P ≤ this (d=2: 4,9,16,64,…)",
)
_parser.add_argument(
    "--obs-std",
    type=float,
    default=0.02,
    dest="obs_std",
    help="Observation-noise std on log-price; default 0.02",
)
_parser.add_argument(
    "--xi",
    type=float,
    default=0.3,
    help="Vol-of-vol ξ; default 0.3",
)
_parser.add_argument(
    "--diff-scale",
    type=float,
    default=1.0,
    dest="diff_scale",
    help="Multiply diffusion L by this factor (process-noise power); default 1.0",
)
_parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Optional output filename tag, e.g. 'xi05' → pareto_budget_vs_rmse_xi05.png",
)
_parser.add_argument(
    "--rmse-comp",
    type=str,
    default="all",
    dest="rmse_comp",
    choices=("all", "X", "v"),
    help="RMSE over all state comps (default), log-price X only, or variance v only",
)
_parser.add_argument("--n-timing", type=int, default=2, dest="n_timing")
_parser.add_argument(
    "--append",
    action="store_true",
    help="Merge into existing raw CSV; skip (seed,family,label,budget) already present",
)
_args = _parser.parse_args()

SEEDS = _parse_ints(_args.seeds)
DELTA_T = float(_args.delta)
HORIZON = float(_args.horizon)
EVAL_DT = float(_args.eval_dt)
ENKF_LIST = _parse_ints(_args.enkf)
PF_LIST = _parse_ints(_args.pf)
N_TIMING = _args.n_timing
OBS_STD = float(_args.obs_std)
XI = float(_args.xi)
DIFF_SCALE = float(_args.diff_scale)
OUT_TAG = f"_{_args.tag}" if _args.tag.strip() else ""
RMSE_COMP = str(_args.rmse_comp)
WSCKF_CONFIGS = default_wsckf_configs(2, max_paths=int(_args.wsckf_max_paths))
if not WSCKF_CONFIGS:
    raise SystemExit(f"no WSC-KF configs from --wsckf-max-paths={_args.wsckf_max_paths}")

PARAMS = HestonParams(meas_std=OBS_STD, xi=XI, diff_scale=DIFF_SCALE)
f_drift, L_fn, Qc_fn, h_obs, R = make_heston_dynamics(PARAMS)


def _score_rmse(est: np.ndarray, truth: np.ndarray) -> float:
    if RMSE_COMP == "all":
        return rmse(est, truth)
    idx = 0 if RMSE_COMP == "X" else 1
    return float(rmse_components(est, truth)[idx])


chol_R = jnp.linalg.cholesky(R)

x_dim, y_dim, noise_dim = 2, 1, 2
DT_INNER = EVAL_DT
CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)

COLORS = {
    "CD-EKF": "#2E86AB",
    "CD-EKF-fine": "#5DADE2",
    "CD-UKF": "#B8860B",
    "CD-UKF-fine": "#E8B923",
    "CD-CKF": "#3DA35D",
    "CD-EnKF": "#F6AE2D",
    "PF": "#F24236",
    "WSC-KF-product": "#6A4C93",
    "WSC-KF-m3": "#6A4C93",
    "WSC-KF-m5": "#C77DFF",
}
DISPLAY_NAME = {
    "WSC-KF-product": "WSC-KF",
    "WSC-KF-m3": "WSC-KF m=3",
    "WSC-KF-m5": "WSC-KF m=5",
    "CD-EKF": "CD-EKF (coarse)",
    "CD-EKF-fine": "CD-EKF (fine)",
    "CD-UKF": "CD-UKF (coarse)",
    "CD-UKF-fine": "CD-UKF (fine)",
    "CD-CKF": "CD-CKF",
    "CD-EnKF": "CD-EnKF",
    "PF": "PF",
}
_LABEL_OFFSET = {
    "CD-EnKF": (0, -10),
    "PF": (0, 7),
    "WSC-KF-product": (0, 8),
    "WSC-KF-m3": (0, 8),
    "WSC-KF-m5": (0, -9),
}


def stack_estimates(m0_arr, means):
    return np.vstack([np.asarray(m0_arr), np.asarray(means)])


def time_call(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return out, float(np.median(samples))


def _as_numpy_drift(drift_fn):
    def drift_np(t: float, x: np.ndarray) -> np.ndarray:
        return np.asarray(drift_fn(t, jnp.asarray(x)), dtype=np.float64)

    return drift_np


def rk4_step(drift_np, t, y, h):
    k1 = drift_np(t, y)
    k2 = drift_np(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = drift_np(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = drift_np(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_to(drift_np, y, t0, t1, dt0):
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


def estimates_on_fine_grid(m_filt, dt_obs, times_fine, drift_fn, dt_integrate):
    drift_np = _as_numpy_drift(drift_fn)
    n_fine = times_fine.shape[0]
    est = np.zeros((n_fine, m_filt.shape[1]), dtype=np.float64)
    est[0] = m_filt[0]
    n_obs = m_filt.shape[0] - 1
    for k in range(n_obs):
        t0 = k * dt_obs
        t1 = (k + 1) * dt_obs
        js = np.where((times_fine > t0 + 1e-12) & (times_fine <= t1 + 1e-9))[0]
        m = m_filt[k].copy()
        t = t0
        for j in js:
            t_target = float(times_fine[j])
            m = rk4_to(drift_np, m, t, t_target, dt_integrate)
            t = t_target
            if abs(t_target - t1) <= 1e-9:
                est[j] = m_filt[k + 1]
                m = m_filt[k + 1].copy()
            else:
                est[j] = m
    return est


if HORIZON / EVAL_DT != int(round(HORIZON / EVAL_DT)):
    raise SystemExit("horizon must be multiple of eval-dt")
N_FINE = int(round(HORIZON / EVAL_DT))
DT_FINE = HORIZON / N_FINE
stride = int(round(DELTA_T / DT_FINE))
if not np.isclose(stride * DT_FINE, DELTA_T):
    raise SystemExit("delta must tile eval-dt")
n_obs = N_FINE // stride
if n_obs * stride != N_FINE:
    raise SystemExit("delta must tile horizon")
dt = stride * DT_FINE
dt0 = DT_INNER
times_fine = np.arange(N_FINE + 1, dtype=np.float64) * DT_FINE
model_inputs = jnp.arange(n_obs + 1)

print(
    f"Heston Pareto: Δt={dt:g}, horizon={HORIZON:g}, n_obs={n_obs}, "
    f"κ={PARAMS.kappa:g}, θ={PARAMS.theta:g}, ξ={PARAMS.xi:g}, ρ={PARAMS.rho:g}, "
    f"diff_scale={PARAMS.diff_scale:g}, obs_std={OBS_STD:g}, rmse={RMSE_COMP}, "
    f"seeds={SEEDS}, "
    f"WSC-KF={[c.label + f'({c.n_paths})' for c in WSCKF_CONFIGS]}"
)

RAW_FIELDS = ["delta_t", "n_obs", "horizon", "seed", "family", "label", "budget", "rmse", "time_s"]
raw_path = os.path.join(OUTPUT_DIR, f"pareto_budget_vs_rmse{OUT_TAG}_raw.csv")
csv_path = os.path.join(OUTPUT_DIR, f"pareto_budget_vs_rmse{OUT_TAG}.csv")
plot_path = os.path.join(OUTPUT_DIR, f"pareto_budget_vs_rmse{OUT_TAG}.png")
raw_rows: list[dict] = []
done_keys: set[tuple[int, str, str, int]] = set()
if _args.append and os.path.isfile(raw_path):
    with open(raw_path, newline="") as f:
        for r in csv.DictReader(f):
            row = {
                "seed": int(r["seed"]),
                "family": r["family"],
                "label": r["label"],
                "budget": int(float(r["budget"])),
                "rmse": float(r["rmse"]),
                "time_s": float(r["time_s"]),
            }
            raw_rows.append(row)
            done_keys.add((row["seed"], row["family"], row["label"], row["budget"]))
    print(
        f"Append mode: loaded {len(raw_rows)} rows "
        f"({len({k[0] for k in done_keys})} seeds already present)"
    )


def _have(seed: int, family: str, label: str, budget: int) -> bool:
    return (seed, family, label, budget) in done_keys


seeds_to_run = [
    s
    for s in SEEDS
    if any(not _have(s, c.family, c.label, c.n_paths) for c in WSCKF_CONFIGS)
    or not _have(s, "CD-EKF", "EKF-coarse", 0)
    or not _have(s, "CD-EKF-fine", "EKF-fine", 0)
    or not _have(s, "CD-UKF", "UKF-coarse", 0)
    or not _have(s, "CD-UKF-fine", "UKF-fine", 0)
    or not _have(s, "CD-CKF", "CKF", 0)
    or any(not _have(s, "CD-EnKF", f"EnKF-{n}", n) for n in ENKF_LIST)
    or any(not _have(s, "PF", f"PF-{n}", n) for n in PF_LIST)
]
if _args.append and done_keys and not seeds_to_run:
    print("No missing (seed, config) rows; regenerating summary/plot from existing raw CSV")

for seed in seeds_to_run:
    print(f"\n=== seed {seed} ===")
    true_fine, ys_np, _times = simulate_heston(
        horizon=HORIZON,
        n_fine=N_FINE,
        p=PARAMS,
        seed=seed,
        obs_stride=stride,
    )
    assert np.allclose(_times, times_fine)
    ys = jnp.asarray(ys_np)
    m0_np, P0_np = init_estimate(PARAMS, true_state=true_fine[0], cov_scale=1.0)
    m0 = jnp.asarray(m0_np, dtype=jnp.float64)
    P0 = jnp.asarray(P0_np, dtype=jnp.float64)
    chol_P0 = jnp.linalg.cholesky(P0)

    def score(family, label, budget, m_filt, wall):
        est = estimates_on_fine_grid(m_filt, dt, times_fine, f_drift, DT_INNER)
        err = _score_rmse(est, true_fine)
        rec = {
            "seed": seed,
            "family": family,
            "label": label,
            "budget": budget,
            "rmse": err,
            "time_s": wall,
        }
        raw_rows.append(rec)
        done_keys.add((seed, family, label, budget))
        print(
            f"  {family:12s} {label:14s} n={budget:4d}  "
            f"RMSE[{RMSE_COMP}]={err:6.4f}  time={1e3 * wall:7.1f} ms"
        )

    for cfg in WSCKF_CONFIGS:
        if _have(seed, cfg.family, cfg.label, cfg.n_paths):
            continue
        n_expect = product_path_count(noise_dim, cfg.k, lyons_degree=cfg.lyons_degree)
        if n_expect != cfg.n_paths:
            raise RuntimeError(f"path count mismatch {cfg}")
        wsckf = build_filter(
            get_init_params=lambda mi: (m0, chol_P0),
            get_dynamics=lambda mi: (f_drift, L_fn),
            get_observations=lambda mi: (
                h_obs,
                chol_R,
                ys[mi - 1],
            ),
            noise_dim=noise_dim,
            dt=dt,
            integration_dt=dt0,
            cubature_order=cfg.k,
            lyons_degree=cfg.lyons_degree,
            multidim_paths="product",
            # Heston drift is Itô; WSC-KF kernels are Stratonovich (same as AZT1D/MFTS).
            sde_form="ito",
        )
        jitted = jit(run_filter, static_argnames=["filter_obj"])
        states, wall = time_call(lambda f=jitted, filt=wsckf: f(filt, model_inputs))
        score(
            cfg.family,
            cfg.label,
            cfg.n_paths,
            stack_estimates(m0, states.mean[1:]),
            wall,
        )

    if not _have(seed, "CD-EKF", "EKF-coarse", 0):
        ekf_c_jit = jit(
            lambda: run_cd_ekf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt0,
                state_order="flowmap", solver="rk4",
            )
        )
        ekf_c_res, wall = time_call(ekf_c_jit)
        score("CD-EKF", "EKF-coarse", 0, stack_estimates(m0, ekf_c_res.means), wall)

    if not _have(seed, "CD-EKF-fine", "EKF-fine", 0):
        ekf_f_jit = jit(
            lambda: run_cd_ekf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt0,
                state_order="first", solver="rk4",
            )
        )
        ekf_f_res, wall = time_call(ekf_f_jit)
        score("CD-EKF-fine", "EKF-fine", 0, stack_estimates(m0, ekf_f_res.means), wall)

    if not _have(seed, "CD-UKF", "UKF-coarse", 0):
        ukf_jit = jit(
            lambda: run_cd_ukf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt0,
                UKFParams(), state_order="flowmap", solver="rk4",
            )
        )
        ukf_res, wall = time_call(ukf_jit)
        score("CD-UKF", "UKF-coarse", 0, stack_estimates(m0, ukf_res.means), wall)

    if not _have(seed, "CD-UKF-fine", "UKF-fine", 0):
        ukf_f_jit = jit(
            lambda: run_cd_ukf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt0,
                UKFParams(), state_order="first", solver="rk4",
            )
        )
        ukf_f_res, wall = time_call(ukf_f_jit)
        score("CD-UKF-fine", "UKF-fine", 0, stack_estimates(m0, ukf_f_res.means), wall)

    if not _have(seed, "CD-CKF", "CKF", 0):
        ckf_jit = jit(
            lambda: run_cd_ukf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt0,
                CKF_PARAMS, state_order="flowmap", solver="rk4",
            )
        )
        ckf_res, wall = time_call(ckf_jit)
        score("CD-CKF", "CKF", 0, stack_estimates(m0, ckf_res.means), wall)

    # Preserve original RNG chain: always split once per listed budget, even if skipped.
    key = random.key(seed)
    for n_part in ENKF_LIST:
        key, sub = random.split(key)
        if _have(seed, "CD-EnKF", f"EnKF-{n_part}", n_part):
            continue
        enkf_jit = jit(
            lambda k, n=n_part: run_cd_enkf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, k,
                n_particles=n, noise_dim=noise_dim, dt0=dt0, solver="rk4",
            )
        )
        enkf_res, wall = time_call(enkf_jit, sub)
        score("CD-EnKF", f"EnKF-{n_part}", n_part, stack_estimates(m0, enkf_res.means), wall)

    for n_part in PF_LIST:
        key, sub = random.split(key)
        if _have(seed, "PF", f"PF-{n_part}", n_part):
            continue
        pf_jit = jit(
            lambda k, n=n_part: run_bootstrap_pf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, k,
                n_particles=n, noise_dim=noise_dim, dt0=dt0, solver="rk4",
            )
        )
        pf_res, wall = time_call(pf_jit, sub)
        score("PF", f"PF-{n_part}", n_part, stack_estimates(m0, pf_res.means), wall)

# ---- CSV ----
with open(raw_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
    w.writeheader()
    for r in sorted(raw_rows, key=lambda x: (x["seed"], x["family"], x["budget"], x["label"])):
        w.writerow({"delta_t": dt, "n_obs": n_obs, "horizon": HORIZON, **r})

keys = sorted({(r["family"], r["label"], r["budget"]) for r in raw_rows})
summary = []
for family, label, budget in keys:
    subset = [
        r for r in raw_rows
        if r["family"] == family and r["label"] == label and r["budget"] == budget
    ]

    def mean_std(field: str) -> tuple[float, float]:
        v = np.asarray([r[field] for r in subset], dtype=np.float64)
        return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)

    rm, rms = mean_std("rmse")
    tm, tms = mean_std("time_s")
    summary.append(
        {
            "family": family,
            "label": label,
            "budget": budget,
            "n_seeds": len(subset),
            "rmse_mean": rm,
            "rmse_std": rms,
            "time_s_mean": tm,
            "time_s_std": tms,
        }
    )

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "delta_t",
            "n_obs",
            "horizon",
            "family",
            "label",
            "budget",
            "n_seeds",
            "rmse_mean",
            "rmse_std",
            "time_s_mean",
            "time_s_std",
        ],
    )
    w.writeheader()
    for r in sorted(summary, key=lambda x: (x["family"], x["budget"], x["label"])):
        w.writerow({"delta_t": dt, "n_obs": n_obs, "horizon": HORIZON, **r})

# ---- plot ----
BAND_SCALE = 0.1


def _annotate_nodes(ax, fam, pts, xs, ys):
    if fam in ("CD-EKF", "CD-EKF-fine", "CD-UKF", "CD-UKF-fine", "CD-CKF"):
        return
    color = COLORS[fam]
    base_dx, base_dy = _LABEL_OFFSET.get(fam, (0, 6))
    for i, (pt, x, y) in enumerate(zip(pts, xs, ys)):
        dy = base_dy if (i % 2 == 0) else (base_dy + (4 if base_dy > 0 else -4))
        ax.annotate(
            str(int(pt["budget"])),
            (x, y),
            textcoords="offset points",
            xytext=(base_dx, dy),
            fontsize=7,
            color=color,
            ha="center",
            va="bottom" if dy >= 0 else "top",
            alpha=0.9,
        )


fig, ax = plt.subplots(figsize=(7.5, 4.5))
# Split WSC-KF by Lyons degree (m=3 vs m=5), matching L-overlay colors.
plot_groups: list[tuple[str, str, Callable[[dict], bool]]] = [
    ("WSC-KF m=3", "WSC-KF-m3", lambda r: r["family"] == "WSC-KF-product" and "-m3-" in r["label"]),
    ("WSC-KF m=5", "WSC-KF-m5", lambda r: r["family"] == "WSC-KF-product" and "-m5-" in r["label"]),
    ("CD-EnKF", "CD-EnKF", lambda r: r["family"] == "CD-EnKF"),
    ("PF", "PF", lambda r: r["family"] == "PF"),
    ("CD-EKF (coarse)", "CD-EKF", lambda r: r["family"] == "CD-EKF"),
    ("CD-EKF (fine)", "CD-EKF-fine", lambda r: r["family"] == "CD-EKF-fine"),
    ("CD-UKF (coarse)", "CD-UKF", lambda r: r["family"] == "CD-UKF"),
    ("CD-UKF (fine)", "CD-UKF-fine", lambda r: r["family"] == "CD-UKF-fine"),
    ("CD-CKF", "CD-CKF", lambda r: r["family"] == "CD-CKF"),
]
all_xs: list[float] = []
n_seeds = len({r["seed"] for r in raw_rows})
wsckf_path_note = ",".join(str(c.n_paths) for c in WSCKF_CONFIGS)
_square = {
    "CD-EKF (coarse)",
    "CD-EKF (fine)",
    "CD-UKF (coarse)",
    "CD-UKF (fine)",
    "CD-CKF",
    "WSC-KF m=3",
    "WSC-KF m=5",
}
_markers = {"WSC-KF m=3": "s", "WSC-KF m=5": "D"}
for disp, color_key, pred in plot_groups:
    pts = [r for r in summary if pred(r)]
    if not pts:
        continue
    pts = sorted(pts, key=lambda r: r["budget"])
    xs = np.asarray([1e3 * r["time_s_mean"] for r in pts])
    ys_plot = np.asarray([r["rmse_mean"] for r in pts])
    ystd = np.asarray([r["rmse_std"] for r in pts]) * BAND_SCALE
    all_xs.extend(xs.tolist())
    color = COLORS[color_key]
    marker = _markers.get(disp, "s" if disp in _square else "o")
    if len(pts) == 1:
        x0 = float(xs[0])
        ax.fill_between(
            [x0 / 1.04, x0 * 1.04],
            [ys_plot[0] - ystd[0]] * 2,
            [ys_plot[0] + ystd[0]] * 2,
            color=color,
            alpha=0.22,
            lw=0,
        )
        ax.plot(xs, ys_plot, marker, color=color, markersize=7, label=disp)
    else:
        ax.fill_between(xs, ys_plot - ystd, ys_plot + ystd, color=color, alpha=0.22, lw=0)
        ax.plot(xs, ys_plot, f"-{marker}", color=color, lw=1.5, markersize=6, label=disp)
    _annotate_nodes(ax, color_key, pts, xs, ys_plot)

ax.set_xlabel("Wall-clock time [ms]")
_rmse_ylab = {
    "all": "Fine-grid RMSE (overall)",
    "X": r"Fine-grid RMSE ($X=\log S$ only)",
    "v": r"Fine-grid RMSE ($v$ only)",
}[RMSE_COMP]
ax.set_ylabel(_rmse_ylab)
ax.set_xscale("log")
ax.set_yscale("linear")
if all_xs:
    ax.set_xlim(min(all_xs) / 1.25, max(all_xs) * 1.35)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="best")
_rmse_note = "" if RMSE_COMP == "all" else rf"; RMSE on ${RMSE_COMP}$"
ax.set_title(
    rf"Heston Pareto at $\Delta t={dt:g}$, $\xi={PARAMS.xi:g}$, "
    rf"$L$-scale={PARAMS.diff_scale:g}, $\rho={PARAMS.rho:g}$ "
    rf"(mean±{BAND_SCALE:g}$\sigma$ over {n_seeds} seeds; "
    rf"horizon={HORIZON:g}; WSC-KF P∈{{{wsckf_path_note}}}{_rmse_note})",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print("-" * 60)
print(f"Wrote {plot_path}")
print(f"Wrote {csv_path}")
print(f"Wrote {raw_path}")
