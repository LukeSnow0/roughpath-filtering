"""Smoke Pareto for Lorenz-63: wall-clock vs fine-grid RMSE.

Mirrors the MFTS budget Pareto style: WSC-KF product path ladder (d=3),
EnKF/PF particle sweeps, multi-seed mean.

Run:
  uv run --extra experiments python experiments/l63_test/run_pareto_budget_smoke.py
  uv run --extra experiments python experiments/l63_test/run_pareto_budget_smoke.py \\
      --seeds 0,1,2,3,4,5,6,7,8 --wsckf-max-paths 64
  uv run --extra experiments python experiments/l63_test/run_pareto_budget_smoke.py \\
      --append --seeds 0,1,2,3,4,5 --wsckf-max-paths 64 \\
      --enkf 25,50,100,200 --pf 50,100,200,500

Outputs:
  ``experiments/l63_test/output/pareto_budget_vs_rmse.png``
  ``experiments/l63_test/output/pareto_budget_vs_rmse.csv``
  ``experiments/l63_test/output/pareto_budget_vs_rmse_raw.csv``
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
# MPS rejects float64; keep x64 on CPU (same pattern as MFTS/movement scripts).
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
from wsckf.library import precompute_cubature_paths, product_path_count
from mfts_test.plot_style import (
    METHOD_COLORS,
    METHOD_MARKERS,
    SINGLE_COLUMN_WIDTH,
    save_publication_figure,
    style_curve_axis,
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
    method: str = "product"  # product | recombined
    family: str = "WSC-KF-product"


def default_wsckf_configs(noise_dim: int = 3, max_paths: int = 8) -> list[WSCKFConfig]:
    """Exact product path counts up to ``max_paths`` (sorted by n_paths).

    For d=3: (m,k) → P = (2^k)^3 for m=3, (3^k)^3 for m=5, so
    8, 27, 64, 512, 729, ...
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
                method="product",
                family="WSC-KF-product",
            )
        )
    return sorted(configs, key=lambda c: c.n_paths)


def recombined_wsckf_configs(path_counts: list[int], k: int = 2) -> list[WSCKFConfig]:
    """Recombined KLV path budgets (arbitrary P; not restricted to product cardinalities)."""
    configs: list[WSCKFConfig] = []
    seen: set[int] = set()
    for n in path_counts:
        if n < 1 or n in seen:
            continue
        seen.add(n)
        configs.append(
            WSCKFConfig(
                label=f"rec-P{n}",
                lyons_degree=5,
                k=k,
                n_paths=n,
                method="recombined",
                family="WSC-KF-recombined",
            )
        )
    return sorted(configs, key=lambda c: c.n_paths)


_parser = argparse.ArgumentParser(description="L63 smoke Pareto: budget vs RMSE")
_parser.add_argument(
    "--seeds",
    type=str,
    default="0,1,2,3,4,5,6,7,8",
    help="Comma-separated truth/filter seeds (default: 9 seeds)",
)
_parser.add_argument("--delta", type=float, default=0.35)
_parser.add_argument("--horizon", type=float, default=14.0)
_parser.add_argument("--eval-dt", type=float, default=0.01, dest="eval_dt")
_parser.add_argument("--spinup", type=int, default=5_000, help="Truth spinup steps")
_parser.add_argument("--enkf", type=str, default="25,50,100")
_parser.add_argument("--pf", type=str, default="50,100,200")
_parser.add_argument(
    "--wsckf-max-paths",
    type=int,
    default=8,
    dest="wsckf_max_paths",
    help="Include exact product WSC-KF configs with P ≤ this (d=3: 8,27,64,…)",
)
_parser.add_argument(
    "--wsckf-recombined",
    type=str,
    default="",
    dest="wsckf_recombined",
    help="Comma-separated recombined path budgets (e.g. 4). Empty = none.",
)
_parser.add_argument(
    "--diff-std",
    type=float,
    default=1.0,
    dest="diff_std",
    help="Process-noise intensity (diffusion scale); default 1.0",
)
_parser.add_argument(
    "--obs-std",
    type=float,
    default=0.3,
    dest="obs_std",
    help="Observation-noise std on y=x1; default 0.3",
)
_parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Optional output filename tag, e.g. 'diff3' → pareto_budget_vs_rmse_diff3.png",
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
SPINUP_STEPS = int(_args.spinup)
ENKF_LIST = _parse_ints(_args.enkf)
PF_LIST = _parse_ints(_args.pf)
N_TIMING = _args.n_timing
DIFF_STD = float(_args.diff_std)
OBS_STD = float(_args.obs_std)
OUT_TAG = f"_{_args.tag}" if _args.tag.strip() else ""
WSCKF_CONFIGS = default_wsckf_configs(3, max_paths=int(_args.wsckf_max_paths))
WSCKF_CONFIGS = sorted(
    WSCKF_CONFIGS + recombined_wsckf_configs(_parse_ints(_args.wsckf_recombined)),
    key=lambda c: (c.n_paths, c.method, c.label),
)
if not WSCKF_CONFIGS:
    raise SystemExit(
        f"no WSC-KF configs from --wsckf-max-paths={_args.wsckf_max_paths} "
        f"and --wsckf-recombined={_args.wsckf_recombined!r}"
    )

lorenz_sigma, lorenz_rho, lorenz_beta = 10.0, 28.0, 8.0 / 3.0
DT_INNER = 0.01
diff_std, obs_std = DIFF_STD, OBS_STD
x_dim, y_dim, noise_dim = 3, 1, 3

H = jnp.array([[1.0, 0.0, 0.0]])
d_obs = jnp.zeros(y_dim)
R = (obs_std**2) * jnp.eye(y_dim)
chol_R = jnp.linalg.cholesky(R)

COLORS = {
    "CD-EKF": METHOD_COLORS["CD-EKF"],
    "CD-UKF": METHOD_COLORS["CD-UKF"],
    "CD-EnKF": METHOD_COLORS["CD-EnKF"],
    "PF": METHOD_COLORS["PF"],
    "WSC-KF-product": METHOD_COLORS["WSC-KF-product"],
    "WSC-KF-recombined": METHOD_COLORS["WSC-KF-recombined"],
}
DISPLAY_NAME = {
    "WSC-KF-product": "WSC-KF",
    "WSC-KF-recombined": "WSC-KF",
    "CD-EKF": "CD-EKF",
    "CD-UKF": "CD-UKF (coarse)",  # flow-map predict
    "CD-EnKF": "CD-EnKF",
    "PF": "PF",
}
_LABEL_OFFSET = {
    "CD-EnKF": (0, -10),
    "PF": (0, 7),
    "WSC-KF-product": (0, 8),
    "WSC-KF-recombined": (0, 8),
}


def lorenz_drift(x):
    return jnp.array(
        [
            lorenz_sigma * (x[1] - x[0]),
            x[0] * (lorenz_rho - x[2]) - x[1],
            x[0] * x[1] - lorenz_beta * x[2],
        ]
    )


def f_drift(t, x):
    return lorenz_drift(x)


def h_obs(x):
    return H @ x + d_obs


def L_fn(t, x):
    return diff_std * jnp.eye(x_dim)


def Qc_fn(t, x):
    return jnp.eye(x_dim)


def heun_step_np(x: np.ndarray, h: float) -> np.ndarray:
    k1 = np.array(
        [
            lorenz_sigma * (x[1] - x[0]),
            x[0] * (lorenz_rho - x[2]) - x[1],
            x[0] * x[1] - lorenz_beta * x[2],
        ],
        dtype=np.float64,
    )
    x_pred = x + h * k1
    k2 = np.array(
        [
            lorenz_sigma * (x_pred[1] - x_pred[0]),
            x_pred[0] * (lorenz_rho - x_pred[2]) - x_pred[1],
            x_pred[0] * x_pred[1] - lorenz_beta * x_pred[2],
        ],
        dtype=np.float64,
    )
    return x + 0.5 * h * (k1 + k2)


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
    f"L63 smoke Pareto: Δt={dt:g}, horizon={HORIZON:g}, n_obs={n_obs}, "
    f"diff_std={diff_std:g}, obs_std={obs_std:g}, "
    f"seeds={SEEDS}, WSC-KF={[c.label + f'({c.n_paths})' for c in WSCKF_CONFIGS]}"
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
    or not _have(s, "CD-EKF", "EKF", 0)
    or not _have(s, "CD-UKF", "UKF-coarse", 0)
    or any(not _have(s, "CD-EnKF", f"EnKF-{n}", n) for n in ENKF_LIST)
    or any(not _have(s, "PF", f"PF-{n}", n) for n in PF_LIST)
]
if _args.append and done_keys and not seeds_to_run:
    print("No missing (seed, config) rows; regenerating summary/plot from existing raw CSV")

for seed in seeds_to_run:
    print(f"\n=== seed {seed} ===")
    rng = np.random.default_rng(seed)
    x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    chol_scale = diff_std * np.sqrt(DT_FINE)
    for _ in range(SPINUP_STEPS):
        x = heun_step_np(x, DT_FINE)
        x = x + chol_scale * rng.standard_normal(x_dim)

    m0_np = x.copy()
    true_fine = [m0_np.copy()]
    ys_fine_list = []
    for _ in range(N_FINE):
        x = heun_step_np(x, DT_FINE)
        x = x + chol_scale * rng.standard_normal(x_dim)
        y = np.array([x[0]], dtype=np.float64) + obs_std * rng.standard_normal(y_dim)
        true_fine.append(x.copy())
        ys_fine_list.append(y)

    true_fine = np.asarray(true_fine, dtype=np.float64)
    ys_fine = jnp.asarray(np.stack(ys_fine_list))
    ys = ys_fine[stride - 1 :: stride]
    m0 = jnp.asarray(m0_np, dtype=jnp.float64)
    P0 = 2.0 * jnp.eye(x_dim)
    chol_P0 = jnp.linalg.cholesky(P0)

    def score(family, label, budget, m_filt, wall):
        est = estimates_on_fine_grid(m_filt, dt, times_fine, f_drift, DT_INNER)
        rmse = float(np.sqrt(np.mean((est - true_fine) ** 2)))
        rec = {
            "seed": seed,
            "family": family,
            "label": label,
            "budget": budget,
            "rmse": rmse,
            "time_s": wall,
        }
        raw_rows.append(rec)
        done_keys.add((seed, family, label, budget))
        print(
            f"  {family:12s} {label:14s} n={budget:4d}  "
            f"RMSE={rmse:6.3f}  time={1e3 * wall:7.1f} ms"
        )

    for cfg in WSCKF_CONFIGS:
        if cfg.method == "product":
            if _have(seed, cfg.family, cfg.label, cfg.n_paths):
                continue
            n_expect = product_path_count(3, cfg.k, lyons_degree=cfg.lyons_degree)
            if n_expect != cfg.n_paths:
                raise RuntimeError(f"path count mismatch {cfg}")
            build_kwargs = dict(
                cubature_order=cfg.k,
                lyons_degree=cfg.lyons_degree,
                multidim_paths="product",
            )
            label = cfg.label
            budget = cfg.n_paths
        else:
            # Build explicitly so scored budget is the actual support size.
            ts_path, pv, pw = precompute_cubature_paths(
                horizon=float(dt),
                noise_dim=noise_dim,
                integration_dt=dt0,
                cubature_order=cfg.k,
                num_paths=cfg.n_paths,
                multidim_method="recombined",
            )
            actual_p = int(pv.shape[0])
            label = f"rec-P{actual_p}"
            budget = actual_p
            if actual_p != cfg.n_paths:
                print(
                    f"  note: requested P={cfg.n_paths} collapsed to actual P={actual_p}"
                )
            if _have(seed, cfg.family, label, budget):
                continue
            build_kwargs = dict(wiener_paths=(ts_path, pv, pw))
        wsckf = build_filter(
            get_init_params=lambda mi: (m0, chol_P0),
            get_dynamics=lambda mi: (
                lambda t, x: lorenz_drift(x),
                lambda t, x: diff_std * jnp.eye(x_dim),
            ),
            get_observations=lambda mi: (
                lambda x: H @ x + d_obs,
                chol_R,
                ys[mi - 1],
            ),
            noise_dim=noise_dim,
            dt=dt,
            integration_dt=dt0,
            **build_kwargs,
        )
        jitted = jit(run_filter, static_argnames=["filter_obj"])
        states, wall = time_call(lambda f=jitted, filt=wsckf: f(filt, model_inputs))
        score(
            cfg.family,
            label,
            budget,
            stack_estimates(m0, states.mean[1:]),
            wall,
        )

    if not _have(seed, "CD-EKF", "EKF", 0):
        ekf_jit = jit(
            lambda: run_cd_ekf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt0,
                state_order="flowmap", solver="rk4",
            )
        )
        ekf_res, wall = time_call(ekf_jit)
        score("CD-EKF", "EKF", 0, stack_estimates(m0, ekf_res.means), wall)

    if not _have(seed, "CD-UKF", "UKF-coarse", 0):
        ukf_jit = jit(
            lambda: run_cd_ukf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt0,
                UKFParams(), state_order="flowmap", solver="rk4",
            )
        )
        ukf_res, wall = time_call(ukf_jit)
        score("CD-UKF", "UKF-coarse", 0, stack_estimates(m0, ukf_res.means), wall)

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

csv_path = os.path.join(OUTPUT_DIR, f"pareto_budget_vs_rmse{OUT_TAG}.csv")
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
BAND_SCALE = 1.0


def _annotate_nodes(ax, fam, pts, xs, ys):
    if fam in ("CD-EKF", "CD-UKF"):
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


fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.65))
# Merge product + recombined onto one WSC-KF curve (sorted by path count).
plot_groups: list[tuple[str, list[str]]] = [
    ("WSC-KF", ["WSC-KF-product", "WSC-KF-recombined"]),
    ("CD-EnKF", ["CD-EnKF"]),
    ("PF", ["PF"]),
    ("CD-EKF", ["CD-EKF"]),
    ("CD-UKF (coarse)", ["CD-UKF"]),
]
all_xs: list[float] = []
n_seeds = len({r["seed"] for r in raw_rows})
wsckf_path_note = ",".join(str(c.n_paths) for c in WSCKF_CONFIGS)
for disp, fams in plot_groups:
    pts = [r for r in summary if r["family"] in fams]
    if not pts:
        continue
    pts = sorted(pts, key=lambda r: r["budget"])
    xs = np.asarray([1e3 * r["time_s_mean"] for r in pts])
    ys = np.asarray([r["rmse_mean"] for r in pts])
    ystd = np.asarray([r["rmse_std"] for r in pts]) * BAND_SCALE
    all_xs.extend(xs.tolist())
    color = COLORS[fams[0]]
    method_key = "WSC-KF" if disp == "WSC-KF" else fams[0]
    marker = METHOD_MARKERS[method_key]
    if len(pts) == 1:
        x0 = float(xs[0])
        ax.fill_between(
            [x0 / 1.04, x0 * 1.04],
            [ys[0] - ystd[0]] * 2,
            [ys[0] + ystd[0]] * 2,
            color=color,
            alpha=0.14,
            lw=0,
        )
        ax.plot(xs, ys, marker, color=color, markerfacecolor="white", markersize=4.5, label=disp)
    else:
        ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=0.14, lw=0)
        ax.plot(xs, ys, f"-{marker}", color=color, markerfacecolor="white", lw=1.25, markersize=4.2, label=disp)
    _annotate_nodes(ax, fams[0], pts, xs, ys)

ax.set_xlabel("Wall-clock time [ms]")
ax.set_ylabel("Fine-grid RMSE")
ax.set_xscale("log")
ax.set_yscale("linear")
if all_xs:
    ax.set_xlim(min(all_xs) / 1.25, max(all_xs) * 1.35)
style_curve_axis(ax, grid_axis="both")
ax.text(
    0.02,
    0.98,
    rf"$\Delta t={dt:g}$; bars: $\pm 1$ SD",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=6.4,
)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3)
fig.subplots_adjust(left=0.19, right=0.98, bottom=0.34, top=0.97)
plot_path = os.path.join(OUTPUT_DIR, f"pareto_budget_vs_rmse{OUT_TAG}.png")
plot_png, plot_pdf = save_publication_figure(fig, plot_path)
plt.close()

print("-" * 60)
print(f"Wrote {plot_png}")
print(f"Wrote {plot_pdf}")
print(f"Wrote {csv_path}")
print(f"Wrote {raw_path}")
