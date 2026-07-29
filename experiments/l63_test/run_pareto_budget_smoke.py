"""Smoke Pareto for Lorenz-63: wall-clock vs fine-grid RMSE.

Mirrors the MFTS budget Pareto style: WS-CKF at the smallest exact product
path count (d=3, m=3,k=1 → 8 paths), EnKF/PF particle sweeps, multi-seed mean.

Run:
  uv run --extra experiments python experiments/l63_test/run_pareto_budget_smoke.py
  uv run --extra experiments python experiments/l63_test/run_pareto_budget_smoke.py \\
      --seeds 0,1,2,3,4,5,6,7,8

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
from crf import build_filter
from crf.library import product_path_count

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
    label: str
    lyons_degree: int
    k: int
    n_paths: int


def default_crf_configs(noise_dim: int = 3) -> list[CrfConfig]:
    """Smallest moment-safe exact product config for d=3: m=3,k=1 → 8 paths."""
    specs = [(3, 1)]
    configs = []
    for m, k in specs:
        n = product_path_count(noise_dim, k, lyons_degree=m)
        configs.append(
            CrfConfig(label=f"prod-m{m}-k{k}", lyons_degree=m, k=k, n_paths=n)
        )
    return configs


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
_parser.add_argument("--n-timing", type=int, default=2, dest="n_timing")
_args = _parser.parse_args()

SEEDS = _parse_ints(_args.seeds)
DELTA_T = float(_args.delta)
HORIZON = float(_args.horizon)
EVAL_DT = float(_args.eval_dt)
SPINUP_STEPS = int(_args.spinup)
ENKF_LIST = _parse_ints(_args.enkf)
PF_LIST = _parse_ints(_args.pf)
N_TIMING = _args.n_timing
CRF_CONFIGS = default_crf_configs(3)

lorenz_sigma, lorenz_rho, lorenz_beta = 10.0, 28.0, 8.0 / 3.0
DT_INNER = 0.01
diff_std, obs_std = 1.0, 0.3
x_dim, y_dim, noise_dim = 3, 1, 3

H = jnp.array([[1.0, 0.0, 0.0]])
d_obs = jnp.zeros(y_dim)
R = (obs_std**2) * jnp.eye(y_dim)
chol_R = jnp.linalg.cholesky(R)

COLORS = {
    "CD-EKF": "#2E86AB",
    "CD-UKF": "#3DA35D",
    "CD-EnKF": "#F6AE2D",
    "PF": "#F24236",
    "CRF-product": "#6A4C93",
}
DISPLAY_NAME = {
    "CRF-product": "WS-CKF",
    "CD-EKF": "CD-EKF",
    "CD-UKF": "CD-UKF",
    "CD-EnKF": "CD-EnKF",
    "PF": "PF",
}
_LABEL_OFFSET = {
    "CD-EnKF": (0, -10),
    "PF": (0, 7),
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
    f"seeds={SEEDS}, CRF={[c.label + f'({c.n_paths})' for c in CRF_CONFIGS]}"
)

raw_rows: list[dict] = []

for seed in SEEDS:
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
        print(
            f"  {family:12s} {label:14s} n={budget:4d}  "
            f"RMSE={rmse:6.3f}  time={1e3 * wall:7.1f} ms"
        )

    for cfg in CRF_CONFIGS:
        n_expect = product_path_count(3, cfg.k, lyons_degree=cfg.lyons_degree)
        if n_expect != cfg.n_paths:
            raise RuntimeError(f"path count mismatch {cfg}")
        crf = build_filter(
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
            cubature_order=cfg.k,
            lyons_degree=cfg.lyons_degree,
            multidim_paths="product",
        )
        jitted = jit(run_filter, static_argnames=["filter_obj"])
        states, wall = time_call(lambda f=jitted, filt=crf: f(filt, model_inputs))
        score(
            "CRF-product",
            cfg.label,
            cfg.n_paths,
            stack_estimates(m0, states.mean[1:]),
            wall,
        )

    ekf_jit = jit(
        lambda: run_cd_ekf(
            f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt0,
            state_order="flowmap", solver="rk4",
        )
    )
    ekf_res, wall = time_call(ekf_jit)
    score("CD-EKF", "EKF", 0, stack_estimates(m0, ekf_res.means), wall)

    ukf_jit = jit(
        lambda: run_cd_ukf(
            f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, dt0,
            UKFParams(), state_order="flowmap", solver="rk4",
        )
    )
    ukf_res, wall = time_call(ukf_jit)
    score("CD-UKF", "UKF", 0, stack_estimates(m0, ukf_res.means), wall)

    key = random.key(seed)
    for n_part in ENKF_LIST:
        key, sub = random.split(key)
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
        pf_jit = jit(
            lambda k, n=n_part: run_bootstrap_pf(
                f_drift, h_obs, L_fn, Qc_fn, R, m0, P0, ys, dt, k,
                n_particles=n, noise_dim=noise_dim, dt0=dt0, solver="rk4",
            )
        )
        pf_res, wall = time_call(pf_jit, sub)
        score("PF", f"PF-{n_part}", n_part, stack_estimates(m0, pf_res.means), wall)

# ---- CSV ----
raw_path = os.path.join(OUTPUT_DIR, "pareto_budget_vs_rmse_raw.csv")
with open(raw_path, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["delta_t", "n_obs", "horizon", "seed", "family", "label", "budget", "rmse", "time_s"],
    )
    w.writeheader()
    for r in raw_rows:
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

csv_path = os.path.join(OUTPUT_DIR, "pareto_budget_vs_rmse.csv")
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
BAND_SCALE = 0.3
WS_CKF_PLOT_BUDGET = CRF_CONFIGS[0].n_paths  # single smallest product config


def _annotate_nodes(ax, fam, pts, xs, ys):
    if fam in ("CD-EKF", "CD-UKF", "CRF-product"):
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
plot_families = ["CRF-product", "CD-EnKF", "PF", "CD-EKF", "CD-UKF"]
all_xs: list[float] = []
n_seeds = len({r["seed"] for r in raw_rows})
for fam in plot_families:
    pts = [r for r in summary if r["family"] == fam]
    if fam == "CRF-product":
        pts = [r for r in pts if r["budget"] == WS_CKF_PLOT_BUDGET]
    if not pts:
        continue
    pts = sorted(pts, key=lambda r: r["budget"])
    xs = np.asarray([1e3 * r["time_s_mean"] for r in pts])
    ys = np.asarray([r["rmse_mean"] for r in pts])
    ystd = np.asarray([r["rmse_std"] for r in pts]) * BAND_SCALE
    all_xs.extend(xs.tolist())
    color = COLORS[fam]
    marker = "s" if fam in ("CD-EKF", "CD-UKF", "CRF-product") else "o"
    disp = DISPLAY_NAME.get(fam, fam)
    if len(pts) == 1:
        x0 = float(xs[0])
        ax.fill_between(
            [x0 / 1.04, x0 * 1.04],
            [ys[0] - ystd[0]] * 2,
            [ys[0] + ystd[0]] * 2,
            color=color,
            alpha=0.22,
            lw=0,
        )
        ax.plot(xs, ys, marker, color=color, markersize=7, label=disp)
    else:
        ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=0.22, lw=0)
        ax.plot(xs, ys, f"-{marker}", color=color, lw=1.5, markersize=6, label=disp)
    _annotate_nodes(ax, fam, pts, xs, ys)

ax.set_xlabel("Wall-clock time [ms]")
ax.set_ylabel("Fine-grid RMSE (overall)")
ax.set_xscale("log")
ax.set_yscale("linear")
if all_xs:
    ax.set_xlim(min(all_xs) / 1.25, max(all_xs) * 1.35)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="best")
ax.set_title(
    rf"L63 smoke Pareto at $\Delta t={dt:g}$ "
    rf"(mean±{BAND_SCALE:g}$\sigma$ over {n_seeds} seeds; "
    rf"horizon={HORIZON:g}; WS-CKF = {WS_CKF_PLOT_BUDGET}-path product)",
    fontsize=11,
)
fig.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "pareto_budget_vs_rmse.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print("-" * 60)
print(f"Wrote {plot_path}")
print(f"Wrote {csv_path}")
print(f"Wrote {raw_path}")
