"""Smoke test: iterated EKF / UKF / WSC-KF Pareto (wall-clock vs MAPE).

Sweeps measurement-update iterations at fixed Δt with one seed and a small
product WSC-KF (9 paths). Intentionally cheap vs ``run_pareto_budget.py``.

Run:
  uv run --extra experiments python experiments/mfts_test/run_pareto_iters_smoke.py

Outputs:
  ``experiments/mfts_test/output/pareto_iters_smoke.{pdf,png}``
  ``experiments/mfts_test/output/pareto_iters_smoke.csv``
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

EXPERIMENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, HERE)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import jit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from experiments.cd_dynamax_mirror.filters import UKFParams, run_cd_ekf, run_cd_ukf
from cuthbert import filter as run_filter
from wsckf import build_filter

from mfts_model import (
    MFTSParams,
    init_estimate,
    make_mfts_dynamics,
    mape_pct,
    simulate_mfts,
)
from plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    METHOD_MARKERS,
    save_publication_figure,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_parser = argparse.ArgumentParser(description="Smoke Pareto: iterated EKF/UKF/WSC-KF")
_parser.add_argument("--seed", type=int, default=100)
_parser.add_argument("--delta", type=float, default=1.0)
_parser.add_argument("--iters", type=str, default="1,2,3,5")
_parser.add_argument("--n-timing", type=int, default=2, dest="n_timing")
_args = _parser.parse_args()

SEED = _args.seed
DELTA_T_MIN = float(_args.delta)
ITERS = [int(x) for x in _args.iters.split(",") if x.strip()]
N_TIMING = _args.n_timing

HORIZON_MIN = 30.0
EVAL_DT_MIN = 0.05
DT_TRUTH = 0.015
DT_FILTER = 0.15
LAMBDA_SIM, SIGMA_SIM = 0.1, 5.0
LAMBDA_EKF_UKF, SIGMA_EKF = 0.0, 5.0
LAMBDA_WSCKF, SIGMA_WSCKF = 2.0e-3, 5.0
SIGMA_UKF = 1.0
F1, F2 = 5.0, 5.0

COLORS = {name: METHOD_COLORS[name] for name in ("EKF", "UKF", "WSC-KF")}


def time_call(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return out, float(np.median(samples))


def stack_estimates(m0_arr, means):
    return np.vstack([np.asarray(m0_arr), np.asarray(means)])


def _as_numpy_drift(drift_fn):
    def drift_np(t_sec: float, x: np.ndarray) -> np.ndarray:
        return np.asarray(drift_fn(t_sec, jnp.asarray(x)), dtype=np.float64)

    return drift_np


def rk4_step(drift_np, t, y, h):
    k1 = drift_np(t, y)
    k2 = drift_np(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = drift_np(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = drift_np(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_to(drift_np, y, t0, t1, dt0):
    span = t1 - t0
    if span <= 0:
        return y.copy()
    n_steps = max(1, int(np.ceil(span / dt0)))
    h = span / n_steps
    t = t0
    for _ in range(n_steps):
        y = rk4_step(drift_np, t, y, h)
        t += h
    return y


def estimates_on_fine_grid(m_filt, dt_obs_sec, times_fine_sec, drift_fn, dt_integrate):
    drift_np = _as_numpy_drift(drift_fn)
    n_fine = times_fine_sec.shape[0]
    est = np.zeros((n_fine, m_filt.shape[1]), dtype=np.float64)
    est[0] = m_filt[0]
    n_obs = m_filt.shape[0] - 1
    for k in range(n_obs):
        t0 = k * dt_obs_sec
        t1 = (k + 1) * dt_obs_sec
        js = np.where((times_fine_sec > t0 + 1e-12) & (times_fine_sec <= t1 + 1e-9))[0]
        m = m_filt[k].copy()
        t = t0
        for j in js:
            t_target = float(times_fine_sec[j])
            m = rk4_to(drift_np, m, t, t_target, dt_integrate)
            t = t_target
            if abs(t_target - t1) <= 1e-9:
                est[j] = m_filt[k + 1]
                m = m_filt[k + 1].copy()
            else:
                est[j] = m
    return est


N_FINE = int(round(HORIZON_MIN / EVAL_DT_MIN))
DT_FINE_MIN = HORIZON_MIN / N_FINE
DT_FINE_SEC = DT_FINE_MIN * 60.0
stride = int(round(DELTA_T_MIN / DT_FINE_MIN))
assert np.isclose(stride * DT_FINE_MIN, DELTA_T_MIN)
n_obs = N_FINE // stride
dt = stride * DT_FINE_MIN * 60.0
dt0 = dt / max(1, int(round(dt / DT_FILTER)))
times_fine_sec = np.arange(N_FINE + 1) * DT_FINE_SEC
model_inputs = jnp.arange(n_obs + 1)

p = MFTSParams()
print(f"Smoke iterated Pareto: Δt={DELTA_T_MIN:g} min, n_obs={n_obs}, iters={ITERS}, seed={SEED}")
true_fine, ys_fine_np, R = simulate_mfts(
    horizon_min=HORIZON_MIN,
    n_obs=N_FINE,
    n_substeps_truth=max(1, int(round(DT_FINE_SEC / DT_TRUTH))),
    p=p,
    lambda_sim=LAMBDA_SIM,
    sigma_sim=SIGMA_SIM,
    F1=F1,
    F2=F2,
    seed=SEED,
)
ys = jnp.asarray(ys_fine_np)[stride - 1 :: stride]
x0, P0 = init_estimate(p, F1, F2, true_state=true_fine[0], cov_scale=1.0)
m0 = jnp.asarray(x0)
P0_j = jnp.asarray(P0)
chol_P0 = jnp.linalg.cholesky(P0_j)
chol_R = jnp.linalg.cholesky(R)

bar_est = lambda t_min, nom=p.F_nominal_est: nom
drift_ekf, L_ekf, Qc, obs_fn, _ = make_mfts_dynamics(
    p, lambda_dist=LAMBDA_EKF_UKF, sigma_dist=SIGMA_EKF,
    bar_F3=bar_est, bar_F4=bar_est, F1=F1, F2=F2,
)
drift_ukf, L_ukf, _, _, _ = make_mfts_dynamics(
    p, lambda_dist=LAMBDA_EKF_UKF, sigma_dist=SIGMA_UKF,
    bar_F3=bar_est, bar_F4=bar_est, F1=F1, F2=F2,
)
drift_wsckf, L_wsckf, _, _, _ = make_mfts_dynamics(
    p, lambda_dist=LAMBDA_WSCKF, sigma_dist=SIGMA_WSCKF,
    bar_F3=bar_est, bar_F4=bar_est, F1=F1, F2=F2,
)

rows: list[dict] = []


def score(family, n_iter, m_filt, final_ll, wall, drift_fn):
    est = estimates_on_fine_grid(m_filt, dt, times_fine_sec, drift_fn, DT_FILTER)
    rec = {
        "family": family,
        "num_iter": n_iter,
        "mapex": mape_pct(est, true_fine, slice(0, 4)),
        "maped": mape_pct(est, true_fine, slice(4, 6)),
        "nll_per_obs": -final_ll / n_obs,
        "time_s": wall,
    }
    rows.append(rec)
    print(
        f"  {family:4s} iter={n_iter}  MAPEx={rec['mapex']:6.2f}  "
        f"MAPEd={rec['maped']:6.2f}  time={1e3*wall:7.1f} ms"
    )


for n_iter in ITERS:
    ekf_jit = jit(
        lambda n=n_iter: run_cd_ekf(
            drift_ekf, obs_fn, L_ekf, Qc, R, m0, P0_j, ys, dt, dt0,
            num_iter=n, state_order="first", solver="rk4",
        )
    )
    res, wall = time_call(ekf_jit)
    score("EKF", n_iter, stack_estimates(m0, res.means), float(res.cum_loglik[-1]), wall, drift_ekf)

    ukf_jit = jit(
        lambda n=n_iter: run_cd_ukf(
            drift_ukf, obs_fn, L_ukf, Qc, R, m0, P0_j, ys, dt, dt0,
            UKFParams(), num_iter=n, state_order="first", solver="rk4",
        )
    )
    res, wall = time_call(ukf_jit)
    score("UKF", n_iter, stack_estimates(m0, res.means), float(res.cum_loglik[-1]), wall, drift_ukf)

    # Exact product WSC-KF: m=5, k=1 → 9 paths
    wsckf = build_filter(
        get_init_params=lambda mi: (m0, chol_P0),
        get_dynamics=lambda mi: (lambda t, x: drift_wsckf(t, x), lambda t, x: L_wsckf(t, x)),
        get_observations=lambda mi: (obs_fn, chol_R, ys[mi - 1]),
        noise_dim=2,
        dt=dt,
        integration_dt=dt0,
        cubature_order=1,
        lyons_degree=5,
        multidim_paths="product",
        num_iter=n_iter,
    )
    wsckf_jit = jit(run_filter, static_argnames=["filter_obj"])
    states, wall = time_call(lambda f=wsckf_jit, filt=wsckf: f(filt, model_inputs))
    score(
        "WSC-KF",
        n_iter,
        stack_estimates(m0, states.mean[1:]),
        float(states.log_normalizing_constant[-1]),
        wall,
        drift_wsckf,
    )

csv_path = os.path.join(OUTPUT_DIR, "pareto_iters_smoke.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["delta_t_min", "seed", "family", "num_iter", "mapex", "maped", "nll_per_obs", "time_s"],
    )
    w.writeheader()
    for r in rows:
        w.writerow({"delta_t_min": DELTA_T_MIN, "seed": SEED, **r})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(DOUBLE_COLUMN_WIDTH, 2.7),
    sharex=True,
)
all_xs: list[float] = []
handles = []
labels = []
label_offsets = {
    "EKF": (-3, 4),
    "UKF": (3, 4),
    "WSC-KF": (0, -8),
}
for panel_idx, (ax, metric, title) in enumerate(
    zip(
        axes,
        ("mapex", "maped"),
        ("Tank states", "Disturbance states"),
    )
):
    for fam in ("EKF", "UKF", "WSC-KF"):
        pts = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["num_iter"])
        xs = np.asarray([1e3 * r["time_s"] for r in pts], dtype=np.float64)
        ys = np.asarray([r[metric] for r in pts], dtype=np.float64)
        all_xs.extend(xs)
        line, = ax.plot(
            xs,
            ys,
            color=COLORS[fam],
            marker=METHOD_MARKERS[fam],
            linestyle="-",
            linewidth=1.25,
            markersize=4.3,
            label=fam,
        )
        if panel_idx == 0:
            handles.append(line)
            labels.append(fam)
        dx, dy = label_offsets[fam]
        for point, x, y in zip(pts, xs, ys):
            ax.annotate(
                str(point["num_iter"]),
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="center",
                va="bottom" if dy >= 0 else "top",
                fontsize=6.1,
                color=COLORS[fam],
            )
    ax.set_xlabel("Wall-clock time [ms]")
    ax.set_ylabel("MAPE [%]")
    ax.set_title(title)
    ax.set_xscale("log")
    style_curve_axis(ax, grid_axis="both")
    ax.margins(y=0.14)
if all_xs:
    axes[0].set_xlim(min(all_xs) / 1.25, max(all_xs) * 1.35)
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=3,
    title="Engineering smoke · node labels: update iterations",
    fontsize=6.8,
    title_fontsize=6.8,
    columnspacing=0.9,
)
fig.subplots_adjust(
    left=0.075,
    right=0.99,
    bottom=0.19,
    top=0.75,
    wspace=0.27,
)
plot_path = os.path.join(OUTPUT_DIR, "pareto_iters_smoke.png")
plot_png_path, plot_pdf_path = save_publication_figure(fig, plot_path)
plt.close()

print("-" * 60)
print(f"Wrote {plot_png_path}")
print(f"Wrote {plot_pdf_path}")
print(f"Wrote {csv_path}")
