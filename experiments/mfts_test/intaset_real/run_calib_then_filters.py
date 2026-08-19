"""Refine InTaSet calibration via EnKF, then bake-off filters.

Two-stage calibration (train window only):
  1. Dynamics ``(k_pump, γ, a_out)`` from EnKF filter-MLE
     (``mle_params_best.json``, or re-run a short Adam if missing).
  2. Noise ``(meas_std, sigma_u)`` via EnKF-AD on ``(log meas_std, log σ_u)``
     with dynamics held fixed — replaces the open-loop high-frequency
     ``meas_std≈0.05`` floor that made likelihoods pathological.

Then compare CD-EKF / CD-EnKF / WSC-KF on held-out windows under:
  * open-loop LS params (``calibrated_params.json``)
  * filter-refined params (``calibrated_params_filter.json``)

Run:
  uv run --extra experiments python \\
      experiments/mfts_test/intaset_real/run_calib_then_filters.py
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
from wsckf.library import default_recombined_num_paths, product_path_count
from experiments.cd_dynamax_mirror.filters import run_cd_ekf, run_cd_enkf

from intaset_io import default_mc_windows, load_named, window_trajectory
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
    PALETTE,
    save_publication_figure,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_parser = argparse.ArgumentParser(description="Filter-refine InTaSet calibration, then test filters")
_parser.add_argument("--source", type=str, default="4T-M2-train")
_parser.add_argument(
    "--open-loop-params",
    type=str,
    default=os.path.join(OUTPUT_DIR, "calibrated_params.json"),
)
_parser.add_argument(
    "--mle-dynamics",
    type=str,
    default=os.path.join(OUTPUT_DIR, "mle_params_best.json"),
    help="Dynamics warm-start from prior EnKF filter-MLE",
)
_parser.add_argument("--fit-min", type=float, default=15.0, dest="fit_min")
_parser.add_argument("--holdout-min", type=float, default=10.0, dest="holdout_min")
_parser.add_argument("--delta", type=float, default=1.0)
_parser.add_argument("--noise-iters", type=int, default=40, dest="noise_iters")
_parser.add_argument("--lr", type=float, default=0.05)
_parser.add_argument("--enkf-particles", type=int, default=100, dest="enkf_particles")
_parser.add_argument("--dt0", type=float, default=0.15)
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--n-windows", type=int, default=2, dest="n_windows")
_parser.add_argument("--horizon-min", type=float, default=10.0, dest="horizon_min")
_parser.add_argument("--n-timing", type=int, default=1, dest="n_timing")
_args = _parser.parse_args()

DT0 = float(_args.dt0)
ENKF_P = int(_args.enkf_particles)
SOLVER = "rk4"


# ---------------------------------------------------------------------------
# Load calibrations
# ---------------------------------------------------------------------------

p_ol, meta_ol = load_params(_args.open_loop_params)
if os.path.exists(_args.mle_dynamics):
    p_dyn, meta_dyn = load_params(_args.mle_dynamics)
    print(f"Dynamics from EnKF-MLE: {_args.mle_dynamics}")
else:
    p_dyn, meta_dyn = p_ol, {"method": "open-loop-fallback"}
    print(f"WARNING: {_args.mle_dynamics} missing; using open-loop dynamics")

print(
    f"Open-loop: k={p_ol.k_pump:.3g} γ=({p_ol.gamma1:.3f},{p_ol.gamma2:.3f}) "
    f"meas_std={p_ol.meas_std:.3f} sigma_u={p_ol.sigma_u:.3g}"
)
print(
    f"Dyn-MLE:   k={p_dyn.k_pump:.3g} γ=({p_dyn.gamma1:.3f},{p_dyn.gamma2:.3f}) "
    f"(noise still open-loop until stage 2)"
)

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
    height_bias=p_dyn.height_bias,
)
grids_full = prepare_filter_grids(
    traj_full_win,
    delta_t_min=float(_args.delta),
    horizon_min=float(_args.fit_min) + float(_args.holdout_min),
    height_bias=p_dyn.height_bias,
)

YS_TRAIN = jnp.asarray(grids_train["ys"])
N_TRAIN = int(grids_train["n_obs"])
DT = float(grids_train["dt_obs_sec"])
T_TRAIN = grids_train["t_fine_sec"]
U_TRAIN = grids_train["u_fine"]
H0 = grids_train["h_fine"][0]

YS_FULL = jnp.asarray(grids_full["ys"])
N_OBS = int(grids_full["n_obs"])
T_FULL = grids_full["t_fine_sec"]
U_FULL = grids_full["u_fine"]


def make_dyn(p: IntaSetParams, t_sec, pumps, *, sigma_u: float | None = None):
    """JAX dynamics; sigma_u override for noise MLE."""
    su = p.sigma_u if sigma_u is None else sigma_u
    su = jnp.asarray(su, dtype=jnp.float64)
    A = jnp.asarray(p.A, dtype=jnp.float64)
    a = jnp.asarray(p.a_out, dtype=jnp.float64)
    g1, g2, k = p.gamma1, p.gamma2, p.k_pump
    t_grid = jnp.asarray(t_sec, dtype=jnp.float64)
    u_grid = jnp.asarray(pumps, dtype=jnp.float64)
    t_lo, t_hi = t_grid[0], t_grid[-1]
    eps = p.h_eps
    g = p.g

    def _pump_frac(t):
        tt = jnp.clip(t, t_lo, t_hi)
        return jnp.array(
            [
                jnp.interp(tt, t_grid, u_grid[:, 0]),
                jnp.interp(tt, t_grid, u_grid[:, 1]),
            ]
        ) / 100.0

    def drift_fn(t, x):
        q = a * jnp.sqrt(2.0 * g * jnp.maximum(x, eps))
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
        s = su
        return jnp.array(
            [
                [g1 * s / A[0], 0.0],
                [0.0, g2 * s / A[1]],
                [0.0, (1.0 - g2) * s / A[2]],
                [(1.0 - g1) * s / A[3], 0.0],
            ]
        )

    def Qc_fn(t, x):
        return jnp.eye(2)

    def obs_fn(x):
        return x

    return drift_fn, L_fn, Qc_fn, obs_fn


def init_for(p: IntaSetParams):
    x0, P0 = init_estimate(H0, p)
    return jnp.asarray(x0), jnp.asarray(P0)


# ---------------------------------------------------------------------------
# Stage 2: noise MLE with fixed dynamics
# ---------------------------------------------------------------------------

m0_dyn, _ = init_for(p_dyn)
_key_noise = jr.PRNGKey(_args.seed + 7)


def enkf_train_ll_noise(log_meas: jnp.ndarray, log_su: jnp.ndarray):
    meas = jnp.exp(log_meas)
    su = jnp.exp(log_su)
    R = jnp.diag(jnp.full(4, meas**2))
    # P0 scales with meas_std (same rule as init_estimate).
    P0 = jnp.eye(4) * (meas**2 + 0.25)
    drift, L, Qc, obs = make_dyn(p_dyn, T_TRAIN, U_TRAIN, sigma_u=su)
    return run_cd_enkf(
        drift,
        obs,
        L,
        Qc,
        R,
        m0_dyn,
        P0,
        YS_TRAIN,
        DT,
        _key_noise,
        n_particles=ENKF_P,
        noise_dim=2,
        dt0=DT0,
        solver=SOLVER,
    ).cum_loglik[-1]


def _ll_theta_noise(th):
    return enkf_train_ll_noise(th[0], th[1])


noise_vg = jit(value_and_grad(_ll_theta_noise))

# Warm start: open-loop sigma_u, but meas_std from open-loop RMSE scale (not 0.05).
rmse_ol = float(meta_ol.get("rmse_fit_cm", 1.0))
meas0 = float(np.clip(0.5 * rmse_ol, 0.2, 2.0))
su0 = float(p_dyn.sigma_u)
th_n = np.array([np.log(meas0), np.log(su0)], dtype=np.float64)

print(
    f"\n=== Stage 2: EnKF noise MLE ({_args.noise_iters} iters, "
    f"{ENKF_P}p, train {N_TRAIN} obs) ==="
)
print(f"  init meas_std={meas0:.3f} (from 0.5×open-loop RMSE)  sigma_u={su0:.3g}")

b1, b2, eps = 0.9, 0.999, 1e-8
m_mom = np.zeros(2)
v_mom = np.zeros(2)
ll_tr = []
for t in range(1, _args.noise_iters + 1):
    val, g = noise_vg(jnp.asarray(th_n))
    ll = float(val)
    g = np.clip(np.asarray(g, dtype=np.float64), -20.0, 20.0)
    ll_tr.append(ll)
    m_mom = b1 * m_mom + (1 - b1) * g
    v_mom = b2 * v_mom + (1 - b2) * g * g
    mhat = m_mom / (1 - b1**t)
    vhat = v_mom / (1 - b2**t)
    th_n = th_n + _args.lr * mhat / (np.sqrt(vhat) + eps)
    if t == 1 or t % 10 == 0 or t == _args.noise_iters:
        print(
            f"  iter {t:3d}: L={ll:.1f}  meas_std={np.exp(th_n[0]):.3f}  "
            f"sigma_u={np.exp(th_n[1]):.3g}"
        )

meas_hat = float(np.exp(th_n[0]))
su_hat = float(np.exp(th_n[1]))
p_filt = IntaSetParams(
    gamma1=p_dyn.gamma1,
    gamma2=p_dyn.gamma2,
    A=p_dyn.A,
    a_out=p_dyn.a_out,
    k_pump=p_dyn.k_pump,
    meas_std=meas_hat,
    height_bias=p_dyn.height_bias,
    sigma_u=su_hat,
    g=p_dyn.g,
    h_eps=p_dyn.h_eps,
)

filt_path = os.path.join(OUTPUT_DIR, "calibrated_params_filter.json")
save_params(
    filt_path,
    p_filt,
    meta={
        "source": _args.source,
        "fit_min": float(_args.fit_min),
        "holdout_start_min": float(_args.fit_min),
        "calibration": "enkf-mle-dynamics + enkf-mle-noise",
        "dynamics_from": _args.mle_dynamics,
        "open_loop_params": _args.open_loop_params,
        "meas_std_init": meas0,
        "noise_iters": int(_args.noise_iters),
        "train_ll_final": ll_tr[-1],
        "rmse_openloop_fit_cm": rmse_ol,
    },
)
print(f"Wrote {filt_path}")
print(f"  filter-calib: k={p_filt.k_pump:.3g} γ=({p_filt.gamma1:.3f},{p_filt.gamma2:.3f})")
print(f"  meas_std={p_filt.meas_std:.3f}  sigma_u={p_filt.sigma_u:.3g}")

# Open-loop MAPE diagnostic
for label, p in (("open-loop", p_ol), ("filter-calib", p_filt)):
    H = grids_train["h_fine"]
    hat = simulate_open_loop(p, T_TRAIN, U_TRAIN, np.maximum(H[0], p.h_eps))
    print(f"  {label:12s} open-loop train MAPE={mape_pct(hat, H):.2f}%")


# ---------------------------------------------------------------------------
# Filter bake-off on held-out windows
# ---------------------------------------------------------------------------

class WSCKFCfg(NamedTuple):
    family: str
    label: str
    n_paths: int
    k: int
    method: str  # "product" | "recombined"
    m: int = 5


def _k_for_delta(dt_min: float, k_ref: float = 0.5, k_max: int = 8) -> int:
    return min(k_max, max(1, int(round(float(dt_min) / k_ref))))


_k_prop = _k_for_delta(float(_args.delta))
_n_prop = default_recombined_num_paths(2, _k_prop)
WSCKF_CFGS = [
    WSCKFCfg("WSC-KF-product", "prod-m5-k1", product_path_count(2, 1, lyons_degree=5), 1, "product", 5),
    WSCKFCfg("WSC-KF-product", "prod-m5-k2", product_path_count(2, 2, lyons_degree=5), 2, "product", 5),
    WSCKFCfg("WSC-KF-recombined", f"k∝Δt-k{_k_prop}", _n_prop, _k_prop, "recombined", 5),
]
ENKF_BUDGETS = [100, 250]

# Evaluation windows start after calibration fit (no leakage).
windows = default_mc_windows(
    sources=[_args.source],
    duration_min=float(_args.horizon_min),
    n_windows=int(_args.n_windows),
    start_min=float(_args.fit_min),
)
print(
    f"\n=== Filter bake-off: {len(windows)}×{_args.horizon_min:g} min windows "
    f"from t≥{_args.fit_min:g} min ==="
)


def estimates_on_fine_grid(m_filt, dt_obs, times_fine, drift_fn, dt_int):
    def drift_np(t, x):
        return np.asarray(drift_fn(t, jnp.asarray(x)), dtype=np.float64)

    def rk4_to(y, t0, t1):
        span = t1 - t0
        if span <= 0:
            return y.copy()
        n_steps = max(1, int(np.ceil(span / dt_int)))
        h = span / n_steps
        t = t0
        for _ in range(n_steps):
            k1 = drift_np(t, y)
            k2 = drift_np(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = drift_np(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = drift_np(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    n_fine = times_fine.shape[0]
    n_obs = m_filt.shape[0] - 1
    est = np.zeros((n_fine, m_filt.shape[1]))
    est[0] = m_filt[0]
    for k in range(n_obs):
        t0 = k * dt_obs
        t1 = (k + 1) * dt_obs
        js = np.where((times_fine > t0 + 1e-12) & (times_fine <= t1 + 1e-9))[0]
        m = m_filt[k].copy()
        t = t0
        for j in js:
            tt = float(times_fine[j])
            m = rk4_to(m, t, tt)
            t = tt
            if abs(tt - t1) <= 1e-9:
                est[j] = m_filt[k + 1]
                m = m_filt[k + 1].copy()
            else:
                est[j] = m
    return est


def run_filters_for_params(tag: str, p: IntaSetParams):
    rows = []
    for win_idx, traj in enumerate(windows):
        grids = prepare_filter_grids(
            traj,
            delta_t_min=float(_args.delta),
            horizon_min=float(_args.horizon_min),
            height_bias=p.height_bias,
        )
        t_fine = grids["t_fine_sec"]
        h_fine = grids["h_fine"]
        u_fine = grids["u_fine"]
        ys = jnp.asarray(grids["ys"])
        n_obs = grids["n_obs"]
        dt = grids["dt_obs_sec"]
        dt0 = dt / max(1, int(round(dt / DT0)))
        m0, P0 = init_estimate(h_fine[0], p)
        m0j, P0j = jnp.asarray(m0), jnp.asarray(P0)
        chol_P0 = jnp.linalg.cholesky(P0j)
        R = jnp.diag(jnp.full(4, p.meas_std**2))
        chol_R = jnp.linalg.cholesky(R)
        drift, L, Qc, obs = make_dyn(p, t_fine, u_fine)

        def score(family, label, budget, m_filt, final_ll, wall):
            est = estimates_on_fine_grid(
                np.asarray(m_filt), dt, t_fine, drift, DT0
            )
            rec = {
                "calib": tag,
                "window": win_idx,
                "family": family,
                "label": label,
                "budget": budget,
                "mape": mape_pct(est, h_fine),
                "nll_per_obs": -float(final_ll) / n_obs,
                "time_s": wall,
            }
            rows.append(rec)
            print(
                f"  [{tag:12s}] w{win_idx} {family:12s} {label:12s}  "
                f"MAPE={rec['mape']:6.2f}  NLL/obs={rec['nll_per_obs']:8.2f}  "
                f"t={1e3*wall:6.1f}ms"
            )

        def time_call(fn):
            out = fn()
            jax.block_until_ready(out)
            samples = []
            for _ in range(max(1, _args.n_timing)):
                t0 = time.perf_counter()
                jax.block_until_ready(fn())
                samples.append(time.perf_counter() - t0)
            return out, float(np.median(samples))

        # EKF
        def _ekf():
            return run_cd_ekf(
                drift, obs, L, Qc, R, m0j, P0j, ys, dt, dt0,
                state_order="first", solver=SOLVER,
            )

        ekf_res, ekf_t = time_call(_ekf)
        m_ekf = np.vstack([m0, np.asarray(ekf_res.means)])
        score("CD-EKF", "EKF", 0, m_ekf, ekf_res.cum_loglik[-1], ekf_t)

        # EnKF budgets
        for np_ in ENKF_BUDGETS:
            key = jr.PRNGKey(_args.seed + 100 * win_idx + np_)

            def _enkf(key=key, n=np_):
                return run_cd_enkf(
                    drift, obs, L, Qc, R, m0j, P0j, ys, dt, key,
                    n_particles=n, noise_dim=2, dt0=dt0, solver=SOLVER,
                )

            enkf_res, enkf_t = time_call(_enkf)
            m_enkf = np.vstack([m0, np.asarray(enkf_res.means)])
            score("CD-EnKF", f"EnKF-{np_}", np_, m_enkf, enkf_res.cum_loglik[-1], enkf_t)

        # WSC-KF (product ladder + k∝Δt recombined mitigation)
        for cfg in WSCKF_CFGS:
            kwargs = dict(
                get_init_params=lambda mi: (m0j, chol_P0),
                get_dynamics=lambda mi: (drift, L),
                get_observations=lambda mi: (obs, chol_R, ys[mi - 1]),
                noise_dim=2,
                dt=dt,
                integration_dt=dt0,
                cubature_order=cfg.k,
                sde_form="ito",
            )
            if cfg.method == "recombined":
                filt = build_filter(
                    **kwargs,
                    num_paths=cfg.n_paths,
                    multidim_paths="recombined",
                )
            else:
                filt = build_filter(
                    **kwargs,
                    lyons_degree=cfg.m,
                    multidim_paths="product",
                )
            run = jit(run_filter, static_argnames=["filter_obj"])
            mis = jnp.arange(n_obs + 1)

            def _wsckf(filt=filt, mis=mis):
                return run(filt, mis)

            st, wsckf_t = time_call(_wsckf)
            score(
                cfg.family,
                cfg.label,
                cfg.n_paths,
                np.asarray(st.mean),
                st.log_normalizing_constant[-1],
                wsckf_t,
            )
    return rows


all_rows: list[dict] = []
all_rows.extend(run_filters_for_params("open-loop", p_ol))
all_rows.extend(run_filters_for_params("filter-calib", p_filt))

raw_path = os.path.join(OUTPUT_DIR, "calib_filter_bakeoff_raw.csv")
with open(raw_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader()
    w.writerows(all_rows)
print(f"\nWrote {raw_path}")

# Aggregate
keys = sorted({(r["calib"], r["family"], r["label"], r["budget"]) for r in all_rows})
summary = []
for calib, fam, lab, bud in keys:
    sub = [
        r
        for r in all_rows
        if r["calib"] == calib and r["family"] == fam and r["label"] == lab and r["budget"] == bud
    ]
    summary.append(
        {
            "calib": calib,
            "family": fam,
            "label": lab,
            "budget": bud,
            "mape_mean": float(np.mean([r["mape"] for r in sub])),
            "mape_std": float(np.std([r["mape"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
            "nll_mean": float(np.mean([r["nll_per_obs"] for r in sub])),
            "nll_std": float(np.std([r["nll_per_obs"] for r in sub], ddof=1) if len(sub) > 1 else 0.0),
            "time_s_mean": float(np.mean([r["time_s"] for r in sub])),
        }
    )

sum_path = os.path.join(OUTPUT_DIR, "calib_filter_bakeoff.csv")
with open(sum_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    for r in sorted(summary, key=lambda x: (x["calib"], x["family"], x["budget"])):
        w.writerow(r)
print(f"Wrote {sum_path}")

# Plot: paired estimates for open-loop vs filter-calibrated parameters.
# Points avoid the ambiguous zero baseline of bars when the NLL panel needs a
# logarithmic scale to show both calibration regimes.
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.65))
# Order methods for display
order = [
    ("CD-EKF", "EKF", 0),
    ("WSC-KF-product", "prod-m5-k1", 9),
    ("WSC-KF-product", "prod-m5-k2", 81),
    ("WSC-KF-recombined", f"k∝Δt-k{_k_prop}", _n_prop),
    ("CD-EnKF", "EnKF-100", 100),
    ("CD-EnKF", "EnKF-250", 250),
]
display_labels = [
    "CD-EKF",
    "WSC-KF\n$P=9$",
    "WSC-KF\n$P=81$",
    f"WSC-KF\n$k={_k_prop}$",
    "CD-EnKF\n$N=100$",
    "CD-EnKF\n$N=250$",
]
x = np.arange(len(order))
offset = 0.14
for ax, metric, std_metric, ylab, title in (
    (axes[0], "mape_mean", "mape_std", "MAPE [%]", "MAPE"),
    (
        axes[1],
        "nll_mean",
        "nll_std",
        "NLL [nats/observation]",
        "NLL",
    ),
):
    ol = []
    fc = []
    ol_std = []
    fc_std = []
    for fam, lab, bud in order:
        ol_r = next(
            r for r in summary if r["calib"] == "open-loop" and r["family"] == fam and r["label"] == lab
        )
        fc_r = next(
            r
            for r in summary
            if r["calib"] == "filter-calib" and r["family"] == fam and r["label"] == lab
        )
        ol.append(ol_r[metric])
        fc.append(fc_r[metric])
        ol_std.append(ol_r[std_metric])
        fc_std.append(fc_r[std_metric])
    ax.errorbar(
        x - offset,
        ol,
        yerr=ol_std,
        label="Open-loop LS",
        fmt="s",
        color=PALETTE["neutral_gray"],
        markerfacecolor="white",
        markeredgewidth=0.8,
        markersize=4.0,
        elinewidth=0.8,
        capsize=2.0,
        capthick=0.8,
        linestyle="none",
        zorder=3,
    )
    ax.errorbar(
        x + offset,
        fc,
        yerr=fc_std,
        label="Filter calibrated",
        fmt="o",
        color=PALETTE["warm_red"],
        markerfacecolor=PALETTE["warm_red"],
        markeredgewidth=0.8,
        markersize=4.0,
        elinewidth=0.8,
        capsize=2.0,
        capthick=0.8,
        linestyle="none",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    if metric == "nll_mean" and min(np.asarray(ol) - np.asarray(ol_std)) > 0.0 and min(
        np.asarray(fc) - np.asarray(fc_std)
    ) > 0.0:
        ax.set_yscale("log")
        ax.set_title(f"{title} (log scale)")
    style_curve_axis(ax, grid_axis="y")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=len(labels),
)
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.23, top=0.78, wspace=0.28)
plot_path = os.path.join(OUTPUT_DIR, "calib_filter_bakeoff.png")
save_publication_figure(fig, plot_path)
plt.close(fig)
print(f"Wrote {plot_path}")

print("\n=== Summary (mean over windows) ===")
print(f"{'calib':12s} {'method':14s} {'MAPE':>8s} {'NLL/obs':>10s}")
for r in sorted(summary, key=lambda x: (x["calib"], x["family"], x["budget"])):
    print(
        f"{r['calib']:12s} {r['label']:14s} {r['mape_mean']:8.2f} {r['nll_mean']:10.2f}"
    )
