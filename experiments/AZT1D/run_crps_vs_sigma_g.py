"""Sweep σ_G scale: does WSC-KF CRPS degrade more slowly than moment / MC filters?

Loads per-subject θ̂ from a cohort params JSON, multiplies σ_G by a ladder of
scales, and scores held-out CRPS / NLL for WSC-KF-4, EKF, UKF, EnKF, PF
(PF uses mixture CRPS).

Outputs under ``output/``:
  crps_vs_sigma_g_scale{_tag}.png
  crps_vs_sigma_g_scale{_tag}.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

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
from jax import jit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from cuthbert import filter as run_filter
from wsckf import build_filter
from wsckf.library import product_path_count
from cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)

from azt1d_data import list_subjects, load_azt1d_series
from glucose_model import (
    GlucoseParams,
    wsckf_glucose_dynamics,
    heldout_crps_per_obs,
    heldout_mean_score,
    init_estimate,
    make_glucose_dynamics,
)
from plot_style import (
    METHOD_COLORS,
    METHOD_MARKERS,
    SINGLE_COLUMN_WIDTH,
    save_publication_figure,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def heldout_nll_per_obs(cum_ll, n_train: int, n_obs: int) -> float:
    ll = np.asarray(cum_ll, dtype=np.float64)
    if ll.shape[0] == n_obs + 1:
        held = float(ll[-1] - ll[n_train])
    elif ll.shape[0] == n_obs:
        base = float(ll[n_train - 1]) if n_train > 0 else 0.0
        held = float(ll[-1] - base)
    else:
        raise ValueError(f"unexpected cum_ll length {ll.shape[0]}")
    return -held / float(n_obs - n_train)


def main():
    ap = argparse.ArgumentParser(description="CRPS vs σ_G scale robustness sweep")
    ap.add_argument(
        "--params-json",
        type=str,
        default=os.path.join(OUTPUT_DIR, "cohort_calibrated_params_dt20_fixed.json"),
    )
    ap.add_argument("--dt-assim", type=float, default=20.0, dest="dt_assim")
    ap.add_argument("--n-obs", type=int, default=2016, dest="n_obs")
    ap.add_argument("--holdout-frac", type=float, default=0.3, dest="holdout_frac")
    ap.add_argument(
        "--scales",
        type=str,
        default="1,2,3,5,8",
        help="comma-separated multipliers on calibrated σ_G",
    )
    ap.add_argument("--subjects", type=str, default="", help="default: all in params json")
    ap.add_argument("--wsckf-paths", type=int, default=4, dest="wsckf_paths")
    ap.add_argument("--enkf", type=int, default=100, help="EnKF particles; 0 = skip")
    ap.add_argument("--pf", type=int, default=100, help="PF particles; 0 = skip")
    ap.add_argument("--dt0", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meas-std", type=float, default=8.0, dest="meas_std")
    ap.add_argument("--out-suffix", type=str, default="dt20", dest="out_suffix")
    args = ap.parse_args()

    base_dt = 5.0
    dt_assim = float(args.dt_assim)
    stride = int(round(dt_assim / base_dt))
    if abs(stride * base_dt - dt_assim) > 1e-6:
        raise SystemExit("--dt-assim must be multiple of 5")

    with open(args.params_json) as f:
        blob = json.load(f)
    theta = {int(r["subject"]): (float(r["p3"]), float(r["sigma_g"])) for r in blob["params"]}
    if args.subjects.strip():
        subjects = _parse_ints(args.subjects)
    else:
        subjects = sorted(theta.keys())

    scales = _parse_floats(args.scales)
    # match product path count
    n_paths = int(args.wsckf_paths)
    m_lyons, k_cub = None, None
    for m, k in [(3, 1), (5, 1), (3, 2), (5, 2), (3, 3)]:
        if product_path_count(2, k, lyons_degree=m) == n_paths:
            m_lyons, k_cub = m, k
            break
    if m_lyons is None:
        raise SystemExit(f"wsckf-paths={n_paths} not on product ladder")

    rows: list[dict] = []
    P_base = GlucoseParams(meas_std=float(args.meas_std))
    print(
        f"σ_G-scale sweep  subjects={subjects}  Δt={dt_assim:g}  "
        f"scales={scales}  WSC-{n_paths}  enkf={args.enkf}  pf={args.pf}  "
        f"θ={args.params_json}"
    )

    for sid in subjects:
        if sid not in theta:
            print(f"  skip S{sid}: no θ")
            continue
        p3, sg0 = theta[sid]
        series = load_azt1d_series(subject=sid, max_obs=int(args.n_obs), dt_min=base_dt)
        idx = np.arange(0, int(series.cgm.shape[0]), stride)
        cgm = np.asarray(series.cgm, dtype=np.float64)[idx]
        n_obs = int(cgm.shape[0])
        n_held = max(1, int(round(n_obs * args.holdout_frac)))
        n_train = n_obs - n_held
        YS = jnp.asarray(cgm.reshape(-1, 1))
        m0_np, P0_np = init_estimate(cgm[:n_train], P_base)
        M0 = jnp.asarray(m0_np)
        P0 = jnp.asarray(P0_np)
        CHOL_P0 = jnp.linalg.cholesky(P0)
        dt = float(dt_assim)
        dt0 = float(args.dt0)

        for scale in scales:
            sg = float(sg0) * float(scale)
            drift, L, Qc, OBS_FN, R = make_glucose_dynamics(
                P_base,
                t_min=series.t_min,
                insulin_u_per_min=series.insulin_u_per_min,
                carb_g_per_min=series.carb_g_per_min,
                p3=p3,
                sigma_g=sg,
            )
            CHOL_R = jnp.linalg.cholesky(R)

            def get_init_params(_mi, M0=M0, CHOL_P0=CHOL_P0):
                return M0, CHOL_P0

            def get_dynamics(mi, drift=drift, L=L, dt=dt):
                return wsckf_glucose_dynamics(drift, L, mi, dt)

            def get_observations(mi, OBS_FN=OBS_FN, CHOL_R=CHOL_R, YS=YS):
                idx_ = mi if not hasattr(mi, "t_idx") else mi.t_idx
                return OBS_FN, CHOL_R, YS[idx_ - 1]

            filt = build_filter(
                get_init_params=get_init_params,
                get_dynamics=get_dynamics,
                get_observations=get_observations,
                noise_dim=2,
                dt=dt,
                integration_dt=dt0,
                cubature_order=k_cub,
                lyons_degree=m_lyons,
                num_paths=None,
                multidim_paths="product",
                sde_form="ito",
                checkpoint_steps=False,
            )
            _run = jit(run_filter, static_argnames=["filter_obj"])

            def score(name, cum_ll, wall, budget=0, *, pred_mu=None, pred_std=None, pred_crps=None):
                nll = heldout_nll_per_obs(cum_ll, n_train, n_obs)
                if pred_crps is not None:
                    crps = heldout_mean_score(pred_crps, n_train, n_obs)
                else:
                    crps = heldout_crps_per_obs(cgm, pred_mu, pred_std, n_train)
                rows.append(
                    {
                        "subject": sid,
                        "scale": float(scale),
                        "sigma_g": sg,
                        "sigma_g0": float(sg0),
                        "p3": float(p3),
                        "filter": name,
                        "budget": budget,
                        "held_nll": nll,
                        "crps": crps,
                        "time_s": wall,
                        "n_obs": n_obs,
                        "n_train": n_train,
                        "n_held": n_held,
                    }
                )
                return nll, crps

            t0 = time.perf_counter()
            st = _run(filt, jnp.arange(n_obs + 1))
            jax.block_until_ready(st.mean)
            nll_w, crps_w = score(
                "WSC-KF",
                st.log_normalizing_constant,
                time.perf_counter() - t0,
                budget=n_paths,
                pred_mu=st.pred_obs_mean,
                pred_std=st.pred_obs_std,
            )

            t0 = time.perf_counter()
            ekf = run_cd_ekf(
                drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, dt0,
                state_order="first", solver="rk4",
            )
            jax.block_until_ready(ekf.means)
            nll_e, crps_e = score(
                "EKF",
                ekf.cum_loglik,
                time.perf_counter() - t0,
                pred_mu=ekf.pred_obs_mean,
                pred_std=ekf.pred_obs_std,
            )

            t0 = time.perf_counter()
            ukf = run_cd_ukf(
                drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, dt0,
                UKFParams(), state_order="first", solver="rk4",
            )
            jax.block_until_ready(ukf.means)
            nll_u, crps_u = score(
                "UKF",
                ukf.cum_loglik,
                time.perf_counter() - t0,
                pred_mu=ukf.pred_obs_mean,
                pred_std=ukf.pred_obs_std,
            )

            key = jr.PRNGKey(args.seed + 17 * sid + int(10 * scale))
            extra = ""
            if args.enkf > 0:
                t0 = time.perf_counter()
                enkf = run_cd_enkf(
                    drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, key,
                    n_particles=int(args.enkf), noise_dim=2, dt0=dt0, solver="rk4",
                )
                jax.block_until_ready(enkf.means)
                _, crps_n = score(
                    "EnKF",
                    enkf.cum_loglik,
                    time.perf_counter() - t0,
                    budget=int(args.enkf),
                    pred_mu=enkf.pred_obs_mean,
                    pred_std=enkf.pred_obs_std,
                )
                extra += f"  EnKF={crps_n:.2f}"

            if args.pf > 0:
                t0 = time.perf_counter()
                pf = run_bootstrap_pf(
                    drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, key,
                    n_particles=int(args.pf), noise_dim=2, dt0=dt0, solver="rk4",
                )
                jax.block_until_ready(pf.means)
                _, crps_p = score(
                    "PF",
                    pf.cum_loglik,
                    time.perf_counter() - t0,
                    budget=int(args.pf),
                    pred_mu=pf.pred_obs_mean,
                    pred_std=pf.pred_obs_std,
                    pred_crps=pf.pred_crps,
                )
                extra += f"  PF={crps_p:.2f}"

            print(
                f"  S{sid:02d} ×{scale:g}: WSC={crps_w:.2f} EKF={crps_e:.2f} "
                f"UKF={crps_u:.2f}{extra}  (NLL W/E={nll_w:.3f}/{nll_e:.3f})"
            )

    if not rows:
        raise SystemExit("no rows")

    tag = str(args.out_suffix).strip().lstrip("_")
    tag = f"_{tag}" if tag else ""
    csv_path = os.path.join(OUTPUT_DIR, f"crps_vs_sigma_g_scale{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # Aggregate mean ± SE over subjects
    filters = []
    for r in rows:
        if r["filter"] not in filters:
            filters.append(r["filter"])
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.55))
    for fam in filters:
        xs, ys, yerr = [], [], []
        for sc in scales:
            sub = [
                float(r["crps"])
                for r in rows
                if r["filter"] == fam and abs(float(r["scale"]) - sc) < 1e-12
            ]
            if not sub:
                continue
            arr = np.asarray(sub, dtype=float)
            xs.append(sc)
            ys.append(float(np.mean(arr)))
            yerr.append(
                float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            )
        ax.errorbar(
            xs, ys, yerr=yerr, color=METHOD_COLORS[fam],
            marker=METHOD_MARKERS[fam], markerfacecolor="white",
            markeredgewidth=0.8, ms=4.2, lw=1.25, capsize=2.0,
            elinewidth=0.7, label=fam,
        )
    ax.set_xlabel(r"Process-noise scale $\sigma_G / \hat\sigma_G$")
    ax.set_ylabel("Held-out CRPS [mg/dL]")
    ax.set_yscale("log")
    style_curve_axis(ax, grid_axis="both")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(3, len(filters)),
    )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.34, top=0.97)
    png = os.path.join(OUTPUT_DIR, f"crps_vs_sigma_g_scale{tag}.png")
    out_png, out_pdf = save_publication_figure(fig, png)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")

    # Print slope-ish summary: ΔCRPS from scale 1 → max
    print("\nΔCRPS (max scale − scale 1), mean over subjects:")
    sc0, sc1 = scales[0], scales[-1]
    for fam in filters:
        dlist = []
        for sid in subjects:
            a = [
                float(r["crps"])
                for r in rows
                if r["filter"] == fam and int(r["subject"]) == sid and abs(float(r["scale"]) - sc0) < 1e-12
            ]
            b = [
                float(r["crps"])
                for r in rows
                if r["filter"] == fam and int(r["subject"]) == sid and abs(float(r["scale"]) - sc1) < 1e-12
            ]
            if a and b:
                dlist.append(b[0] - a[0])
        if dlist:
            print(f"  {fam:7s}  ΔCRPS={np.mean(dlist):+.2f}  (±{np.std(dlist, ddof=1)/np.sqrt(len(dlist)):.2f} SE)")


if __name__ == "__main__":
    main()
