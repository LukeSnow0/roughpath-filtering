"""MAPE vs multiplicative disturbance intensity on Nielsen MFTS.

Always uses state-dependent diffusion
  g(F) = σ √(1 + (F / F_c)²)
on (F3, F4). Sweeps a scale on σ (truth + filters share the same σ) at fixed
WSC-KF path count (=4 by default) and compares to moment-ODE / ensemble filters.

Hypothesis: as σ grows, freeze-at-mean EKF/UKF/CKF degrade faster than WSC@4
(and EnKF/PF) on MAPE — especially MAPEd on the disturbance channels.

Outputs under ``output/``:
  mape_vs_sigma_mult{_tag}.png
  mape_vs_sigma_mult{_tag}.csv
  mape_vs_sigma_mult{_tag}_raw.csv

Example:
  uv run --extra experiments python experiments/mfts_test/run_mape_vs_sigma_mult.py
  uv run --extra experiments python experiments/mfts_test/run_mape_vs_sigma_mult.py \\
      --scales 1,1.5,2,3,4 --seeds 100,101,102,103,104 --enkf 100 --pf 100
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
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
for pth in (HERE, EXPERIMENTS_DIR, ROOT_DIR):
    sys.path.insert(0, pth)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import jit, random

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

from mfts_model import (
    MFTSParams,
    init_estimate,
    make_mfts_dynamics,
    mape_pct,
    simulate_mfts,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)

COLORS = {
    "WSC-KF-4": "#6A4C93",
    "WSC-KF-9": "#9B6BC9",
    "EKF": "#2E86AB",
    "UKF": "#E8B923",
    "CKF": "#3DA35D",
    "EnKF": "#D97706",
    "PF": "#F24236",
}

PATH_TO_CFG = {
    4: (3, 1),
    9: (5, 1),
    16: (3, 2),
    64: (3, 3),
    81: (5, 2),
}


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def mape_at_obs(
    m_filt: np.ndarray,
    true_fine: np.ndarray,
    stride: int,
    state_idx: slice,
) -> float:
    """MAPE of filter means vs truth subsampled at observation times (incl. t0)."""
    true_obs = true_fine[::stride]
    if true_obs.shape[0] != m_filt.shape[0]:
        # fine grid is N_FINE+1; obs means are n_obs+1 with samples at stride indices
        true_obs = true_fine[np.arange(0, true_fine.shape[0], stride)]
    n = min(m_filt.shape[0], true_obs.shape[0])
    return mape_pct(m_filt[:n], true_obs[:n], state_idx)


def stack_estimates(m0_arr, means):
    return np.vstack([np.asarray(m0_arr), np.asarray(means)])


def main():
    ap = argparse.ArgumentParser(
        description="MFTS MAPE vs multiplicative σ scale (WSC@4 vs others)"
    )
    ap.add_argument(
        "--scales",
        type=str,
        default="1,1.5,2,3,4",
        help="multipliers on base σ (truth + filters)",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="100,101,102,103,104",
        help="truth/filter seeds",
    )
    ap.add_argument("--sigma0", type=float, default=5.0, help="base σ at scale=1")
    ap.add_argument(
        "--fc",
        type=float,
        default=150.0,
        help="F_c in g(F)=σ√(1+(F/Fc)²); lower → stronger state-dependence",
    )
    ap.add_argument("--delta", type=float, default=1.0, help="obs interval [min]")
    ap.add_argument("--eval-dt", type=float, default=0.05, dest="eval_dt")
    ap.add_argument("--horizon", type=float, default=30.0, help="horizon [min]")
    ap.add_argument(
        "--wsckf-paths",
        type=str,
        default="4,9",
        dest="wsckf_paths",
        help="comma-separated product path counts (e.g. 4,9)",
    )
    ap.add_argument("--enkf", type=int, default=100, help="EnKF particles; 0=skip")
    ap.add_argument("--pf", type=int, default=100, help="PF particles; 0=skip")
    ap.add_argument("--solver", type=str, default="rk4")
    ap.add_argument("--out-suffix", type=str, default="", dest="out_suffix")
    args = ap.parse_args()

    scales = _parse_floats(args.scales)
    seeds = _parse_ints(args.seeds)
    if not scales or not seeds:
        raise SystemExit("need nonempty --scales and --seeds")

    wsckf_path_list = _parse_ints(args.wsckf_paths)
    wsckf_cfgs: list[tuple[int, int, int]] = []  # (n_paths, lyons_degree, cubature_order)
    for n_paths in wsckf_path_list:
        if n_paths not in PATH_TO_CFG:
            raise SystemExit(
                f"--wsckf-paths entry {n_paths} not in product ladder "
                f"{sorted(PATH_TO_CFG)}"
            )
        lyons_degree, cubature_order = PATH_TO_CFG[n_paths]
        n_expect = product_path_count(2, cubature_order, lyons_degree=lyons_degree)
        if n_expect != n_paths:
            raise SystemExit(f"config mismatch: expect {n_expect} paths for m,k")
        wsckf_cfgs.append((n_paths, lyons_degree, cubature_order))
    if not wsckf_cfgs:
        raise SystemExit("need at least one --wsckf-paths entry")

    HORIZON_MIN = float(args.horizon)
    DT_TRUTH = 0.015
    DT_FILTER = 0.15
    LAMBDA_SIM = 0.1
    LAMBDA_EKF_UKF = 0.0
    LAMBDA_ENKF_PF = 2.0e-3
    F_NOISE_SCALE = float(args.fc)
    SIGMA0 = float(args.sigma0)
    F1, F2 = 5.0, 5.0
    DELTA_T_MIN = float(args.delta)
    EVAL_DT_MIN = float(args.eval_dt)
    SOLVER = args.solver

    # Match run_pareto_budget: redefine fine dt from integer step count so
    # Δt=1 with eval-dt=0.05 tiles cleanly.
    N_FINE = int(round(HORIZON_MIN / EVAL_DT_MIN))
    DT_FINE_MIN = HORIZON_MIN / N_FINE
    DT_FINE_SEC = DT_FINE_MIN * 60.0
    N_SUBSTEPS_TRUTH_FINE = max(1, int(round(DT_FINE_SEC / DT_TRUTH)))
    stride = int(round(DELTA_T_MIN / DT_FINE_MIN))
    if stride < 1 or not np.isclose(
        stride * DT_FINE_MIN, DELTA_T_MIN, rtol=0.0, atol=1e-12
    ):
        raise SystemExit(
            f"--delta={DELTA_T_MIN} must be an integer multiple of "
            f"fine dt={DT_FINE_MIN} (from horizon/eval-dt)"
        )
    n_obs = N_FINE // stride
    if n_obs * stride != N_FINE:
        raise SystemExit("delta does not tile fine grid")
    dt_min = stride * DT_FINE_MIN
    dt = dt_min * 60.0
    n_substeps_filter = max(1, int(round(dt / DT_FILTER)))
    dt0 = dt / n_substeps_filter
    model_inputs = jnp.arange(n_obs + 1)

    p = MFTSParams()
    bar_est = lambda t_min, nom=p.F_nominal_est: nom

    tag = str(args.out_suffix).strip().lstrip("_")
    tag = f"_{tag}" if tag else ""

    print(
        f"MFTS MAPE vs σ×  multiplicative g(F), Fc={F_NOISE_SCALE:g}, "
        f"σ0={SIGMA0:g}, Δt={dt_min:g} min, n_obs={n_obs}, "
        f"WSC paths={[n for n, _, _ in wsckf_cfgs]}, seeds={seeds}, scales={scales}",
        flush=True,
    )

    raw_rows: list[dict] = []

    def score_row(
        *,
        scale: float,
        sigma: float,
        seed: int,
        name: str,
        budget: int,
        m_filt: np.ndarray,
        final_ll: float,
        wall_s: float,
        true_fine: np.ndarray,
    ):
        rec = {
            "scale": scale,
            "sigma": sigma,
            "fc": F_NOISE_SCALE,
            "seed": seed,
            "filter": name,
            "budget": budget,
            "mapex": mape_at_obs(m_filt, true_fine, stride, slice(0, 4)),
            "maped": mape_at_obs(m_filt, true_fine, stride, slice(4, 6)),
            "nll_per_obs": -final_ll / n_obs,
            "time_s": wall_s,
        }
        raw_rows.append(rec)
        print(
            f"  σ×{scale:g} seed={seed}  {name:8s} n={budget:4d}  "
            f"MAPEx={rec['mapex']:6.2f}  MAPEd={rec['maped']:6.2f}  "
            f"t={1e3 * wall_s:6.1f} ms",
            flush=True,
        )

    for scale in scales:
        sigma = SIGMA0 * float(scale)
        dyn_kw = dict(
            noise="multiplicative",
            F_noise_scale=F_NOISE_SCALE,
            F1=F1,
            F2=F2,
        )
        drift_ekf, L_ekf, Qc, obs_fn, _ = make_mfts_dynamics(
            p,
            lambda_dist=LAMBDA_EKF_UKF,
            sigma_dist=sigma,
            bar_F3=bar_est,
            bar_F4=bar_est,
            **dyn_kw,
        )
        # Shared σ for UKF/CKF under multiplicative (fair vs path filters).
        drift_ukf, L_ukf, _, _, _ = make_mfts_dynamics(
            p,
            lambda_dist=LAMBDA_EKF_UKF,
            sigma_dist=sigma,
            bar_F3=bar_est,
            bar_F4=bar_est,
            **dyn_kw,
        )
        drift_ens, L_ens, _, _, _ = make_mfts_dynamics(
            p,
            lambda_dist=LAMBDA_ENKF_PF,
            sigma_dist=sigma,
            bar_F3=bar_est,
            bar_F4=bar_est,
            **dyn_kw,
        )

        for seed in seeds:
            print(f"\n=== scale×{scale:g}  σ={sigma:g}  seed {seed} ===")
            true_fine, ys_fine_np, R = simulate_mfts(
                horizon_min=HORIZON_MIN,
                n_obs=N_FINE,
                n_substeps_truth=N_SUBSTEPS_TRUTH_FINE,
                p=p,
                lambda_sim=LAMBDA_SIM,
                sigma_sim=sigma,
                F1=F1,
                F2=F2,
                seed=seed,
                noise="multiplicative",
                F_noise_scale=F_NOISE_SCALE,
            )
            ys_fine = jnp.asarray(ys_fine_np)
            ys = ys_fine[stride - 1 :: stride]
            assert ys.shape[0] == n_obs

            x0, P0 = init_estimate(p, F1, F2, true_state=true_fine[0], cov_scale=1.0)
            m0 = jnp.asarray(x0, dtype=jnp.float64)
            P0_j = jnp.asarray(P0, dtype=jnp.float64)
            chol_P0 = jnp.linalg.cholesky(P0_j)
            chol_R = jnp.linalg.cholesky(R)

            # --- WSC-KF (product ladder path counts) ---
            for n_paths, lyons_degree, cubature_order in wsckf_cfgs:
                filt = build_filter(
                    get_init_params=lambda mi: (m0, chol_P0),
                    get_dynamics=lambda mi: (
                        lambda t, x: drift_ens(t, x),
                        lambda t, x: L_ens(t, x),
                    ),
                    get_observations=lambda mi: (obs_fn, chol_R, ys[mi - 1]),
                    noise_dim=2,
                    dt=dt,
                    integration_dt=dt0,
                    cubature_order=cubature_order,
                    lyons_degree=lyons_degree,
                    multidim_paths="product",
                )
                jitted = jit(run_filter, static_argnames=["filter_obj"])
                t0 = time.perf_counter()
                st = jitted(filt, model_inputs)
                jax.block_until_ready(st.mean)
                wall = time.perf_counter() - t0
                score_row(
                    scale=scale,
                    sigma=sigma,
                    seed=seed,
                    name=f"WSC-KF-{n_paths}",
                    budget=n_paths,
                    m_filt=stack_estimates(m0, st.mean[1:]),
                    final_ll=float(st.log_normalizing_constant[-1]),
                    wall_s=wall,
                    true_fine=true_fine,
                )

            t0 = time.perf_counter()
            ekf = run_cd_ekf(
                drift_ekf,
                obs_fn,
                L_ekf,
                Qc,
                R,
                m0,
                P0_j,
                ys,
                dt,
                dt0,
                state_order="first",
                solver=SOLVER,
            )
            jax.block_until_ready(ekf.means)
            score_row(
                scale=scale,
                sigma=sigma,
                seed=seed,
                name="EKF",
                budget=0,
                m_filt=stack_estimates(m0, ekf.means),
                final_ll=float(ekf.cum_loglik[-1]),
                wall_s=time.perf_counter() - t0,
                true_fine=true_fine,
            )

            t0 = time.perf_counter()
            ukf = run_cd_ukf(
                drift_ukf,
                obs_fn,
                L_ukf,
                Qc,
                R,
                m0,
                P0_j,
                ys,
                dt,
                dt0,
                UKFParams(),
                state_order="first",
                solver=SOLVER,
            )
            jax.block_until_ready(ukf.means)
            score_row(
                scale=scale,
                sigma=sigma,
                seed=seed,
                name="UKF",
                budget=0,
                m_filt=stack_estimates(m0, ukf.means),
                final_ll=float(ukf.cum_loglik[-1]),
                wall_s=time.perf_counter() - t0,
                true_fine=true_fine,
            )

            t0 = time.perf_counter()
            ckf = run_cd_ukf(
                drift_ukf,
                obs_fn,
                L_ukf,
                Qc,
                R,
                m0,
                P0_j,
                ys,
                dt,
                dt0,
                CKF_PARAMS,
                state_order="first",
                solver=SOLVER,
            )
            jax.block_until_ready(ckf.means)
            score_row(
                scale=scale,
                sigma=sigma,
                seed=seed,
                name="CKF",
                budget=0,
                m_filt=stack_estimates(m0, ckf.means),
                final_ll=float(ckf.cum_loglik[-1]),
                wall_s=time.perf_counter() - t0,
                true_fine=true_fine,
            )

            key = random.key(seed + 17 * int(1000 * scale))
            if args.enkf > 0:
                key, sub = random.split(key)
                t0 = time.perf_counter()
                enkf = run_cd_enkf(
                    drift_ens,
                    obs_fn,
                    L_ens,
                    Qc,
                    R,
                    m0,
                    P0_j,
                    ys,
                    dt,
                    sub,
                    n_particles=int(args.enkf),
                    noise_dim=2,
                    dt0=dt0,
                    solver=SOLVER,
                )
                jax.block_until_ready(enkf.means)
                score_row(
                    scale=scale,
                    sigma=sigma,
                    seed=seed,
                    name="EnKF",
                    budget=int(args.enkf),
                    m_filt=stack_estimates(m0, enkf.means),
                    final_ll=float(enkf.cum_loglik[-1]),
                    wall_s=time.perf_counter() - t0,
                    true_fine=true_fine,
                )

            if args.pf > 0:
                key, sub = random.split(key)
                t0 = time.perf_counter()
                pf = run_bootstrap_pf(
                    drift_ens,
                    obs_fn,
                    L_ens,
                    Qc,
                    R,
                    m0,
                    P0_j,
                    ys,
                    dt,
                    sub,
                    n_particles=int(args.pf),
                    noise_dim=2,
                    dt0=dt0,
                    solver=SOLVER,
                )
                jax.block_until_ready(pf.means)
                score_row(
                    scale=scale,
                    sigma=sigma,
                    seed=seed,
                    name="PF",
                    budget=int(args.pf),
                    m_filt=stack_estimates(m0, pf.means),
                    final_ll=float(pf.cum_loglik[-1]),
                    wall_s=time.perf_counter() - t0,
                    true_fine=true_fine,
                )

    if not raw_rows:
        raise SystemExit("no rows")

    raw_path = os.path.join(OUTPUT_DIR, f"mape_vs_sigma_mult{tag}_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    print(f"Wrote {raw_path}")

    # Aggregate mean ± std over seeds
    filters: list[str] = []
    for r in raw_rows:
        if r["filter"] not in filters:
            filters.append(r["filter"])

    summary: list[dict] = []
    for fam in filters:
        for sc in scales:
            sub = [
                r
                for r in raw_rows
                if r["filter"] == fam and abs(float(r["scale"]) - sc) < 1e-12
            ]
            if not sub:
                continue
            mapex = np.asarray([r["mapex"] for r in sub], dtype=float)
            maped = np.asarray([r["maped"] for r in sub], dtype=float)
            nll = np.asarray([r["nll_per_obs"] for r in sub], dtype=float)
            summary.append(
                {
                    "scale": sc,
                    "sigma": SIGMA0 * sc,
                    "fc": F_NOISE_SCALE,
                    "filter": fam,
                    "budget": int(sub[0]["budget"]),
                    "n_seeds": len(sub),
                    "mapex_mean": float(np.mean(mapex)),
                    "mapex_std": float(np.std(mapex, ddof=1)) if len(sub) > 1 else 0.0,
                    "maped_mean": float(np.mean(maped)),
                    "maped_std": float(np.std(maped, ddof=1)) if len(sub) > 1 else 0.0,
                    "nll_mean": float(np.mean(nll)),
                    "nll_std": float(np.std(nll, ddof=1)) if len(sub) > 1 else 0.0,
                }
            )

    csv_path = os.path.join(OUTPUT_DIR, f"mape_vs_sigma_mult{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {csv_path}")

    def _series(fam: str, metric: str):
        xs, ys, yerr = [], [], []
        for sc in scales:
            row = next(
                (
                    s
                    for s in summary
                    if s["filter"] == fam and abs(float(s["scale"]) - sc) < 1e-12
                ),
                None,
            )
            if row is None:
                continue
            xs.append(sc)
            ys.append(row[f"{metric}_mean"])
            yerr.append(0.3 * row[f"{metric}_std"])  # match Pareto plot band style
        return xs, ys, yerr

    def _finite_mask(ys, yerr):
        ys = np.asarray(ys, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        ok = np.isfinite(ys) & np.isfinite(yerr) & (ys < 1e3)
        return ok

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained", sharex=True)
    for ax, metric, ylab in zip(
        axes,
        ("mapex", "maped"),
        ("MAPEx [%] (tanks)", "MAPEd [%] (disturbances)"),
    ):
        for fam in filters:
            xs, ys, yerr = _series(fam, metric)
            if not xs:
                continue
            xs_a = np.asarray(xs, dtype=float)
            ok = _finite_mask(ys, yerr)
            if not np.any(ok):
                continue
            is_wsc = fam.startswith("WSC-KF")
            ax.errorbar(
                xs_a[ok],
                np.asarray(ys)[ok],
                yerr=np.asarray(yerr)[ok],
                color=COLORS.get(fam, "0.3"),
                marker="*" if is_wsc else "s",
                ms=11 if is_wsc else 7,
                lw=1.7,
                capsize=3,
                label=fam,
            )
        ax.set_xlabel(r"disturbance-noise scale $\sigma / \sigma_0$")
        ax.set_ylabel(ylab)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        rf"MFTS: obs-time MAPE vs multiplicative $\sigma$ "
        rf"($g(F)=\sigma\sqrt{{1+(F/F_c)^2}}$, $F_c$={F_NOISE_SCALE:g}, "
        rf"$\sigma_0$={SIGMA0:g}, $\Delta t$={dt_min:g} min, "
        rf"{len(seeds)} seeds; band = $0.3\sigma$)",
        fontsize=11,
    )
    png = os.path.join(OUTPUT_DIR, f"mape_vs_sigma_mult{tag}.png")
    fig.savefig(png, dpi=170)
    plt.close(fig)
    print(f"Wrote {png}")

    # ΔMAPE from scale min → max (mean over seeds, paired)
    sc0, sc1 = scales[0], scales[-1]
    print(f"\nΔMAPE (scale {sc1:g} − scale {sc0:g}), mean over seeds:")
    for fam in filters:
        for metric, label in (("mapex", "MAPEx"), ("maped", "MAPEd")):
            dlist = []
            for seed in seeds:
                a = [
                    float(r[metric])
                    for r in raw_rows
                    if r["filter"] == fam
                    and int(r["seed"]) == seed
                    and abs(float(r["scale"]) - sc0) < 1e-12
                ]
                b = [
                    float(r[metric])
                    for r in raw_rows
                    if r["filter"] == fam
                    and int(r["seed"]) == seed
                    and abs(float(r["scale"]) - sc1) < 1e-12
                ]
                if a and b:
                    dlist.append(b[0] - a[0])
            if dlist:
                se = (
                    float(np.std(dlist, ddof=1) / np.sqrt(len(dlist)))
                    if len(dlist) > 1
                    else 0.0
                )
                print(f"  {fam:7s}  Δ{label}={np.mean(dlist):+.2f}  (±{se:.2f} SE)")


if __name__ == "__main__":
    main()
