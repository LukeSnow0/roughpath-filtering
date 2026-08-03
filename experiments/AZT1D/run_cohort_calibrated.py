"""Cohort analysis: per-subject EnKF θ̂_i, then filter bakeoff aggregated over patients.

Protocol
--------
For each AZT1D subject ``i``:
  1. Adam EnKF-MLE of ``(p₃, σ_G)`` on that subject's train prefix → ``θ̂_i``.
  2. Run WSC-KF / EKF / UKF / CKF / EnKF / PF at ``θ̂_i`` (no per-filter refit).
  3. Record held-out self-NLL/obs and MAPE.

Aggregate mean ± SE across subjects and plot bar charts.

This is the natural multi-patient extension of the single-subject calibrated bakeoff:
patient-specific calibration, then population summary of filter performance.

Run:
  uv run --extra experiments python experiments/AZT1D/run_cohort_calibrated.py
  uv run --extra experiments python experiments/AZT1D/run_cohort_calibrated.py \\
      --n-obs 2016 --mle-iters 25 --subjects 1,2,3

Outputs under ``experiments/AZT1D/output/``:
  ``cohort_calibrated_per_subject.csv``, ``cohort_calibrated_summary.csv``,
  ``cohort_calibrated_bakeoff.png``, ``cohort_calibrated_params.json``
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass

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
from jax import jit, value_and_grad

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from cuthbert import filter as run_filter
from crf import build_filter
from crf.library import product_path_count
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
    heldout_crps_per_obs,
    heldout_mean_score,
    init_estimate,
    make_glucose_dynamics,
    mape_glucose,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)

_parser = argparse.ArgumentParser(description="AZT1D cohort: per-subject θ̂_i + filter bakeoff")
_parser.add_argument(
    "--subjects",
    type=str,
    default="",
    help="comma-separated subject ids (default: all)",
)
_parser.add_argument("--n-obs", type=int, default=2016, dest="n_obs", help="CGM samples on 5-min grid (~7d)")
_parser.add_argument(
    "--dt-assim",
    type=float,
    default=5.0,
    dest="dt_assim",
    help="assimilation interval [min]; subsample every dt_assim/5 from the 5-min grid",
)
_parser.add_argument("--holdout-frac", type=float, default=0.3, dest="holdout_frac")
_parser.add_argument("--mle-iters", type=int, default=25, dest="mle_iters")
_parser.add_argument("--lr", type=float, default=0.1)
_parser.add_argument("--p3-init", type=float, default=5e-4, dest="p3_init")
_parser.add_argument("--sigma-g-init", type=float, default=1.5, dest="sigma_g_init")
_parser.add_argument("--dt0", type=float, default=1.0)
_parser.add_argument(
    "--crf-paths",
    type=str,
    default="4",
    dest="crf_paths",
    help="comma-separated product path counts (matched to m,k ladder)",
)
_parser.add_argument(
    "--enkf",
    type=str,
    default="25,500,1000",
    help="comma-separated EnKF particle counts",
)
_parser.add_argument(
    "--pf",
    type=str,
    default="25,100,500",
    help="comma-separated PF particle counts",
)
_parser.add_argument("--mle-enkf-particles", type=int, default=100, dest="mle_enkf_particles")
_parser.add_argument("--solver", type=str, default="rk4", choices=("rk4", "diffrax"))
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--skip-ukf", action="store_true")
_parser.add_argument(
    "--with-ckf",
    action="store_true",
    help="include CD-CKF in the bakeoff (omitted by default)",
)
_parser.add_argument("--skip-pf", action="store_true")
_parser.add_argument("--meas-std", type=float, default=8.0, dest="meas_std")
_parser.add_argument(
    "--replace-pareto-plot",
    action="store_true",
    help="also write cohort Pareto to pareto_assimilation.png",
)
_args = _parser.parse_args()


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
    n_paths: int
    k: int
    lyons_degree: int


def product_configs_for_paths(wanted: list[int], noise_dim: int = 2) -> list[CrfConfig]:
    """Map requested path counts onto the standard product (m,k) ladder."""
    specs = [(3, 1), (5, 1), (3, 2), (3, 3), (5, 2), (3, 4), (5, 3)]
    by_n: dict[int, CrfConfig] = {}
    for m, k in specs:
        n = product_path_count(noise_dim, k, lyons_degree=m)
        if n not in by_n:
            by_n[n] = CrfConfig(label=f"prod-m{m}-k{k}", n_paths=n, k=k, lyons_degree=m)
    out: list[CrfConfig] = []
    missing: list[int] = []
    for n in wanted:
        if n in by_n:
            out.append(by_n[n])
        else:
            missing.append(n)
    if missing:
        avail = sorted(by_n)
        raise SystemExit(
            f"product path counts {missing} not available; choose from {avail}"
        )
    return out


if _args.subjects.strip():
    SUBJECTS = _parse_ints(_args.subjects)
else:
    SUBJECTS = sorted(list_subjects())

CRF_CONFIGS = product_configs_for_paths(_parse_ints(_args.crf_paths))
ENKF_LIST = _parse_ints(_args.enkf)
PF_LIST = _parse_ints(_args.pf)
MLE_ENKF_PARTICLES = int(_args.mle_enkf_particles)
SOLVER = _args.solver
P_BASE = GlucoseParams(meas_std=float(_args.meas_std))
BASE_DT = 5.0
DT_ASSIM = float(_args.dt_assim)
if DT_ASSIM < BASE_DT - 1e-9:
    raise SystemExit(f"--dt-assim must be ≥ {BASE_DT:g} min")
STRIDE = int(round(DT_ASSIM / BASE_DT))
if abs(STRIDE * BASE_DT - DT_ASSIM) > 1e-6:
    raise SystemExit(
        f"--dt-assim={DT_ASSIM:g} must be an integer multiple of {BASE_DT:g} min"
    )
OUT_TAG = "" if abs(DT_ASSIM - BASE_DT) < 1e-9 else f"_dt{int(round(DT_ASSIM))}"
if not CRF_CONFIGS:
    raise SystemExit("--crf-paths must be non-empty")
if not ENKF_LIST:
    raise SystemExit("--enkf must be non-empty")
if not _args.skip_pf and not PF_LIST:
    raise SystemExit("--pf must be non-empty unless --skip-pf")


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


def run_subject(subject: int) -> tuple[dict, list[dict]]:
    series = load_azt1d_series(subject=subject, max_obs=int(_args.n_obs), dt_min=BASE_DT)
    # Keep fine insulin/carb for continuous input interp; assimilate every STRIDE-th CGM.
    idx = np.arange(0, int(series.cgm.shape[0]), STRIDE)
    cgm = np.asarray(series.cgm, dtype=np.float64)[idx]
    YS = jnp.asarray(cgm.reshape(-1, 1))
    N_OBS = int(YS.shape[0])
    min_obs = 40 if DT_ASSIM > BASE_DT + 1e-9 else 100
    if N_OBS < min_obs:
        raise RuntimeError(f"subject {subject}: too few obs ({N_OBS})")
    if not (0.05 <= _args.holdout_frac <= 0.5):
        raise SystemExit("--holdout-frac must be in [0.05, 0.5]")
    N_HELD = max(1, int(round(N_OBS * _args.holdout_frac)))
    N_TRAIN = N_OBS - N_HELD
    min_train = 20 if DT_ASSIM > BASE_DT + 1e-9 else 40
    if N_TRAIN < min_train:
        raise RuntimeError(f"subject {subject}: train too short ({N_TRAIN})")
    YS_TRAIN = YS[:N_TRAIN]
    DT = float(DT_ASSIM)
    DT0 = float(_args.dt0)

    P = P_BASE
    m0_np, P0_np = init_estimate(cgm[:N_TRAIN], P)
    M0 = jnp.asarray(m0_np)
    P0 = jnp.asarray(P0_np)
    CHOL_P0 = jnp.linalg.cholesky(P0)
    _, _, _, OBS_FN, R = make_glucose_dynamics(
        P,
        t_min=series.t_min,
        insulin_u_per_min=series.insulin_u_per_min,
        carb_g_per_min=series.carb_g_per_min,
    )
    CHOL_R = jnp.linalg.cholesky(R)

    def _dyn(p3, sigma_g):
        return make_glucose_dynamics(
            P,
            t_min=series.t_min,
            insulin_u_per_min=series.insulin_u_per_min,
            carb_g_per_min=series.carb_g_per_min,
            p3=p3,
            sigma_g=sigma_g,
        )

    def enkf_ll_train(p3, sigma_g, key):
        drift, L, Qc, _, _ = _dyn(p3, sigma_g)
        return run_cd_enkf(
            drift, OBS_FN, L, Qc, R, M0, P0, YS_TRAIN, DT, key,
            n_particles=MLE_ENKF_PARTICLES, noise_dim=2, dt0=DT0, solver=SOLVER,
        ).cum_loglik[-1]

    key_ad = jr.PRNGKey(_args.seed + 17 * subject)
    vg = jit(
        value_and_grad(
            lambda th: enkf_ll_train(jnp.exp(th[0]), jnp.exp(th[1]), key_ad)
        )
    )
    th = np.array([np.log(_args.p3_init), np.log(_args.sigma_g_init)], dtype=float)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(2)
    v = np.zeros(2)
    p3_hat = float(_args.p3_init)
    sg_hat = float(_args.sigma_g_init)
    for t in range(1, _args.mle_iters + 1):
        val, g = vg(jnp.asarray(th))
        ll = float(val)
        g = np.asarray(g, dtype=float)
        if not np.isfinite(ll) or not np.all(np.isfinite(g)):
            break
        p3_hat = float(np.exp(th[0]))
        sg_hat = float(np.exp(th[1]))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        th = th + _args.lr * mhat / (np.sqrt(vhat) + eps)
    # final point
    p3_hat = float(np.exp(th[0]))
    sg_hat = float(np.exp(th[1]))

    params = {
        "subject": subject,
        "p3": p3_hat,
        "sigma_g": sg_hat,
        "dt_assim": DT_ASSIM,
        "n_obs": N_OBS,
        "n_train": N_TRAIN,
        "n_held": N_HELD,
        "span_days": float(series.span_days),
    }

    drift, L, Qc, _, _ = _dyn(p3_hat, sg_hat)

    def build_crf(cfg: CrfConfig):
        def get_init_params(_mi):
            return M0, CHOL_P0

        def get_dynamics(_mi):
            return drift, L

        def get_observations(mi):
            idx = mi if not hasattr(mi, "t_idx") else mi.t_idx
            return OBS_FN, CHOL_R, YS[idx - 1]

        return build_filter(
            get_init_params=get_init_params,
            get_dynamics=get_dynamics,
            get_observations=get_observations,
            noise_dim=2,
            dt=DT,
            integration_dt=DT0,
            cubature_order=cfg.k,
            lyons_degree=cfg.lyons_degree,
            num_paths=None,
            multidim_paths="product",
            sde_form="ito",
            checkpoint_steps=False,
        )

    _run = jit(run_filter, static_argnames=["filter_obj"])
    rows: list[dict] = []
    mi = jnp.arange(N_OBS + 1)

    def record(
        name: str,
        cum_ll,
        means,
        wall_s: float,
        budget: int = 0,
        label: str = "",
        *,
        pred_mu=None,
        pred_std=None,
        pred_crps=None,
        crps_kind: str = "gaussian",
    ):
        means_np = np.asarray(means)
        g = means_np if means_np.ndim == 1 else means_np[:, 0]
        if g.shape[0] == N_OBS + 1:
            g = g[1:]
        nll = heldout_nll_per_obs(cum_ll, N_TRAIN, N_OBS)
        mape = mape_glucose(g, cgm)
        if pred_crps is not None:
            crps = heldout_mean_score(pred_crps, N_TRAIN, N_OBS)
        elif pred_mu is not None and pred_std is not None:
            crps = heldout_crps_per_obs(cgm, pred_mu, pred_std, N_TRAIN)
        else:
            crps = float("nan")
        rows.append(
            {
                "subject": subject,
                "filter": name,
                "budget": budget,
                "label": label or (name if budget == 0 else f"{name}-{budget}"),
                "p3": p3_hat,
                "sigma_g": sg_hat,
                "held_nll": nll,
                "crps": crps,
                "crps_kind": crps_kind,
                "mape": mape,
                "time_s": wall_s,
                "n_obs": N_OBS,
                "n_train": N_TRAIN,
                "n_held": N_HELD,
            }
        )
        print(
            f"  S{subject:02d} {name:6s} n={budget:4d}  "
            f"NLL={nll:.4f}  CRPS={crps:.3f}  MAPE={mape:.2f}%  t={1e3*wall_s:.0f}ms"
        )

    for cfg in CRF_CONFIGS:
        filt = build_crf(cfg)
        t0 = time.perf_counter()
        st = _run(filt, mi)
        jax.block_until_ready(st.mean)
        record(
            "WSC-KF",
            st.log_normalizing_constant,
            st.mean,
            time.perf_counter() - t0,
            budget=cfg.n_paths,
            label=cfg.label,
            pred_mu=st.pred_obs_mean,
            pred_std=st.pred_obs_std,
        )

    t0 = time.perf_counter()
    ekf = run_cd_ekf(
        drift, OBS_FN, L, Qc, R, M0, P0, YS, DT, DT0,
        state_order="first", solver=SOLVER,
    )
    jax.block_until_ready(ekf.means)
    record(
        "EKF", ekf.cum_loglik, ekf.means, time.perf_counter() - t0,
        pred_mu=ekf.pred_obs_mean, pred_std=ekf.pred_obs_std,
    )

    if not _args.skip_ukf:
        t0 = time.perf_counter()
        ukf = run_cd_ukf(
            drift, OBS_FN, L, Qc, R, M0, P0, YS, DT, DT0,
            UKFParams(), state_order="first", solver=SOLVER,
        )
        jax.block_until_ready(ukf.means)
        record(
            "UKF", ukf.cum_loglik, ukf.means, time.perf_counter() - t0,
            pred_mu=ukf.pred_obs_mean, pred_std=ukf.pred_obs_std,
        )

    if _args.with_ckf:
        t0 = time.perf_counter()
        ckf = run_cd_ukf(
            drift, OBS_FN, L, Qc, R, M0, P0, YS, DT, DT0,
            CKF_PARAMS, state_order="first", solver=SOLVER,
        )
        jax.block_until_ready(ckf.means)
        record(
            "CKF", ckf.cum_loglik, ckf.means, time.perf_counter() - t0,
            pred_mu=ckf.pred_obs_mean, pred_std=ckf.pred_obs_std,
        )

    key = jr.PRNGKey(_args.seed + 101 * subject)
    for n_part in ENKF_LIST:
        t0 = time.perf_counter()
        enkf = run_cd_enkf(
            drift, OBS_FN, L, Qc, R, M0, P0, YS, DT, key,
            n_particles=n_part, noise_dim=2, dt0=DT0, solver=SOLVER,
        )
        jax.block_until_ready(enkf.means)
        record(
            "EnKF",
            enkf.cum_loglik,
            enkf.means,
            time.perf_counter() - t0,
            budget=n_part,
            pred_mu=enkf.pred_obs_mean,
            pred_std=enkf.pred_obs_std,
        )

    if not _args.skip_pf:
        for n_part in PF_LIST:
            t0 = time.perf_counter()
            pf = run_bootstrap_pf(
                drift, OBS_FN, L, Qc, R, M0, P0, YS, DT, key,
                n_particles=n_part, noise_dim=2, dt0=DT0, solver=SOLVER,
            )
            jax.block_until_ready(pf.means)
            record(
                "PF",
                pf.cum_loglik,
                pf.means,
                time.perf_counter() - t0,
                budget=n_part,
                pred_mu=pf.pred_obs_mean,
                pred_std=pf.pred_obs_std,
                pred_crps=pf.pred_crps,
                crps_kind="mixture",
            )

    return params, rows


def main():
    print(
        f"cohort subjects={SUBJECTS}  n_obs≤{_args.n_obs} (5-min)  "
        f"Δt={DT_ASSIM:g} min (stride={STRIDE})  "
        f"MLE iters={_args.mle_iters} (EnKF-{MLE_ENKF_PARTICLES})  "
        f"WSC-KF product={[c.n_paths for c in CRF_CONFIGS]}  EnKF={ENKF_LIST}  "
        f"PF={PF_LIST if not _args.skip_pf else []}"
    )
    all_rows: list[dict] = []
    all_params: list[dict] = []
    failures: list[str] = []

    for sid in SUBJECTS:
        try:
            t0 = time.perf_counter()
            params, rows = run_subject(sid)
            all_params.append(params)
            all_rows.extend(rows)
            print(f"  subject {sid} done in {time.perf_counter()-t0:.1f}s")
        except Exception as e:
            msg = f"subject {sid}: {type(e).__name__}: {e}"
            print(f"  FAIL {msg}")
            failures.append(msg)

    if not all_rows:
        raise SystemExit("no subjects succeeded")

    per_path = os.path.join(OUTPUT_DIR, f"cohort_calibrated_per_subject{OUT_TAG}.csv")
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Aggregate across subjects by (filter, budget)
    keys = []
    for r in all_rows:
        k = (r["filter"], int(r["budget"]))
        if k not in keys:
            keys.append(k)
    summary = []
    for name, bud in keys:
        sub = [r for r in all_rows if r["filter"] == name and int(r["budget"]) == bud]
        nll = np.array([r["held_nll"] for r in sub], dtype=float)
        crps = np.array([r["crps"] for r in sub], dtype=float)
        mape = np.array([r["mape"] for r in sub], dtype=float)
        tim = np.array([r["time_s"] for r in sub], dtype=float)
        n = len(sub)
        lab = sub[0].get("label") or (name if bud == 0 else f"{name}-{bud}")
        summary.append(
            {
                "filter": name,
                "budget": bud,
                "label": lab,
                "crps_kind": sub[0].get("crps_kind", "gaussian"),
                "n_subjects": n,
                "nll_mean": float(np.mean(nll)),
                "nll_std": float(np.std(nll, ddof=1)) if n > 1 else 0.0,
                "nll_se": float(np.std(nll, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "crps_mean": float(np.mean(crps)),
                "crps_std": float(np.std(crps, ddof=1)) if n > 1 else 0.0,
                "crps_se": float(np.std(crps, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "mape_mean": float(np.mean(mape)),
                "mape_std": float(np.std(mape, ddof=1)) if n > 1 else 0.0,
                "mape_se": float(np.std(mape, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "time_s_mean": float(np.mean(tim)),
                "time_s_std": float(np.std(tim, ddof=1)) if n > 1 else 0.0,
                "time_s_se": float(np.std(tim, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            }
        )

    sum_path = os.path.join(OUTPUT_DIR, f"cohort_calibrated_summary{OUT_TAG}.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    params_path = os.path.join(OUTPUT_DIR, f"cohort_calibrated_params{OUT_TAG}.json")
    with open(params_path, "w") as f:
        json.dump(
            {"dt_assim": DT_ASSIM, "params": all_params, "failures": failures},
            f,
            indent=2,
        )

    colors = {
        "WSC-KF": "#6A4C93",
        "EKF": "#2E86AB",
        "UKF": "#E8B923",
        "CKF": "#3DA35D",
        "EnKF": "#D97706",
        "PF": "#F24236",
    }
    families = ["WSC-KF", "EKF", "UKF", "CKF", "EnKF", "PF"]
    if not _args.with_ckf:
        families = [f for f in families if f != "CKF"]

    # Pareto: NLL / MAPE vs wall-clock
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, ykey, yerrkey, ylab in zip(
        axes,
        ("nll_mean", "mape_mean"),
        ("nll_se", "mape_se"),
        ("Held-out NLL / obs", "MAPE vs CGM [%]"),
    ):
        for fam in families:
            pts = [s for s in summary if s["filter"] == fam]
            if not pts:
                continue
            pts = sorted(pts, key=lambda s: s["time_s_mean"])
            xs = [1e3 * s["time_s_mean"] for s in pts]
            ys = [s[ykey] for s in pts]
            yerr = [s[yerrkey] for s in pts]
            xerr = [1e3 * s["time_s_se"] for s in pts]
            ax.errorbar(
                xs, ys, xerr=xerr, yerr=yerr,
                fmt="o-", color=colors[fam], markersize=7, capsize=2.5,
                linewidth=1.2 if len(pts) > 1 else 0.0,
                label=fam, zorder=3,
            )
            for s, x, y in zip(pts, xs, ys):
                if s["budget"]:
                    ax.annotate(
                        str(s["budget"]), (x, y),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=colors[fam],
                    )
        ax.set_xlabel("Wall-clock [ms] (mean ± SE over subjects)")
        ax.set_ylabel(ylab)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(frameon=False, fontsize=8)

    n_ok = len(all_params)
    fig.suptitle(
        f"AZT1D cohort Pareto: per-subject EnKF $\\hat\\theta_i$, then budget ladders\n"
        f"N={n_ok} subjects; $\\Delta t$={DT_ASSIM:g} min; "
        f"WSC-KF product paths={[c.n_paths for c in CRF_CONFIGS]}; "
        f"EnKF/PF={ENKF_LIST}",
        fontsize=11,
    )
    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"cohort_calibrated_pareto{OUT_TAG}.png")
    fig.savefig(png_path, dpi=160)
    if _args.replace_pareto_plot:
        alt = os.path.join(OUTPUT_DIR, "pareto_assimilation.png")
        fig.savefig(alt, dpi=160)
        print(f"Wrote {alt}")
    plt.close(fig)

    # CRPS vs wall-clock (mean ± scaled std over subjects; PF = mixture CRPS)
    err_scale = 0.25  # display ± (err_scale * std); keeps ratios, reduces clutter
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for fam in families:
        pts = [s for s in summary if s["filter"] == fam]
        if not pts:
            continue
        pts = sorted(pts, key=lambda s: s["time_s_mean"])
        xs = np.asarray([1e3 * s["time_s_mean"] for s in pts])
        ys = np.asarray([s["crps_mean"] for s in pts])
        yerr = err_scale * np.asarray([s["crps_std"] for s in pts])
        xerr = err_scale * np.asarray([1e3 * s["time_s_std"] for s in pts])
        if len(pts) > 1:
            ax.plot(xs, ys, "-", color=colors[fam], linewidth=1.2, zorder=2)
        ax.errorbar(
            xs, ys, xerr=xerr, yerr=yerr,
            fmt="o", color=colors[fam], markersize=7, capsize=2.5,
            elinewidth=1.0, linestyle="none",
            label=fam, zorder=3,
        )
        for s, x, y in zip(pts, xs, ys):
            if s["budget"]:
                ax.annotate(
                    str(s["budget"]), (x, y),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color=colors[fam],
                )
    ax.set_xlabel(
        rf"Wall-clock [ms] (mean $\pm$ {err_scale:g}\,std over subjects)"
    )
    ax.set_ylabel("Held-out CRPS")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"AZT1D cohort CRPS Pareto (PF=mixture; $\\Delta t$={DT_ASSIM:g} min)\n"
        f"N={n_ok} subjects; per-subject EnKF $\\hat\\theta_i$; "
        f"WSC-KF paths={[c.n_paths for c in CRF_CONFIGS]}; "
        rf"bars = $\pm${err_scale:g} std",
        fontsize=11,
    )
    fig.tight_layout()
    crps_png = os.path.join(OUTPUT_DIR, f"cohort_calibrated_crps{OUT_TAG}.png")
    fig.savefig(crps_png, dpi=160)
    # Also refresh the untagged path when running a non-default Δt, after
    # preserving any existing Δt=5 figure.
    if abs(DT_ASSIM - BASE_DT) > 1e-9:
        untagged = os.path.join(OUTPUT_DIR, "cohort_calibrated_crps.png")
        dt5 = os.path.join(OUTPUT_DIR, "cohort_calibrated_crps_dt5.png")
        if os.path.isfile(untagged) and not os.path.isfile(dt5):
            import shutil
            shutil.copy2(untagged, dt5)
            print(f"Preserved {dt5}")
        fig.savefig(untagged, dpi=160)
        print(f"Wrote {untagged}")
    plt.close(fig)

    print(f"\nWrote {per_path}")
    print(f"Wrote {sum_path}")
    print(f"Wrote {params_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {crps_png}")
    print("\nCohort summary (mean ± SE for NLL/MAPE; CRPS mean ± std):")
    for s in summary:
        print(
            f"  {s['label']:12s}  NLL={s['nll_mean']:.4f}±{s['nll_se']:.4f}  "
            f"CRPS={s['crps_mean']:.3f}±{s['crps_std']:.3f}  "
            f"MAPE={s['mape_mean']:.2f}±{s['mape_se']:.2f}%  "
            f"t={1e3*s['time_s_mean']:.0f}ms  (n={s['n_subjects']})"
        )
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for m in failures:
            print(f"  {m}")


if __name__ == "__main__":
    main()