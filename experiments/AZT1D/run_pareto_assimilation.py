"""Pareto curves at one or more assimilation intervals (default Δt = 5 / 20 min).

For a fixed calendar span and EnKF-calibrated ``θ̂``, resample AZT1D onto each
assimilation grid, run the budget ladder over multiple non-overlapping
time-window seeds, then write a separate cohort-style Pareto plot per ``Δt``.

Run:
  uv run --extra experiments python experiments/AZT1D/run_pareto_assimilation.py
  uv run --extra experiments python experiments/AZT1D/run_pareto_assimilation.py \\
      --deltas 5,20 --seeds 0,1,2 --span-min 10080
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

from azt1d_data import load_azt1d_series
from glucose_model import (
    GlucoseParams,
    wsckf_glucose_dynamics,
    heldout_crps_per_obs,
    heldout_mean_score,
    init_estimate,
    make_glucose_dynamics,
    mape_glucose,
)
from plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    METHOD_MARKERS,
    PALETTE,
    SINGLE_COLUMN_WIDTH,
    save_publication_figure,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)

_parser = argparse.ArgumentParser(
    description="AZT1D Pareto overlay across assimilation intervals"
)
_parser.add_argument("--subject", type=int, default=1)
_parser.add_argument(
    "--span-min",
    type=float,
    default=10080.0,
    dest="span_min",
    help="fixed calendar span [min]; default 10080 = 7 days",
)
_parser.add_argument(
    "--deltas",
    type=str,
    default="5,20",
    help="comma-separated assimilation intervals [min]",
)
_parser.add_argument(
    "--seeds",
    type=str,
    default="0,1,2",
    help="window / RNG seeds (indices into starts spaced by --seed-stride-min)",
)
_parser.add_argument(
    "--seed-stride-min",
    type=float,
    default=None,
    dest="seed_stride_min",
    help="minutes between window starts (default: --span-min = non-overlapping)",
)
_parser.add_argument("--holdout-frac", type=float, default=0.3, dest="holdout_frac")
_parser.add_argument("--dt0", type=float, default=1.0)
_parser.add_argument("--wsckf-max-paths", type=int, default=4, dest="wsckf_max_paths")
_parser.add_argument("--enkf", type=str, default="25,50,100,500")
_parser.add_argument("--pf", type=str, default="25,50,100,500")
_parser.add_argument("--n-timing", type=int, default=1, dest="n_timing")
_parser.add_argument("--solver", type=str, default="rk4", choices=("rk4", "diffrax"))
_parser.add_argument("--skip-ekf", action="store_true")
_parser.add_argument("--skip-ukf", action="store_true")
_parser.add_argument("--skip-ckf", action="store_true")
_parser.add_argument("--skip-pf", action="store_true")
_parser.add_argument("--meas-std", type=float, default=8.0, dest="meas_std")
_parser.add_argument("--p3", type=float, default=None)
_parser.add_argument("--sigma-g", type=float, default=None, dest="sigma_g")
_parser.add_argument(
    "--theta-json",
    type=str,
    default=os.path.join(OUTPUT_DIR, "calibrated_params_enkf.json"),
    dest="theta_json",
)
_parser.add_argument(
    "--output-prefix",
    type=str,
    default="pareto_assimilation",
    dest="output_prefix",
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


def _parse_floats(s: str) -> list[float]:
    return [float(tok.strip()) for tok in s.split(",") if tok.strip()]


@dataclass(frozen=True)
class WSCKFConfig:
    family: str
    label: str
    n_paths: int
    k: int = 1
    lyons_degree: int = 5
    method: str = "product"


def default_product_ladder(noise_dim: int = 2, max_paths: int = 81) -> list[WSCKFConfig]:
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
                family="WSC-KF-product",
                label=f"prod-m{m}-k{k}",
                n_paths=n,
                k=k,
                lyons_degree=m,
                method="product",
            )
        )
    return sorted(configs, key=lambda c: c.n_paths)


_p3 = _args.p3
_sigma_g = _args.sigma_g
_meas_std = float(_args.meas_std)
if _args.theta_json and os.path.isfile(_args.theta_json):
    with open(_args.theta_json) as f:
        blob = json.load(f)
    _p3 = float(blob["p3"])
    _sigma_g = float(blob["sigma_g"])
    if "meas_std" in blob:
        _meas_std = float(blob["meas_std"])
elif _args.theta_json:
    raise SystemExit(f"missing --theta-json: {_args.theta_json}")
_p3 = float(GlucoseParams().p3 if _p3 is None else _p3)
_sigma_g = float(GlucoseParams().sigma_g if _sigma_g is None else _sigma_g)

DELTAS = _parse_floats(_args.deltas)
SEEDS = _parse_ints(_args.seeds)
ENKF_LIST = _parse_ints(_args.enkf)
PF_LIST = _parse_ints(_args.pf)
N_TIMING = max(1, int(_args.n_timing))
SOLVER = _args.solver
WSCKF_CONFIGS = default_product_ladder(2, max_paths=int(_args.wsckf_max_paths))
SPAN_MIN = float(_args.span_min)
SEED_STRIDE_MIN = float(
    _args.seed_stride_min if _args.seed_stride_min is not None else SPAN_MIN
)
if SEED_STRIDE_MIN <= 0:
    raise SystemExit("--seed-stride-min must be positive")
if not (0.05 <= _args.holdout_frac <= 0.5):
    raise SystemExit("--holdout-frac must be in [0.05, 0.5]")
if not WSCKF_CONFIGS:
    raise SystemExit(f"--wsckf-max-paths={_args.wsckf_max_paths} yields no product configs")
if not SEEDS:
    raise SystemExit("--seeds must be non-empty")

P = GlucoseParams(meas_std=_meas_std, p3=_p3, sigma_g=_sigma_g)
print(
    f"θ̂: p3={P.p3:.6e}  σ_G={P.sigma_g:.6f}  R^0.5={P.meas_std:g}  "
    f"span={SPAN_MIN:g} min (~{SPAN_MIN/1440:.2f} d)  "
    f"stride={SEED_STRIDE_MIN:g} min  Δt={DELTAS}  seeds={SEEDS}"
)


def heldout_nll_per_obs(cum_ll, n_train: int, n_obs: int) -> float:
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
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(n_timing):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return out, float(np.mean(times))


def run_one_delta(dt_assim: float, seed: int) -> list[dict]:
    """One (Δt, seed) window: start at ``seed * SEED_STRIDE_MIN``."""
    start_min = float(seed) * SEED_STRIDE_MIN
    n_span = max(20, int(round(SPAN_MIN / dt_assim)) + 1)
    i0 = int(round(start_min / dt_assim))
    # Load enough grid to cover this window from subject start.
    max_obs = i0 + n_span + 2
    series = load_azt1d_series(
        subject=int(_args.subject),
        max_obs=max_obs,
        dt_min=float(dt_assim),
    )
    if series.n_obs <= i0 + 20:
        raise SystemExit(
            f"Δt={dt_assim} seed={seed}: not enough data "
            f"(need start index {i0}, have n={series.n_obs})"
        )
    i1 = min(series.n_obs, i0 + n_span)
    t_full = series.t_min
    t_min = t_full[i0:i1] - t_full[i0]
    cgm = series.cgm[i0:i1]
    insulin = series.insulin_u_per_min[i0:i1]
    carb = series.carb_g_per_min[i0:i1]

    DT = float(dt_assim)
    DT0 = float(_args.dt0)
    YS = jnp.asarray(cgm.reshape(-1, 1))
    N_OBS = int(YS.shape[0])
    N_HELD = max(1, int(round(N_OBS * _args.holdout_frac)))
    N_TRAIN = N_OBS - N_HELD
    if N_TRAIN < 10:
        raise SystemExit(f"Δt={dt_assim} seed={seed}: train too short (n_train={N_TRAIN})")

    m0_np, P0_np = init_estimate(cgm, P)
    M0 = jnp.asarray(m0_np)
    P0 = jnp.asarray(P0_np)
    CHOL_P0 = jnp.linalg.cholesky(P0)

    DRIFT, L_FN, QC, OBS_FN, R = make_glucose_dynamics(
        P,
        t_min=t_min,
        insulin_u_per_min=insulin,
        carb_g_per_min=carb,
    )
    CHOL_R = jnp.linalg.cholesky(R)

    print(
        f"\n=== Δt={dt_assim:g} min  seed={seed}  start={start_min:g} min  "
        f"N={N_OBS}  train={N_TRAIN} held={N_HELD}  span≈{t_min[-1]/1440:.2f} d ==="
    )

    raw: list[dict] = []

    def score(
        family: str,
        label: str,
        budget: int,
        cum_ll,
        means,
        wall_s: float,
        *,
        pred_mu=None,
        pred_std=None,
        pred_crps=None,
        crps_kind: str = "gaussian",
    ):
        nll_h = heldout_nll_per_obs(cum_ll, N_TRAIN, N_OBS)
        means_np = np.asarray(means)
        g_hat = means_np if means_np.ndim == 1 else means_np[:, 0]
        mape = mape_glucose(g_hat, cgm)
        if pred_crps is not None:
            crps = heldout_mean_score(pred_crps, N_TRAIN, N_OBS)
        elif pred_mu is None or pred_std is None:
            crps = float("nan")
        else:
            crps = heldout_crps_per_obs(cgm, pred_mu, pred_std, N_TRAIN)
        rec = {
            "delta_t_min": float(dt_assim),
            "subject": int(series.subject),
            "n_obs": N_OBS,
            "n_train": N_TRAIN,
            "n_held": N_HELD,
            "dt0": DT0,
            "solver": SOLVER,
            "p3": P.p3,
            "sigma_g": P.sigma_g,
            "meas_std": P.meas_std,
            "seed": seed,
            "start_min": start_min,
            "family": family,
            "label": label,
            "budget": budget,
            "nll_held_per_obs": nll_h,
            "crps_held": crps,
            "crps_kind": crps_kind,
            "mape_pct": mape,
            "time_s": wall_s,
            "time_per_obs_ms": 1e3 * wall_s / max(N_OBS, 1),
        }
        raw.append(rec)
        print(
            f"  {family:12s} {label:12s} n={budget:4d}  "
            f"NLL={nll_h:7.3f}  CRPS={crps:6.3f}  MAPE={mape:5.2f}%  "
            f"time={1e3*wall_s:7.1f} ms"
        )

    def build_wsckf(cfg: WSCKFConfig):
        model_inputs = jnp.arange(N_OBS + 1)

        def get_init_params(_mi):
            return M0, CHOL_P0

        def get_dynamics(mi):
            return wsckf_glucose_dynamics(DRIFT, L_FN, mi, DT)

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

    _run_wsckf = jit(run_filter, static_argnames=["filter_obj"])
    for cfg in WSCKF_CONFIGS:
        filt, mi = build_wsckf(cfg)
        states, wall = time_call(lambda f=filt, m=mi: _run_wsckf(f, m))
        score(
            cfg.family,
            cfg.label,
            cfg.n_paths,
            cum_ll=states.log_normalizing_constant,
            means=states.mean,
            wall_s=wall,
            pred_mu=states.pred_obs_mean,
            pred_std=states.pred_obs_std,
        )

    if not _args.skip_ekf:
        ekf_jit = jit(
            lambda: run_cd_ekf(
                DRIFT, OBS_FN, L_FN, QC, R, M0, P0, YS, DT, DT0,
                state_order="first", solver=SOLVER,
            )
        )
        ekf_res, wall = time_call(ekf_jit)
        score(
            "CD-EKF", "EKF", 0, ekf_res.cum_loglik, ekf_res.means, wall,
            pred_mu=ekf_res.pred_obs_mean, pred_std=ekf_res.pred_obs_std,
        )

    if not _args.skip_ukf:
        ukf_jit = jit(
            lambda: run_cd_ukf(
                DRIFT, OBS_FN, L_FN, QC, R, M0, P0, YS, DT, DT0,
                UKFParams(), state_order="first", solver=SOLVER,
            )
        )
        ukf_res, wall = time_call(ukf_jit)
        score(
            "CD-UKF", "UKF", 0, ukf_res.cum_loglik, ukf_res.means, wall,
            pred_mu=ukf_res.pred_obs_mean, pred_std=ukf_res.pred_obs_std,
        )

    if not _args.skip_ckf:
        ckf_jit = jit(
            lambda: run_cd_ukf(
                DRIFT, OBS_FN, L_FN, QC, R, M0, P0, YS, DT, DT0,
                CKF_PARAMS, state_order="first", solver=SOLVER,
            )
        )
        ckf_res, wall = time_call(ckf_jit)
        score(
            "CD-CKF", "CKF", 0, ckf_res.cum_loglik, ckf_res.means, wall,
            pred_mu=ckf_res.pred_obs_mean, pred_std=ckf_res.pred_obs_std,
        )

    for n_part in ENKF_LIST:
        key = jr.key(seed + 17 * n_part + int(10 * dt_assim))

        def _enkf(k=key, n=n_part):
            return run_cd_enkf(
                DRIFT, OBS_FN, L_FN, QC, R, M0, P0, YS, DT, k,
                n_particles=n, noise_dim=2, dt0=DT0, solver=SOLVER,
            )

        enkf_res, wall = time_call(jit(_enkf))
        score(
            "CD-EnKF", f"EnKF-{n_part}", n_part, enkf_res.cum_loglik, enkf_res.means, wall,
            pred_mu=enkf_res.pred_obs_mean, pred_std=enkf_res.pred_obs_std,
        )

    if not _args.skip_pf:
        for n_part in PF_LIST:
            key = jr.key(1000 + seed + 31 * n_part + int(10 * dt_assim))

            def _pf(k=key, n=n_part):
                return run_bootstrap_pf(
                    DRIFT, OBS_FN, L_FN, QC, R, M0, P0, YS, DT, k,
                    n_particles=n, noise_dim=2, dt0=DT0, solver=SOLVER,
                )

            pf_res, wall = time_call(jit(_pf))
            score(
                "PF", f"PF-{n_part}", n_part, pf_res.cum_loglik, pf_res.means, wall,
                pred_mu=pf_res.pred_obs_mean, pred_std=pf_res.pred_obs_std,
                pred_crps=pf_res.pred_crps, crps_kind="mixture",
            )

    return raw


def aggregate(raw_rows: list[dict]) -> list[dict]:
    keys = sorted(
        {
            (r["delta_t_min"], r["family"], r["label"], r["budget"])
            for r in raw_rows
        }
    )
    out: list[dict] = []
    for dt, fam, lab, bud in keys:
        sub = [
            r
            for r in raw_rows
            if r["delta_t_min"] == dt
            and r["family"] == fam
            and r["label"] == lab
            and r["budget"] == bud
        ]
        base = {k: sub[0][k] for k in (
            "delta_t_min", "subject", "n_obs", "n_train", "n_held", "dt0",
            "solver", "p3", "sigma_g", "meas_std", "family", "label", "budget",
            "crps_kind",
        )}
        n = len(sub)
        nll_std = float(
            np.std([r["nll_held_per_obs"] for r in sub], ddof=1) if n > 1 else 0.0
        )
        crps_std = float(
            np.std([r["crps_held"] for r in sub], ddof=1) if n > 1 else 0.0
        )
        mape_std = float(np.std([r["mape_pct"] for r in sub], ddof=1) if n > 1 else 0.0)
        time_std = float(
            np.std([r["time_s"] for r in sub], ddof=1) if n > 1 else 0.0
        )
        time_per_obs_std = float(
            np.std([r["time_per_obs_ms"] for r in sub], ddof=1)
            if n > 1
            else 0.0
        )
        out.append(
            {
                **base,
                "nll_held_mean": float(np.mean([r["nll_held_per_obs"] for r in sub])),
                "nll_held_std": nll_std,
                "nll_held_se": nll_std / np.sqrt(n) if n > 1 else 0.0,
                "crps_held_mean": float(np.mean([r["crps_held"] for r in sub])),
                "crps_held_std": crps_std,
                "crps_held_se": crps_std / np.sqrt(n) if n > 1 else 0.0,
                "mape_mean": float(np.mean([r["mape_pct"] for r in sub])),
                "mape_std": mape_std,
                "mape_se": mape_std / np.sqrt(n) if n > 1 else 0.0,
                "time_s_mean": float(np.mean([r["time_s"] for r in sub])),
                "time_s_std": time_std,
                "time_s_se": time_std / np.sqrt(n) if n > 1 else 0.0,
                "time_per_obs_ms_mean": float(np.mean([r["time_per_obs_ms"] for r in sub])),
                "time_per_obs_ms_std": time_per_obs_std,
                "time_per_obs_ms_se": (
                    time_per_obs_std / np.sqrt(n) if n > 1 else 0.0
                ),
                "n_runs": n,
            }
        )
    return out


def plot_one_delta(
    summary_dt: list[dict],
    png_path: str,
    *,
    per_obs: bool = False,
    metrics: tuple[tuple[str, str, str], ...] = (
        ("nll_held_mean", "nll_held_se", "Held-out NLL [nats/observation]"),
        ("mape_mean", "mape_se", "Full-record MAPE [%]"),
    ),
) -> tuple[str, str]:
    """Single-Δt Pareto in the cohort style (one series per filter)."""
    DISPLAY = {
        "WSC-KF-product": "WSC-KF",
        "CD-EKF": "EKF",
        "CD-UKF": "UKF",
        "CD-CKF": "CKF",
        "CD-EnKF": "EnKF",
        "PF": "PF",
    }
    families = ("WSC-KF-product", "CD-EKF", "CD-UKF", "CD-CKF", "CD-EnKF", "PF")
    s0 = summary_dt[0]
    crps_figure = all(metric[0].startswith("crps_") for metric in metrics)

    n_panel = len(metrics)
    figure_width = DOUBLE_COLUMN_WIDTH if n_panel > 1 else SINGLE_COLUMN_WIDTH
    figure_height = 2.55 if n_panel > 1 else 2.45
    fig, axes = plt.subplots(
        1,
        n_panel,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )
    axes = axes[0]
    xkey = "time_per_obs_ms_mean" if per_obs else "time_s_mean"
    xlabel = (
        "Mean time / observation [ms]"
        if per_obs
        else "Mean wall time [ms]"
    )
    for panel_idx, (ax, (ykey, yerrkey, ylab)) in enumerate(zip(axes, metrics)):
        for fam in families:
            pts = [s for s in summary_dt if s["family"] == fam]
            if not pts:
                continue
            pts = sorted(pts, key=lambda s: s[xkey])
            xs = (
                [s[xkey] for s in pts]
                if per_obs
                else [1e3 * s["time_s_mean"] for s in pts]
            )
            xerr = (
                [s["time_per_obs_ms_se"] for s in pts]
                if per_obs
                else [1e3 * s["time_s_se"] for s in pts]
            )
            ys = [s[ykey] for s in pts]
            yerr = [float(s.get(yerrkey, 0.0)) for s in pts]
            display_name = DISPLAY[fam]
            legend_name = (
                f"{display_name} (mixture)"
                if crps_figure and fam == "PF"
                else display_name
            )
            ax.errorbar(
                xs,
                ys,
                xerr=xerr,
                yerr=yerr,
                fmt=METHOD_MARKERS[display_name],
                color=METHOD_COLORS[fam],
                markerfacecolor="white",
                markeredgewidth=0.85,
                markersize=4.4,
                capsize=2.0,
                capthick=0.7,
                elinewidth=0.7,
                linestyle="-" if len(pts) > 1 else "none",
                linewidth=1.1,
                label=legend_name,
                zorder=3,
            )
            for s, x, y in zip(pts, xs, ys):
                bud = int(s["budget"])
                if bud:
                    ax.annotate(
                        str(bud),
                        (x, y),
                        textcoords="offset points",
                        xytext=(3, 3),
                        fontsize=6.2,
                        color=METHOD_COLORS[fam],
                    )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_xscale("log")
        style_curve_axis(ax)
        if n_panel > 1:
            ax.set_title(f"({chr(ord('a') + panel_idx)})")

    axes[0].text(
        0.02,
        0.98,
        r"error bars: $\pm 1$ SE",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=PALETTE["neutral_gray"],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(3 if n_panel == 1 else 6, len(labels)),
    )
    fig.subplots_adjust(
        left=0.12 if n_panel > 1 else 0.19,
        right=0.985 if n_panel > 1 else 0.97,
        bottom=0.27 if n_panel > 1 else 0.31,
        top=0.94 if n_panel > 1 else 0.98,
        wspace=0.34,
    )
    written_png, written_pdf = save_publication_figure(fig, png_path)
    plt.close(fig)
    return str(written_png), str(written_pdf)


def plot_by_delta(summary: list[dict], prefix_path: str) -> list[str]:
    """Write NLL/MAPE and CRPS Pareto plots per assimilation interval."""
    base, ext = os.path.splitext(prefix_path)
    if ext.lower() != ".png":
        base, ext = prefix_path, ".png"
    deltas = sorted({float(s["delta_t_min"]) for s in summary})
    written: list[str] = []
    nll_mape = (
        ("nll_held_mean", "nll_held_se", "Held-out NLL [nats/observation]"),
        ("mape_mean", "mape_se", "Full-record MAPE [%]"),
    )
    crps_only = (
        ("crps_held_mean", "crps_held_se", "Held-out CRPS [mg/dL]"),
    )
    for dt in deltas:
        sub = [s for s in summary if float(s["delta_t_min"]) == dt]
        for per_obs, suffix in ((False, ""), (True, "_per_obs")):
            tag = f"{base}_dt{dt:g}{suffix}{ext}"
            written.extend(
                plot_one_delta(sub, tag, per_obs=per_obs, metrics=nll_mape)
            )
            tag_c = f"{base}_dt{dt:g}_crps{suffix}{ext}"
            written.extend(
                plot_one_delta(
                    sub,
                    tag_c,
                    per_obs=per_obs,
                    metrics=crps_only,
                ),
            )
    return written


def main():
    raw_all: list[dict] = []
    for dt in DELTAS:
        for seed in SEEDS:
            raw_all.extend(run_one_delta(dt, seed))

    summary = aggregate(raw_all)
    prefix = str(_args.output_prefix).strip() or "pareto_assimilation"
    raw_path = os.path.join(OUTPUT_DIR, f"{prefix}_raw.csv")
    csv_path = os.path.join(OUTPUT_DIR, f"{prefix}.csv")
    png_path = os.path.join(OUTPUT_DIR, f"{prefix}.png")

    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_all[0].keys()))
        w.writeheader()
        w.writerows(raw_all)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    written = plot_by_delta(summary, png_path)
    print(f"\nWrote {raw_path}")
    print(f"Wrote {csv_path}")
    for p in written:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
