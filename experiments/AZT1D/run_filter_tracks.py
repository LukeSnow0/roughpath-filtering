"""Overlay CGM with filter mean tracks at EnKF-calibrated θ̂.

Uses per-subject ``θ̂_i`` from ``cohort_calibrated_params.json`` when present,
else ``calibrated_params_enkf.json``. Produces a two-panel figure:
  * full series — CGM + WSC-KF mean ±1σ
  * held-out zoom — CGM + WSC-KF / EKF / EnKF / PF means

Run:
  uv run --extra experiments python experiments/AZT1D/run_filter_tracks.py
  uv run --extra experiments python experiments/AZT1D/run_filter_tracks.py \\
      --subject 1 --enkf 500 --pf 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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
from crf import build_filter
from crf.library import product_path_count
from cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)

from azt1d_data import load_azt1d_series
from glucose_model import GlucoseParams, init_estimate, make_glucose_dynamics

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "WSC-KF": "#6A4C93",
    "EKF": "#2E86AB",
    "UKF": "#E8B923",
    "EnKF": "#D97706",
    "PF": "#F24236",
    "CGM": "#4A4A4A",
}


def _align_G(means, n_obs: int) -> np.ndarray:
    m = np.asarray(means)
    g = m if m.ndim == 1 else m[:, 0]
    if g.shape[0] == n_obs + 1:
        return g[1:]
    return g[:n_obs]


def _align_std(covs_or_chol, n_obs: int, *, from_chol: bool = False) -> np.ndarray:
    if from_chol:
        chol = np.asarray(covs_or_chol)
        # diag of LLᵀ is sum of squares of row entries
        var = np.sum(chol[..., 0, :] ** 2, axis=-1)
    else:
        covs = np.asarray(covs_or_chol)
        var = covs[:, 0, 0]
    if var.shape[0] == n_obs + 1:
        var = var[1:]
    return np.sqrt(np.maximum(var[:n_obs], 0.0))


def _load_theta(subject: int) -> tuple[float, float, str]:
    cohort = os.path.join(OUTPUT_DIR, "cohort_calibrated_params.json")
    single = os.path.join(OUTPUT_DIR, "calibrated_params_enkf.json")
    if os.path.isfile(cohort):
        with open(cohort) as f:
            blob = json.load(f)
        for row in blob.get("params", []):
            if int(row["subject"]) == subject:
                return float(row["p3"]), float(row["sigma_g"]), "cohort θ̂_i"
    if os.path.isfile(single):
        with open(single) as f:
            blob = json.load(f)
        if int(blob.get("subject", -1)) == subject:
            return float(blob["p3"]), float(blob["sigma_g"]), "EnKF-MLE θ̂"
        if "p3" in blob:
            return float(blob["p3"]), float(blob["sigma_g"]), "EnKF-MLE θ̂"
    raise SystemExit(
        f"no calibrated θ for subject {subject}; run run_cohort_calibrated.py "
        "or run_glucose_mle.py first"
    )


def _product_cfg(n_paths: int, noise_dim: int = 2):
    specs = [(3, 1), (5, 1), (3, 2), (3, 3), (5, 2), (3, 4), (5, 3)]
    for m, k in specs:
        n = product_path_count(noise_dim, k, lyons_degree=m)
        if n == n_paths:
            return m, k
    raise SystemExit(f"product path count {n_paths} not on ladder")


def main():
    ap = argparse.ArgumentParser(description="AZT1D CGM + filter track overlay")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n-obs", type=int, default=2016, dest="n_obs")
    ap.add_argument(
        "--dt-assim",
        type=float,
        default=5.0,
        dest="dt_assim",
        help="assimilation interval [min]; subsample every dt_assim/5 from the 5-min grid",
    )
    ap.add_argument("--holdout-frac", type=float, default=0.3, dest="holdout_frac")
    ap.add_argument("--crf-paths", type=int, default=4, dest="crf_paths")
    ap.add_argument("--enkf", type=int, default=500)
    ap.add_argument("--pf", type=int, default=500)
    ap.add_argument("--dt0", type=float, default=1.0)
    ap.add_argument("--solver", type=str, default="rk4", choices=("rk4", "diffrax"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meas-std", type=float, default=8.0, dest="meas_std")
    ap.add_argument("--skip-ukf", action="store_true")
    ap.add_argument(
        "--zoom-hours",
        type=float,
        default=0.0,
        dest="zoom_hours",
        help="held-out panel width in hours ending at series end (0 = full held-out)",
    )
    ap.add_argument(
        "--zoom-t",
        type=str,
        default="",
        dest="zoom_t",
        help="held-out panel time window as 't0,t1' in hours (overrides --zoom-hours)",
    )
    args = ap.parse_args()

    base_dt = 5.0
    dt_assim = float(args.dt_assim)
    if dt_assim < base_dt - 1e-9:
        raise SystemExit(f"--dt-assim must be ≥ {base_dt:g} min")
    stride = int(round(dt_assim / base_dt))
    if abs(stride * base_dt - dt_assim) > 1e-6:
        raise SystemExit(
            f"--dt-assim={dt_assim:g} must be an integer multiple of {base_dt:g} min"
        )

    p3, sigma_g, theta_src = _load_theta(args.subject)
    series = load_azt1d_series(subject=args.subject, max_obs=args.n_obs, dt_min=base_dt)
    # Keep fine insulin/carb for continuous input interpolation; assimilate every
    # ``stride``-th CGM on a Δt = dt_assim grid.
    cgm_fine = np.asarray(series.cgm)
    t_fine = np.asarray(series.t_min)
    idx = np.arange(0, cgm_fine.shape[0], stride)
    cgm = cgm_fine[idx]
    t_min = t_fine[idx]
    n_obs = int(cgm.shape[0])
    n_held = max(1, int(round(n_obs * args.holdout_frac)))
    n_train = n_obs - n_held
    t_h = t_min / 60.0
    dt = float(dt_assim)
    dt0 = float(args.dt0)

    P = GlucoseParams(meas_std=float(args.meas_std))
    m0_np, P0_np = init_estimate(cgm, P)
    M0 = jnp.asarray(m0_np)
    P0 = jnp.asarray(P0_np)
    CHOL_P0 = jnp.linalg.cholesky(P0)
    drift, L, Qc, OBS_FN, R = make_glucose_dynamics(
        P,
        t_min=series.t_min,
        insulin_u_per_min=series.insulin_u_per_min,
        carb_g_per_min=series.carb_g_per_min,
        p3=p3,
        sigma_g=sigma_g,
    )
    CHOL_R = jnp.linalg.cholesky(R)
    YS = jnp.asarray(cgm.reshape(-1, 1))

    tracks: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}

    # ---- WSC-KF ----
    m_lyons, k_cub = _product_cfg(args.crf_paths)

    def get_init_params(_mi):
        return M0, CHOL_P0

    def get_dynamics(_mi):
        return drift, L

    def get_observations(mi):
        idx = mi if not hasattr(mi, "t_idx") else mi.t_idx
        return OBS_FN, CHOL_R, YS[idx - 1]

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
    st = _run(filt, jnp.arange(n_obs + 1))
    jax.block_until_ready(st.mean)
    wsc_label = f"WSC-KF-{args.crf_paths}"
    tracks[wsc_label] = _align_G(st.mean, n_obs)
    stds[wsc_label] = _align_std(st.chol_cov, n_obs, from_chol=True)

    # ---- EKF / UKF ----
    ekf = run_cd_ekf(
        drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, dt0,
        state_order="first", solver=args.solver,
    )
    jax.block_until_ready(ekf.means)
    tracks["EKF"] = _align_G(ekf.means, n_obs)
    stds["EKF"] = _align_std(ekf.covs, n_obs)

    if not args.skip_ukf:
        ukf = run_cd_ukf(
            drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, dt0,
            UKFParams(), state_order="first", solver=args.solver,
        )
        jax.block_until_ready(ukf.means)
        tracks["UKF"] = _align_G(ukf.means, n_obs)
        stds["UKF"] = _align_std(ukf.covs, n_obs)

    key = jr.PRNGKey(args.seed + 101 * args.subject)
    enkf = run_cd_enkf(
        drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, key,
        n_particles=args.enkf, noise_dim=2, dt0=dt0, solver=args.solver,
    )
    jax.block_until_ready(enkf.means)
    enkf_label = f"EnKF-{args.enkf}"
    tracks[enkf_label] = _align_G(enkf.means, n_obs)
    stds[enkf_label] = _align_std(enkf.covs, n_obs)

    pf = run_bootstrap_pf(
        drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, key,
        n_particles=args.pf, noise_dim=2, dt0=dt0, solver=args.solver,
    )
    jax.block_until_ready(pf.means)
    pf_label = f"PF-{args.pf}"
    tracks[pf_label] = _align_G(pf.means, n_obs)
    stds[pf_label] = _align_std(pf.covs, n_obs)

    # ---- figure ----
    t_split = float(t_h[n_train])
    if args.zoom_t.strip():
        parts = [p.strip() for p in args.zoom_t.split(",")]
        if len(parts) != 2:
            raise SystemExit("--zoom-t must be 't0,t1' in hours")
        t_zoom0, t_zoom1 = float(parts[0]), float(parts[1])
        if t_zoom1 <= t_zoom0:
            raise SystemExit("--zoom-t requires t1 > t0")
    elif args.zoom_hours > 0:
        t_zoom0 = max(t_split, float(t_h[-1]) - args.zoom_hours)
        t_zoom1 = float(t_h[-1])
    else:
        t_zoom0 = t_split
        t_zoom1 = float(t_h[-1])
    zoom = (t_h >= t_zoom0 - 1e-9) & (t_h <= t_zoom1 + 1e-9)
    if not np.any(zoom):
        raise SystemExit(f"no samples in zoom window [{t_zoom0:g}, {t_zoom1:g}] h")

    fig, axes = plt.subplots(
        2, 1, figsize=(11.0, 6.4), sharex=False,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.32},
        layout="constrained",
    )

    # Panel A: full series + WSC band
    ax = axes[0]
    ax.plot(t_h[:n_train], cgm[:n_train], "o", ms=5.0, color="#1A1A1A",
            alpha=0.9, label="CGM (train)", zorder=5,
            markeredgecolor="white", markeredgewidth=0.35)
    ax.plot(t_h[n_train:], cgm[n_train:], "o", ms=5.0, color="#555555",
            alpha=0.92, label="CGM (held-out)", zorder=5,
            markeredgecolor="white", markeredgewidth=0.35)
    g = tracks[wsc_label]
    s = stds[wsc_label]
    ax.fill_between(
        t_h, g - s, g + s, color=COLORS["WSC-KF"], alpha=0.18, linewidth=0, zorder=1,
        label=r"WSC-KF $\pm 1\sigma$",
    )
    ax.plot(t_h, g, "-", color=COLORS["WSC-KF"], lw=1.35, zorder=3, label=wsc_label)
    ax.axvline(t_split, color="#2E86AB", ls="--", lw=0.9, alpha=0.8, zorder=0)
    ax.set_ylabel("glucose [mg/dL]")
    ax.set_xlim(float(t_h[0]), float(t_h[-1]))
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax.set_title(
        f"Subject {args.subject}: CGM + {wsc_label}  "
        f"($\\Delta t$={dt_assim:g} min; {theta_src}: $p_3$={p3:.2e}, $\\sigma_G$={sigma_g:.3g})",
        fontsize=11,
    )

    # Panel B: held-out multi-filter overlay with ±1σ bands
    ax = axes[1]
    ax.plot(
        t_h[zoom], cgm[zoom], "o", ms=6.5, color="#1A1A1A",
        alpha=0.92, label="CGM", zorder=6,
        markeredgecolor="white", markeredgewidth=0.4,
    )
    order = [wsc_label, "EKF"]
    if "UKF" in tracks:
        order.append("UKF")
    order.extend([enkf_label, pf_label])
    lw = {wsc_label: 1.8, "EKF": 1.15, "UKF": 1.0, enkf_label: 1.15, pf_label: 1.15}
    ls = {wsc_label: "-", "EKF": "-", "UKF": "--", enkf_label: "-", pf_label: "-"}
    band_alpha = {
        wsc_label: 0.22, "EKF": 0.12, "UKF": 0.10,
        enkf_label: 0.14, pf_label: 0.14,
    }

    def _fam(name: str) -> str:
        if name.startswith("WSC"):
            return "WSC-KF"
        if name.startswith("EnKF"):
            return "EnKF"
        if name.startswith("PF"):
            return "PF"
        return name

    for name in order:
        color = COLORS[_fam(name)]
        g = tracks[name][zoom]
        if name in stds:
            s = stds[name][zoom]
            ax.fill_between(
                t_h[zoom], g - s, g + s,
                color=color, alpha=band_alpha[name], linewidth=0, zorder=1,
            )
        ax.plot(
            t_h[zoom], g, ls[name], color=color,
            lw=lw[name], label=rf"{name} $\pm 1\sigma$", zorder=3, alpha=0.95,
        )
    if t_zoom0 <= t_split + 1e-6 <= t_zoom1:
        ax.axvline(t_split, color="#2E86AB", ls="--", lw=0.9, alpha=0.8, zorder=0)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("glucose [mg/dL]")
    ax.set_xlim(t_zoom0, t_zoom1)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax.set_title(
        f"Held-out filter means $\\pm 1\\sigma$ (same EnKF $\\hat\\theta$; "
        f"$\\Delta t$={dt_assim:g} min; $t\\in[{t_zoom0:g},{t_zoom1:g}]$ h)",
        fontsize=11,
    )

    # y-limits from visible window (include ±1σ so bands aren't clipped)
    y_parts = [cgm[zoom]]
    for name in order:
        g = tracks[name][zoom]
        if name in stds:
            s = stds[name][zoom]
            y_parts.extend([g - s, g + s])
        else:
            y_parts.append(g)
    y_src = np.concatenate([np.asarray(p, dtype=float).ravel() for p in y_parts])
    y_lo = float(np.nanmin(y_src) - 10)
    y_hi = float(np.nanmax(y_src) + 10)
    axes[1].set_ylim(y_lo, y_hi)
    y_lo_full = float(np.nanpercentile(cgm, 1) - 15)
    y_hi_full = float(np.nanpercentile(cgm, 99) + 15)
    axes[0].set_ylim(y_lo_full, y_hi_full)

    # annotate holdout on top panel
    axes[0].annotate(
        "held-out",
        xy=(t_split, y_hi_full),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=8,
        color="#2E86AB",
        va="top",
    )

    out = os.path.join(OUTPUT_DIR, f"filter_tracks_dt{int(round(dt_assim))}.png")
    fig.savefig(out, dpi=170)
    latest = os.path.join(OUTPUT_DIR, "filter_tracks.png")
    if os.path.abspath(out) != os.path.abspath(latest):
        fig.savefig(latest, dpi=170)
    print(f"Wrote {out}")
    if os.path.abspath(out) != os.path.abspath(latest):
        print(f"Wrote {latest}")
    print(f"  θ: p3={p3:.6g}  sigma_g={sigma_g:.6g}  ({theta_src})")
    print(f"  Δt={dt_assim:g} min (stride={stride})  n_obs={n_obs}  n_train={n_train}  n_held={n_held}")


if __name__ == "__main__":
    main()
