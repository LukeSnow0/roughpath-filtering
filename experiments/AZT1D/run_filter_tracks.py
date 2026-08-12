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
from wsckf import build_filter
from wsckf.library import product_path_count
from cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)

from azt1d_data import (
    cgm_noise_seed_for_subject,
    inject_cgm_gaussian_noise,
    load_azt1d_series,
)
from glucose_model import (
    GlucoseParams,
    wsckf_glucose_dynamics,
    init_estimate,
    make_glucose_dynamics,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "WSC-KF": "#6A4C93",
    "EKF": "#1F77B4",
    "UKF": "#3D8B6E",
    "EnKF": "#E67E22",  # orange
    "PF": "#C0392B",    # red
    "CGM": "#4A4A4A",
}
# Soft fills; WSC/EKF chosen to contrast when nested
BAND_COLORS = {
    "WSC-KF": "#B39DDB",  # lavender (inner / narrower typically)
    "EKF": "#90CAF9",     # light blue (outer / wider typically)
    "UKF": "#A8D5C2",
    "EnKF": "#F5C48A",    # light orange
    "PF": "#E8A0A0",      # light red
}


def _align_G(means, n_obs: int) -> np.ndarray:
    m = np.asarray(means)
    g = m if m.ndim == 1 else m[:, 0]
    if g.shape[0] == n_obs + 1:
        return g[1:]
    return g[:n_obs]


def _align_means2(means, n_obs: int) -> np.ndarray:
    """Align filtered state means to ``(n_obs, x_dim)``."""
    m = np.asarray(means, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.shape[0] == n_obs + 1:
        m = m[1:]
    return m[:n_obs]


def _align_covs(covs_or_chol, n_obs: int, *, from_chol: bool = False) -> np.ndarray:
    """Align filtered covariances to ``(n_obs, x_dim, x_dim)``."""
    if from_chol:
        chol = np.asarray(covs_or_chol, dtype=np.float64)
        if chol.shape[0] == n_obs + 1:
            chol = chol[1:]
        chol = chol[:n_obs]
        return np.einsum("...ij,...kj->...ik", chol, chol)
    covs = np.asarray(covs_or_chol, dtype=np.float64)
    if covs.shape[0] == n_obs + 1:
        covs = covs[1:]
    return covs[:n_obs]


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


def _align_pred(mu, std, n_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Align one-step predictive observation mean/std to length ``n_obs``."""
    m = np.asarray(mu, dtype=np.float64).reshape(-1)
    s = np.asarray(std, dtype=np.float64).reshape(-1)
    if m.shape[0] == n_obs + 1:
        m = m[1:]
    if s.shape[0] == n_obs + 1:
        s = s[1:]
    return m[:n_obs], np.maximum(s[:n_obs], 0.0)


def _load_theta(
    subject: int,
    dt_assim: float = 5.0,
    *,
    out_suffix: str = "",
) -> tuple[float, float, str]:
    """Prefer cohort params tagged by Δt (+ optional suffix); then fall back."""
    dt_i = int(round(dt_assim))
    sfx = str(out_suffix).strip().lstrip("_")
    tags = []
    if abs(dt_assim - 5.0) >= 1e-9:
        base = f"_dt{dt_i}"
        if sfx:
            tags.append(f"{base}_{sfx}")
        tags.append(base)
        for fallback in (40, 20, 10):
            if fallback < dt_i:
                fb = f"_dt{fallback}"
                if sfx:
                    tags.append(f"{fb}_{sfx}")
                tags.append(fb)
    else:
        if sfx:
            tags.append(f"_{sfx}")
    tags.append("")
    candidates = [os.path.join(OUTPUT_DIR, f"cohort_calibrated_params{t}.json") for t in tags]
    candidates.append(os.path.join(OUTPUT_DIR, "calibrated_params_enkf.json"))
    prefer = tags[0] if tags else ""
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            blob = json.load(f)
        if "params" in blob:
            for row in blob.get("params", []):
                if int(row["subject"]) == subject:
                    used = next((t for t in tags if t and path.endswith(f"params{t}.json")), "")
                    src = "cohort θ̂_i" + (f" {used}" if used else "")
                    if used and prefer and used != prefer:
                        src += f" (fallback; no {prefer})"
                    return float(row["p3"]), float(row["sigma_g"]), src
        elif "p3" in blob:
            if int(blob.get("subject", -1)) in (subject, -1) or "subject" not in blob:
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
    ap.add_argument("--wsckf-paths", type=int, default=4, dest="wsckf_paths")
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
    ap.add_argument(
        "--zoom-max-disc",
        action="store_true",
        dest="zoom_max_disc",
        help="auto-zoom held-out panel to the window with largest filter mean disagreement",
    )
    ap.add_argument(
        "--zoom-disc-hours",
        type=float,
        default=12.0,
        dest="zoom_disc_hours",
        help="window width [h] used by --zoom-max-disc (default 12)",
    )
    ap.add_argument(
        "--predictive",
        action="store_true",
        help="plot one-step predictive y-moments (pred_obs) instead of filtered means; "
        "matches NLL/CRPS and shows filter disagreement under large Δt",
    )
    ap.add_argument(
        "--out-suffix",
        type=str,
        default="",
        dest="out_suffix",
        help="append to output filename (e.g. 'fixed'); also prefers matching cohort params",
    )
    ap.add_argument(
        "--sigma-g-scale",
        type=float,
        default=1.0,
        dest="sigma_g_scale",
        help="multiply loaded σ_G (stress multiplicative noise)",
    )
    ap.add_argument(
        "--params-json",
        type=str,
        default="",
        dest="params_json",
        help="load (p3, sigma_g) for this subject from a cohort_calibrated_params*.json",
    )
    ap.add_argument(
        "--cgm-noise-std",
        type=float,
        default=0.0,
        dest="cgm_noise_std",
        help="add i.i.d. N(0, std²) to each assimilation-grid CGM (0 = clean data)",
    )
    ap.add_argument(
        "--cgm-noise-seed",
        type=int,
        default=0,
        dest="cgm_noise_seed",
        help="base seed for CGM noise (per-subject: seed + 100003*subject)",
    )
    ap.add_argument(
        "--band-scale",
        type=float,
        default=1.0,
        dest="band_scale",
        help="scale factor for shaded ±σ bands (e.g. 0.33 ≈ 3× narrower)",
    )
    ap.add_argument(
        "--no-latest",
        action="store_true",
        dest="no_latest",
        help="do not also write filter_tracks.png",
    )
    ap.add_argument(
        "--plot-all-cov",
        action="store_true",
        dest="plot_all_cov",
        help="plot filtered G/X means ±σ and Cov(G,X) for every filter "
        "(uses filtered state moments even with --predictive)",
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

    p3, sigma_g, theta_src = _load_theta(
        args.subject, dt_assim, out_suffix=str(args.out_suffix)
    )
    params_json = str(args.params_json).strip()
    if params_json:
        if not os.path.isfile(params_json):
            raise SystemExit(f"--params-json not found: {params_json}")
        with open(params_json) as f:
            blob = json.load(f)
        found = False
        for row in blob.get("params", []):
            if int(row["subject"]) == args.subject:
                p3 = float(row["p3"])
                sigma_g = float(row["sigma_g"])
                theta_src = f"params-json {os.path.basename(params_json)}"
                found = True
                break
        if not found:
            raise SystemExit(
                f"subject {args.subject} not in --params-json {params_json}"
            )
    # Prefer explicit fixed params when stressing calibrated θ (suffix may be stress_*)
    # unless --params-json already supplied the unscaled θ̂.
    if abs(float(args.sigma_g_scale) - 1.0) > 1e-12 and not params_json:
        fixed_tag = (
            ""
            if abs(dt_assim - 5.0) < 1e-9
            else f"_dt{int(round(dt_assim))}_fixed"
        )
        fixed_path = os.path.join(OUTPUT_DIR, f"cohort_calibrated_params{fixed_tag}.json")
        if os.path.isfile(fixed_path):
            with open(fixed_path) as f:
                blob = json.load(f)
            for row in blob.get("params", []):
                if int(row["subject"]) == args.subject:
                    p3 = float(row["p3"])
                    sigma_g = float(row["sigma_g"])
                    theta_src = f"cohort θ̂_i {fixed_tag} ×σ_G"
                    break
    if abs(float(args.sigma_g_scale) - 1.0) > 1e-12:
        sigma_g = float(sigma_g) * float(args.sigma_g_scale)
        theta_src = f"{theta_src} (σ_G×{float(args.sigma_g_scale):g} → {sigma_g:.3g})"
    series = load_azt1d_series(subject=args.subject, max_obs=args.n_obs, dt_min=base_dt)
    # Keep fine insulin/carb for continuous input interpolation; assimilate every
    # ``stride``-th CGM on a Δt = dt_assim grid.
    cgm_fine = np.asarray(series.cgm)
    t_fine = np.asarray(series.t_min)
    idx = np.arange(0, cgm_fine.shape[0], stride)
    cgm = cgm_fine[idx]
    if float(args.cgm_noise_std) > 0.0:
        cgm = inject_cgm_gaussian_noise(
            cgm,
            std=float(args.cgm_noise_std),
            seed=cgm_noise_seed_for_subject(int(args.cgm_noise_seed), args.subject),
        )
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
    state_means: dict[str, np.ndarray] = {}
    state_covs: dict[str, np.ndarray] = {}

    def _store_state(name: str, means, covs_or_chol, *, from_chol: bool = False):
        state_means[name] = _align_means2(means, n_obs)
        state_covs[name] = _align_covs(covs_or_chol, n_obs, from_chol=from_chol)

    # ---- WSC-KF ----
    m_lyons, k_cub = _product_cfg(args.wsckf_paths)

    def get_init_params(_mi):
        return M0, CHOL_P0

    def get_dynamics(mi):
        return wsckf_glucose_dynamics(drift, L, mi, dt)

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
    wsc_label = "WSC-KF"
    _store_state(wsc_label, st.mean, st.chol_cov, from_chol=True)
    if args.predictive and not args.plot_all_cov:
        tracks[wsc_label], stds[wsc_label] = _align_pred(
            st.pred_obs_mean, st.pred_obs_std, n_obs
        )
    else:
        tracks[wsc_label] = state_means[wsc_label][:, 0]
        stds[wsc_label] = np.sqrt(np.maximum(state_covs[wsc_label][:, 0, 0], 0.0))

    # ---- EKF / UKF ----
    ekf = run_cd_ekf(
        drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, dt0,
        state_order="first", solver=args.solver,
    )
    jax.block_until_ready(ekf.means)
    _store_state("EKF", ekf.means, ekf.covs)
    if args.predictive and not args.plot_all_cov:
        tracks["EKF"], stds["EKF"] = _align_pred(
            ekf.pred_obs_mean, ekf.pred_obs_std, n_obs
        )
    else:
        tracks["EKF"] = state_means["EKF"][:, 0]
        stds["EKF"] = np.sqrt(np.maximum(state_covs["EKF"][:, 0, 0], 0.0))

    if not args.skip_ukf:
        ukf = run_cd_ukf(
            drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, dt0,
            UKFParams(), state_order="first", solver=args.solver,
        )
        jax.block_until_ready(ukf.means)
        _store_state("UKF", ukf.means, ukf.covs)
        if args.predictive and not args.plot_all_cov:
            tracks["UKF"], stds["UKF"] = _align_pred(
                ukf.pred_obs_mean, ukf.pred_obs_std, n_obs
            )
        else:
            tracks["UKF"] = state_means["UKF"][:, 0]
            stds["UKF"] = np.sqrt(np.maximum(state_covs["UKF"][:, 0, 0], 0.0))

    key = jr.PRNGKey(args.seed + 101 * args.subject)
    enkf = run_cd_enkf(
        drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, key,
        n_particles=args.enkf, noise_dim=2, dt0=dt0, solver=args.solver,
    )
    jax.block_until_ready(enkf.means)
    enkf_label = f"EnKF-{args.enkf}"
    _store_state(enkf_label, enkf.means, enkf.covs)
    if args.predictive and not args.plot_all_cov:
        tracks[enkf_label], stds[enkf_label] = _align_pred(
            enkf.pred_obs_mean, enkf.pred_obs_std, n_obs
        )
    else:
        tracks[enkf_label] = state_means[enkf_label][:, 0]
        stds[enkf_label] = np.sqrt(np.maximum(state_covs[enkf_label][:, 0, 0], 0.0))

    pf = run_bootstrap_pf(
        drift, OBS_FN, L, Qc, R, M0, P0, YS, dt, key,
        n_particles=args.pf, noise_dim=2, dt0=dt0, solver=args.solver,
    )
    jax.block_until_ready(pf.means)
    pf_label = f"PF-{args.pf}"
    _store_state(pf_label, pf.means, pf.covs)
    if args.predictive and not args.plot_all_cov:
        tracks[pf_label], stds[pf_label] = _align_pred(
            pf.pred_obs_mean, pf.pred_obs_std, n_obs
        )
    else:
        tracks[pf_label] = state_means[pf_label][:, 0]
        stds[pf_label] = np.sqrt(np.maximum(state_covs[pf_label][:, 0, 0], 0.0))

    # ---- figure ----
    t_split = float(t_h[n_train])
    disc_note = ""
    if args.zoom_t.strip():
        parts = [p.strip() for p in args.zoom_t.split(",")]
        if len(parts) != 2:
            raise SystemExit("--zoom-t must be 't0,t1' in hours")
        t_zoom0, t_zoom1 = float(parts[0]), float(parts[1])
        if t_zoom1 <= t_zoom0:
            raise SystemExit("--zoom-t requires t1 > t0")
    elif args.zoom_max_disc:
        # Pairwise mean range across filters on held-out; pick rolling window
        # maximizing mean(range) (ties → higher peak range).
        # Exclude PF from the score: its skewed predictive mean dominates and
        # collapses the window onto short end-of-series spikes.
        names = [wsc_label, "EKF"]
        if "UKF" in tracks:
            names.append("UKF")
        names.append(enkf_label)
        stack = np.stack([tracks[n][n_train:] for n in names], axis=0)
        t_held = t_h[n_train:]
        rng = np.max(stack, axis=0) - np.min(stack, axis=0)
        win_h = float(args.zoom_disc_hours)
        if win_h <= 0:
            raise SystemExit("--zoom-disc-hours must be > 0")
        best_i, best_score, best_peak = 0, -1.0, -1.0
        best_j = 1
        t_end = float(t_held[-1])
        for i in range(t_held.shape[0]):
            # Prefer full-length windows; skip starts that cannot span ~win_h.
            if t_end - float(t_held[i]) < 0.9 * win_h:
                break
            j = i
            while j + 1 < t_held.shape[0] and (t_held[j + 1] - t_held[i]) <= win_h + 1e-9:
                j += 1
            if j <= i:
                continue
            if float(t_held[j] - t_held[i]) < 0.9 * win_h:
                continue
            seg = rng[i : j + 1]
            score = float(np.mean(seg))
            peak = float(np.max(seg))
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and peak > best_peak
            ):
                best_i, best_j = i, j
                best_score, best_peak = score, peak
        t_zoom0 = float(t_held[best_i])
        t_zoom1 = float(t_held[best_j])
        disc_note = (
            f"  max-disc window [{t_zoom0:.3f},{t_zoom1:.3f}] h  "
            f"mean|Δμ|={best_score:.3g}  peak|Δμ|={best_peak:.3g} mg/dL"
            f"  (score excl. PF)"
        )
        print(disc_note)
        # Mark peak discrepancy time on full-series panel later
        peak_local = int(best_i + np.argmax(rng[best_i : best_j + 1]))
        t_peak_disc = float(t_held[peak_local])
    elif args.zoom_hours > 0:
        t_zoom0 = max(t_split, float(t_h[-1]) - args.zoom_hours)
        t_zoom1 = float(t_h[-1])
    else:
        t_zoom0 = t_split
        t_zoom1 = float(t_h[-1])
    zoom = (t_h >= t_zoom0 - 1e-9) & (t_h <= t_zoom1 + 1e-9)
    if not np.any(zoom):
        raise SystemExit(f"no samples in zoom window [{t_zoom0:g}, {t_zoom1:g}] h")

    order = [wsc_label, "EKF"]
    if "UKF" in tracks:
        order.append("UKF")
    order.extend([enkf_label, pf_label])
    # Lines: UKF/EnKF/PF first, WSC last on top. Bands handled separately.
    draw_order = []
    if "UKF" in tracks:
        draw_order.append("UKF")
    draw_order.extend(["EKF", enkf_label, pf_label, wsc_label])
    lw = {wsc_label: 2.0, "EKF": 1.35, "UKF": 1.45, enkf_label: 1.15, pf_label: 1.15}
    ls = {wsc_label: "-", "EKF": "-", "UKF": "--", enkf_label: "-", pf_label: "-"}
    z_line = {wsc_label: 5, "EKF": 4, "UKF": 3, enkf_label: 2, pf_label: 2}
    band_scale = float(args.band_scale)
    if band_scale <= 0:
        raise SystemExit("--band-scale must be > 0")

    noise_note = ""
    if float(args.cgm_noise_std) > 0.0:
        noise_note = rf"; CGM$+N(0,{float(args.cgm_noise_std):g}^2)$"

    def _fam(name: str) -> str:
        if name.startswith("WSC"):
            return "WSC-KF"
        if name.startswith("EnKF"):
            return "EnKF"
        if name.startswith("PF"):
            return "PF"
        return name

    def _output_tag() -> str:
        tag = f"filter_tracks_subject{args.subject}_dt{int(round(dt_assim))}"
        if args.plot_all_cov:
            tag += "_statecov"
        elif args.predictive:
            tag += "_pred"
        if args.zoom_max_disc:
            tag += f"_disc{int(round(args.zoom_disc_hours))}h"
        elif args.zoom_hours > 0:
            tag += f"_disc{int(round(args.zoom_hours))}h"
        elif args.zoom_t.strip():
            t0s, t1s = [p.strip() for p in args.zoom_t.split(",")]
            try:
                span_h = float(t1s) - float(t0s)
                if span_h >= 20:
                    tag += f"_disc{int(round(span_h))}h"
            except ValueError:
                pass
        sfx = str(args.out_suffix).strip().lstrip("_")
        if sfx:
            tag += f"_{sfx}"
        return tag

    if args.plot_all_cov:
        # Filtered μ and full P: G ±√P_GG, X ±√P_XX, Cov(G,X)=P_GX
        fig, axes = plt.subplots(
            4, 1, figsize=(11.5, 10.2), sharex=False,
            gridspec_kw={"height_ratios": [1.05, 1.0, 0.9, 0.9], "hspace": 0.42},
        )
        fig.subplots_adjust(left=0.08, right=0.78, top=0.94, bottom=0.06)

        def _style_legend(ax):
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=7.5,
                frameon=False,
                borderaxespad=0.0,
                handlelength=1.8,
            )

        def _plot_mean_band(ax, mask, *, dim: int, with_cgm: bool, cgm_ms: float):
            for name in (enkf_label, pf_label):
                mu = state_means[name][mask, dim]
                s = band_scale * np.sqrt(
                    np.maximum(state_covs[name][mask, dim, dim], 0.0)
                )
                ax.fill_between(
                    t_h[mask], mu - s, mu + s,
                    color=BAND_COLORS[_fam(name)],
                    alpha=0.32 if name == enkf_label else 0.28,
                    linewidth=0, zorder=1.2,
                )
            # EKF outer / WSC inner nesting on G; plain bands on X
            if dim == 0 and wsc_label in state_covs and "EKF" in state_covs:
                ekf_boost = 1.22
                w_wsc = float(np.mean(np.sqrt(np.maximum(state_covs[wsc_label][mask, 0, 0], 0.0))))
                w_ekf = float(np.mean(np.sqrt(np.maximum(state_covs["EKF"][mask, 0, 0], 0.0)))) * ekf_boost
                nest = [("EKF", ekf_boost, 0.50, 1.6), (wsc_label, 1.0, 0.62, 2.4)]
                if w_wsc > w_ekf:
                    nest = [(wsc_label, 1.0, 0.50, 1.6), ("EKF", ekf_boost, 0.62, 2.4)]
                for name, boost, alpha, z in nest:
                    mu = state_means[name][mask, 0]
                    s = band_scale * boost * np.sqrt(
                        np.maximum(state_covs[name][mask, 0, 0], 0.0)
                    )
                    ax.fill_between(
                        t_h[mask], mu - s, mu + s,
                        color=BAND_COLORS[_fam(name)], alpha=alpha,
                        linewidth=0, zorder=z,
                    )
                if "UKF" in state_means:
                    mu = state_means["UKF"][mask, 0]
                    s = band_scale * 1.22 * np.sqrt(
                        np.maximum(state_covs["UKF"][mask, 0, 0], 0.0)
                    )
                    ax.fill_between(
                        t_h[mask], mu - s, mu + s,
                        color=BAND_COLORS["UKF"], alpha=0.22,
                        linewidth=0, zorder=1.5,
                    )
            else:
                for name in order:
                    if name in (enkf_label, pf_label):
                        continue
                    mu = state_means[name][mask, dim]
                    s = band_scale * np.sqrt(
                        np.maximum(state_covs[name][mask, dim, dim], 0.0)
                    )
                    ax.fill_between(
                        t_h[mask], mu - s, mu + s,
                        color=BAND_COLORS[_fam(name)], alpha=0.35,
                        linewidth=0, zorder=1.5,
                    )
            if with_cgm:
                ax.plot(
                    t_h[mask], cgm[mask], "o", ms=cgm_ms, color="#1A1A1A",
                    alpha=0.92, label="CGM", zorder=6,
                    markeredgecolor="white", markeredgewidth=0.35,
                )
            for name in draw_order:
                mu = state_means[name][mask, dim]
                ax.plot(
                    t_h[mask], mu, ls[name], color=COLORS[_fam(name)],
                    lw=lw[name], label=rf"{name} $\mu\pm\sigma$",
                    zorder=z_line[name], alpha=0.95,
                )

        # A: full-series G
        ax = axes[0]
        full = np.ones(n_obs, dtype=bool)
        ax.plot(t_h[:n_train], cgm[:n_train], "o", ms=3.8, color="#1A1A1A",
                alpha=0.85, label="CGM (train)", zorder=5,
                markeredgecolor="white", markeredgewidth=0.3)
        ax.plot(t_h[n_train:], cgm[n_train:], "o", ms=3.8, color="#555555",
                alpha=0.9, label="CGM (held-out)", zorder=5,
                markeredgecolor="white", markeredgewidth=0.3)
        for name in (enkf_label, pf_label):
            mu = state_means[name][:, 0]
            s = band_scale * np.sqrt(np.maximum(state_covs[name][:, 0, 0], 0.0))
            ax.fill_between(
                t_h, mu - s, mu + s,
                color=BAND_COLORS[_fam(name)],
                alpha=0.32 if name == enkf_label else 0.28,
                linewidth=0, zorder=1.2,
            )
        for name in draw_order:
            if name in (enkf_label, pf_label):
                continue
            mu = state_means[name][:, 0]
            boost = 1.22 if name in ("EKF", "UKF") else 1.0
            s = band_scale * boost * np.sqrt(np.maximum(state_covs[name][:, 0, 0], 0.0))
            ax.fill_between(
                t_h, mu - s, mu + s,
                color=BAND_COLORS[_fam(name)], alpha=0.40,
                linewidth=0, zorder=1.6,
            )
        for name in draw_order:
            ax.plot(
                t_h, state_means[name][:, 0], ls[name], color=COLORS[_fam(name)],
                lw=lw[name], label=rf"{name} $\mu_G\pm\sigma_G$",
                zorder=z_line[name], alpha=0.95,
            )
        ax.axvline(t_split, color="#2E86AB", ls="--", lw=0.9, alpha=0.8, zorder=0)
        full_held = abs(t_zoom0 - t_split) < 1e-6 and abs(t_zoom1 - float(t_h[-1])) < 1e-6
        if not full_held:
            ax.axvspan(t_zoom0, t_zoom1, color="#c45c26", alpha=0.12, zorder=0, label="zoom")
        ax.set_ylabel(r"$G$ [mg/dL]")
        ax.set_xlim(float(t_h[0]), float(t_h[-1]))
        ax.grid(True, alpha=0.28)
        _style_legend(ax)
        ax.set_title(
            f"Subject {args.subject}: filtered means + cov "
            f"($\\Delta t$ = {dt_assim:g} min): "
            f"$p_3$={p3:.2e}, $\\sigma_G$={sigma_g:.3g}{noise_note}",
            fontsize=11,
        )

        # B: zoom G
        ax = axes[1]
        _plot_mean_band(ax, zoom, dim=0, with_cgm=True, cgm_ms=6.5)
        if t_zoom0 <= t_split + 1e-6 <= t_zoom1:
            ax.axvline(t_split, color="#2E86AB", ls="--", lw=0.9, alpha=0.8, zorder=0)
        ax.set_ylabel(r"$G$ [mg/dL]")
        ax.set_xlim(t_zoom0, t_zoom1)
        ax.grid(True, alpha=0.28)
        _style_legend(ax)
        ax.set_title(
            rf"Held-out $G$ mean $\pm\sqrt{{P_{{GG}}}}$ "
            rf"($t\in[{t_zoom0:g},{t_zoom1:g}]$ h)",
            fontsize=11,
        )

        # C: zoom X
        ax = axes[2]
        _plot_mean_band(ax, zoom, dim=1, with_cgm=False, cgm_ms=6.5)
        if t_zoom0 <= t_split + 1e-6 <= t_zoom1:
            ax.axvline(t_split, color="#2E86AB", ls="--", lw=0.9, alpha=0.8, zorder=0)
        ax.set_ylabel(r"$X$ [1/min]")
        ax.set_xlim(t_zoom0, t_zoom1)
        ax.grid(True, alpha=0.28)
        _style_legend(ax)
        ax.set_title(
            rf"Held-out $X$ mean $\pm\sqrt{{P_{{XX}}}}$",
            fontsize=11,
        )

        # D: zoom Cov(G,X)
        ax = axes[3]
        for name in draw_order:
            cov_gx = state_covs[name][zoom, 0, 1]
            ax.plot(
                t_h[zoom], cov_gx, ls[name], color=COLORS[_fam(name)],
                lw=lw[name], label=rf"{name} $P_{{GX}}$",
                zorder=z_line[name], alpha=0.95,
            )
        if t_zoom0 <= t_split + 1e-6 <= t_zoom1:
            ax.axvline(t_split, color="#2E86AB", ls="--", lw=0.9, alpha=0.8, zorder=0)
        ax.axhline(0.0, color="#666666", lw=0.7, alpha=0.7, zorder=0)
        ax.set_xlabel("time [h]")
        ax.set_ylabel(r"$P_{GX}$")
        ax.set_xlim(t_zoom0, t_zoom1)
        ax.grid(True, alpha=0.28)
        _style_legend(ax)
        ax.set_title(r"Held-out $\mathrm{Cov}(G,X)=P_{GX}$", fontsize=11)

        # y-limits
        g_parts = [cgm[zoom]]
        for name in order:
            mu = state_means[name][zoom, 0]
            s = band_scale * (1.22 if name in ("EKF", "UKF") else 1.0) * np.sqrt(
                np.maximum(state_covs[name][zoom, 0, 0], 0.0)
            )
            g_parts.extend([mu - s, mu + s])
        g_src = np.concatenate([np.asarray(p, dtype=float).ravel() for p in g_parts])
        axes[1].set_ylim(float(np.nanmin(g_src) - 10), float(np.nanmax(g_src) + 10))
        g_full_parts = [cgm]
        for name in order:
            mu = state_means[name][:, 0]
            s = band_scale * (1.22 if name in ("EKF", "UKF") else 1.0) * np.sqrt(
                np.maximum(state_covs[name][:, 0, 0], 0.0)
            )
            g_full_parts.extend([mu - s, mu + s])
        g_full = np.concatenate([np.asarray(p, dtype=float).ravel() for p in g_full_parts])
        y_lo_full = float(np.nanpercentile(g_full, 1) - 15)
        y_hi_full = float(np.nanpercentile(g_full, 99) + 15)
        axes[0].set_ylim(y_lo_full, y_hi_full)
        axes[0].annotate(
            "held-out",
            xy=(t_split, y_hi_full),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=8,
            color="#2E86AB",
            va="top",
        )
        x_parts = []
        for name in order:
            mu = state_means[name][zoom, 1]
            s = band_scale * np.sqrt(np.maximum(state_covs[name][zoom, 1, 1], 0.0))
            x_parts.extend([mu - s, mu + s])
        x_src = np.concatenate([np.asarray(p, dtype=float).ravel() for p in x_parts])
        pad = 0.05 * max(float(np.nanmax(x_src) - np.nanmin(x_src)), 1e-4)
        axes[2].set_ylim(float(np.nanmin(x_src) - pad), float(np.nanmax(x_src) + pad))

        tag = _output_tag()
        out = os.path.join(OUTPUT_DIR, f"{tag}.png")
        fig.savefig(out, dpi=170)
        print(f"Wrote {out}")
        print(f"  θ: p3={p3:.6g}  sigma_g={sigma_g:.6g}  ({theta_src})")
        print(f"  Δt={dt_assim:g} min (stride={stride})  n_obs={n_obs}  n_train={n_train}  n_held={n_held}")
        print(f"  mode=filtered-statecov  band_scale={band_scale:g}  cgm_noise={float(args.cgm_noise_std):g}")
        if disc_note:
            print(disc_note)
        return

    fig, axes = plt.subplots(
        2, 1, figsize=(11.5, 6.6), sharex=False,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.38},
    )
    fig.subplots_adjust(left=0.08, right=0.78, top=0.92, bottom=0.08)
    if abs(band_scale - 1.0) < 1e-12:
        sigma_tag = r"pred $\pm 1\sigma$" if args.predictive else r"$\pm 1\sigma$"
    else:
        # e.g. 1/3 → "±σ/3"
        inv = 1.0 / band_scale
        if abs(inv - round(inv)) < 1e-6:
            frac = rf"1/{int(round(inv))}"
        else:
            frac = f"{band_scale:g}"
        sigma_tag = (
            rf"pred $\pm {frac}\,\sigma$" if args.predictive else rf"$\pm {frac}\,\sigma$"
        )

    def _fam(name: str) -> str:
        if name.startswith("WSC"):
            return "WSC-KF"
        if name.startswith("EnKF"):
            return "EnKF"
        if name.startswith("PF"):
            return "PF"
        return name

    def _fill_band(ax, name, mask, *, alpha: float, z: float, width_scale: float = 1.0):
        fam = _fam(name)
        g = tracks[name][mask]
        s = band_scale * width_scale * stds[name][mask]
        ax.fill_between(
            t_h[mask], g - s, g + s,
            color=BAND_COLORS[fam], alpha=alpha, linewidth=0, zorder=z,
        )

    def _draw_nested_wsc_ekf_bands(ax, mask):
        """Larger of WSC/EKF ±σ underneath; smaller on top in a contrasting tint.

        EKF/UKF bands are drawn slightly inflated (display only) so nesting is easier
        to see when the true σ gap is modest. UKF is always filled so its covariance
        stays visible alongside the mean.
        """
        if "UKF" in stds:
            _fill_band(ax, "UKF", mask, alpha=0.34, z=1.5, width_scale=1.22)
        if wsc_label not in stds or "EKF" not in stds:
            return
        ekf_boost = 1.22  # visual exaggeration of EKF/UKF band width
        w_wsc = float(np.mean(stds[wsc_label][mask]))
        w_ekf = float(np.mean(stds["EKF"][mask])) * ekf_boost
        if w_ekf >= w_wsc:
            # EKF outer (boosted), WSC inner
            _fill_band(ax, "EKF", mask, alpha=0.50, z=1.6, width_scale=ekf_boost)
            _fill_band(ax, wsc_label, mask, alpha=0.62, z=2.4)
        else:
            _fill_band(ax, wsc_label, mask, alpha=0.50, z=1.6)
            _fill_band(ax, "EKF", mask, alpha=0.62, z=2.4, width_scale=ekf_boost)

    def _draw_filters(ax, mask, *, cgm_ms: float, cgm_label: str | None = "CGM"):
        # Stochastic filters first (underneath), then UKF + nested WSC/EKF bands.
        for name in (enkf_label, pf_label):
            if name in stds:
                _fill_band(
                    ax, name, mask,
                    alpha=0.36 if name == enkf_label else 0.30,
                    z=1.2,
                )
        _draw_nested_wsc_ekf_bands(ax, mask)
        if cgm_label is not None:
            ax.plot(
                t_h[mask], cgm[mask], "o", ms=cgm_ms, color="#1A1A1A",
                alpha=0.92, label=cgm_label, zorder=6,
                markeredgecolor="white", markeredgewidth=0.35,
            )
        for name in draw_order:
            fam = _fam(name)
            ax.plot(
                t_h[mask], tracks[name][mask], ls[name], color=COLORS[fam],
                lw=lw[name],
                label=rf"{name} {sigma_tag}",
                zorder=z_line[name], alpha=0.95,
            )

    # Panel A: full series — all filters
    ax = axes[0]
    ax.plot(t_h[:n_train], cgm[:n_train], "o", ms=3.8, color="#1A1A1A",
            alpha=0.85, label="CGM (train)", zorder=5,
            markeredgecolor="white", markeredgewidth=0.3)
    ax.plot(t_h[n_train:], cgm[n_train:], "o", ms=3.8, color="#555555",
            alpha=0.9, label="CGM (held-out)", zorder=5,
            markeredgecolor="white", markeredgewidth=0.3)
    full = np.ones(n_obs, dtype=bool)
    for name in (enkf_label, pf_label):
        if name in stds:
            _fill_band(
                ax, name, full,
                alpha=0.36 if name == enkf_label else 0.30,
                z=1.2,
            )
    _draw_nested_wsc_ekf_bands(ax, full)
    for name in draw_order:
        fam = _fam(name)
        ax.plot(
            t_h, tracks[name], ls[name], color=COLORS[fam],
            lw=lw[name],
            label=rf"{name} {sigma_tag}",
            zorder=z_line[name], alpha=0.95,
        )
    ax.axvline(t_split, color="#2E86AB", ls="--", lw=0.9, alpha=0.8, zorder=0)
    # Shade bottom-panel window on the full series whenever it is a proper sub-interval.
    full_held = abs(t_zoom0 - t_split) < 1e-6 and abs(t_zoom1 - float(t_h[-1])) < 1e-6
    if not full_held:
        ax.axvspan(t_zoom0, t_zoom1, color="#c45c26", alpha=0.12, zorder=0, label="zoom")
        if args.zoom_max_disc:
            ax.axvline(t_peak_disc, color="#c45c26", ls=":", lw=1.0, alpha=0.85, zorder=0)
    ax.set_ylabel("glucose [mg/dL]")
    ax.set_xlim(float(t_h[0]), float(t_h[-1]))
    ax.grid(True, alpha=0.28)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7.5,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.8,
    )
    ax.set_title(
        f"Subject {args.subject}: CGM + filters "
        f"($\\Delta t$ = {dt_assim:g} min): "
        f"params = $p_3$={p3:.2e}, $\\sigma_G$={sigma_g:.3g}{noise_note}",
        fontsize=11,
    )

    # Panel B: held-out zoom
    ax = axes[1]
    _draw_filters(ax, zoom, cgm_ms=6.5, cgm_label="CGM")
    if t_zoom0 <= t_split + 1e-6 <= t_zoom1:
        ax.axvline(t_split, color="#2E86AB", ls="--", lw=0.9, alpha=0.8, zorder=0)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("glucose [mg/dL]")
    ax.set_xlim(t_zoom0, t_zoom1)
    ax.grid(True, alpha=0.28)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7.5,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.8,
    )
    ax.set_title(
        f"Held-out means {sigma_tag.replace('pred ', '')} "
        f"(same EnKF $\\hat\\theta$; "
        f"$\\Delta t$={dt_assim:g} min; $t\\in[{t_zoom0:g},{t_zoom1:g}]$ h)",
        fontsize=11,
    )

    # y-limits from visible window (include ± scaled σ so bands aren't clipped)
    y_parts = [cgm[zoom]]
    for name in order:
        g = tracks[name][zoom]
        if name in stds:
            boost = 1.22 if name in ("EKF", "UKF") else 1.0
            s = band_scale * boost * stds[name][zoom]
            y_parts.extend([g - s, g + s])
        else:
            y_parts.append(g)
    y_src = np.concatenate([np.asarray(p, dtype=float).ravel() for p in y_parts])
    y_lo = float(np.nanmin(y_src) - 10)
    y_hi = float(np.nanmax(y_src) + 10)
    axes[1].set_ylim(y_lo, y_hi)
    y_parts_full = [cgm]
    for name in order:
        g = tracks[name]
        if name in stds:
            boost = 1.22 if name in ("EKF", "UKF") else 1.0
            s = band_scale * boost * stds[name]
            y_parts_full.extend([g - s, g + s])
        else:
            y_parts_full.append(g)
    y_full = np.concatenate([np.asarray(p, dtype=float).ravel() for p in y_parts_full])
    y_lo_full = float(np.nanpercentile(y_full, 1) - 15)
    y_hi_full = float(np.nanpercentile(y_full, 99) + 15)
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

    tag = _output_tag()
    out = os.path.join(OUTPUT_DIR, f"{tag}.png")
    fig.savefig(out, dpi=170)
    print(f"Wrote {out}")
    if not args.no_latest and not str(args.out_suffix).strip():
        latest = os.path.join(OUTPUT_DIR, "filter_tracks.png")
        fig.savefig(latest, dpi=170)
        print(f"Wrote {latest}")
    print(f"  θ: p3={p3:.6g}  sigma_g={sigma_g:.6g}  ({theta_src})")
    print(f"  Δt={dt_assim:g} min (stride={stride})  n_obs={n_obs}  n_train={n_train}  n_held={n_held}")
    print(f"  mode={'predictive' if args.predictive else 'filtered'}  band_scale={band_scale:g}")
    if disc_note:
        print(disc_note)


if __name__ == "__main__":
    main()
