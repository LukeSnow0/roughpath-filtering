"""Temporal weak-error δ-scaling of WSC-KF vs CD filters on Heston ``(S, v)``.

Textbook Itô Heston (not log-price)::

    dS = μ S dt + S √v dW¹
    dv = κ(θ − v) dt + ξ √v (ρ dW¹ + √(1−ρ²) dW²)

``E[LLᵀ]_{SS} = E[S² v]`` does not close at second order, so CD freeze-at-mean
moment ODEs are *not* the true covariance equations (unlike ``(log S, v)``).

Protocol: one predict step from a point mass ``P0 = 0``.
- Mean errors use the closed affine mean (:func:`heston_model.exact_heston_sv_mean`).
- Cov errors use a high-budget antithetic MC reference with the same C^∞
  diffusion map as the filters.

Diffusion uses ``φ(v)=(v²+ε²)^{1/4}`` in place of ``√v`` (``ε=θ``).

Cubature paths are explicit multi-D Lyons–Victoir (``multidim_method="lv"``).

TME-2 / TME-3 (Zhao et al., TAC 2021) are overlaid as generator-Taylor
point-mass predictors of the same Itô dynamics.

Run:
  uv run --extra experiments python experiments/heston_test/weak_error_delta_scaling.py
  uv run --extra experiments python experiments/heston_test/weak_error_delta_scaling.py \\
      --mc-paths 4000000 --mc-dt 1e-4
  uv run --extra experiments python experiments/heston_test/weak_error_delta_scaling.py \\
      --plot-from-csv weak_error_delta_scaling_sv.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, ROOT_DIR)

from cd_dynamax_mirror.filters import UKFParams, _ekf_predict, _ukf_predict, _ukf_weights
from heston_model import (
    HestonParams,
    exact_heston_sv_mean,
    make_heston_sv_dynamics,
    mc_heston_sv_mean_cov,
)
from tme import tme_mean_cov
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAMS = HestonParams()
V0 = float(PARAMS.theta if PARAMS.v0 is None else PARAMS.v0)
S0 = float(np.exp(PARAMS.x0))
X0 = np.array([S0, V0], dtype=np.float64)

INNER_STEPS = 128
SMOOTH_VOL_EPS = float(PARAMS.theta)
ITO_DRIFT, DIFFUSION, QC_FN, _, _ = make_heston_sv_dynamics(
    PARAMS, smooth_vol_eps=SMOOTH_VOL_EPS
)
STRAT_DRIFT, STRAT_DIFF = ito_to_stratonovich_dynamics(ITO_DRIFT, DIFFUSION)

CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)
TME_ORDERS = (2, 3)
TME_COLORS = {2: "#E69F00", 3: "#8DA35B"}
TME_FMTS = {2: "v--", 3: "P--"}
MC_CACHE_DIR = os.path.join(OUTPUT_DIR, "mc_cache")
os.makedirs(MC_CACHE_DIR, exist_ok=True)


def wsckf_predict(delta: float, lyons_degree: int, k: int = 1):
    integration_dt = delta / INNER_STEPS
    ts, path_values, path_weights = precompute_cubature_paths(
        horizon=float(delta),
        noise_dim=2,
        integration_dt=integration_dt,
        cubature_order=int(k),
        multidim_method="lv",
        lyons_degree=lyons_degree,
    )
    mean = jnp.asarray(X0)
    chol_cov = jnp.zeros((2, 2))
    pred_mean, pred_chol, _, _ = predict(
        mean,
        chol_cov,
        jnp.asarray(ts),
        jnp.asarray(path_values),
        jnp.asarray(path_weights),
        STRAT_DRIFT,
        STRAT_DIFF,
    )
    P = pred_chol @ pred_chol.T
    return np.asarray(pred_mean), np.asarray(P)


def cd_predict(delta: float, which: str):
    m = jnp.asarray(X0)
    P = jnp.zeros((2, 2))
    n_inner = INNER_STEPS
    dt0 = jnp.asarray(delta / n_inner, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(delta, dtype=jnp.float64)
    if which == "ekf":
        m_pred, P_pred = _ekf_predict(
            m,
            P,
            ITO_DRIFT,
            DIFFUSION,
            QC_FN,
            t0,
            t1,
            dt0,
            state_order="first",
            solver="rk4",
            n_steps=n_inner,
        )
    else:
        n = 2
        p = CKF_PARAMS if which == "ckf" else UKFParams()
        lamb = p.alpha**2 * (n + p.kappa) - n
        w_mean, w_cov, W_matrix = _ukf_weights(n, p.alpha, p.beta, lamb)
        m_pred, P_pred = _ukf_predict(
            m,
            P,
            ITO_DRIFT,
            DIFFUSION,
            QC_FN,
            t0,
            t1,
            dt0,
            n,
            lamb,
            w_mean,
            w_cov,
            W_matrix,
            state_order="first",
            solver="rk4",
            n_steps=n_inner,
        )
    return np.asarray(m_pred), np.asarray(P_pred)


def tme_predict(delta: float, order: int):
    """Point-mass TME-M predict (Zhao et al., TAC 2021)."""
    return tme_mean_cov(X0, float(delta), int(order), ITO_DRIFT, DIFFUSION)


def _mc_cache_path(delta: float, *, n_paths: int, n_steps: int, seed: int, n_seeds: int) -> str:
    tag = (
        f"sv_d{float(delta):.12g}_n{int(n_paths)}_steps{int(n_steps)}"
        f"_seed{int(seed)}_seeds{int(n_seeds)}_eps{SMOOTH_VOL_EPS:g}"
    )
    return os.path.join(MC_CACHE_DIR, f"{tag}.npz")


def mc_ref_cov(
    delta: float,
    *,
    n_paths: int,
    n_steps: int,
    seed: int,
    n_seeds: int,
    batch_paths: int,
) -> np.ndarray:
    """High-budget MC covariance, cached under ``output/mc_cache/``."""
    path = _mc_cache_path(
        delta, n_paths=n_paths, n_steps=n_steps, seed=seed, n_seeds=n_seeds
    )
    if os.path.isfile(path):
        data = np.load(path)
        return np.asarray(data["P_ref"], dtype=np.float64)

    P_acc = np.zeros((2, 2), dtype=np.float64)
    for s in range(max(1, int(n_seeds))):
        _, P_s = mc_heston_sv_mean_cov(
            float(delta),
            X0,
            PARAMS,
            n_paths=int(n_paths),
            n_steps=int(n_steps),
            seed=int(seed) + s,
            smooth_vol_eps=SMOOTH_VOL_EPS,
            batch_paths=int(batch_paths),
        )
        P_acc += np.asarray(P_s, dtype=np.float64)
    P_ref = P_acc / float(max(1, int(n_seeds)))
    np.savez(path, P_ref=P_ref)
    return P_ref


def fit_slope(deltas, errs, dmin=None, dmax=0.35, floor=1e-15):
    d = np.asarray(deltas, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    mask = (e > floor) & (d <= dmax)
    if dmin is not None:
        mask &= d >= float(dmin)
    if mask.sum() < 2:
        mask = e > floor
    if mask.sum() < 2:
        return float("nan")
    if np.all(e[mask] <= floor * 10):
        return float("nan")
    return float(np.polyfit(np.log(d[mask]), np.log(e[mask]), 1)[0])


def guide(ax, deltas, anchor_err, anchor_d, slope, label, color="k", ls="--"):
    d = np.asarray(deltas)
    c = anchor_err / (anchor_d**slope)
    (line,) = ax.loglog(
        d, c * d**slope, ls, color=color, alpha=0.75, lw=1.6, label=label
    )
    return line


def _plot_scaling(
    deltas,
    series: dict[str, tuple[np.ndarray, str, str]],
    *,
    ylabel: str,
    png_path: str,
    guides: list[tuple[float, np.ndarray, str, str]],
    ncol: int = 3,
    fit_dmin: float | None = None,
    fit_dmax: float = 0.35,
    xlim: tuple[float, float] | None = None,
):
    AXIS_FS, LEGEND_FS, TICK_FS = 13, 11, 11
    rng_floor = 1e-18
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    handles = []
    for name, (errs, color, fmt) in series.items():
        slope = fit_slope(deltas, errs, dmin=fit_dmin, dmax=fit_dmax)
        label = name if not np.isfinite(slope) else rf"{name} (fit={slope:.2f})"
        (ln,) = ax.loglog(
            deltas, np.maximum(errs, rng_floor), fmt, color=color, label=label
        )
        handles.append(ln)
    guide_handles = []
    for slope, anchor_errs, lab, ls in guides:
        mid = min(8, len(deltas) - 1)
        guide_handles.append(
            guide(
                ax,
                deltas,
                max(float(anchor_errs[mid]), rng_floor),
                float(deltas[mid]),
                slope,
                lab,
                color="0.35",
                ls=ls,
            )
        )
    ax.set_xlabel(r"assimilation interval $\tau$", fontsize=AXIS_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    x0, x1 = (deltas[0], deltas[-1]) if xlim is None else xlim
    ax.set_xlim(x0, x1)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(
        handles=[*handles, *guide_handles],
        fontsize=LEGEND_FS,
        ncol=ncol,
        columnspacing=1.0,
        handletextpad=0.4,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_scaling_plots(
    plot_rows: list[dict[str, float]],
    *,
    plot_tag: str,
    fit_dmin: float,
    fit_dmax: float,
    xlim: tuple[float, float] | None,
    skip_cov: bool,
    wsckf_colors: dict[int, str] | None = None,
):
    if wsckf_colors is None:
        wsckf_colors = {3: "#6A4C93", 5: "#9B5DE5"}

    def _col(name: str) -> np.ndarray:
        return np.asarray([float(r[name]) for r in plot_rows], dtype=np.float64)

    def _nonzero(series: dict) -> dict:
        return {
            name: spec
            for name, spec in series.items()
            if np.nanmax(np.asarray(spec[0], dtype=np.float64)) > 1e-15
        }

    plot_deltas = _col("delta")
    mean_series = _nonzero(
        {
            r"WSC-KF $m$=3": (_col("wsckf_mean_err_m3"), wsckf_colors[3], "o-"),
            r"WSC-KF $m$=5": (_col("wsckf_mean_err_m5"), wsckf_colors[5], "o-"),
            r"TME-2": (_col("tme_mean_err_m2"), TME_COLORS[2], TME_FMTS[2]),
            r"TME-3": (_col("tme_mean_err_m3"), TME_COLORS[3], TME_FMTS[3]),
            "CD-EKF": (_col("ekf_mean_err"), "C3", "s-"),
            "CD-UKF": (_col("ukf_mean_err"), "C0", "^-"),
            "CD-CKF": (_col("ckf_mean_err"), "#2E86AB", "D-"),
        }
    )
    mean_guides = [
        (2.0, _col("wsckf_mean_err_m3"), r"$\propto\tau^{2}$", "--"),
        (3.0, _col("wsckf_mean_err_m5"), r"$\propto\tau^{3}$", ":"),
    ]
    if "TME-3" in mean_series:
        mean_guides.append(
            (4.0, _col("tme_mean_err_m3"), r"$\propto\tau^{4}$", "-.")
        )
    png_mean = os.path.join(OUTPUT_DIR, f"weak_error_delta_scaling_sv_mean{plot_tag}.png")
    _plot_scaling(
        plot_deltas,
        mean_series,
        ylabel=r"$\|\hat m - m\|_2$",
        png_path=png_mean,
        guides=mean_guides,
        ncol=3,
        fit_dmin=fit_dmin,
        fit_dmax=fit_dmax,
        xlim=xlim,
    )
    print(f"\nSaved plot -> {png_mean}")

    if skip_cov:
        return
    cov_series = {
        r"WSC-KF $m$=3": (_col("wsckf_cov_err_m3"), wsckf_colors[3], "o-"),
        r"WSC-KF $m$=5": (_col("wsckf_cov_err_m5"), wsckf_colors[5], "o-"),
        r"TME-2": (_col("tme_cov_err_m2"), TME_COLORS[2], TME_FMTS[2]),
        r"TME-3": (_col("tme_cov_err_m3"), TME_COLORS[3], TME_FMTS[3]),
        "CD-EKF": (_col("ekf_cov_err"), "C3", "s-"),
        "CD-UKF": (_col("ukf_cov_err"), "C0", "^-"),
        "CD-CKF": (_col("ckf_cov_err"), "#2E86AB", "D-"),
    }
    png_cov = os.path.join(OUTPUT_DIR, f"weak_error_delta_scaling_sv_cov{plot_tag}.png")
    _plot_scaling(
        plot_deltas,
        cov_series,
        ylabel=r"$\|\hat P - P\|_F$",
        png_path=png_cov,
        guides=[
            (2.0, _col("ekf_cov_err"), r"$\propto\tau^{2}$", "--"),
            (3.0, _col("wsckf_cov_err_m5"), r"$\propto\tau^{3}$", ":"),
            (4.0, _col("tme_cov_err_m3"), r"$\propto\tau^{4}$", "-."),
        ],
        ncol=3,
        fit_dmin=fit_dmin,
        fit_dmax=fit_dmax,
        xlim=xlim,
    )
    print(f"Saved plot -> {png_cov}")


def main():
    parser = argparse.ArgumentParser(description="Heston (S,v) weak-error δ-scaling")
    parser.add_argument(
        "--mc-paths",
        type=int,
        default=2_000_000,
        dest="mc_paths",
        help="Independent Z draws for cov MC (antithetic → 2× trajectories)",
    )
    parser.add_argument(
        "--mc-steps",
        type=int,
        default=2048,
        dest="mc_steps",
        help="Fixed Euler steps for every δ (shared across methods)",
    )
    parser.add_argument(
        "--mc-batch",
        type=int,
        default=150_000,
        dest="mc_batch",
        help="Path batch size for streaming MC moment accumulation",
    )
    parser.add_argument(
        "--mc-dt",
        type=float,
        default=1e-4,
        dest="mc_dt",
        help="Unused if --mc-steps is set; kept for CLI compatibility",
    )
    parser.add_argument("--mc-seed", type=int, default=0, dest="mc_seed")
    parser.add_argument(
        "--mc-seeds",
        type=int,
        default=1,
        dest="mc_seeds",
        help="Independent MC seeds averaged into P_ref (reduces floor)",
    )
    parser.add_argument(
        "--delta-min",
        type=float,
        default=0.15,
        dest="delta_min",
        help="Smallest assimilation interval δ",
    )
    parser.add_argument(
        "--delta-max",
        type=float,
        default=0.55,
        dest="delta_max",
        help="Largest assimilation interval δ",
    )
    parser.add_argument(
        "--n-deltas",
        type=int,
        default=10,
        dest="n_deltas",
        help="Number of log-spaced δ values",
    )
    parser.add_argument(
        "--fit-dmin",
        type=float,
        default=0.15,
        dest="fit_dmin",
        help="Lower δ for reported/plot slope fits",
    )
    parser.add_argument(
        "--fit-dmax",
        type=float,
        default=0.50,
        dest="fit_dmax",
        help="Upper δ for reported/plot slope fits",
    )
    parser.add_argument(
        "--skip-cov",
        action="store_true",
        dest="skip_cov",
        help="Only compute mean errors (no MC cov)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for output filenames (e.g. _zoom)",
    )
    parser.add_argument(
        "--append-csv",
        type=str,
        default="",
        dest="append_csv",
        help="Merge new rows into this existing CSV (by delta); also rewrite plots from merged data",
    )
    parser.add_argument(
        "--xlim-min",
        type=float,
        default=None,
        dest="xlim_min",
        help="Optional plot x-axis lower limit",
    )
    parser.add_argument(
        "--xlim-max",
        type=float,
        default=None,
        dest="xlim_max",
        help="Optional plot x-axis upper limit",
    )
    parser.add_argument(
        "--plot-from-csv",
        type=str,
        default="",
        dest="plot_from_csv",
        help="Only rewrite mean/cov plots from an existing CSV (no new solves)",
    )
    args = parser.parse_args()

    if args.plot_from_csv:
        csv_in = args.plot_from_csv
        if not os.path.isabs(csv_in):
            csv_in = os.path.join(OUTPUT_DIR, csv_in)
        plot_rows = []
        with open(csv_in, newline="") as f:
            for r in csv.DictReader(f):
                plot_rows.append({k: float(r[k]) for k in r if r[k] != ""})
        base = os.path.basename(csv_in)
        prefix = "weak_error_delta_scaling_sv"
        plot_tag = args.tag
        if base.startswith(prefix) and base.endswith(".csv"):
            plot_tag = base[len(prefix) : -4]
        xlim = None
        if args.xlim_min is not None or args.xlim_max is not None:
            xlim = (
                float(args.xlim_min if args.xlim_min is not None else args.delta_min),
                float(args.xlim_max if args.xlim_max is not None else args.delta_max),
            )
        _write_scaling_plots(
            plot_rows,
            plot_tag=plot_tag,
            fit_dmin=args.fit_dmin,
            fit_dmax=args.fit_dmax,
            xlim=xlim,
            skip_cov=args.skip_cov or ("wsckf_cov_err_m3" not in plot_rows[0]),
        )
        return

    deltas = np.logspace(np.log10(args.delta_min), np.log10(args.delta_max), int(args.n_deltas))
    wsckf_degrees = (3, 5)
    wsckf_colors = {3: "#6A4C93", 5: "#9B5DE5"}
    tag = args.tag
    xlim = None
    if args.xlim_min is not None or args.xlim_max is not None:
        xlim = (
            float(args.xlim_min if args.xlim_min is not None else args.delta_min),
            float(args.xlim_max if args.xlim_max is not None else args.delta_max),
        )

    print(
        f"Heston (S,v) C^∞-vol weak-error δ-scaling  "
        f"φ(v)=(v²+ε²)^{{1/4}} ε={SMOOTH_VOL_EPS:g}  "
        f"κ={PARAMS.kappa:g} θ={PARAMS.theta:g} ξ={PARAMS.xi:g} ρ={PARAMS.rho:g}  "
        f"x0={X0.tolist()}  (P0=0)\n"
        f"δ ∈ [{args.delta_min:g}, {args.delta_max:g}] ({len(deltas)} pts); "
        f"fit window [{args.fit_dmin:g}, {args.fit_dmax:g}]\n"
        f"MC cov: n_paths={args.mc_paths} (×2 antithetic) ×{args.mc_seeds} seed(s), "
        f"steps={args.mc_steps}, batch={args.mc_batch}, smooth_vol_eps={SMOOTH_VOL_EPS:g}\n"
    )

    # --- Mean (exact reference) ---
    print("Compiling TME generator iterates (JAX jacfwd)...", flush=True)
    tme_predict(float(deltas[0]), order=max(TME_ORDERS))
    print("  TME ready.", flush=True)

    ws_mean_err: dict[int, np.ndarray] = {}
    ws_cov_err: dict[int, np.ndarray] = {m: [] for m in wsckf_degrees}
    tme_mean_err: dict[int, np.ndarray] = {}
    tme_cov_err: dict[int, list[float]] = {m: [] for m in TME_ORDERS}
    for m in wsckf_degrees:
        em = []
        for d in deltas:
            m_ref = exact_heston_sv_mean(float(d), X0, PARAMS)
            m_hat, _ = wsckf_predict(float(d), lyons_degree=m, k=1)
            em.append(float(np.linalg.norm(m_hat - m_ref)))
        ws_mean_err[m] = np.array(em)

    for m in TME_ORDERS:
        em = []
        for d in deltas:
            m_ref = exact_heston_sv_mean(float(d), X0, PARAMS)
            m_hat, _ = tme_predict(float(d), order=m)
            em.append(float(np.linalg.norm(m_hat - m_ref)))
        tme_mean_err[m] = np.array(em)

    ekf_mean, ukf_mean, ckf_mean = [], [], []
    ekf_cov, ukf_cov, ckf_cov = [], [], []
    for d in deltas:
        m_ref = exact_heston_sv_mean(float(d), X0, PARAMS)
        for which, em_list in (
            ("ekf", ekf_mean),
            ("ukf", ukf_mean),
            ("ckf", ckf_mean),
        ):
            m_hat, _ = cd_predict(float(d), which)
            em_list.append(float(np.linalg.norm(m_hat - m_ref)))
    ekf_mean, ukf_mean, ckf_mean = map(np.array, (ekf_mean, ukf_mean, ckf_mean))

    print(
        f"Mean weak-error slope vs delta "
        f"(exact mean ref, fit window δ∈[{args.fit_dmin:g},{args.fit_dmax:g}]):"
    )
    for m in wsckf_degrees:
        print(
            f"    WSC-KF m={m}: slope = "
            f"{fit_slope(deltas, ws_mean_err[m], dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
            f"   (theory {(m + 1) / 2:.1f})"
        )
    for m in TME_ORDERS:
        print(
            f"    TME-{m}   : slope = "
            f"{fit_slope(deltas, tme_mean_err[m], dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
            f"   (Taylor remainder O(τ^{m + 1}))"
        )
    print(
        f"    CD-EKF  : slope = "
        f"{fit_slope(deltas, ekf_mean, dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
    )
    print(
        f"    CD-UKF  : slope = "
        f"{fit_slope(deltas, ukf_mean, dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
    )
    print(
        f"    CD-CKF  : slope = "
        f"{fit_slope(deltas, ckf_mean, dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
    )

    # --- Cov (high-budget MC reference, same φ) ---
    if not args.skip_cov:
        print("\nComputing high-budget MC covariance references...")
        for i, d in enumerate(deltas):
            # Fixed step count avoids per-δ JIT blowups and keeps large-δ
            # runs from OOMing; bias is shared across methods at each δ.
            n_steps = int(args.mc_steps)
            n_seeds = max(1, int(args.mc_seeds))
            print(
                f"  δ={float(d):.4g}  MC steps={n_steps}  "
                f"paths={args.mc_paths}×2 ×{n_seeds} seed(s) ...",
                flush=True,
            )
            P_ref = mc_ref_cov(
                float(d),
                n_paths=int(args.mc_paths),
                n_steps=n_steps,
                seed=int(args.mc_seed) + i * 1009,
                n_seeds=n_seeds,
                batch_paths=int(args.mc_batch),
            )

            for m in wsckf_degrees:
                _, P_hat = wsckf_predict(float(d), lyons_degree=m, k=1)
                ws_cov_err[m].append(float(np.linalg.norm(P_hat - P_ref, ord="fro")))

            for m in TME_ORDERS:
                _, P_hat = tme_predict(float(d), order=m)
                tme_cov_err[m].append(float(np.linalg.norm(P_hat - P_ref, ord="fro")))

            for which, clist in (
                ("ekf", ekf_cov),
                ("ukf", ukf_cov),
                ("ckf", ckf_cov),
            ):
                _, P_hat = cd_predict(float(d), which)
                clist.append(float(np.linalg.norm(P_hat - P_ref, ord="fro")))

        for m in wsckf_degrees:
            ws_cov_err[m] = np.array(ws_cov_err[m])
        for m in TME_ORDERS:
            tme_cov_err[m] = np.array(tme_cov_err[m])
        ekf_cov, ukf_cov, ckf_cov = map(np.array, (ekf_cov, ukf_cov, ckf_cov))

        print(
            f"\nCov weak-error slope vs delta "
            f"(MC ref, fit window δ∈[{args.fit_dmin:g},{args.fit_dmax:g}]):"
        )
        for m in wsckf_degrees:
            print(
                f"    WSC-KF m={m}: slope = "
                f"{fit_slope(deltas, ws_cov_err[m], dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
                f"   (theory {(m + 1) / 2:.1f})"
            )
        for m in TME_ORDERS:
            print(
                f"    TME-{m}   : slope = "
                f"{fit_slope(deltas, tme_cov_err[m], dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
                f"   (Taylor remainder O(τ^{m + 1}))"
            )
        print(
            f"    CD-EKF  : slope = "
            f"{fit_slope(deltas, ekf_cov, dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
        )
        print(
            f"    CD-UKF  : slope = "
            f"{fit_slope(deltas, ukf_cov, dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
        )
        print(
            f"    CD-CKF  : slope = "
            f"{fit_slope(deltas, ckf_cov, dmin=args.fit_dmin, dmax=args.fit_dmax):.2f}"
        )

    csv_path = os.path.join(OUTPUT_DIR, f"weak_error_delta_scaling_sv{tag}.csv")
    header = (
        ["delta"]
        + [f"wsckf_mean_err_m{m}" for m in wsckf_degrees]
        + [f"tme_mean_err_m{m}" for m in TME_ORDERS]
        + ["ekf_mean_err", "ukf_mean_err", "ckf_mean_err"]
    )
    if not args.skip_cov:
        header += (
            [f"wsckf_cov_err_m{m}" for m in wsckf_degrees]
            + [f"tme_cov_err_m{m}" for m in TME_ORDERS]
            + ["ekf_cov_err", "ukf_cov_err", "ckf_cov_err"]
        )

    new_rows: list[dict[str, float]] = []
    for i, d in enumerate(deltas):
        row = {
            "delta": float(d),
            "wsckf_mean_err_m3": float(ws_mean_err[3][i]),
            "wsckf_mean_err_m5": float(ws_mean_err[5][i]),
            "tme_mean_err_m2": float(tme_mean_err[2][i]),
            "tme_mean_err_m3": float(tme_mean_err[3][i]),
            "ekf_mean_err": float(ekf_mean[i]),
            "ukf_mean_err": float(ukf_mean[i]),
            "ckf_mean_err": float(ckf_mean[i]),
        }
        if not args.skip_cov:
            row.update(
                {
                    "wsckf_cov_err_m3": float(ws_cov_err[3][i]),
                    "wsckf_cov_err_m5": float(ws_cov_err[5][i]),
                    "tme_cov_err_m2": float(tme_cov_err[2][i]),
                    "tme_cov_err_m3": float(tme_cov_err[3][i]),
                    "ekf_cov_err": float(ekf_cov[i]),
                    "ukf_cov_err": float(ukf_cov[i]),
                    "ckf_cov_err": float(ckf_cov[i]),
                }
            )
        new_rows.append(row)

    def _write_rows(path: str, rows: list[dict[str, float]]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in header})

    _write_rows(csv_path, new_rows)

    plot_rows = new_rows
    if args.append_csv:
        merge_path = args.append_csv
        if not os.path.isabs(merge_path):
            merge_path = os.path.join(OUTPUT_DIR, merge_path)
        existing: list[dict[str, float]] = []
        if os.path.isfile(merge_path):
            with open(merge_path, newline="") as f:
                for r in csv.DictReader(f):
                    existing.append({k: float(r[k]) for k in r if r[k] != ""})
        by_delta = {round(float(r["delta"]), 12): r for r in existing}
        for r in new_rows:
            key = round(float(r["delta"]), 12)
            by_delta[key] = {**by_delta.get(key, {}), **r}
        plot_rows = [by_delta[k] for k in sorted(by_delta)]
        _write_rows(merge_path, plot_rows)
        print(f"Append-merged {len(new_rows)} rows into {merge_path} (now {len(plot_rows)} rows)")

    plot_tag = tag
    if args.append_csv:
        base = os.path.basename(args.append_csv)
        prefix = "weak_error_delta_scaling_sv"
        if base.startswith(prefix) and base.endswith(".csv"):
            plot_tag = base[len(prefix) : -4]

    _write_scaling_plots(
        plot_rows,
        plot_tag=plot_tag,
        fit_dmin=args.fit_dmin,
        fit_dmax=args.fit_dmax,
        xlim=xlim,
        skip_cov=args.skip_cov,
        wsckf_colors=wsckf_colors,
    )
    print(f"Saved data -> {csv_path}")


if __name__ == "__main__":
    main()
