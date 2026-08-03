"""Fixed-Δt cost–accuracy Pareto on real InTaSet 4-tank recordings.

Mirrors ``experiments/mfts_test/run_pareto_budget.py``, but:

* Observations / "truth" come from InTaSet tank-level sensors (not a simulator).
* Pump commands drive a Johansson-style continuous–discrete model.
* Monte Carlo averaging is over **non-overlapping time windows** of the long
  training records (InTaSet has no i.i.d. replicates). Default: 3×10 min
  windows from ``4T-M2-train`` (fewer than the simulated 5-seed sweep).

Optional ``--k-proportional`` adds a recombined CRF with
``k ∝ Δt`` (capped by ``--k-max``) to mitigate long-interval cubature truncation.

Run:
  uv run --extra experiments python \\
      experiments/mfts_test/intaset_real/run_pareto_budget.py
  uv run --extra experiments python \\
      experiments/mfts_test/intaset_real/run_pareto_budget.py \\
      --k-proportional --params output/calibrated_params_filter.json
  uv run --extra experiments python \\
      experiments/mfts_test/intaset_real/run_pareto_budget.py \\
      --sources 4T-M2-train --n-windows 3 --horizon-min 10

Outputs under ``experiments/mfts_test/intaset_real/output/``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)
# MPS backend rejects float64; keep parity with simulated MFTS (x64 on CPU).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

EXPERIMENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, EXPERIMENTS_DIR)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import jit, random

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from experiments.cd_dynamax_mirror.filters import (
    UKFParams,
    run_bootstrap_pf,
    run_cd_ekf,
    run_cd_enkf,
    run_cd_ukf,
)
from cuthbert import filter as run_filter
from crf import build_filter
from crf.library import default_recombined_num_paths, product_path_count

from intaset_io import default_mc_windows
from intaset_model import (
    IntaSetParams,
    init_estimate,
    load_params,
    make_intaset_dynamics,
    mape_pct,
    prepare_filter_grids,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(EXPERIMENTS_DIR, "multiplicative_noise_test"))
try:
    from lyons_victoir_paths import precompute_lv_paths  # type: ignore
except ImportError:  # pragma: no cover
    precompute_lv_paths = None


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
    family: str
    label: str
    n_paths: int
    k: int = 1
    lyons_degree: int = 5
    method: str = "product"  # "product" | "recombined" | "lv"


def default_product_ladder(noise_dim: int = 2, max_paths: int = 100) -> list[CrfConfig]:
    specs = [(3, 1), (5, 1), (3, 2), (3, 3), (5, 2), (3, 4), (5, 3)]
    configs: list[CrfConfig] = []
    seen: set[int] = set()
    for m, k in specs:
        n = product_path_count(noise_dim, k, lyons_degree=m)
        if n > max_paths or n in seen:
            continue
        seen.add(n)
        configs.append(
            CrfConfig(
                family="CRF-product",
                label=f"prod-m{m}-k{k}",
                n_paths=n,
                k=k,
                lyons_degree=m,
                method="product",
            )
        )
    return sorted(configs, key=lambda c: c.n_paths)


def lv_configs(noise_dim: int = 2) -> list[CrfConfig]:
    if precompute_lv_paths is None:
        raise SystemExit("--include-lv requires lyons_victoir_paths")
    counts = {3: 2 * noise_dim, 5: (3**noise_dim) * 2}
    return [
        CrfConfig(
            family="CRF-LV",
            label=f"LV-m{m}",
            n_paths=counts[m],
            k=1,
            lyons_degree=m,
            method="lv",
        )
        for m in (3, 5)
    ]


def k_for_delta(dt_min: float, k_ref_min: float, k_max: int) -> int:
    return min(int(k_max), max(1, int(round(float(dt_min) / float(k_ref_min)))))


def kprop_recombined_config(
    dt_min: float,
    *,
    k_ref_min: float,
    k_max: int,
    num_paths: int | None = None,
    k_fixed: int | None = None,
) -> CrfConfig:
    """Recombined CRF with macro-segment count k ∝ Δt (or an explicit ``k_fixed``)."""
    k = int(k_fixed) if k_fixed is not None else k_for_delta(dt_min, k_ref_min, k_max)
    k = max(1, k)
    n = default_recombined_num_paths(2, k) if num_paths is None else max(1, int(num_paths))
    label = f"k∝Δt-k{k}" if k_fixed is None else f"recomb-k{k}"
    return CrfConfig(
        family="CRF-recombined",
        label=label,
        n_paths=n,
        k=k,
        lyons_degree=5,
        method="recombined",
    )


_parser = argparse.ArgumentParser(
    description="InTaSet real-data Pareto: cubature paths vs particles"
)
_parser.add_argument(
    "--sources",
    type=str,
    default="4T-M2-train",
    help="Comma-separated InTaSet recordings "
    "(4T-M1-train,4T-M2-train,4T-M1-test,4T-M2-test)",
)
_parser.add_argument(
    "--n-windows",
    type=int,
    default=2,
    dest="n_windows",
    help="Non-overlapping windows per source (default 2; after calibration holdout)",
)
_parser.add_argument(
    "--horizon-min",
    type=float,
    default=10.0,
    dest="horizon_min",
    help="Window length in minutes (default 10; shorter than sim 30 for speed)",
)
_parser.add_argument(
    "--start-min",
    type=float,
    default=None,
    dest="start_min",
    help="Skip initial minutes (default: meta.holdout_start_min from --params, else 0)",
)
_parser.add_argument(
    "--params",
    type=str,
    default=os.path.join(OUTPUT_DIR, "calibrated_params.json"),
    help="JSON from calibrate_open_loop.py (default: output/calibrated_params.json)",
)
_parser.add_argument(
    "--filter-seed",
    type=int,
    default=100,
    dest="filter_seed",
    help="Base RNG seed for stochastic filters (EnKF/PF); offset by window index",
)
_parser.add_argument("--delta", type=float, default=1.0, help="Δt in minutes")
_parser.add_argument(
    "--crf-max-paths",
    type=int,
    default=81,
    dest="crf_max_paths",
)
_parser.add_argument(
    "--k-proportional",
    action="store_true",
    dest="k_proportional",
    help="Add recombined CRF with k=min(--k-max, max(1, round(Δt/--k-ref-min))) "
    "to mitigate long-Δt cubature truncation",
)
_parser.add_argument(
    "--k-ref-min",
    type=float,
    default=0.5,
    dest="k_ref_min",
    help="Target macro-segment length [min] for --k-proportional (default 0.5)",
)
_parser.add_argument(
    "--k-max",
    type=int,
    default=8,
    dest="k_max",
    help="Cap on k under --k-proportional (default 8)",
)
_parser.add_argument(
    "--k-prop-paths",
    type=str,
    default="default",
    dest="k_prop_paths",
    help="Path budget for k∝Δt CRF: 'default' or an integer",
)
_parser.add_argument(
    "--k-prop-ks",
    type=str,
    default="",
    dest="k_prop_ks",
    help="Comma-separated fixed k values for recombined CRF "
    "(e.g. 3,4). If set, these are used instead of k∝Δt from --k-ref-min. "
    "Implies recombined CRF configs even without --k-proportional.",
)
_parser.add_argument(
    "--skip-product-crf",
    action="store_true",
    dest="skip_product_crf",
    help="Omit the product Lyons ladder (useful with --k-proportional alone)",
)
_parser.add_argument("--include-lv", action="store_true")
_parser.add_argument(
    "--enkf",
    type=str,
    default="50,100,250,500",
    help="EnKF particle counts (fewer than sim default to save time)",
)
_parser.add_argument(
    "--pf",
    type=str,
    default="50,100,250,500",
    help="PF particle counts",
)
_parser.add_argument("--solver", choices=("rk4", "diffrax"), default="rk4")
_parser.add_argument("--n-timing", type=int, default=2, dest="n_timing")
_parser.add_argument(
    "--sigma-u",
    type=float,
    default=None,
    dest="sigma_u",
    help="Override pump-flow process-noise [cm^3/s] (default: from --params)",
)
_parser.add_argument(
    "--only",
    type=str,
    default="",
    help="Comma-separated families to run (e.g. CD-EKF,CRF-product)",
)
_args = _parser.parse_args()

SOURCES = [s.strip() for s in _args.sources.split(",") if s.strip()]
N_WINDOWS = int(_args.n_windows)
HORIZON_MIN = float(_args.horizon_min)
DELTA_T_MIN = float(_args.delta)
SOLVER = _args.solver
N_TIMING = _args.n_timing
FILTER_SEED0 = int(_args.filter_seed)
ENKF_PARTICLE_LIST = _parse_ints(_args.enkf)
PF_PARTICLE_LIST = _parse_ints(_args.pf)
CRF_CONFIGS: list[CrfConfig] = []
if not _args.skip_product_crf:
    CRF_CONFIGS.extend(default_product_ladder(2, max_paths=_args.crf_max_paths))
if _args.include_lv:
    CRF_CONFIGS.extend(lv_configs(2))
n_override = None
tok = _args.k_prop_paths.strip().lower()
if tok not in ("", "default"):
    n_override = int(tok)
k_fixed_list = _parse_ints(_args.k_prop_ks) if _args.k_prop_ks.strip() else []
if k_fixed_list:
    for k_fix in k_fixed_list:
        CRF_CONFIGS.append(
            kprop_recombined_config(
                DELTA_T_MIN,
                k_ref_min=float(_args.k_ref_min),
                k_max=int(_args.k_max),
                num_paths=n_override,
                k_fixed=k_fix,
            )
        )
elif _args.k_proportional:
    if float(_args.k_ref_min) <= 0.0:
        raise SystemExit("--k-ref-min must be > 0")
    CRF_CONFIGS.append(
        kprop_recombined_config(
            DELTA_T_MIN,
            k_ref_min=float(_args.k_ref_min),
            k_max=int(_args.k_max),
            num_paths=n_override,
        )
    )
CRF_CONFIGS = sorted(CRF_CONFIGS, key=lambda c: (c.family, c.n_paths, c.label))
ONLY_FAMILIES: Optional[set[str]] = None
if _args.only.strip():
    ONLY_FAMILIES = {s.strip() for s in _args.only.split(",") if s.strip()}

if not SOURCES:
    raise SystemExit("--sources must list at least one recording")
if N_WINDOWS < 1:
    raise SystemExit("--n-windows must be ≥ 1")
if ONLY_FAMILIES is None:
    if not CRF_CONFIGS:
        raise SystemExit(
            "No CRF configs (raise --crf-max-paths, or pass --k-proportional)"
        )
    if not ENKF_PARTICLE_LIST or not PF_PARTICLE_LIST:
        raise SystemExit("--enkf / --pf must list at least one particle count")


def _want(family: str) -> bool:
    return ONLY_FAMILIES is None or family in ONLY_FAMILIES


# ---------------------------------------------------------------------------
# Plant parameters (prefer calibrated JSON)
# ---------------------------------------------------------------------------

PARAMS_PATH = _args.params
PARAMS_META: dict = {}
if PARAMS_PATH and os.path.exists(PARAMS_PATH):
    p, PARAMS_META = load_params(PARAMS_PATH)
    print(f"Loaded calibrated params from {PARAMS_PATH}")
else:
    p = IntaSetParams()
    if PARAMS_PATH:
        print(f"WARNING: --params {PARAMS_PATH} not found; using defaults")

if _args.start_min is not None:
    START_MIN = float(_args.start_min)
else:
    START_MIN = float(PARAMS_META.get("holdout_start_min", 0.0))

sigma_u = float(_args.sigma_u) if _args.sigma_u is not None else float(p.sigma_u)
DT_FILTER = 0.15  # s, RK4 substep for filters / fine-grid reconstruction
SIGMA_U_EKF_UKF = sigma_u
SIGMA_U_ENS = sigma_u
SIGMA_U_UKF = max(1.0, 0.2 * sigma_u)

CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)

COLORS = {
    "CD-EKF": "#2E86AB",
    "CD-UKF": "#E8B923",
    "CD-CKF": "#3DA35D",
    "CD-EnKF": "#D97706",
    "PF": "#F24236",
    "CRF-product": "#6A4C93",
    "CRF-LV": "#C44536",
    "CRF-recombined": "#8B1E3F",
}
DISPLAY_NAME = {
    "CRF-product": "WS-CKF",
    "CRF-LV": "WS-CKF-LV",
    "CRF-recombined": r"WS-CKF $k\propto\Delta t$",
    "CD-EKF": "CD-EKF",
    "CD-UKF": "CD-UKF",
    "CD-CKF": "CD-CKF",
    "CD-EnKF": "CD-EnKF",
    "PF": "PF",
}
_LABEL_OFFSET = {
    "CRF-product": (0, 7),
    "CD-EnKF": (0, -10),
    "PF": (0, 7),
}
_LABEL_ABOVE_BUDGETS = {"CD-EnKF": {50, 100}}


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
    def drift_np(t_sec: float, x: np.ndarray) -> np.ndarray:
        return np.asarray(drift_fn(t_sec, jnp.asarray(x)), dtype=np.float64)

    return drift_np


def rk4_step(drift_np, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = drift_np(t, y)
    k2 = drift_np(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = drift_np(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = drift_np(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_to(drift_np, y: np.ndarray, t0: float, t1: float, dt0: float) -> np.ndarray:
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


def estimates_on_fine_grid(
    m_filt: np.ndarray,
    dt_obs_sec: float,
    times_fine_sec: np.ndarray,
    drift_fn,
    dt_integrate: float,
) -> np.ndarray:
    drift_np = _as_numpy_drift(drift_fn)
    n_fine = times_fine_sec.shape[0]
    d = m_filt.shape[1]
    n_obs = m_filt.shape[0] - 1
    est = np.zeros((n_fine, d), dtype=np.float64)
    est[0] = m_filt[0]

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


# ---------------------------------------------------------------------------
# Load windows (= pseudo-MC realizations)
# ---------------------------------------------------------------------------

windows = default_mc_windows(
    sources=SOURCES,
    duration_min=HORIZON_MIN,
    n_windows=N_WINDOWS,
    start_min=START_MIN,
)
print(
    f"InTaSet Pareto: sources={SOURCES}, windows={len(windows)}, "
    f"start_min={START_MIN:g}, horizon={HORIZON_MIN:g} min, Δt={DELTA_T_MIN:g} min"
)
print(
    f"  params: k_pump={p.k_pump:.3g} γ=({p.gamma1:.3f},{p.gamma2:.3f}) "
    f"meas_std={p.meas_std:.3f} sigma_u={sigma_u:.3g}"
)
print(
    "  CRF: "
    + ", ".join(f"{c.label}({c.method},k={c.k},P={c.n_paths})" for c in CRF_CONFIGS)
)
for w in windows:
    print(f"  {w.source}: n={w.n}, dt≈{w.dt_sec:.3f}s, dur={w.duration_sec/60:.2f} min")

raw_rows: list[dict] = []


def score(
    family: str,
    label: str,
    budget: int,
    window_id: int,
    source: str,
    m_filt: np.ndarray,
    final_ll: float,
    wall_s: float,
    drift_fn,
    h_fine: np.ndarray,
    times_fine_sec: np.ndarray,
    dt_obs_sec: float,
    n_obs: int,
):
    est_fine = estimates_on_fine_grid(
        m_filt, dt_obs_sec, times_fine_sec, drift_fn, DT_FILTER
    )
    rec = {
        "window": window_id,
        "source": source,
        "family": family,
        "label": label,
        "budget": budget,
        "mape": mape_pct(est_fine, h_fine, slice(0, 4)),
        "nll_per_obs": -final_ll / n_obs,
        "time_s": wall_s,
    }
    raw_rows.append(rec)
    print(
        f"  {source:18s} {family:12s} {label:14s} n={budget:4d}  "
        f"MAPE={rec['mape']:6.2f}  time={1e3 * wall_s:7.1f} ms"
    )


def build_crf(cfg: CrfConfig, drift, L, obs_fn, m0, chol_P0, chol_R, ys, dt, dt0):
    wiener_paths = None
    if cfg.method == "lv":
        assert precompute_lv_paths is not None
        wiener_paths = precompute_lv_paths(
            horizon=dt,
            noise_dim=2,
            lyons_degree=cfg.lyons_degree,
            integration_dt=dt0,
        )
        n_actual = int(np.asarray(wiener_paths[2]).shape[0])
        if n_actual != cfg.n_paths:
            raise RuntimeError(
                f"LV m={cfg.lyons_degree}: expected {cfg.n_paths} paths, got {n_actual}"
            )
        return build_filter(
            get_init_params=lambda mi: (m0, chol_P0),
            get_dynamics=lambda mi: (
                lambda t, x: drift(t, x),
                lambda t, x: L(t, x),
            ),
            get_observations=lambda mi: (obs_fn, chol_R, ys[mi - 1]),
            noise_dim=2,
            dt=dt,
            integration_dt=dt0,
            cubature_order=cfg.k,
            lyons_degree=cfg.lyons_degree,
            multidim_paths="product",
            wiener_paths=wiener_paths,
        )
    if cfg.method == "recombined":
        return build_filter(
            get_init_params=lambda mi: (m0, chol_P0),
            get_dynamics=lambda mi: (
                lambda t, x: drift(t, x),
                lambda t, x: L(t, x),
            ),
            get_observations=lambda mi: (obs_fn, chol_R, ys[mi - 1]),
            noise_dim=2,
            dt=dt,
            integration_dt=dt0,
            cubature_order=cfg.k,
            num_paths=cfg.n_paths,
            multidim_paths="recombined",
            sde_form="ito",
        )
    # product Lyons–Victoir
    return build_filter(
        get_init_params=lambda mi: (m0, chol_P0),
        get_dynamics=lambda mi: (
            lambda t, x: drift(t, x),
            lambda t, x: L(t, x),
        ),
        get_observations=lambda mi: (obs_fn, chol_R, ys[mi - 1]),
        noise_dim=2,
        dt=dt,
        integration_dt=dt0,
        cubature_order=cfg.k,
        lyons_degree=cfg.lyons_degree,
        multidim_paths="product",
        sde_form="ito",
    )


for win_idx, traj in enumerate(windows):
    print(f"\n=== window {win_idx}: {traj.source} ===")
    grids = prepare_filter_grids(
        traj,
        delta_t_min=DELTA_T_MIN,
        horizon_min=HORIZON_MIN,
        height_bias=p.height_bias,
    )
    t_fine = grids["t_fine_sec"]
    h_fine = grids["h_fine"]
    u_fine = grids["u_fine"]
    ys_np = grids["ys"]
    n_obs = grids["n_obs"]
    dt = grids["dt_obs_sec"]
    dt0 = dt / max(1, int(round(dt / DT_FILTER)))
    times_fine_sec = t_fine
    ys = jnp.asarray(ys_np)
    model_inputs = jnp.arange(n_obs + 1)

    # Dynamics use the full fine pump schedule (known input).
    drift_ekf, L_ekf, Qc, obs_fn, R = make_intaset_dynamics(
        p, t_sec=t_fine, pumps=u_fine, sigma_u=SIGMA_U_EKF_UKF
    )
    drift_ukf, L_ukf, _, _, _ = make_intaset_dynamics(
        p, t_sec=t_fine, pumps=u_fine, sigma_u=SIGMA_U_UKF
    )
    drift_ens, L_ens, _, _, _ = make_intaset_dynamics(
        p, t_sec=t_fine, pumps=u_fine, sigma_u=SIGMA_U_ENS
    )

    x0, P0 = init_estimate(h_fine[0], p)
    m0 = jnp.asarray(x0, dtype=jnp.float64)
    P0_j = jnp.asarray(P0, dtype=jnp.float64)
    chol_P0 = jnp.linalg.cholesky(P0_j)
    chol_R = jnp.linalg.cholesky(R)

    print(
        f"  n_obs={n_obs}, dt_obs={dt:g}s, fine_dt={grids['dt_fine_sec']:g}s, "
        f"n_fine={grids['n_fine']}"
    )

    for cfg in CRF_CONFIGS:
        if not _want(cfg.family):
            continue
        if cfg.method == "product":
            n_expect = product_path_count(2, cfg.k, lyons_degree=cfg.lyons_degree)
            if n_expect != cfg.n_paths:
                raise RuntimeError(f"config mismatch: {cfg} vs product count {n_expect}")
        crf = build_crf(
            cfg, drift_ens, L_ens, obs_fn, m0, chol_P0, chol_R, ys, dt, dt0
        )
        jitted = jit(run_filter, static_argnames=["filter_obj"])
        states, wall = time_call(lambda f=jitted, filt=crf: f(filt, model_inputs))
        score(
            cfg.family,
            cfg.label,
            cfg.n_paths,
            win_idx,
            traj.source,
            stack_estimates(m0, states.mean[1:]),
            float(states.log_normalizing_constant[-1]),
            wall,
            drift_ens,
            h_fine,
            times_fine_sec,
            dt,
            n_obs,
        )

    if _want("CD-EKF"):
        ekf_jit = jit(
            lambda: run_cd_ekf(
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
        )
        ekf_res, wall = time_call(ekf_jit)
        score(
            "CD-EKF",
            "EKF",
            0,
            win_idx,
            traj.source,
            stack_estimates(m0, ekf_res.means),
            float(ekf_res.cum_loglik[-1]),
            wall,
            drift_ekf,
            h_fine,
            times_fine_sec,
            dt,
            n_obs,
        )

    if _want("CD-UKF"):
        ukf_jit = jit(
            lambda: run_cd_ukf(
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
        )
        ukf_res, wall = time_call(ukf_jit)
        score(
            "CD-UKF",
            "UKF",
            0,
            win_idx,
            traj.source,
            stack_estimates(m0, ukf_res.means),
            float(ukf_res.cum_loglik[-1]),
            wall,
            drift_ukf,
            h_fine,
            times_fine_sec,
            dt,
            n_obs,
        )

    if _want("CD-CKF"):
        ckf_jit = jit(
            lambda: run_cd_ukf(
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
        )
        ckf_res, wall = time_call(ckf_jit)
        score(
            "CD-CKF",
            "CKF",
            0,
            win_idx,
            traj.source,
            stack_estimates(m0, ckf_res.means),
            float(ckf_res.cum_loglik[-1]),
            wall,
            drift_ukf,
            h_fine,
            times_fine_sec,
            dt,
            n_obs,
        )

    key = random.key(FILTER_SEED0 + win_idx)
    if _want("CD-EnKF"):
        for n_part in ENKF_PARTICLE_LIST:
            key, sub = random.split(key)
            enkf_jit = jit(
                lambda k, n=n_part: run_cd_enkf(
                    drift_ens,
                    obs_fn,
                    L_ens,
                    Qc,
                    R,
                    m0,
                    P0_j,
                    ys,
                    dt,
                    k,
                    n_particles=n,
                    noise_dim=2,
                    dt0=dt0,
                    solver=SOLVER,
                )
            )
            enkf_res, wall = time_call(enkf_jit, sub)
            score(
                "CD-EnKF",
                f"EnKF-{n_part}",
                n_part,
                win_idx,
                traj.source,
                stack_estimates(m0, enkf_res.means),
                float(enkf_res.cum_loglik[-1]),
                wall,
                drift_ens,
                h_fine,
                times_fine_sec,
                dt,
                n_obs,
            )

    if _want("PF"):
        for n_part in PF_PARTICLE_LIST:
            key, sub = random.split(key)
            pf_jit = jit(
                lambda k, n=n_part: run_bootstrap_pf(
                    drift_ens,
                    obs_fn,
                    L_ens,
                    Qc,
                    R,
                    m0,
                    P0_j,
                    ys,
                    dt,
                    k,
                    n_particles=n,
                    noise_dim=2,
                    dt0=dt0,
                    solver=SOLVER,
                )
            )
            pf_res, wall = time_call(pf_jit, sub)
            score(
                "PF",
                f"PF-{n_part}",
                n_part,
                win_idx,
                traj.source,
                stack_estimates(m0, pf_res.means),
                float(pf_res.cum_loglik[-1]),
                wall,
                drift_ens,
                h_fine,
                times_fine_sec,
                dt,
                n_obs,
            )


# ---------------------------------------------------------------------------
# Aggregate across windows
# ---------------------------------------------------------------------------

n_windows_total = len({(r["source"], r["window"]) for r in raw_rows})
n_obs_meta = grids["n_obs"]

raw_path = os.path.join(OUTPUT_DIR, "pareto_budget_vs_mape_raw.csv")
with open(raw_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "delta_t_min",
            "n_obs",
            "window",
            "source",
            "family",
            "label",
            "budget",
            "mape",
            "nll_per_obs",
            "time_s",
        ],
    )
    writer.writeheader()
    for rec in raw_rows:
        writer.writerow({"delta_t_min": DELTA_T_MIN, "n_obs": n_obs_meta, **rec})

keys = sorted({(r["family"], r["label"], r["budget"]) for r in raw_rows})
summary: list[dict] = []
for family, label, budget in keys:
    subset = [
        r
        for r in raw_rows
        if r["family"] == family and r["label"] == label and r["budget"] == budget
    ]

    def _mean_std(field: str) -> tuple[float, float]:
        vals = np.asarray([r[field] for r in subset], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    mape_m, mape_s = _mean_std("mape")
    nll, nll_s = _mean_std("nll_per_obs")
    tm, tm_s = _mean_std("time_s")
    summary.append(
        {
            "family": family,
            "label": label,
            "budget": budget,
            "n_windows": len(subset),
            "mape_mean": mape_m,
            "mape_std": mape_s,
            "nll_per_obs_mean": nll,
            "nll_per_obs_std": nll_s,
            "time_s_mean": tm,
            "time_s_std": tm_s,
        }
    )

csv_path = os.path.join(OUTPUT_DIR, "pareto_budget_vs_mape.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "delta_t_min",
            "n_obs",
            "family",
            "label",
            "budget",
            "n_windows",
            "mape_mean",
            "mape_std",
            "nll_per_obs_mean",
            "nll_per_obs_std",
            "time_s_mean",
            "time_s_std",
        ],
    )
    writer.writeheader()
    for rec in sorted(summary, key=lambda r: (r["family"], r["budget"], r["label"])):
        writer.writerow({"delta_t_min": DELTA_T_MIN, "n_obs": n_obs_meta, **rec})


# ---------------------------------------------------------------------------
# Plot mean ± std Pareto (height MAPE only — no disturbance states)
# ---------------------------------------------------------------------------

BAND_SCALE = 0.3
# Single moment-safe WS-CKF point (higher path budgets don't help much here).
WS_CKF_PLOT_BUDGETS = {4}
PARTICLE_PLOT_HIDE_BUDGETS = {25, 1000}
plot_families = [
    "CRF-product",
    "CRF-recombined",
    "CD-EnKF",
    "PF",
    "CD-EKF",
    "CD-UKF",
    "CD-CKF",
]


def _annotate_nodes(ax, fam, pts, xs, ys):
    if fam in ("CD-EKF", "CD-UKF", "CD-CKF", "CRF-product", "CRF-recombined"):
        return
    color = COLORS.get(fam, "#333333")
    base_dx, base_dy = _LABEL_OFFSET.get(fam, (0, 6))
    above = _LABEL_ABOVE_BUDGETS.get(fam, set())
    for i, (pt, x, y) in enumerate(zip(pts, xs, ys)):
        budget = int(pt["budget"])
        if budget in above:
            dx, dy = 0, 7
        else:
            dx = base_dx
            dy = base_dy if (i % 2 == 0) else (base_dy + (4 if base_dy > 0 else -4))
        ax.annotate(
            str(budget),
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7,
            color=color,
            ha="center",
            va="bottom" if dy >= 0 else "top",
            alpha=0.9,
        )


def _plot_pts_for_family(fam: str) -> list[dict]:
    pts = [r for r in summary if r["family"] == fam]
    if fam == "CRF-product":
        pts = [r for r in pts if r["budget"] in WS_CKF_PLOT_BUDGETS]
    elif fam in ("CD-EnKF", "PF"):
        pts = [r for r in pts if r["budget"] not in PARTICLE_PLOT_HIDE_BUDGETS]
    return pts


def _plot_pareto_metric(metric: str, ylabel: str, title: str, out_name: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    all_xs: list[float] = []
    for fam in plot_families:
        pts = _plot_pts_for_family(fam)
        if not pts:
            continue
        pts = sorted(pts, key=lambda r: r["budget"])
        xs = np.asarray([1e3 * r["time_s_mean"] for r in pts])
        ys = np.asarray([r[f"{metric}_mean"] for r in pts])
        ystd = np.asarray([r[f"{metric}_std"] for r in pts]) * BAND_SCALE
        all_xs.extend(xs.tolist())
        color = COLORS.get(fam, "#333333")
        if fam in ("CRF-product", "CRF-recombined"):
            marker = "*"
        elif fam in ("CD-EKF", "CD-UKF", "CD-CKF"):
            marker = "s"
        else:
            marker = "o"
        disp = DISPLAY_NAME.get(fam, fam)
        if len(pts) == 1:
            x0 = float(xs[0])
            ax.fill_between(
                [x0 / 1.04, x0 * 1.04],
                [ys[0] - ystd[0]] * 2,
                [ys[0] + ystd[0]] * 2,
                color=color,
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(
                xs,
                ys,
                marker,
                color=color,
                markersize=12 if marker == "*" else 7,
                label=disp,
            )
        else:
            ax.fill_between(
                xs, ys - ystd, ys + ystd, color=color, alpha=0.22, linewidth=0
            )
            ax.plot(
                xs,
                ys,
                f"-{marker}",
                color=color,
                lw=1.5,
                markersize=10 if marker == "*" else 6,
                label=disp,
            )
        _annotate_nodes(ax, fam, pts, xs, ys)
    ax.set_xlabel("Wall-clock time [ms]")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if all_xs:
        ax.set_xlim(min(all_xs) / 1.25, max(all_xs) * 1.35)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


_ws_ckf_budgets = ",".join(str(b) for b in sorted(WS_CKF_PLOT_BUDGETS))
_title_suffix = (
    rf"at $\Delta t={DELTA_T_MIN:g}$ min "
    rf"(mean±{BAND_SCALE:g}$\sigma$ over {n_windows_total} windows; "
    rf"WS-CKF paths = {_ws_ckf_budgets})"
)
mape_plot_path = _plot_pareto_metric(
    "mape",
    "Height MAPE [%]",
    rf"InTaSet Pareto: MAPE vs wall-clock {_title_suffix}",
    "pareto_budget_vs_mape.png",
)
nll_plot_path = _plot_pareto_metric(
    "nll_per_obs",
    "Mean NLL per observation",
    rf"InTaSet Pareto: NLL vs wall-clock {_title_suffix}",
    "pareto_budget_vs_nll.png",
)

print("-" * 72)
print(f"Wrote {mape_plot_path}")
print(f"Wrote {nll_plot_path}")
print(f"Wrote {csv_path}")
print(f"Wrote {raw_path}")
