"""Fixed-Δt cost–accuracy Pareto over exact cubature path counts and particles.

WSC-KF uses **product** Lyons–Victoir constructions so the path count is exact
(no recombined collapse). Default ladder for ``d=2``:

  m=3,k=1 → 4 | m=5,k=1 → 9 | m=3,k=2 → 16 | m=3,k=3 → 64 | m=5,k=2 → 81

Optional ``--include-lv`` adds true multi-D LV (m=3 → 4, m=5 → 18).

Accuracy and wall-clock are averaged over multiple truth/filter seeds; the plot
shows mean ± 1 std.

Run:
  uv run --extra experiments python experiments/mfts_test/run_pareto_budget.py
  uv run --extra experiments python experiments/mfts_test/run_pareto_budget.py \\
      --seeds 100,101,102,103,104,105,106,107,108,109 --include-lv

Outputs:
  ``experiments/mfts_test/output/pareto_budget_vs_mape.{pdf,png}``
  ``experiments/mfts_test/output/pareto_budget_vs_nll.{pdf,png}``
  ``experiments/mfts_test/output/pareto_budget_vs_mape.csv``       (across-seed summary)
  ``experiments/mfts_test/output/pareto_budget_vs_mape_raw.csv``   (per-seed)
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
# MPS rejects float64; keep x64 on CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

EXPERIMENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
sys.path.insert(0, EXPERIMENTS_DIR)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
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
from wsckf import build_filter
from wsckf.library import default_num_paths, precompute_cubature_paths, product_path_count

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
    PALETTE,
    SINGLE_COLUMN_WIDTH,
    save_publication_figure,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
class WSCKFConfig:
    """Exact-count WSC-KF Wiener construction."""

    family: str  # "WSC-KF-product" | "WSC-KF-LV"
    label: str
    n_paths: int
    k: int = 1
    lyons_degree: int = 5
    method: str = "product"  # product | lv


def default_product_ladder(noise_dim: int = 2, max_paths: int = 100) -> list[WSCKFConfig]:
    """Exact product path counts up to ``max_paths`` (sorted by n_paths)."""
    specs = [
        (3, 1),
        (5, 1),
        (3, 2),
        (3, 3),
        (5, 2),
        (3, 4),
        (5, 3),
    ]
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


def lv_configs(noise_dim: int = 2) -> list[WSCKFConfig]:
    return [
        WSCKFConfig(
            family="WSC-KF-LV",
            label=f"LV-m{m}",
            n_paths=default_num_paths(noise_dim, 1, "lv", lyons_degree=m),
            k=1,
            lyons_degree=m,
            method="lv",
        )
        for m in (3, 5)
    ]


_parser = argparse.ArgumentParser(
    description="MFTS multi-seed Pareto: exact cubature paths vs particles"
)
_parser.add_argument(
    "--seeds",
    type=str,
    default="100,101,102,103,104,105,106,107,108,109",
    help="Comma-separated truth/filter seeds to average over (default: 100–109)",
)
_parser.add_argument(
    "--delta",
    type=float,
    default=1.0,
    help="Fixed observation interval Δt in minutes (default 1.0)",
)
_parser.add_argument("--eval-dt", type=float, default=0.05, dest="eval_dt")
_parser.add_argument(
    "--wsckf-max-paths",
    type=int,
    default=100,
    dest="wsckf_max_paths",
    help="Include product (m,k) configs with exact path count ≤ this (default 100)",
)
_parser.add_argument(
    "--include-lv",
    action="store_true",
    help="Also plot true multi-D Lyons–Victoir (exact 4 and 18 paths for d=2)",
)
_parser.add_argument(
    "--enkf",
    type=str,
    default="25,50,100,250,500,1000",
    help="Comma-separated EnKF particle counts",
)
_parser.add_argument(
    "--pf",
    type=str,
    default="25,50,100,250,500,1000",
    help="Comma-separated PF particle counts",
)
_parser.add_argument("--solver", choices=("rk4", "diffrax"), default="rk4")
_parser.add_argument(
    "--n-timing",
    type=int,
    default=3,
    dest="n_timing",
    help="Timed repeats per (seed, config) after JIT warmup",
)
_parser.add_argument(
    "--append",
    action="store_true",
    help="Keep existing pareto_budget_vs_mape_raw.csv rows and only run "
    "seeds not already present, then re-aggregate/plot.",
)
_parser.add_argument(
    "--merge",
    action="store_true",
    help="Keep existing raw rows; re-run the requested seeds (optionally "
    "restricted by --only) and replace matching (seed, family, label) rows.",
)
_parser.add_argument(
    "--only",
    type=str,
    default="",
    help="Comma-separated families to run (e.g. CD-CKF). Use with --merge "
    "to fill a family into an existing Pareto CSV without re-running others.",
)
_parser.add_argument(
    "--noise",
    choices=("additive", "multiplicative"),
    default="additive",
    help="Disturbance diffusion: additive (Nielsen) or multiplicative "
    "g(F)=σ√(1+(F/Fc)²) on F3,F4",
)
_parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Output filename tag. Empty → '' for additive, '_mult' for multiplicative. "
    "Example: --tag mult → pareto_budget_vs_mape_mult.png",
)
_parser.add_argument(
    "--sigma-scale",
    type=float,
    default=1.0,
    dest="sigma_scale",
    help="Multiply base disturbance σ (truth + filters) by this factor "
    "(e.g. 4 for σ/σ₀=2² under multiplicative g(F))",
)
_parser.add_argument(
    "--mape-mode",
    choices=("fine", "obs"),
    default="fine",
    dest="mape_mode",
    help="MAPE scoring: fine-grid (Nielsen open-loop fill) or obs-time "
    "(filter means vs truth at assimilation times only)",
)
_parser.add_argument(
    "--shared-sigma",
    action="store_true",
    dest="shared_sigma",
    help="Use the same disturbance σ for UKF/CKF as for EKF/EnKF/PF/WSC "
    "(overrides Nielsen's tighter additive UKF σ=1)",
)
_parser.add_argument(
    "--shared-lambda",
    action="store_true",
    dest="shared_lambda",
    help="Use the same OU rate λ for EKF/UKF/CKF and EnKF/PF/WSC "
    "(overrides Nielsen's λ=0 vs λ=2e-3 split)",
)
_parser.add_argument(
    "--lambda-dist",
    type=float,
    default=None,
    dest="lambda_dist",
    help="Filter OU rate λ on (F3,F4). Default: Nielsen split, or 2e-3 "
    "when --shared-lambda is set. Ignored for truth (always λ_sim=0.1).",
)
_args = _parser.parse_args()

SEEDS = _parse_ints(_args.seeds)
DELTA_T_MIN = float(_args.delta)
EVAL_DT_MIN = float(_args.eval_dt)
SOLVER = _args.solver
N_TIMING = _args.n_timing
NOISE_MODE = _args.noise
SIGMA_SCALE = float(_args.sigma_scale)
MAPE_MODE = _args.mape_mode
if _args.tag.strip():
    OUT_TAG = f"_{_args.tag.strip()}"
else:
    parts: list[str] = []
    if NOISE_MODE == "multiplicative":
        parts.append("mult")
    if abs(SIGMA_SCALE - 1.0) > 1e-12:
        parts.append(f"sg{SIGMA_SCALE:g}")
    if MAPE_MODE == "obs":
        parts.append("obs")
    OUT_TAG = ("_" + "_".join(parts)) if parts else ""
ENKF_PARTICLE_LIST = _parse_ints(_args.enkf)
PF_PARTICLE_LIST = _parse_ints(_args.pf)
WSCKF_CONFIGS = default_product_ladder(2, max_paths=_args.wsckf_max_paths)
if _args.include_lv:
    WSCKF_CONFIGS = sorted(WSCKF_CONFIGS + lv_configs(2), key=lambda c: (c.n_paths, c.family))
ONLY_FAMILIES: Optional[set[str]] = None
if _args.only.strip():
    ONLY_FAMILIES = {s.strip() for s in _args.only.split(",") if s.strip()}
if not SEEDS:
    raise SystemExit("--seeds must list at least one seed")
if ONLY_FAMILIES is None:
    if not WSCKF_CONFIGS:
        raise SystemExit("No WSC-KF configs (raise --wsckf-max-paths?)")
    if not ENKF_PARTICLE_LIST or not PF_PARTICLE_LIST:
        raise SystemExit("--enkf / --pf must list at least one particle count")
if _args.append and _args.merge:
    raise SystemExit("Use only one of --append / --merge")
if ONLY_FAMILIES is not None and not (_args.merge or _args.append):
    raise SystemExit("--only requires --merge (or --append) to avoid wiping raw CSV")


def _want(family: str) -> bool:
    return ONLY_FAMILIES is None or family in ONLY_FAMILIES


def _load_raw_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "seed": int(r["seed"]),
                    "family": r["family"],
                    "label": r["label"],
                    "budget": int(float(r["budget"])),
                    "mapex": float(r["mapex"]),
                    "maped": float(r["maped"]),
                    "nll_per_obs": float(r["nll_per_obs"]),
                    "time_s": float(r["time_s"]),
                }
            )
    return rows


RAW_PATH = os.path.join(OUTPUT_DIR, f"pareto_budget_vs_mape{OUT_TAG}_raw.csv")
EXISTING_RAW: list[dict] = []
if _args.append or _args.merge:
    if not os.path.exists(RAW_PATH):
        raise SystemExit(f"--append/--merge requires existing {RAW_PATH}")
    EXISTING_RAW = _load_raw_rows(RAW_PATH)
    if _args.append:
        have = {r["seed"] for r in EXISTING_RAW}
        skipped = [s for s in SEEDS if s in have]
        SEEDS = [s for s in SEEDS if s not in have]
        print(
            f"--append: keeping {len(EXISTING_RAW)} rows from seeds {sorted(have)}; "
            f"skip {skipped}; will run {SEEDS}"
        )
        if not SEEDS:
            raise SystemExit("Nothing to run: all requested seeds already in raw CSV")
    else:
        print(
            f"--merge: keeping {len(EXISTING_RAW)} existing rows; "
            f"will re-run seeds={SEEDS}"
            + (f" families={sorted(ONLY_FAMILIES)}" if ONLY_FAMILIES else "")
        )

HORIZON_MIN = 30.0
DT_TRUTH = 0.015
DT_FILTER = 0.15
LAMBDA_SIM = 0.1
SIGMA0 = 5.0
SIGMA_SIM = SIGMA0 * SIGMA_SCALE
# Nielsen split: EKF/UKF λ=0, EnKF/PF/WSC λ=2e-3. --shared-lambda equalizes.
_LAM_DEFAULT_PATH = 2.0e-3
if _args.lambda_dist is not None:
    LAMBDA_EKF_UKF = float(_args.lambda_dist)
    LAMBDA_ENKF_PF = float(_args.lambda_dist)
elif bool(_args.shared_lambda):
    LAMBDA_EKF_UKF = _LAM_DEFAULT_PATH
    LAMBDA_ENKF_PF = _LAM_DEFAULT_PATH
else:
    LAMBDA_EKF_UKF = 0.0
    LAMBDA_ENKF_PF = _LAM_DEFAULT_PATH
SIGMA_EKF_ENKF_PF = SIGMA0 * SIGMA_SCALE
# Additive Nielsen bakeoff uses a tighter UKF process noise (σ=1); pass
# --shared-sigma (or use multiplicative mode) for a fair common-σ comparison.
if NOISE_MODE == "multiplicative" or bool(_args.shared_sigma):
    SIGMA_UKF = SIGMA_EKF_ENKF_PF
else:
    SIGMA_UKF = 1.0 * SIGMA_SCALE
F_NOISE_SCALE = 150.0
F1, F2 = 5.0, 5.0

# Spherical-radial cubature ≡ UKF with λ = 0 (Arasaratnam–Haykin CD-CKF).
CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)

COLORS = {
    name: METHOD_COLORS[name]
    for name in (
        "CD-EKF",
        "CD-UKF-fine",
        "CD-UKF-coarse",
        "CD-CKF",
        "CD-EnKF",
        "PF",
        "WSC-KF-product",
        "WSC-KF-LV",
    )
}

# Legend / axis display names (CSV family keys stay stable).
DISPLAY_NAME = {
    "WSC-KF-product": "WSC-KF",
    "WSC-KF-LV": "WSC-KF-LV",
    "CD-EKF": "CD-EKF",
    "CD-UKF-fine": "CD-UKF (fine)",
    "CD-UKF-coarse": "CD-UKF (coarse)",
    "CD-CKF": "CD-CKF",
    "CD-EnKF": "CD-EnKF",
    "PF": "PF",
}

# Vertical label offsets (points) per family — keep curves readable on log-x.
_LABEL_OFFSET = {
    "WSC-KF-product": (0, 7),
    "CD-EnKF": (0, -10),
    "PF": (0, 7),
}


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


def stride_for_delta(dt_min: float, eval_dt: float) -> int:
    stride = int(round(dt_min / eval_dt))
    if stride < 1 or not np.isclose(stride * eval_dt, dt_min, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Delta_t={dt_min} min must be an integer multiple of "
            f"--eval-dt={eval_dt} min (got stride={stride})"
        )
    return stride


if HORIZON_MIN / EVAL_DT_MIN != int(round(HORIZON_MIN / EVAL_DT_MIN)):
    raise SystemExit(
        f"Horizon {HORIZON_MIN} min must be an integer multiple of "
        f"--eval-dt={EVAL_DT_MIN} min"
    )
N_FINE = int(round(HORIZON_MIN / EVAL_DT_MIN))
DT_FINE_MIN = HORIZON_MIN / N_FINE
DT_FINE_SEC = DT_FINE_MIN * 60.0
N_SUBSTEPS_TRUTH_FINE = max(1, int(round(DT_FINE_SEC / DT_TRUTH)))

stride = stride_for_delta(DELTA_T_MIN, DT_FINE_MIN)
n_obs = N_FINE // stride
if n_obs * stride != N_FINE:
    raise SystemExit(
        f"Delta_t={DELTA_T_MIN} min (stride={stride}) does not tile "
        f"the fine grid of {N_FINE} steps"
    )
dt_min = stride * DT_FINE_MIN
dt = dt_min * 60.0
n_substeps_filter = max(1, int(round(dt / DT_FILTER)))
dt0 = dt / n_substeps_filter
times_fine_sec = np.arange(N_FINE + 1) * DT_FINE_SEC
model_inputs = jnp.arange(n_obs + 1)

p = MFTSParams()
bar_est = lambda t_min, nom=p.F_nominal_est: nom
_dyn_kw = dict(noise=NOISE_MODE, F_noise_scale=F_NOISE_SCALE, F1=F1, F2=F2)
drift_ekf, L_ekf, Qc, obs_fn, _ = make_mfts_dynamics(
    p,
    lambda_dist=LAMBDA_EKF_UKF,
    sigma_dist=SIGMA_EKF_ENKF_PF,
    bar_F3=bar_est,
    bar_F4=bar_est,
    **_dyn_kw,
)
drift_ukf, L_ukf, _, _, _ = make_mfts_dynamics(
    p,
    lambda_dist=LAMBDA_EKF_UKF,
    sigma_dist=SIGMA_UKF,
    bar_F3=bar_est,
    bar_F4=bar_est,
    **_dyn_kw,
)
drift_ens, L_ens, _, _, _ = make_mfts_dynamics(
    p,
    lambda_dist=LAMBDA_ENKF_PF,
    sigma_dist=SIGMA_EKF_ENKF_PF,
    bar_F3=bar_est,
    bar_F4=bar_est,
    **_dyn_kw,
)

print(
    f"Δt={dt_min:g} min, n_obs={n_obs}, noise={NOISE_MODE}, "
    f"σ×={SIGMA_SCALE:g}, mape={MAPE_MODE}, "
    f"λ_ekf={LAMBDA_EKF_UKF:g}, λ_ens={LAMBDA_ENKF_PF:g}, "
    f"seeds={SEEDS}, "
    f"WSC-KF configs={[c.label + f'({c.n_paths})' for c in WSCKF_CONFIGS]}"
)

raw_rows: list[dict] = []


def mape_at_obs(m_filt: np.ndarray, true_fine: np.ndarray, state_idx: slice) -> float:
    """MAPE of filter means vs truth at t0 and each observation time."""
    true_obs = true_fine[::stride]
    n = min(m_filt.shape[0], true_obs.shape[0])
    return mape_pct(m_filt[:n], true_obs[:n], state_idx)


def score(
    family: str,
    label: str,
    budget: int,
    seed: int,
    m_filt: np.ndarray,
    final_ll: float,
    wall_s: float,
    drift_fn,
    true_fine: np.ndarray,
):
    if MAPE_MODE == "obs":
        mapex = mape_at_obs(m_filt, true_fine, slice(0, 4))
        maped = mape_at_obs(m_filt, true_fine, slice(4, 6))
    else:
        est_fine = estimates_on_fine_grid(
            m_filt, dt, times_fine_sec, drift_fn, DT_FILTER
        )
        mapex = mape_pct(est_fine, true_fine, slice(0, 4))
        maped = mape_pct(est_fine, true_fine, slice(4, 6))
    rec = {
        "seed": seed,
        "family": family,
        "label": label,
        "budget": budget,
        "mapex": mapex,
        "maped": maped,
        "nll_per_obs": -final_ll / n_obs,
        "time_s": wall_s,
    }
    raw_rows.append(rec)
    print(
        f"  seed={seed}  {family:12s} {label:14s} n={budget:4d}  "
        f"MAPEx={rec['mapex']:6.2f}  MAPEd={rec['maped']:6.2f}  "
        f"time={1e3 * wall_s:7.1f} ms"
    )


def build_wsckf(cfg: WSCKFConfig, m0, chol_P0, chol_R, ys):
    wiener_paths = None
    if cfg.method == "lv":
        wiener_paths = precompute_cubature_paths(
            horizon=dt,
            noise_dim=2,
            integration_dt=dt0,
            cubature_order=1,
            multidim_method="lv",
            lyons_degree=cfg.lyons_degree,
        )
        # Verify exact count.
        n_actual = int(np.asarray(wiener_paths[2]).shape[0])
        if n_actual != cfg.n_paths:
            raise RuntimeError(
                f"LV m={cfg.lyons_degree}: expected {cfg.n_paths} paths, got {n_actual}"
            )
    return build_filter(
        get_init_params=lambda mi: (m0, chol_P0),
        get_dynamics=lambda mi: (
            lambda t, x: drift_ens(t, x),
            lambda t, x: L_ens(t, x),
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


for seed in SEEDS:
    print(f"\n=== seed {seed} ===")
    true_fine, ys_fine_np, R = simulate_mfts(
        horizon_min=HORIZON_MIN,
        n_obs=N_FINE,
        n_substeps_truth=N_SUBSTEPS_TRUTH_FINE,
        p=p,
        lambda_sim=LAMBDA_SIM,
        sigma_sim=SIGMA_SIM,
        F1=F1,
        F2=F2,
        seed=seed,
        noise=NOISE_MODE,
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

    # WSC-KF exact-count configs
    for cfg in WSCKF_CONFIGS:
        if not _want(cfg.family):
            continue
        if cfg.method == "product":
            n_expect = product_path_count(2, cfg.k, lyons_degree=cfg.lyons_degree)
            if n_expect != cfg.n_paths:
                raise RuntimeError(f"config mismatch: {cfg} vs product count {n_expect}")
        wsckf = build_wsckf(cfg, m0, chol_P0, chol_R, ys)
        jitted = jit(run_filter, static_argnames=["filter_obj"])
        states, wall = time_call(lambda f=jitted, filt=wsckf: f(filt, model_inputs))
        score(
            cfg.family,
            cfg.label,
            cfg.n_paths,
            seed,
            stack_estimates(m0, states.mean[1:]),
            float(states.log_normalizing_constant[-1]),
            wall,
            drift_ens,
            true_fine,
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
            seed,
            stack_estimates(m0, ekf_res.means),
            float(ekf_res.cum_loglik[-1]),
            wall,
            drift_ekf,
            true_fine,
        )

    if _want("CD-UKF-fine") or _want("CD-UKF"):
        # Moment-ODE predict (canonical continuous–discrete UKF).
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
            "CD-UKF-fine",
            "UKF-fine",
            0,
            seed,
            stack_estimates(m0, ukf_res.means),
            float(ukf_res.cum_loglik[-1]),
            wall,
            drift_ukf,
            true_fine,
        )

    if _want("CD-UKF-coarse"):
        # Flow-map predict (discrete UKF on the SDE flow over each obs interval).
        ukf_c_jit = jit(
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
                state_order="flowmap",
                solver=SOLVER,
            )
        )
        ukf_c_res, wall = time_call(ukf_c_jit)
        score(
            "CD-UKF-coarse",
            "UKF-coarse",
            0,
            seed,
            stack_estimates(m0, ukf_c_res.means),
            float(ukf_c_res.cum_loglik[-1]),
            wall,
            drift_ukf,
            true_fine,
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
            seed,
            stack_estimates(m0, ckf_res.means),
            float(ckf_res.cum_loglik[-1]),
            wall,
            drift_ukf,
            true_fine,
        )

    key = random.key(seed)
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
                seed,
                stack_estimates(m0, enkf_res.means),
                float(enkf_res.cum_loglik[-1]),
                wall,
                drift_ens,
                true_fine,
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
                seed,
                stack_estimates(m0, pf_res.means),
                float(pf_res.cum_loglik[-1]),
                wall,
                drift_ens,
                true_fine,
            )


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------

raw_path = RAW_PATH
if _args.merge and raw_rows:
    replaced = {(r["seed"], r["family"], r["label"]) for r in raw_rows}
    EXISTING_RAW = [
        r
        for r in EXISTING_RAW
        if (r["seed"], r["family"], r["label"]) not in replaced
    ]
raw_rows = EXISTING_RAW + raw_rows
with open(raw_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "delta_t_min",
            "n_obs",
            "seed",
            "family",
            "label",
            "budget",
            "mapex",
            "maped",
            "nll_per_obs",
            "time_s",
        ],
    )
    writer.writeheader()
    for rec in raw_rows:
        writer.writerow({"delta_t_min": dt_min, "n_obs": n_obs, **rec})

keys = sorted({(r["family"], r["label"], r["budget"]) for r in raw_rows})
summary: list[dict] = []
for family, label, budget in keys:
    subset = [
        r for r in raw_rows if r["family"] == family and r["label"] == label and r["budget"] == budget
    ]
    def _mean_std(field: str) -> tuple[float, float]:
        vals = np.asarray([r[field] for r in subset], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    mx, mx_s = _mean_std("mapex")
    md, md_s = _mean_std("maped")
    nll, nll_s = _mean_std("nll_per_obs")
    tm, tm_s = _mean_std("time_s")
    summary.append(
        {
            "family": family,
            "label": label,
            "budget": budget,
            "n_seeds": len(subset),
            "mapex_mean": mx,
            "mapex_std": mx_s,
            "maped_mean": md,
            "maped_std": md_s,
            "nll_per_obs_mean": nll,
            "nll_per_obs_std": nll_s,
            "time_s_mean": tm,
            "time_s_std": tm_s,
        }
    )

csv_path = os.path.join(OUTPUT_DIR, f"pareto_budget_vs_mape{OUT_TAG}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "delta_t_min",
            "n_obs",
            "family",
            "label",
            "budget",
            "n_seeds",
            "mapex_mean",
            "mapex_std",
            "maped_mean",
            "maped_std",
            "nll_per_obs_mean",
            "nll_per_obs_std",
            "time_s_mean",
            "time_s_std",
        ],
    )
    writer.writeheader()
    for rec in sorted(summary, key=lambda r: (r["family"], r["budget"], r["label"])):
        writer.writerow({"delta_t_min": dt_min, "n_obs": n_obs, **rec})

# ---------------------------------------------------------------------------
# Plot mean ± 1 SD Pareto (WSC-KF-LV omitted from figure; kept in CSV)
# ---------------------------------------------------------------------------


# Additive: single moment-safe WSC-KF point. Multiplicative: full product ladder
# so path-order / Lévy-area effects are visible on the Pareto.
WSC_KF_PLOT_BUDGETS: set[int] | None = (
    None if NOISE_MODE == "multiplicative" else {4}
)
# Keep 25 / 1000 in CSV; omit from figure for a cleaner mid-range Pareto.
PARTICLE_PLOT_HIDE_BUDGETS = {25, 1000}
plot_families = [
    "WSC-KF-product",
    "CD-EnKF",
    "PF",
    "CD-EKF",
    "CD-UKF-fine",
    "CD-UKF-coarse",
    "CD-CKF",
]


def _plot_pts_for_family(fam: str) -> list[dict]:
    pts = [r for r in summary if r["family"] == fam]
    if fam == "WSC-KF-product" and WSC_KF_PLOT_BUDGETS is not None:
        pts = [r for r in pts if r["budget"] in WSC_KF_PLOT_BUDGETS]
    elif fam in ("CD-EnKF", "PF"):
        pts = [r for r in pts if r["budget"] not in PARTICLE_PLOT_HIDE_BUDGETS]
    return pts


def _annotate_budget_endpoints(ax, fam, pts, xs, ys) -> None:
    """Label only the smallest and largest variable path/particle budgets."""
    if len(pts) < 2:
        return

    color = COLORS.get(fam, PALETTE["neutral_gray"])
    prefix = "P" if fam.startswith("WSC-KF") else "N"
    dx, dy = _LABEL_OFFSET.get(fam, (0, 6))
    for idx in (0, len(pts) - 1):
        budget = int(pts[idx]["budget"])
        ax.annotate(
            f"{prefix}={budget}",
            (xs[idx], ys[idx]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.2,
            color=color,
            ha="center",
            va="bottom" if dy >= 0 else "top",
            alpha=0.9,
        )


def _plot_family(ax, fam: str, metric: str):
    """Draw one family with its actual across-seed ±1 SD in both coordinates."""
    pts = sorted(_plot_pts_for_family(fam), key=lambda r: r["budget"])
    if not pts:
        return None

    xs = np.asarray([1e3 * r["time_s_mean"] for r in pts], dtype=np.float64)
    xstd = np.asarray([1e3 * r["time_s_std"] for r in pts], dtype=np.float64)
    ys = np.asarray([r[f"{metric}_mean"] for r in pts], dtype=np.float64)
    ystd = np.asarray([r[f"{metric}_std"] for r in pts], dtype=np.float64)
    color = COLORS.get(fam, PALETTE["neutral_gray"])
    marker = METHOD_MARKERS.get(fam, "o")

    ax.errorbar(
        xs,
        ys,
        xerr=xstd,
        yerr=ystd,
        fmt="none",
        ecolor=color,
        elinewidth=0.65,
        capsize=1.5,
        capthick=0.65,
        alpha=0.48,
        zorder=1,
    )
    line, = ax.plot(
        xs,
        ys,
        color=color,
        marker=marker,
        linestyle="-" if len(pts) > 1 else "none",
        linewidth=1.25,
        markersize=7.0 if marker == "*" else 4.2,
        markeredgecolor=color,
        markeredgewidth=0.7,
        label=DISPLAY_NAME.get(fam, fam),
        zorder=2,
    )
    return line, pts, xs, ys


def _style_pareto_axis(ax, ylabel: str) -> None:
    ax.set_xlabel("Wall-clock time [ms]")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    style_curve_axis(ax, grid_axis="both")
    ax.margins(x=0.08, y=0.12)


def _plot_pareto_nll(out_name: str):
    """Single-column NLL Pareto with an external family legend."""
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.65))
    handles = []
    labels = []
    for fam in plot_families:
        plotted = _plot_family(ax, fam, "nll_per_obs")
        if plotted is None:
            continue
        line, pts, xs, ys = plotted
        handles.append(line)
        labels.append(DISPLAY_NAME.get(fam, fam))
        _annotate_budget_endpoints(ax, fam, pts, xs, ys)

    _style_pareto_axis(ax, "NLL\n[nats/observation]")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        fontsize=6.4,
        columnspacing=0.65,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.17, top=0.72)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    paths = save_publication_figure(fig, out_path)
    plt.close()
    return paths


def _plot_pareto_mape(out_name: str):
    """Full-width MAPEx/MAPEd panels with one family legend."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.45),
        sharex=True,
    )
    handles = []
    labels = []
    for panel_idx, (ax, metric, panel_title) in enumerate(
        zip(
            axes,
            ("mapex", "maped"),
            ("Tank states", "Disturbance states"),
        )
    ):
        for fam in plot_families:
            plotted = _plot_family(ax, fam, metric)
            if plotted is None:
                continue
            line, pts, xs, ys = plotted
            if panel_idx == 0:
                handles.append(line)
                labels.append(DISPLAY_NAME.get(fam, fam))
                _annotate_budget_endpoints(ax, fam, pts, xs, ys)
        _style_pareto_axis(ax, "MAPE [%]")
        ax.set_title(panel_title)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=max(1, len(handles)),
        fontsize=6.7,
        columnspacing=0.65,
        handletextpad=0.35,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.19,
        top=0.76,
        wspace=0.27,
    )
    out_path = os.path.join(OUTPUT_DIR, out_name)
    paths = save_publication_figure(fig, out_path)
    plt.close()
    return paths


mape_png_path, mape_pdf_path = _plot_pareto_mape(
    f"pareto_budget_vs_mape{OUT_TAG}.png"
)
nll_png_path, nll_pdf_path = _plot_pareto_nll(
    f"pareto_budget_vs_nll{OUT_TAG}.png"
)

print("-" * 72)
print(f"Wrote {mape_png_path}")
print(f"Wrote {mape_pdf_path}")
print(f"Wrote {nll_png_path}")
print(f"Wrote {nll_pdf_path}")
print(f"Wrote {csv_path}")
print(f"Wrote {raw_path}")
