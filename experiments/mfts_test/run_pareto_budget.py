"""Fixed-Δt cost–accuracy Pareto over exact cubature path counts and particles.

CRF uses **product** Lyons–Victoir constructions so the path count is exact
(no recombined collapse). Default ladder for ``d=2``:

  m=3,k=1 → 4 | m=5,k=1 → 9 | m=3,k=2 → 16 | m=3,k=3 → 64 | m=5,k=2 → 81

Optional ``--include-lv`` adds true multi-D LV (m=3 → 4, m=5 → 18).

Accuracy and wall-clock are averaged over multiple truth/filter seeds; the plot
shows mean ± 1 std.

Run:
  uv run --extra experiments python experiments/mfts_test/run_pareto_budget.py
  uv run --extra experiments python experiments/mfts_test/run_pareto_budget.py \\
      --seeds 100,101,102,103,104 --include-lv

Outputs:
  ``experiments/mfts_test/output/pareto_budget_vs_mape.png``
  ``experiments/mfts_test/output/pareto_budget_vs_nll.png``
  ``experiments/mfts_test/output/pareto_budget_vs_mape.csv``       (seed means)
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
from crf import build_filter
from crf.library import product_path_count

from mfts_model import (
    MFTSParams,
    init_estimate,
    make_mfts_dynamics,
    mape_pct,
    simulate_mfts,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional true multi-D LV (exact counts: m3→4, m5→18 for d=2).
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
    """Exact-count CRF Wiener construction."""

    family: str  # "CRF-product" | "CRF-LV"
    label: str
    n_paths: int
    k: int = 1
    lyons_degree: int = 5
    method: str = "product"  # product | lv


def default_product_ladder(noise_dim: int = 2, max_paths: int = 100) -> list[CrfConfig]:
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
        raise SystemExit("--include-lv requires lyons_victoir_paths in multiplicative_noise_test")
    # Counts from Stroud / GH×Bernoulli (see lyons_victoir_paths.describe).
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


_parser = argparse.ArgumentParser(
    description="MFTS multi-seed Pareto: exact cubature paths vs particles"
)
_parser.add_argument(
    "--seeds",
    type=str,
    default="100,101,102,103,104",
    help="Comma-separated truth/filter seeds to average over",
)
_parser.add_argument(
    "--delta",
    type=float,
    default=1.0,
    help="Fixed observation interval Δt in minutes (default 1.0)",
)
_parser.add_argument("--eval-dt", type=float, default=0.05, dest="eval_dt")
_parser.add_argument(
    "--crf-max-paths",
    type=int,
    default=100,
    dest="crf_max_paths",
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
_args = _parser.parse_args()

SEEDS = _parse_ints(_args.seeds)
DELTA_T_MIN = float(_args.delta)
EVAL_DT_MIN = float(_args.eval_dt)
SOLVER = _args.solver
N_TIMING = _args.n_timing
NOISE_MODE = _args.noise
if _args.tag.strip():
    OUT_TAG = f"_{_args.tag.strip()}"
elif NOISE_MODE == "multiplicative":
    OUT_TAG = "_mult"
else:
    OUT_TAG = ""
ENKF_PARTICLE_LIST = _parse_ints(_args.enkf)
PF_PARTICLE_LIST = _parse_ints(_args.pf)
CRF_CONFIGS = default_product_ladder(2, max_paths=_args.crf_max_paths)
if _args.include_lv:
    CRF_CONFIGS = sorted(CRF_CONFIGS + lv_configs(2), key=lambda c: (c.n_paths, c.family))
ONLY_FAMILIES: Optional[set[str]] = None
if _args.only.strip():
    ONLY_FAMILIES = {s.strip() for s in _args.only.split(",") if s.strip()}
if not SEEDS:
    raise SystemExit("--seeds must list at least one seed")
if ONLY_FAMILIES is None:
    if not CRF_CONFIGS:
        raise SystemExit("No CRF configs (raise --crf-max-paths?)")
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
SIGMA_SIM = 5.0
LAMBDA_EKF_UKF = 0.0
LAMBDA_ENKF_PF = 2.0e-3
SIGMA_EKF_ENKF_PF = 5.0
# Additive Nielsen bakeoff uses a tighter UKF process noise; under multiplicative
# diffusion keep a shared σ so moment-ODE vs path filters are comparable.
SIGMA_UKF = 5.0 if NOISE_MODE == "multiplicative" else 1.0
F_NOISE_SCALE = 150.0
F1, F2 = 5.0, 5.0

# Spherical-radial cubature ≡ UKF with λ = 0 (Arasaratnam–Haykin CD-CKF).
CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)

COLORS = {
    "CD-EKF": "#2E86AB",
    "CD-UKF-fine": "#E8B923",  # yellow — moment-ODE predict
    "CD-UKF-coarse": "#B8860B",  # darkgoldenrod — flow-map predict
    "CD-CKF": "#3DA35D",  # green
    "CD-EnKF": "#D97706",  # amber-orange (away from PF red / UKF yellow)
    "PF": "#F24236",
    "CRF-product": "#6A4C93",
    "CRF-LV": "#C44536",
}

# Legend / axis display names (CSV family keys stay stable).
DISPLAY_NAME = {
    "CRF-product": "WS-CKF",
    "CRF-LV": "WS-CKF-LV",
    "CD-EKF": "CD-EKF",
    "CD-UKF-fine": "CD-UKF (fine)",
    "CD-UKF-coarse": "CD-UKF (coarse)",
    "CD-CKF": "CD-CKF",
    "CD-EnKF": "CD-EnKF",
    "PF": "PF",
}

# Vertical label offsets (points) per family — keep curves readable on log-x.
_LABEL_OFFSET = {
    "CRF-product": (0, 7),
    "CD-EnKF": (0, -10),
    "PF": (0, 7),
}
# CD-EnKF early nodes sit under the curve; pin those labels above instead.
_LABEL_ABOVE_BUDGETS = {
    "CD-EnKF": {50, 100},
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
    f"Δt={dt_min:g} min, n_obs={n_obs}, noise={NOISE_MODE}, seeds={SEEDS}, "
    f"CRF configs={[c.label + f'({c.n_paths})' for c in CRF_CONFIGS]}"
)

raw_rows: list[dict] = []


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
    est_fine = estimates_on_fine_grid(
        m_filt, dt, times_fine_sec, drift_fn, DT_FILTER
    )
    rec = {
        "seed": seed,
        "family": family,
        "label": label,
        "budget": budget,
        "mapex": mape_pct(est_fine, true_fine, slice(0, 4)),
        "maped": mape_pct(est_fine, true_fine, slice(4, 6)),
        "nll_per_obs": -final_ll / n_obs,
        "time_s": wall_s,
    }
    raw_rows.append(rec)
    print(
        f"  seed={seed}  {family:12s} {label:14s} n={budget:4d}  "
        f"MAPEx={rec['mapex']:6.2f}  MAPEd={rec['maped']:6.2f}  "
        f"time={1e3 * wall_s:7.1f} ms"
    )


def build_crf(cfg: CrfConfig, m0, chol_P0, chol_R, ys):
    wiener_paths = None
    if cfg.method == "lv":
        assert precompute_lv_paths is not None
        wiener_paths = precompute_lv_paths(
            horizon=dt,
            noise_dim=2,
            lyons_degree=cfg.lyons_degree,
            integration_dt=dt0,
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

    # CRF exact-count configs
    for cfg in CRF_CONFIGS:
        if not _want(cfg.family):
            continue
        if cfg.method == "product":
            n_expect = product_path_count(2, cfg.k, lyons_degree=cfg.lyons_degree)
            if n_expect != cfg.n_paths:
                raise RuntimeError(f"config mismatch: {cfg} vs product count {n_expect}")
        crf = build_crf(cfg, m0, chol_P0, chol_R, ys)
        jitted = jit(run_filter, static_argnames=["filter_obj"])
        states, wall = time_call(lambda f=jitted, filt=crf: f(filt, model_inputs))
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
n_seeds_total = len({r["seed"] for r in raw_rows})
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
# Plot mean ± std Pareto (CRF-LV omitted from figure; kept in CSV)
# ---------------------------------------------------------------------------


def _annotate_nodes(ax, fam, pts, xs, ys):
    """Label multi-budget nodes with path/particle count (EKF/UKF/CKF via legend)."""
    if fam in ("CD-EKF", "CD-UKF-fine", "CD-UKF-coarse", "CD-CKF", "CRF-product"):
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
            # Slight stagger on dense product ladder so early nodes don't collide.
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


BAND_SCALE = 0.3  # visual width of shaded ±σ band
# Additive: single moment-safe WS-CKF point. Multiplicative: full product ladder
# so path-order / Lévy-area effects are visible on the Pareto.
WS_CKF_PLOT_BUDGETS: set[int] | None = (
    None if NOISE_MODE == "multiplicative" else {4}
)
# Keep 25 / 1000 in CSV; omit from figure for a cleaner mid-range Pareto.
PARTICLE_PLOT_HIDE_BUDGETS = {25, 1000}
plot_families = [
    "CRF-product",
    "CD-EnKF",
    "PF",
    "CD-EKF",
    "CD-UKF-fine",
    "CD-UKF-coarse",
    "CD-CKF",
]


def _plot_pts_for_family(fam: str) -> list[dict]:
    pts = [r for r in summary if r["family"] == fam]
    if fam == "CRF-product" and WS_CKF_PLOT_BUDGETS is not None:
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
        if fam == "CRF-product":
            marker = "*"
        elif fam in ("CD-EKF", "CD-UKF-fine", "CD-UKF-coarse", "CD-CKF"):
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
    ax.set_yscale("linear")
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


def _plot_pareto_mapex_maped(out_name: str, title: str = "") -> str:
    """MAPEx (solid) and MAPEd (dashed) on one axes; color = filter family."""

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    all_xs: list[float] = []
    metric_style = {
        # Single-point filters keep modest bands; PF/EnKF curves use much lighter fill.
        "mapex": {
            "ls": "-",
            "alpha_band_point": 0.18,
            "alpha_band_curve": 0.035,
            "markerfacecolor": None,
            "z": 3,
        },
        "maped": {
            "ls": "--",
            "alpha_band_point": 0.12,
            "alpha_band_curve": 0.025,
            "markerfacecolor": "white",
            "z": 2,
        },
    }
    for fam in plot_families:
        pts = _plot_pts_for_family(fam)
        if not pts:
            continue
        pts = sorted(pts, key=lambda r: r["budget"])
        xs = np.asarray([1e3 * r["time_s_mean"] for r in pts])
        all_xs.extend(xs.tolist())
        color = COLORS.get(fam, "#333333")
        if fam == "CRF-product":
            marker = "*"
        elif fam in ("CD-EKF", "CD-UKF-fine", "CD-UKF-coarse", "CD-CKF"):
            marker = "s"
        else:
            marker = "o"
        disp = DISPLAY_NAME.get(fam, fam)
        for metric, sty in metric_style.items():
            ys = np.asarray([r[f"{metric}_mean"] for r in pts])
            ystd = np.asarray([r[f"{metric}_std"] for r in pts]) * BAND_SCALE
            mfc = color if sty["markerfacecolor"] is None else sty["markerfacecolor"]
            label = f"{disp} · MAPEx" if metric == "mapex" else f"{disp} · MAPEd"
            ms = 12 if marker == "*" else 7
            if len(pts) == 1:
                x0 = float(xs[0])
                ax.fill_between(
                    [x0 / 1.04, x0 * 1.04],
                    [ys[0] - ystd[0]] * 2,
                    [ys[0] + ystd[0]] * 2,
                    color=color,
                    alpha=sty["alpha_band_point"],
                    linewidth=0,
                    zorder=sty["z"] - 1,
                )
                ax.plot(
                    xs,
                    ys,
                    marker,
                    color=color,
                    markersize=ms,
                    markerfacecolor=mfc,
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                    linestyle="none",
                    label=label,
                    zorder=sty["z"],
                )
            else:
                ax.fill_between(
                    xs,
                    ys - ystd,
                    ys + ystd,
                    color=color,
                    alpha=sty["alpha_band_curve"],
                    linewidth=0,
                    zorder=sty["z"] - 1,
                )
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    linestyle=sty["ls"],
                    color=color,
                    lw=1.5,
                    markersize=6 if marker != "*" else 10,
                    markerfacecolor=mfc,
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    label=label,
                    zorder=sty["z"],
                )
        # Particle/path labels once (on MAPEx).
        _annotate_nodes(
            ax,
            fam,
            pts,
            xs,
            np.asarray([r["mapex_mean"] for r in pts]),
        )

    ax.set_xlabel("Wall-clock time [ms]", fontsize=13)
    ax.set_ylabel("MAPE [%]", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if all_xs:
        ax.set_xlim(min(10.0, min(all_xs) / 1.25), max(1e3, max(all_xs) * 1.35))
    ax.set_xticks([1e1, 1e2, 1e3])
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, which="both", alpha=0.3)
    # Compact legend: family colors + metric style keys.
    handles, labels = ax.get_legend_handles_labels()
    # Keep family·metric entries but add style cue at top via title is enough;
    # split into two columns for readability.
    ax.legend(
        handles,
        labels,
        fontsize=10,
        loc="best",
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
        title=r"solid = MAPEx,  dashed/open = MAPEd",
        title_fontsize=10,
    )
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


_ws_ckf_budgets = (
    "ladder"
    if WS_CKF_PLOT_BUDGETS is None
    else ",".join(str(b) for b in sorted(WS_CKF_PLOT_BUDGETS))
)
_noise_note = "multiplicative $g(F)$" if NOISE_MODE == "multiplicative" else "additive"
_title_suffix = (
    rf"at $\Delta t={dt_min:g}$ min, {_noise_note} "
    rf"(mean±{BAND_SCALE:g}$\sigma$ over {n_seeds_total} seeds; "
    rf"WS-CKF paths = {_ws_ckf_budgets})"
)
mape_plot_path = _plot_pareto_mapex_maped(
    f"pareto_budget_vs_mape{OUT_TAG}.png",
    title=rf"MFTS Pareto: MAPE vs wall-clock {_title_suffix}",
)
nll_plot_path = _plot_pareto_metric(
    "nll_per_obs",
    "Mean NLL per observation",
    rf"MFTS Pareto: NLL vs wall-clock {_title_suffix}",
    f"pareto_budget_vs_nll{OUT_TAG}.png",
)

print("-" * 72)
print(f"Wrote {mape_plot_path}")
print(f"Wrote {nll_plot_path}")
print(f"Wrote {csv_path}")
print(f"Wrote {raw_path}")
