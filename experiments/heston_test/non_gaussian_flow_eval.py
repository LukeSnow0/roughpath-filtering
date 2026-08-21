"""Non-Gaussian one-step transition diagnostics for Heston in ``x=[X,v]``.

Claim (1) experiment: compare one-step predictive-law fidelity of WSC-KF against
Gaussian-surrogate forecasts (CD-CKF) under refinement, plus EnKF / bootstrap PF /
KLV-PF (Crisan–Ortiz-Latorre) particle forecasts.

Run:
  uv run --extra experiments python experiments/heston_test/non_gaussian_flow_eval.py
  # add/refresh KLV-PF only on an existing CSV:
  uv run --extra experiments python experiments/heston_test/non_gaussian_flow_eval.py --append-klv

Outputs under ``experiments/heston_test/output/non_gaussian/``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

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

from cd_dynamax_mirror.filters import (
    UKFParams,
    _ekf_predict,
    _enkf_forecast,
    _ukf_predict,
    _ukf_weights,
)
from heston_model import HestonParams, make_heston_dynamics
from klv_particle_filter import klv_pf_forecast
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict


OUTPUT_DIR = os.path.join(HERE, "output", "non_gaussian")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass(frozen=True)
class Regime:
    name: str
    params: HestonParams


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


def _simulate_transition_mc(
    *,
    x0: np.ndarray,
    p: HestonParams,
    delta: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """High-fidelity Euler reference for one-step transition from point mass."""
    rng = np.random.default_rng(seed)
    dt = float(delta) / int(n_steps)
    sqrt_dt = np.sqrt(dt)
    x = np.tile(np.asarray(x0, dtype=np.float64), (n_paths, 1))
    rho = p.rho
    rho_bar = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    for _ in range(n_steps):
        z = rng.standard_normal((n_paths, 2))
        dW = z * sqrt_dt
        v = np.maximum(x[:, 1], p.v_eps)
        sv = p.diff_scale * np.sqrt(v)
        dX = (p.mu - 0.5 * x[:, 1]) * dt + sv * dW[:, 0]
        dv = p.kappa * (p.theta - x[:, 1]) * dt + p.xi * sv * (
            rho * dW[:, 0] + rho_bar * dW[:, 1]
        )
        x[:, 0] += dX
        x[:, 1] += dv
    return x


def _weighted_stats(x1: np.ndarray, x0: np.ndarray, w: np.ndarray):
    """Return key non-Gaussian diagnostics on ``v`` and mixed increments."""
    w = np.asarray(w, dtype=np.float64)
    w = w / np.sum(w)
    dx = x1[:, 0] - float(x0[0])
    dv = x1[:, 1] - float(x0[1])
    v1 = x1[:, 1]

    mv = float(np.sum(w * v1))
    cv = v1 - mv
    var_v = float(np.sum(w * cv * cv))
    std_v = float(np.sqrt(max(var_v, 1e-30)))
    skew_v = float(np.sum(w * (cv / std_v) ** 3))
    kurt_v = float(np.sum(w * (cv / std_v) ** 4))

    metrics = {
        "v_skew": skew_v,
        "v_kurt": kurt_v,
        "E_dX_dv": float(np.sum(w * dx * dv)),
        "E_dX2_dv": float(np.sum(w * dx * dx * dv)),
    }
    return metrics


def _gaussian_moment_proxy(
    mean: np.ndarray, cov: np.ndarray, x0: np.ndarray, n_draws: int, seed: int
) -> dict[str, float]:
    """Approximate nonlinear moments implied by a Gaussian surrogate N(mean,cov)."""
    rng = np.random.default_rng(seed)
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-14)
    cov_psd = (evecs * evals) @ evecs.T
    draws = rng.multivariate_normal(mean=mean, cov=cov_psd, size=n_draws)
    w = np.full(n_draws, 1.0 / n_draws, dtype=np.float64)
    return _weighted_stats(draws, x0=x0, w=w)


def _predict_wsckf_pointmass(
    p: HestonParams,
    x0: np.ndarray,
    delta: float,
    *,
    lyons_degree: int,
    k: int,
    integration_dt: float,
):
    drift, L_fn, _, _, _ = make_heston_dynamics(p)
    strat_drift, strat_L = ito_to_stratonovich_dynamics(drift, L_fn)
    ts, path_values, path_weights = precompute_cubature_paths(
        horizon=float(delta),
        noise_dim=2,
        integration_dt=float(integration_dt),
        cubature_order=int(k),
        multidim_method="lv",
        lyons_degree=int(lyons_degree),
    )
    mean, chol, samples, weights = predict(
        mean=jnp.asarray(x0, dtype=jnp.float64),
        chol_cov=jnp.zeros((2, 2), dtype=jnp.float64),
        ts=jnp.asarray(ts),
        path_values=jnp.asarray(path_values),
        path_weights=jnp.asarray(path_weights),
        drift_fn=strat_drift,
        diffusion_fn=strat_L,
    )
    cov = np.asarray(chol @ chol.T)
    return np.asarray(mean), cov, np.asarray(samples), np.asarray(weights)


def _prepare_wsckf_predictor(
    p: HestonParams,
    x0: np.ndarray,
    delta: float,
    *,
    lyons_degree: int,
    k: int,
    integration_dt: float,
) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build a zero-arg WSC predictor with precomputed paths outside timing."""
    drift, L_fn, _, _, _ = make_heston_dynamics(p)
    strat_drift, strat_L = ito_to_stratonovich_dynamics(drift, L_fn)
    ts, path_values, path_weights = precompute_cubature_paths(
        horizon=float(delta),
        noise_dim=2,
        integration_dt=float(integration_dt),
        cubature_order=int(k),
        multidim_method="lv",
        lyons_degree=int(lyons_degree),
    )

    def _run():
        mean, chol, samples, weights = predict(
            mean=jnp.asarray(x0, dtype=jnp.float64),
            chol_cov=jnp.zeros((2, 2), dtype=jnp.float64),
            ts=jnp.asarray(ts),
            path_values=jnp.asarray(path_values),
            path_weights=jnp.asarray(path_weights),
            drift_fn=strat_drift,
            diffusion_fn=strat_L,
        )
        cov = np.asarray(chol @ chol.T)
        return np.asarray(mean), cov, np.asarray(samples), np.asarray(weights)

    return _run


def _predict_cd_ckf_pointmass(
    p: HestonParams,
    x0: np.ndarray,
    delta: float,
    *,
    n_inner: int,
):
    drift, L_fn, Qc_fn, _, _ = make_heston_dynamics(p)
    ckf = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)
    n = 2
    lamb = ckf.alpha**2 * (n + ckf.kappa) - n
    w_mean, w_cov, W = _ukf_weights(n, ckf.alpha, ckf.beta, lamb)
    m_pred, P_pred = _ukf_predict(
        jnp.asarray(x0, dtype=jnp.float64),
        jnp.zeros((2, 2), dtype=jnp.float64),
        drift,
        L_fn,
        Qc_fn,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(delta, dtype=jnp.float64),
        jnp.asarray(delta / n_inner, dtype=jnp.float64),
        n,
        lamb,
        w_mean,
        w_cov,
        W,
        state_order="flowmap",
        solver="rk4",
        n_steps=int(n_inner),
    )
    return np.asarray(m_pred), np.asarray(P_pred)


def _predict_cd_ekf_pointmass(
    p: HestonParams,
    x0: np.ndarray,
    delta: float,
    *,
    n_inner: int,
):
    drift, L_fn, Qc_fn, _, _ = make_heston_dynamics(p)
    m_pred, P_pred = _ekf_predict(
        jnp.asarray(x0, dtype=jnp.float64),
        jnp.zeros((2, 2), dtype=jnp.float64),
        drift,
        L_fn,
        Qc_fn,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(delta, dtype=jnp.float64),
        jnp.asarray(delta / n_inner, dtype=jnp.float64),
        state_order="flowmap",
        solver="rk4",
        n_steps=int(n_inner),
    )
    return np.asarray(m_pred), np.asarray(P_pred)


def _predict_cd_ukf_pointmass(
    p: HestonParams,
    x0: np.ndarray,
    delta: float,
    *,
    n_inner: int,
):
    drift, L_fn, Qc_fn, _, _ = make_heston_dynamics(p)
    ukf = UKFParams()
    n = 2
    lamb = ukf.alpha**2 * (n + ukf.kappa) - n
    w_mean, w_cov, W = _ukf_weights(n, ukf.alpha, ukf.beta, lamb)
    m_pred, P_pred = _ukf_predict(
        jnp.asarray(x0, dtype=jnp.float64),
        jnp.zeros((2, 2), dtype=jnp.float64),
        drift,
        L_fn,
        Qc_fn,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(delta, dtype=jnp.float64),
        jnp.asarray(delta / n_inner, dtype=jnp.float64),
        n,
        lamb,
        w_mean,
        w_cov,
        W,
        state_order="flowmap",
        solver="rk4",
        n_steps=int(n_inner),
    )
    return np.asarray(m_pred), np.asarray(P_pred)


def _predict_particle_forecast_pointmass(
    p: HestonParams,
    x0: np.ndarray,
    delta: float,
    *,
    n_inner: int,
    n_particles: int,
    seed: int,
):
    drift, L_fn, Qc_fn, _, _ = make_heston_dynamics(p)
    particles0 = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], n_particles, axis=0)
    particles1 = _enkf_forecast(
        key=jax.random.key(int(seed)),
        particles=jnp.asarray(particles0, dtype=jnp.float64),
        f=drift,
        L_fn=L_fn,
        Qc_fn=Qc_fn,
        t0=0.0,
        t1=float(delta),
        dt0=float(delta / n_inner),
        noise_dim=2,
        cov_rescaling=1.0,
        solver="rk4",
        n_steps=int(n_inner),
        sde_form="ito",
    )
    p1 = np.asarray(particles1, dtype=np.float64)
    w = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
    return p1, w


def _time_call(fn, *args, repeats: int = 7, **kwargs):
    """Time a JAX-backed call using median wall-clock over repeated executions."""
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - t0)
    return out, float(np.median(samples))


def _time_interleaved_runners(
    runners: dict[str, Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    *,
    rounds: int = 9,
    seed: int = 0,
) -> dict[str, float]:
    """Benchmark runners with interleaved randomized execution order.

    This reduces systematic order bias (e.g., always timing k=3 after k=2) by
    shuffling execution each round and taking a robust median per runner.
    """
    keys = list(runners.keys())
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in keys}

    # Warm-up all runners once (also primes any JIT traces).
    for k in keys:
        out = runners[k]()
        jax.block_until_ready(out)

    for _ in range(int(rounds)):
        order = keys.copy()
        rng.shuffle(order)
        for k in order:
            t0 = time.perf_counter()
            out = runners[k]()
            jax.block_until_ready(out)
            samples[k].append(time.perf_counter() - t0)

    return {k: float(np.median(v)) for k, v in samples.items()}


def main():
    parser = argparse.ArgumentParser(description="Heston non-Gaussian flow evaluation")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--mc-paths", type=int, default=300000, dest="mc_paths")
    parser.add_argument("--mc-steps", type=int, default=1024, dest="mc_steps")
    parser.add_argument("--gauss-draws", type=int, default=200000, dest="gauss_draws")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated seed list. Overrides --seed when provided.",
    )
    parser.add_argument(
        "--klv",
        type=str,
        default="500,1000,2000,5000",
        help="Comma-separated KLV-PF particle counts (empty = skip).",
    )
    parser.add_argument(
        "--klv-macro",
        type=int,
        default=8,
        dest="klv_macro",
        help="Number of KLV cubature macro-steps over Δ.",
    )
    parser.add_argument(
        "--klv-lyons",
        type=int,
        default=5,
        dest="klv_lyons",
        help="Lyons–Victoir degree m for KLV-PF local cubature.",
    )
    parser.add_argument(
        "--klv-k",
        type=int,
        default=1,
        dest="klv_k",
        help="Cubature order k for KLV-PF local paths.",
    )
    parser.add_argument(
        "--append-klv",
        action="store_true",
        dest="append_klv",
        help="Only run KLV-PF, merge into existing CSV, and replot.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge new rows into existing CSV (replace matching regime/seed/method/metric/refinement).",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default="mild,stress",
        help="Comma-separated regimes to run (mild,stress).",
    )
    args = parser.parse_args()

    x0 = np.array([0.0, 0.04], dtype=np.float64)
    delta = float(args.delta)

    all_regimes = [
        Regime("mild", HestonParams(xi=0.3, rho=-0.4, meas_std=0.02, v0=float(x0[1]))),
        Regime("stress", HestonParams(xi=1.0, rho=-0.9, meas_std=0.02, v0=float(x0[1]))),
    ]
    wanted = {s.strip() for s in str(args.regimes).split(",") if s.strip()}
    regimes = [r for r in all_regimes if r.name in wanted]
    if not regimes:
        raise SystemExit(f"No valid regimes in --regimes={args.regimes!r}")
    csv_path = os.path.join(OUTPUT_DIR, "non_gaussian_flow_eval.csv")
    existing_rows: list[dict[str, float | str | int]] = []
    if (args.append_klv or args.append) and os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            existing_rows = [dict(r) for r in csv.DictReader(f)]
        if args.append_klv and not args.append:
            existing_rows = [r for r in existing_rows if r.get("method") != "KLV-PF-forecast"]
        for r in existing_rows:
            r["seed"] = int(r["seed"])
            r["refinement"] = int(float(r["refinement"]))
            for k in ("value", "reference", "abs_error", "time_s"):
                r[k] = float(r[k])

    if args.seeds.strip():
        seeds = _parse_ints(args.seeds)
    elif args.append_klv and existing_rows:
        seeds = sorted({int(r["seed"]) for r in existing_rows})
    else:
        seeds = [int(args.seed)]

    cd_refinement = [1, 2, 4, 8]
    enkf_particles = [500, 1000, 2000, 5000]
    pf_particles = [500, 1000, 2000, 5000]
    klv_particles = _parse_ints(args.klv) if str(args.klv).strip() else []
    klv_macro = int(args.klv_macro)
    klv_m = int(args.klv_lyons)
    klv_k = int(args.klv_k)
    rows: list[dict[str, float | str | int]] = []
    skip_baselines = bool(args.append_klv)

    for reg in regimes:
        print(f"\n=== regime={reg.name} xi={reg.params.xi:g} rho={reg.params.rho:g} ===")
        for seed in seeds:
            print(f"--- seed={seed} ---")
            if skip_baselines and existing_rows:
                ref_rows = [
                    r
                    for r in existing_rows
                    if r["regime"] == reg.name and int(r["seed"]) == int(seed)
                ]
                if not ref_rows:
                    print(f"  skip: no existing rows for {reg.name}/seed={seed}")
                    continue
                ref = {}
                for metric in ("v_skew", "v_kurt", "E_dX_dv", "E_dX2_dv"):
                    vals = [float(r["reference"]) for r in ref_rows if r["metric"] == metric]
                    if not vals:
                        raise SystemExit(f"missing reference for {metric} in {reg.name}/{seed}")
                    ref[metric] = float(np.mean(vals))
            else:
                truth = _simulate_transition_mc(
                    x0=x0,
                    p=reg.params,
                    delta=delta,
                    n_steps=int(args.mc_steps),
                    n_paths=int(args.mc_paths),
                    seed=int(seed),
                )
                w_truth = np.full(truth.shape[0], 1.0 / truth.shape[0], dtype=np.float64)
                ref = _weighted_stats(truth, x0=x0, w=w_truth)

            if not skip_baselines:
                ws_models: list[tuple[str, int, dict[str, float], float]] = []
                ws_seeds = {(3, 1): 11, (3, 2): 12, (5, 1): 13, (5, 2): 14}
                ws_runners: dict[str, Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
                ws_meta: dict[str, tuple[int, int]] = {}
                for m in (3, 5):
                    for k in (1, 2, 3):
                        key = f"m{m}-k{k}"
                        ws_runner = _prepare_wsckf_predictor(
                            reg.params,
                            x0,
                            delta,
                            lyons_degree=m,
                            k=k,
                            integration_dt=delta / 256,
                        )
                        ws_runners[key] = ws_runner
                        ws_meta[key] = (m, k)

                ws_times = _time_interleaved_runners(
                    ws_runners, rounds=11, seed=10_000 + int(seed)
                )
                for key, ws_runner in ws_runners.items():
                    m, k = ws_meta[key]
                    m_hat, P_hat, samples, w_hat = ws_runner()
                    t_ws = ws_times[key]
                    ws_models.append(
                        (f"WSC-KF-m{m}-k{k}-cloud", k, _weighted_stats(samples, x0=x0, w=w_hat), t_ws)
                    )
                    if k == 1:
                        ws_models.append(
                            (
                                f"WSC-KF-m{m}-gauss",
                                k,
                                _gaussian_moment_proxy(
                                    m_hat,
                                    P_hat,
                                    x0,
                                    args.gauss_draws,
                                    int(seed) + ws_seeds[(m, k)],
                                ),
                                t_ws,
                            )
                        )

                for method, ws_k, metrics, runtime_s in ws_models:
                    for metric, val in metrics.items():
                        err = abs(val - ref[metric])
                        rows.append(
                            {
                                "regime": reg.name,
                                "seed": int(seed),
                                "method": method,
                                "refinement": ws_k,
                                "metric": metric,
                                "value": val,
                                "reference": ref[metric],
                                "abs_error": err,
                                "time_s": runtime_s,
                            }
                        )
                        print(f"  {method:16s} {metric:10s} abs_err={err:.3e}")

                for r in cd_refinement:
                    n_inner = 256 * int(r)
                    (m_ekf, P_ekf), ekf_time = _time_call(
                        _predict_cd_ekf_pointmass, reg.params, x0, delta, n_inner=n_inner
                    )
                    ekf_metrics = _gaussian_moment_proxy(
                        m_ekf, P_ekf, x0, args.gauss_draws, int(seed) + 200 + r
                    )
                    for metric, val in ekf_metrics.items():
                        err = abs(val - ref[metric])
                        rows.append(
                            {
                                "regime": reg.name,
                                "seed": int(seed),
                                "method": "CD-EKF-gauss",
                                "refinement": r,
                                "metric": metric,
                                "value": val,
                                "reference": ref[metric],
                                "abs_error": err,
                                "time_s": ekf_time,
                            }
                        )
                        print(f"  CD-EKF x{r:<2d}      {metric:10s} abs_err={err:.3e}")

                    (m_ukf, P_ukf), ukf_time = _time_call(
                        _predict_cd_ukf_pointmass, reg.params, x0, delta, n_inner=n_inner
                    )
                    ukf_metrics = _gaussian_moment_proxy(
                        m_ukf, P_ukf, x0, args.gauss_draws, int(seed) + 300 + r
                    )
                    for metric, val in ukf_metrics.items():
                        err = abs(val - ref[metric])
                        rows.append(
                            {
                                "regime": reg.name,
                                "seed": int(seed),
                                "method": "CD-UKF-gauss",
                                "refinement": r,
                                "metric": metric,
                                "value": val,
                                "reference": ref[metric],
                                "abs_error": err,
                                "time_s": ukf_time,
                            }
                        )
                        print(f"  CD-UKF x{r:<2d}      {metric:10s} abs_err={err:.3e}")

                    (m_ckf, P_ckf), ckf_time = _time_call(
                        _predict_cd_ckf_pointmass, reg.params, x0, delta, n_inner=n_inner
                    )
                    metrics = _gaussian_moment_proxy(
                        m_ckf, P_ckf, x0, args.gauss_draws, int(seed) + 100 + r
                    )
                    for metric, val in metrics.items():
                        err = abs(val - ref[metric])
                        rows.append(
                            {
                                "regime": reg.name,
                                "seed": int(seed),
                                "method": "CD-CKF-gauss",
                                "refinement": r,
                                "metric": metric,
                                "value": val,
                                "reference": ref[metric],
                                "abs_error": err,
                                "time_s": ckf_time,
                            }
                        )
                        print(f"  CD-CKF x{r:<2d}      {metric:10s} abs_err={err:.3e}")

                for n_part in enkf_particles:
                    (parts, wts), enkf_time = _time_call(
                        _predict_particle_forecast_pointmass,
                        reg.params,
                        x0,
                        delta,
                        n_inner=512,
                        n_particles=int(n_part),
                        seed=int(seed) + 400 + int(n_part),
                    )
                    metrics = _weighted_stats(parts, x0=x0, w=wts)
                    for metric, val in metrics.items():
                        err = abs(val - ref[metric])
                        rows.append(
                            {
                                "regime": reg.name,
                                "seed": int(seed),
                                "method": "CD-EnKF-forecast",
                                "refinement": int(n_part),
                                "metric": metric,
                                "value": val,
                                "reference": ref[metric],
                                "abs_error": err,
                                "time_s": enkf_time,
                            }
                        )
                        print(f"  EnKF n={n_part:<4d}   {metric:10s} abs_err={err:.3e}")

                for n_part in pf_particles:
                    (parts, wts), pf_time = _time_call(
                        _predict_particle_forecast_pointmass,
                        reg.params,
                        x0,
                        delta,
                        n_inner=512,
                        n_particles=int(n_part),
                        seed=int(seed) + 500 + int(n_part),
                    )
                    metrics = _weighted_stats(parts, x0=x0, w=wts)
                    for metric, val in metrics.items():
                        err = abs(val - ref[metric])
                        rows.append(
                            {
                                "regime": reg.name,
                                "seed": int(seed),
                                "method": "Bootstrap-PF-forecast",
                                "refinement": int(n_part),
                                "metric": metric,
                                "value": val,
                                "reference": ref[metric],
                                "abs_error": err,
                                "time_s": pf_time,
                            }
                        )
                        print(f"  PF n={n_part:<4d}     {metric:10s} abs_err={err:.3e}")

            for n_part in klv_particles:
                (parts, wts), klv_time = _time_call(
                    klv_pf_forecast,
                    x0=x0,
                    p=reg.params,
                    delta=delta,
                    n_particles=int(n_part),
                    n_macro=klv_macro,
                    lyons_degree=klv_m,
                    cubature_order=klv_k,
                    seed=int(seed) + 600 + int(n_part),
                )
                metrics = _weighted_stats(parts, x0=x0, w=wts)
                for metric, val in metrics.items():
                    err = abs(val - ref[metric])
                    rows.append(
                        {
                            "regime": reg.name,
                            "seed": int(seed),
                            "method": "KLV-PF-forecast",
                            "refinement": int(n_part),
                            "metric": metric,
                            "value": val,
                            "reference": ref[metric],
                            "abs_error": err,
                            "time_s": klv_time,
                        }
                    )
                    print(f"  KLV-PF n={n_part:<4d}  {metric:10s} abs_err={err:.3e}")

    if args.append_klv:
        rows = existing_rows + rows
    elif args.append and existing_rows:

        def _row_key(r):
            return (
                str(r["regime"]),
                int(r["seed"]),
                str(r["method"]),
                int(r["refinement"]),
                str(r["metric"]),
            )

        new_keys = {_row_key(r) for r in rows}
        kept = [r for r in existing_rows if _row_key(r) not in new_keys]
        rows = kept + rows

    csv_path = os.path.join(OUTPUT_DIR, "non_gaussian_flow_eval.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "regime",
                "seed",
                "method",
                "refinement",
                "metric",
                "value",
                "reference",
                "abs_error",
                "time_s",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    metrics_to_plot = ["v_skew", "v_kurt", "E_dX_dv", "E_dX2_dv"]
    metric_title = {
        "v_skew": r"Skewness error of $v_{t+\Delta}$",
        "v_kurt": r"Kurtosis error of $v_{t+\Delta}$",
        "E_dX_dv": r"Mixed moment error $E[\Delta X\,\Delta v]$",
        "E_dX2_dv": r"Mixed moment error $E[(\Delta X)^2\Delta v]$",
    }
    for reg in ("mild", "stress"):
        # Plot A: CKF refinement trend with WSC reference lines.
        fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.4))
        axs = axs.ravel()
        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i]
            ckf = [
                r
                for r in rows
                if r["regime"] == reg and r["method"] == "CD-CKF-gauss" and r["metric"] == metric
            ]
            ckf = sorted(ckf, key=lambda r: int(r["refinement"]))
            xs = np.array([int(r["refinement"]) for r in ckf], dtype=np.float64)
            ys = np.array([float(r["abs_error"]) for r in ckf], dtype=np.float64)
            ax.plot(xs, ys, "o-", color="#2E86AB", lw=1.8, label="CD-CKF Gaussian")

            for method, color, disp in (
                ("WSC-KF-m3-k1-cloud", "#6A4C93", "WSC-KF m=3, k=1 cloud"),
                ("WSC-KF-m5-k1-cloud", "#9B5DE5", "WSC-KF m=5, k=1 cloud"),
            ):
                wr = [
                    r
                    for r in rows
                    if r["regime"] == reg and r["method"] == method and r["metric"] == metric
                ]
                y_mean = float(np.mean([float(r["abs_error"]) for r in wr]))
                ax.axhline(
                    y_mean,
                    color=color,
                    linestyle="--",
                    lw=1.6,
                    label=disp,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{int(x)}x" for x in xs])
            ax.set_yscale("log")
            ax.set_title(metric_title[metric], fontsize=10)
            ax.set_xlabel("CD-CKF forecast refinement")
            ax.set_ylabel("Absolute error vs MC reference")
            ax.grid(True, which="both", alpha=0.3)

        handles, labels = axs[0].get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        fig.legend(
            list(uniq.values()),
            list(uniq.keys()),
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )
        fig.suptitle(
            f"Heston one-step transition diagnostics ({reg}): CKF refinement vs WSC cloud",
            y=0.98,
        )
        fig.tight_layout()
        png_ref = os.path.join(OUTPUT_DIR, f"non_gaussian_flow_eval_refinement_{reg}.png")
        fig.savefig(png_ref, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot -> {png_ref}")

        # Plot B: direct method comparison at a fixed CKF refinement (x8).
        fig2, ax2 = plt.subplots(figsize=(10.5, 4.6))
        methods = [
            ("WSC-KF-m5-k1-cloud", "WSC-KF m=5, k=1 cloud", "#9B5DE5"),
            ("WSC-KF-m5-gauss", "WSC-KF m=5, k=1 Gaussian collapse", "#B8A1F2"),
            ("CD-CKF-gauss", "CD-CKF Gaussian (x8)", "#2E86AB"),
        ]
        xloc = np.arange(len(metrics_to_plot), dtype=np.float64)
        width = 0.25
        for j, (method, label, color) in enumerate(methods):
            vals = []
            for metric in metrics_to_plot:
                candidates = [
                    r
                    for r in rows
                    if r["regime"] == reg and r["method"] == method and r["metric"] == metric
                ]
                if method == "CD-CKF-gauss":
                    candidates = [r for r in candidates if int(r["refinement"]) == 8]
                vals.append(float(np.mean([float(r["abs_error"]) for r in candidates])))
            ax2.bar(xloc + (j - 1) * width, vals, width=width, label=label, color=color)

        ax2.set_yscale("log")
        ax2.set_xticks(xloc)
        ax2.set_xticklabels(
            [r"$\mathrm{skew}(v)$", r"$\mathrm{kurt}(v)$", r"$E[\Delta X\Delta v]$", r"$E[(\Delta X)^2\Delta v]$"]
        )
        ax2.set_ylabel("Absolute error vs MC reference")
        ax2.set_title(f"Direct comparison ({reg}): non-Gaussian cloud vs Gaussian surrogates")
        ax2.grid(True, axis="y", which="both", alpha=0.3)
        ax2.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        fig2.tight_layout()
        png_sum = os.path.join(OUTPUT_DIR, f"non_gaussian_flow_eval_summary_{reg}.png")
        fig2.savefig(png_sum, dpi=160, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved plot -> {png_sum}")

    # Plot C: combined Pareto across both regimes.
    fig3, ax3 = plt.subplots(figsize=(10.8, 5.2))
    pareto_methods = [
        ("WSC-KF-m3-k1-cloud", "WSC m=3 k=1", "#6A4C93"),
        ("WSC-KF-m3-k2-cloud", "WSC m=3 k=2", "#9670C4"),
        ("WSC-KF-m3-k3-cloud", "WSC m=3 k=3", "#B8A1D9"),
        ("WSC-KF-m5-k1-cloud", "WSC m=5 k=1", "#9B5DE5"),
        ("WSC-KF-m5-k2-cloud", "WSC m=5 k=2", "#C29CF0"),
        ("WSC-KF-m5-k3-cloud", "WSC m=5 k=3", "#DEC9F7"),
        ("CD-EKF-gauss", "EKF x8", "#FF7F50"),
        ("CD-UKF-gauss", "UKF x8", "#F4A261"),
        ("CD-CKF-gauss", "CKF x1", "#2E86AB"),
        ("CD-CKF-gauss", "CKF x2", "#4EA5D9"),
        ("CD-CKF-gauss", "CKF x4", "#7CC4E4"),
        ("CD-CKF-gauss", "CKF x8", "#A8DADC"),
        ("CD-EnKF-forecast", "EnKF n=500", "#BC4749"),
        ("CD-EnKF-forecast", "EnKF n=1000", "#9B2226"),
        ("CD-EnKF-forecast", "EnKF n=2000", "#6A040F"),
        ("CD-EnKF-forecast", "EnKF n=5000", "#4A020A"),
        ("Bootstrap-PF-forecast", "PF n=500", "#90BE6D"),
        ("Bootstrap-PF-forecast", "PF n=1000", "#43AA8B"),
        ("Bootstrap-PF-forecast", "PF n=2000", "#4D908E"),
        ("Bootstrap-PF-forecast", "PF n=5000", "#277DA1"),
        ("KLV-PF-forecast", "KLV-PF n=500", "#F77F00"),
        ("KLV-PF-forecast", "KLV-PF n=1000", "#FCBF49"),
        ("KLV-PF-forecast", "KLV-PF n=2000", "#E09F3E"),
        ("KLV-PF-forecast", "KLV-PF n=5000", "#D62828"),
    ]
    regime_markers = {"mild": "o", "stress": "s"}
    def _method_rows(regime: str, method_key: str, refinement: int | None):
        subset = [
            r
            for r in rows
            if r["regime"] == regime and r["method"] == method_key
        ]
        if refinement is not None:
            subset = [r for r in subset if int(r["refinement"]) == refinement]
        return subset

    points: dict[tuple[str, str], tuple[float, float, float, float, str]] = {}
    for regime, marker in regime_markers.items():
        for method_key, label, color in pareto_methods:
            refinement = None
            if label.startswith("CKF x") or label.startswith("EKF x") or label.startswith("UKF x"):
                refinement = int(label.split("x", 1)[1])
            elif "n=" in label:
                refinement = int(label.split("n=", 1)[1])
            elif "k=" in label:
                refinement = int(label.split("k=", 1)[1])
            subset = _method_rows(regime, method_key, refinement)
            if not subset:
                continue
            seed_ids = sorted({int(r["seed"]) for r in subset})
            agg_by_seed: list[float] = []
            t_by_seed_ms: list[float] = []
            for sid in seed_ids:
                ss = [r for r in subset if int(r["seed"]) == sid]
                metrics_unique = sorted({str(r["metric"]) for r in ss})
                errs = np.array(
                    [
                        float(np.mean([float(r["abs_error"]) for r in ss if str(r["metric"]) == m]))
                        for m in metrics_unique
                    ],
                    dtype=np.float64,
                )
                scales = np.array(
                    [
                        max(
                            1e-15,
                            float(
                                np.mean(
                                    [
                                        abs(float(r["reference"]))
                                        for r in ss
                                        if str(r["metric"]) == m
                                    ]
                                )
                            ),
                        )
                        for m in metrics_unique
                    ],
                    dtype=np.float64,
                )
                rel = np.maximum(errs / scales, 1e-15)
                agg_by_seed.append(float(np.exp(np.mean(np.log(rel)))))
                t_by_seed_ms.append(1e3 * float(np.mean([float(r["time_s"]) for r in ss])))

            agg = float(np.mean(agg_by_seed))
            agg_std = float(np.std(agg_by_seed, ddof=1)) if len(agg_by_seed) > 1 else 0.0
            t_ms = float(np.mean(t_by_seed_ms))
            t_std = float(np.std(t_by_seed_ms, ddof=1)) if len(t_by_seed_ms) > 1 else 0.0
            ax3.scatter(
                t_ms,
                agg,
                s=80,
                marker=marker,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax3.errorbar(
                t_ms,
                agg,
                xerr=t_std,
                yerr=agg_std,
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                alpha=0.7,
                capsize=2,
                zorder=0,
            )
            points[(regime, label)] = (t_ms, agg, t_std, agg_std, color)

    # Connect trajectories requested by user:
    # WSC m=3 : k=1 -> k=2
    # WSC m=5 : k=1 -> k=2
    # EnKF and PF: n=500 -> 1000 -> 2000
    connect_groups = [
        ["WSC m=3 k=1", "WSC m=3 k=2", "WSC m=3 k=3"],
        ["WSC m=5 k=1", "WSC m=5 k=2", "WSC m=5 k=3"],
        ["EnKF n=500", "EnKF n=1000", "EnKF n=2000", "EnKF n=5000"],
        ["PF n=500", "PF n=1000", "PF n=2000", "PF n=5000"],
    ]
    for regime in regime_markers:
        for labels in connect_groups:
            seq = [points[(regime, lab)] for lab in labels if (regime, lab) in points]
            if len(seq) >= 2:
                xs = [p[0] for p in seq]
                ys = [p[1] for p in seq]
                ax3.plot(
                    xs,
                    ys,
                    "-",
                    color=seq[0][4],
                    lw=1.4,
                    alpha=0.8,
                    zorder=1,
                )

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("One-step forecast wall-clock [ms]")
    ax3.set_ylabel("Geometric mean relative error across non-Gaussian metrics")
    ax3.set_title("Heston non-Gaussian Pareto: mild vs stress regimes")
    ax3.grid(True, which="both", alpha=0.3)
    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=label,
            markersize=8,
        )
        for _, label, color in pareto_methods
    ]
    regime_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="0.25", linestyle="None", label="mild", markersize=8
        ),
        plt.Line2D(
            [0], [0], marker="s", color="0.25", linestyle="None", label="stress", markersize=8
        ),
    ]
    leg1 = ax3.legend(
        handles=method_handles,
        title="Filter / configuration",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )
    ax3.add_artist(leg1)
    ax3.legend(
        handles=regime_handles,
        title="Regime marker",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0.0,
        fontsize=9,
        title_fontsize=9,
    )
    fig3.tight_layout()
    png_pareto = os.path.join(OUTPUT_DIR, "non_gaussian_flow_eval_pareto.png")
    fig3.savefig(png_pareto, dpi=160, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved plot -> {png_pareto}")

    print(f"Saved data -> {csv_path}")

    # Refresh the 2x2 per-metric Pareto panel used in the paper figures.
    try:
        from plot_non_gaussian_metric_paretos import main as _plot_metric_paretos

        _plot_metric_paretos()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not refresh metric Pareto panel ({exc})")


if __name__ == "__main__":
    main()
