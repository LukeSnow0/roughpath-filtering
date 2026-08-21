"""Heston benchmarks for WSC-KF claim (3): deterministic, parallel forecast ODEs.

Parts:
  A) Determinism: run-to-run metric/time variance (WSC/CD vs EnKF/PF)
  B) Parallel scaling: throughput vs batch/worker count
  C) Fixed wall-clock budget: best non-Gaussian error achievable
  D) Adjoint gradients: calibrate vol-of-vol xi via filter log-likelihood

Run:
  uv run --extra experiments python experiments/heston_test/parallel_determinism_benchmark.py
  uv run --extra experiments python experiments/heston_test/parallel_determinism_benchmark.py \\
      --parts A,B,C,D --seeds 100,101,102 --repeats 20 --workers 1,2,4,8

Outputs under ``experiments/heston_test/output/parallel_determinism/``.
"""

from __future__ import annotations

import argparse
import csv
import os
import resource
import subprocess
import sys
import time
from typing import Callable, NamedTuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, vmap, value_and_grad

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, ROOT_DIR)

from cd_dynamax_mirror.filters import run_bootstrap_pf, run_cd_ekf, run_cd_enkf
from cuthbert import filter as run_filter
from heston_model import HestonParams, init_estimate, make_heston_dynamics, simulate_heston
from non_gaussian_flow_eval import (
    _gaussian_moment_proxy,
    _predict_cd_ckf_pointmass,
    _predict_particle_forecast_pointmass,
    _prepare_wsckf_predictor,
    _simulate_transition_mc,
    _weighted_stats,
)
from wsckf import build_filter
from wsckf.library import ito_to_stratonovich_dynamics, lv_path_count, precompute_cubature_paths, predict

OUTPUT_DIR = os.path.join(HERE, "output", "parallel_determinism")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DELTA = 0.05
INTEGRATION_DT = DELTA / 256
N_INNER = 256
X0 = np.array([0.0, 0.04], dtype=np.float64)
STRESS = HestonParams(xi=1.0, rho=-0.9, meas_std=0.02, v0=float(X0[1]))
SPATIAL_NODES = 4  # degree-3 spherical cubature in d=2


def _heston_dynamics_for_xi(p_template: HestonParams, xi):
    """JAX-friendly Heston dynamics with differentiable vol-of-vol ``xi``."""
    mu, kappa, theta, rho = p_template.mu, p_template.kappa, p_template.theta, p_template.rho
    v_eps = p_template.v_eps
    diff_scale = p_template.diff_scale
    rho_bar = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    xi_j = jnp.asarray(xi, dtype=jnp.float64)

    def drift_fn(t, x):
        v = x[1]
        return jnp.array([mu - 0.5 * v, kappa * (theta - v)])

    def L_fn(t, x):
        sqrt_v = jnp.sqrt(jnp.maximum(x[1], v_eps))
        return diff_scale * jnp.array(
            [
                [sqrt_v, 0.0],
                [xi_j * rho * sqrt_v, xi_j * rho_bar * sqrt_v],
            ]
        )

    def Qc_fn(t, x):
        return jnp.eye(2)

    def obs_fn(x):
        return x[:1]

    R = jnp.array([[p_template.meas_std**2]], dtype=jnp.float64)
    return drift_fn, L_fn, Qc_fn, obs_fn, R


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
    out: list[float] = []
    seen: set[float] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        x = float(tok)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _agg_rel_error(metrics: dict[str, float], ref: dict[str, float]) -> float:
    rel = []
    for k, v in metrics.items():
        scale = max(1e-15, abs(ref[k]))
        rel.append(abs(v - ref[k]) / scale)
    return float(np.exp(np.mean(np.log(np.maximum(rel, 1e-15)))))


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class ForecastCfg(NamedTuple):
    label: str
    family: str
    kind: str  # wsc | ckf | enkf | pf
    lyons_degree: int = 5
    k: int = 1
    n_particles: int = 0


def _method_configs_a() -> list[ForecastCfg]:
    return [
        ForecastCfg("WSC m=5 k=1", "WSC-KF", "wsc", lyons_degree=5, k=1),
        ForecastCfg("WSC m=5 k=2", "WSC-KF", "wsc", lyons_degree=5, k=2),
        ForecastCfg("CD-CKF x8", "CD-CKF", "ckf"),
        ForecastCfg("EnKF n=2000", "EnKF", "enkf", n_particles=2000),
        ForecastCfg("PF n=2000", "PF", "pf", n_particles=2000),
    ]


def _forecast_metrics(cfg: ForecastCfg, p: HestonParams, *, seed: int) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
    if cfg.kind == "wsc":
        runner = _prepare_wsckf_predictor(
            p, X0, DELTA, lyons_degree=cfg.lyons_degree, k=cfg.k, integration_dt=INTEGRATION_DT
        )
        _, _, samples, weights = runner()
        jax.block_until_ready(samples)
        metrics = _weighted_stats(samples, x0=X0, w=weights)
    elif cfg.kind == "ckf":
        m, P = _predict_cd_ckf_pointmass(p, X0, DELTA, n_inner=N_INNER * 8)
        metrics = _gaussian_moment_proxy(m, P, X0, n_draws=50_000, seed=0)
    else:
        parts, weights = _predict_particle_forecast_pointmass(
            p, X0, DELTA, n_inner=512, n_particles=cfg.n_particles, seed=seed
        )
        jax.block_until_ready(parts)
        metrics = _weighted_stats(parts, x0=X0, w=weights)
    wall = time.perf_counter() - t0
    return metrics, wall


def _parallel_units(cfg: ForecastCfg) -> int:
    if cfg.kind == "wsc":
        return SPATIAL_NODES * lv_path_count(2, cfg.k, lyons_degree=cfg.lyons_degree)
    if cfg.kind in ("enkf", "pf"):
        return cfg.n_particles
    return 1


# ---------------------------------------------------------------------------
# Part A — determinism
# ---------------------------------------------------------------------------


def part_a_determinism(
    p: HestonParams, ref: dict[str, float], *, repeats: int, seed_base: int
) -> list[dict]:
    rows: list[dict] = []
    for cfg in _method_configs_a():
        metric_runs: list[dict[str, float]] = []
        agg_runs: list[float] = []
        times: list[float] = []
        for r in range(repeats):
            # Deterministic methods: seed ignored. Stochastic: fresh key each repeat.
            run_seed = seed_base if cfg.kind in ("wsc", "ckf") else seed_base + r
            metrics, wall = _forecast_metrics(cfg, p, seed=run_seed)
            metric_runs.append(metrics)
            agg_runs.append(_agg_rel_error(metrics, ref))
            times.append(wall)

        row_base = {
            "part": "A",
            "method": cfg.label,
            "family": cfg.family,
            "time_mean_ms": 1e3 * float(np.mean(times)),
            "time_std_ms": 1e3 * float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "agg_error_std": float(np.std(agg_runs, ddof=1)) if len(agg_runs) > 1 else 0.0,
        }
        for metric_name in ref:
            vals = [m[metric_name] for m in metric_runs]
            rows.append(
                {
                    **row_base,
                    "metric": metric_name,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
            )
        print(
            f"  A {cfg.label:16s} agg_err_std={row_base['agg_error_std']:.3e} "
            f"time_std={row_base['time_std_ms']:.2f} ms"
        )
    return rows


# ---------------------------------------------------------------------------
# Part B — parallel scaling (vmap batch)
# ---------------------------------------------------------------------------


def _build_batched_wsc(p: HestonParams, *, lyons_degree: int, k: int, batch: int):
    drift, L_fn, _, _, _ = make_heston_dynamics(p)
    strat_drift, strat_L = ito_to_stratonovich_dynamics(drift, L_fn)
    ts, path_values, path_weights = precompute_cubature_paths(
        horizon=float(DELTA),
        noise_dim=2,
        integration_dt=float(INTEGRATION_DT),
        cubature_order=int(k),
        multidim_method="lv",
        lyons_degree=int(lyons_degree),
    )

    @jit
    def _batch_predict(x0_batch):
        def one(x0):
            _, _, samples, weights = predict(
                x0,
                jnp.zeros((2, 2), dtype=jnp.float64),
                ts,
                path_values,
                path_weights,
                strat_drift,
                strat_L,
            )
            return jnp.sum(weights * samples[:, 1])

        return vmap(one)(x0_batch)

    x0_batch = jnp.tile(jnp.asarray(X0), (batch, 1))
    return _batch_predict, x0_batch


def _build_batched_enkf(p: HestonParams, *, n_particles: int, batch: int):
    drift, L_fn, Qc_fn, _, _ = make_heston_dynamics(p)
    from cd_dynamax_mirror.filters import _enkf_forecast

    particles0 = jnp.tile(jnp.asarray(X0)[None, :], (n_particles, 1))

    @jit
    def _batch(key_batch):
        def one(key):
            parts = _enkf_forecast(
                key,
                particles0,
                drift,
                L_fn,
                Qc_fn,
                0.0,
                float(DELTA),
                float(DELTA / 512),
                2,
                1.0,
                solver="rk4",
                n_steps=512,
                sde_form="ito",
            )
            return jnp.mean(parts[:, 1])

        return vmap(one)(key_batch)

    keys = jr.split(jr.key(0), batch)
    return _batch, keys


def _probe_single_forecast(method: str, *, workers: int, rounds: int = 5) -> float:
    """Time one forecast in a fresh subprocess with thread count capped at ``workers``."""
    env = os.environ.copy()
    w = str(workers)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = w
    cmd = [
        sys.executable,
        os.path.join(HERE, "parallel_forecast_probe.py"),
        "--method",
        method,
        "--rounds",
        str(rounds),
    ]
    out = subprocess.check_output(cmd, env=env, text=True).strip()
    return float(out.splitlines()[-1])


def part_b_parallel(p: HestonParams, *, workers: list[int], rounds: int = 5) -> list[dict]:
    rows: list[dict] = []
    specs = [
        ("WSC m=5 k=2", "wsc", SPATIAL_NODES * lv_path_count(2, 2, 5)),
        ("EnKF n=2000", "enkf", 2000),
        ("CD-CKF x8", "ckf", 1),
    ]
    for label, kind, units in specs:
        t1 = None
        for w in workers:
            med = _probe_single_forecast(kind, workers=w, rounds=rounds)
            if t1 is None:
                t1 = med
            speedup = t1 / med if med > 0 else float("nan")
            efficiency = speedup / w if w > 0 else float("nan")
            throughput = units / med
            rows.append(
                {
                    "part": "B",
                    "method": label,
                    "workers": w,
                    "time_s": med,
                    "throughput_units_per_s": throughput,
                    "speedup_vs_1": speedup,
                    "efficiency": efficiency,
                    "parallel_units_per_forecast": units,
                }
            )
            print(
                f"  B {label:16s} W={w:2d}  {1e3*med:7.1f} ms  "
                f"speedup={speedup:4.2f}  eff={efficiency:4.2f}"
            )
    return rows


# ---------------------------------------------------------------------------
# Part C — fixed wall-clock budget
# ---------------------------------------------------------------------------


def _time_forecast(cfg: ForecastCfg, p: HestonParams, *, seed: int, batch: int) -> float:
    if batch == 1:
        _, wall = _forecast_metrics(cfg, p, seed=seed)
        return wall
    if cfg.kind == "wsc":
        fn, args = _build_batched_wsc(p, lyons_degree=cfg.lyons_degree, k=cfg.k, batch=batch)
    else:
        fn, args = _build_batched_enkf(p, n_particles=cfg.n_particles, batch=batch)
    ts = []
    for r in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / batch


def _error_forecast(cfg: ForecastCfg, p: HestonParams, ref: dict[str, float], *, seed: int) -> float:
    metrics, _ = _forecast_metrics(cfg, p, seed=seed)
    return _agg_rel_error(metrics, ref)


def part_c_budget(
    p: HestonParams,
    ref: dict[str, float],
    *,
    budgets_ms: list[float],
    worker_counts: list[int],
    timing_seed: int,
) -> list[dict]:
    rows: list[dict] = []
    candidates: list[tuple[str, ForecastCfg, float, float, int]] = []

    for k in (1, 2, 3):
        cfg = ForecastCfg(f"WSC m=5 k={k}", "WSC-KF", "wsc", lyons_degree=5, k=k)
        for w in worker_counts:
            wall = _time_forecast(cfg, p, seed=timing_seed, batch=w)
            err = _error_forecast(cfg, p, ref, seed=timing_seed)
            candidates.append((cfg.label, cfg, wall, err, w))

    for n in (500, 1000, 2000, 5000):
        for kind, fam in (("enkf", "EnKF"), ("pf", "PF")):
            cfg = ForecastCfg(f"{fam} n={n}", fam, kind, n_particles=n)
            for w in worker_counts:
                wall = _time_forecast(cfg, p, seed=timing_seed + n, batch=w)
                err = _error_forecast(cfg, p, ref, seed=timing_seed + n)
                candidates.append((cfg.label, cfg, wall, err, w))

    for budget_ms in budgets_ms:
        budget_s = budget_ms / 1e3
        for w in worker_counts:
            feasible = [c for c in candidates if c[2] <= budget_s and c[4] == w]
            if not feasible:
                continue
            best = min(feasible, key=lambda c: c[3])
            rows.append(
                {
                    "part": "C",
                    "budget_ms": budget_ms,
                    "workers": w,
                    "method": best[0],
                    "time_ms": 1e3 * best[2],
                    "agg_rel_error": best[3],
                }
            )
            print(
                f"  C budget={budget_ms:5.0f} ms W={w} -> {best[0]:16s} "
                f"err={best[3]:.3e} ({1e3*best[2]:.1f} ms)"
            )
    return rows


# ---------------------------------------------------------------------------
# Part D — adjoint gradient calibration of xi
# ---------------------------------------------------------------------------


class XiInputs(NamedTuple):
    idx: jnp.ndarray
    xi: jnp.ndarray


def _simulate_heston_track(p: HestonParams, *, n_obs: int, seed: int):
    horizon = n_obs * DELTA
    n_fine = int(round(horizon / (DELTA / 4)))
    true_fine, ys_np, _ = simulate_heston(
        horizon=horizon,
        n_fine=n_fine,
        p=p,
        seed=seed,
        obs_stride=max(1, n_fine // n_obs),
    )
    return true_fine, jnp.asarray(ys_np)


def _build_xi_ll(
    p_template: HestonParams,
    ys: jnp.ndarray,
    *,
    method: str,
    checkpoint: bool,
    n_particles: int,
    key,
):
    m0_np, P0_np = init_estimate(p_template, true_state=X0, cov_scale=1.0)
    m0 = jnp.asarray(m0_np)
    P0 = jnp.asarray(P0_np)
    chol_P0 = jnp.linalg.cholesky(P0)
    _, _, Qc_base, h_obs, R = make_heston_dynamics(p_template)
    chol_R = jnp.linalg.cholesky(R)
    n_obs = ys.shape[0]
    model_inputs = jnp.arange(n_obs + 1)

    if method == "WSC":

        def get_init(_):
            return m0, chol_P0

        def get_dyn(inp: XiInputs):
            drift, L, _, _, _ = _heston_dynamics_for_xi(p_template, inp.xi)
            return drift, L

        def get_obs(inp: XiInputs):
            return h_obs, chol_R, ys[inp.idx - 1]

        filt = build_filter(
            get_init_params=get_init,
            get_dynamics=get_dyn,
            get_observations=get_obs,
            noise_dim=2,
            dt=DELTA,
            integration_dt=INTEGRATION_DT,
            cubature_order=1,
            lyons_degree=5,
            multidim_paths="lv",
            checkpoint_steps=checkpoint,
            sde_form="ito",
        )
        run = jit(run_filter, static_argnames=["filter_obj"])

        def ll(log_xi: jnp.ndarray) -> jnp.ndarray:
            xi = jnp.exp(log_xi)
            mi = XiInputs(idx=model_inputs, xi=jnp.full((n_obs + 1,), xi))
            return run(filt, mi).log_normalizing_constant[-1]

        return ll

    def dyn_for(xi_val):
        return _heston_dynamics_for_xi(p_template, xi_val)

    if method == "CD-EKF":

        def ll(log_xi: jnp.ndarray) -> jnp.ndarray:
            xi = jnp.exp(log_xi)
            drift, L, Qc, _, _ = dyn_for(xi)
            res = run_cd_ekf(
                drift, h_obs, L, Qc, R, m0, P0, ys, DELTA, INTEGRATION_DT,
                state_order="flowmap", solver="rk4",
            )
            return res.cum_loglik[-1]

        return ll

    if method == "CD-EnKF":

        def ll(log_xi: jnp.ndarray) -> jnp.ndarray:
            xi = jnp.exp(log_xi)
            drift, L, Qc, _, _ = dyn_for(xi)
            res = run_cd_enkf(
                drift, h_obs, L, Qc, R, m0, P0, ys, DELTA, key,
                n_particles=n_particles, noise_dim=2, dt0=INTEGRATION_DT, solver="rk4",
            )
            return res.cum_loglik[-1]

        return ll

    def ll(log_xi: jnp.ndarray) -> jnp.ndarray:
        xi = jnp.exp(log_xi)
        drift, L, Qc, _, _ = dyn_for(xi)
        res = run_bootstrap_pf(
            drift, h_obs, L, Qc, R, m0, P0, ys, DELTA, key,
            n_particles=n_particles, noise_dim=2, dt0=INTEGRATION_DT, solver="rk4",
        )
        return res.cum_loglik[-1]

    return ll


def _adam_fit(
    grad_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    log_xi0: float,
    *,
    steps: int = 60,
    lr: float = 0.02,
    log_xi_min: float = -1.5,
    log_xi_max: float = 0.5,
) -> tuple[float, int, list[float]]:
    alpha = jnp.asarray(log_xi0, dtype=jnp.float64)
    losses = []
    for t in range(steps):
        loss, g = grad_fn(alpha)
        loss_f, g_f = float(loss), float(g)
        losses.append(loss_f)
        if not np.isfinite(loss_f) or not np.isfinite(g_f):
            return float(np.exp(alpha)), t, losses
        alpha = jnp.clip(alpha + lr * g, log_xi_min, log_xi_max)
        if t > 8 and abs(losses[-1] - losses[-2]) < 1e-4:
            return float(np.exp(alpha)), t + 1, losses
    return float(np.exp(alpha)), steps, losses


def part_d_adjoint(
    p_true: HestonParams,
    *,
    n_obs: int,
    seed: int,
    xi_true: float,
    grad_repeats: int,
) -> list[dict]:
    rows: list[dict] = []
    _, ys = _simulate_heston_track(p_true, n_obs=n_obs, seed=seed)
    log_xi0 = float(np.log(0.85 * xi_true))
    a0 = jnp.asarray(log_xi0, dtype=jnp.float64)

    specs = [
        ("WSC-KF-adjoint", "WSC", True, 0, False),
        ("WSC-KF-FD", "WSC", False, 0, True),
        ("CD-EKF-AD", "CD-EKF", False, 0, False),
        ("EnKF-AD", "CD-EnKF", False, 2000, False),
        ("PF-AD", "PF", False, 2000, False),
    ]

    for label, method, ckpt, n_part, use_fd in specs:
        if use_fd:
            ll = _build_xi_ll(p_true, ys, method="WSC", checkpoint=False, n_particles=0, key=None)

            def grad_fn(a):
                eps = 1e-4
                lp = ll(a + eps)
                lm = ll(a - eps)
                return 0.5 * (lp + lm), (lp - lm) / (2 * eps)

            fit_grad_fn = grad_fn
        else:
            key = jr.key(seed + 9000) if method in ("CD-EnKF", "PF") else None
            ll = _build_xi_ll(
                p_true, ys, method=method, checkpoint=ckpt, n_particles=n_part, key=key,
            )
            fit_grad_fn = jit(value_and_grad(ll))

        jax.block_until_ready(fit_grad_fn(a0))
        peak_mb = _peak_rss_mb()

        grad_samples = []
        for r in range(grad_repeats):
            if method in ("CD-EnKF", "PF"):
                key_r = jr.key(seed + 9100 + r)
                ll_r = _build_xi_ll(
                    p_true, ys, method=method, checkpoint=False, n_particles=n_part, key=key_r,
                )
                _, g = jit(value_and_grad(ll_r))(a0)
            elif use_fd:
                _, g = grad_fn(a0)
            else:
                _, g = fit_grad_fn(a0)
            grad_samples.append(float(g))

        grad_finite = [g for g in grad_samples if np.isfinite(g)]
        grad_std = float(np.std(grad_finite, ddof=1)) if len(grad_finite) > 1 else 0.0
        grad_mean = float(np.mean(grad_finite)) if grad_finite else float("nan")

        fit_lr = 0.005 if label == "WSC-KF-adjoint" else 0.02
        xi_hat, n_iter, losses = _adam_fit(fit_grad_fn, log_xi0, lr=fit_lr)
        rows.append(
            {
                "part": "D",
                "method": label,
                "family": method,
                "xi_true": xi_true,
                "xi_hat": xi_hat,
                "xi_abs_err": abs(xi_hat - xi_true),
                "grad_mean": grad_mean,
                "grad_std": grad_std,
                "n_iter": n_iter,
                "peak_rss_mb": peak_mb,
                "final_nll": losses[-1] if losses else float("nan"),
            }
        )
        print(
            f"  D {label:16s} xi_hat={xi_hat:.3f} err={abs(xi_hat-xi_true):.3f} "
            f"grad_std={grad_std:.2e} rss={peak_mb:.1f} MB"
        )
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_all(a_rows, b_rows, c_rows, d_rows):
    if a_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        methods = sorted({r["method"] for r in a_rows})
        vals = []
        for m in methods:
            sub = [r for r in a_rows if r["method"] == m]
            vals.append(float(sub[0]["agg_error_std"]))
        ax.bar(methods, vals, color="#6A4C93")
        ax.set_yscale("log")
        ax.set_ylabel("Run-to-run std (aggregate rel. error)")
        ax.set_title("A: Determinism — forecast variability across repeats")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "determinism_variance.png"), dpi=150)
        plt.close(fig)

    if b_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in sorted({r["method"] for r in b_rows}):
            sub = sorted([r for r in b_rows if r["method"] == method], key=lambda r: r["workers"])
            ax.plot(
                [r["workers"] for r in sub],
                [r["speedup_vs_1"] for r in sub],
                "o-",
                label=method,
            )
        ax.set_xlabel("Parallel batch size (independent forecasts)")
        ax.set_ylabel("Speedup vs batch=1")
        ax.set_title("B: Parallel forecast scaling (vmap batch)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "parallel_speedup.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for method in sorted({r["method"] for r in b_rows}):
            sub = sorted([r for r in b_rows if r["method"] == method], key=lambda r: r["workers"])
            ax.loglog(
                [r["workers"] for r in sub],
                [r["throughput_units_per_s"] for r in sub],
                "o-",
                label=method,
            )
        ax.set_xlabel("Parallel batch size")
        ax.set_ylabel("Independent ODE units / s")
        ax.set_title("B: Forecast throughput vs parallelism")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "parallel_throughput.png"), dpi=150)
        plt.close(fig)

    if c_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for w, ls in ((1, "-"), (8, "--")):
            sub = [r for r in c_rows if r["workers"] == w]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: r["budget_ms"])
            ax.loglog(
                [r["budget_ms"] for r in sub],
                [r["agg_rel_error"] for r in sub],
                f"o{ls}",
                label=f"W={w}",
            )
            for r in sub:
                ax.annotate(
                    r["method"].split()[0],
                    (r["budget_ms"], r["agg_rel_error"]),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        ax.set_xlabel("Wall-clock budget [ms]")
        ax.set_ylabel("Best aggregate rel. error")
        ax.set_title("C: Accuracy at fixed compute budget")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "budget_tradeoff.png"), dpi=150)
        plt.close(fig)

    if d_rows:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        methods = [r["method"] for r in d_rows]
        x = np.arange(len(methods))
        ax1.bar(x - 0.2, [r["grad_std"] for r in d_rows], width=0.4, label="grad std", color="#6A4C93")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.2, [r["xi_abs_err"] for r in d_rows], width=0.4, label="|xi_hat-xi|", color="#F4A261")
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=15, ha="right")
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax1.set_ylabel("Gradient std")
        ax2.set_ylabel("|xi_hat - xi_true|")
        ax1.set_title("D: Adjoint calibration of xi")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "adjoint_calibration.png"), dpi=150)
        plt.close(fig)


def _read_csv_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _coerce_csv_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    int_keys = {"workers", "repeats", "n_iter", "parallel_units_per_forecast"}
    float_keys = {
        "mean", "std", "time_mean_ms", "time_std_ms", "agg_error_std",
        "time_s", "speedup_vs_1", "efficiency", "throughput_units_per_s",
        "budget_ms", "time_ms", "agg_rel_error", "xi_true", "xi_hat", "xi_abs_err",
        "grad_mean", "grad_std", "peak_rss_mb", "final_nll",
    }
    for row in rows:
        rec = dict(row)
        for k in int_keys:
            if k in rec and rec[k] not in ("", None):
                rec[k] = int(float(rec[k]))
        for k in float_keys:
            if k in rec and rec[k] not in ("", None):
                rec[k] = float(rec[k])
        out.append(rec)
    return out


def _plot_from_csv() -> None:
    a_rows = _coerce_csv_rows(_read_csv_rows(os.path.join(OUTPUT_DIR, "determinism_variance.csv")))
    b_rows = _coerce_csv_rows(_read_csv_rows(os.path.join(OUTPUT_DIR, "parallel_speedup.csv")))
    c_rows = _coerce_csv_rows(_read_csv_rows(os.path.join(OUTPUT_DIR, "budget_tradeoff.csv")))
    d_rows = _coerce_csv_rows(_read_csv_rows(os.path.join(OUTPUT_DIR, "adjoint_calibration.csv")))
    _plot_all(a_rows, b_rows, c_rows, d_rows)
    print(f"Replotted from CSVs in {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="WSC-KF parallel/determinism benchmark (A-D)")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate PNGs from existing CSVs in the output subfolder",
    )
    parser.add_argument("--parts", type=str, default="A,B,C,D", help="Comma-separated parts to run")
    parser.add_argument("--seeds", type=str, default="100,101,102")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--workers", type=str, default="1,2,4,8")
    parser.add_argument("--budget-workers", type=str, default="1,8", dest="budget_workers")
    parser.add_argument("--budgets-ms", type=str, default="50,100,200,500", dest="budgets_ms")
    parser.add_argument("--mc-paths", type=int, default=300_000)
    parser.add_argument("--n-obs", type=int, default=8, dest="n_obs")
    parser.add_argument("--grad-repeats", type=int, default=5)
    args = parser.parse_args()

    if args.plot_only:
        _plot_from_csv()
        return

    parts = {p.strip().upper() for p in args.parts.split(",") if p.strip()}
    seeds = _parse_ints(args.seeds)
    workers = _parse_ints(args.workers)
    budget_workers = _parse_ints(args.budget_workers)
    budgets_ms = _parse_floats(args.budgets_ms)

    p = STRESS
    xi_true = p.xi
    a_rows: list[dict] = []
    b_rows: list[dict] = []
    c_rows: list[dict] = []
    d_rows: list[dict] = []

    ref = None
    if parts & {"A", "C"}:
        print(f"Building MC reference ({args.mc_paths} paths)...")
        ref = _weighted_stats(
            _simulate_transition_mc(
                x0=X0, p=p, delta=DELTA, n_steps=1024, n_paths=args.mc_paths, seed=seeds[0]
            ),
            x0=X0,
            w=np.full(args.mc_paths, 1.0 / args.mc_paths),
        )

    if "A" in parts:
        print(f"\n=== Part A: determinism ({args.repeats} repeats) ===")
        a_rows = part_a_determinism(p, ref, repeats=args.repeats, seed_base=seeds[0])
        _write_csv(os.path.join(OUTPUT_DIR, "determinism_variance.csv"), a_rows)

    if "B" in parts:
        print("\n=== Part B: parallel scaling ===")
        b_rows = part_b_parallel(p, workers=workers)
        _write_csv(os.path.join(OUTPUT_DIR, "parallel_speedup.csv"), b_rows)

    if "C" in parts:
        print("\n=== Part C: fixed wall-clock budget ===")
        c_rows = part_c_budget(
            p, ref, budgets_ms=budgets_ms, worker_counts=budget_workers, timing_seed=seeds[0]
        )
        _write_csv(os.path.join(OUTPUT_DIR, "budget_tradeoff.csv"), c_rows)

    if "D" in parts:
        print("\n=== Part D: adjoint xi calibration ===")
        d_rows = part_d_adjoint(
            p, n_obs=args.n_obs, seed=seeds[0], xi_true=xi_true, grad_repeats=args.grad_repeats
        )
        _write_csv(os.path.join(OUTPUT_DIR, "adjoint_calibration.csv"), d_rows)

    _plot_all(a_rows, b_rows, c_rows, d_rows)
    print(f"\nOutputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
