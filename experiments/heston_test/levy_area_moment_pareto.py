"""One-step Lévy-area second-moment Pareto at the filter's Δ=0.5.

Score |Ê[(x²)²] − δ²/2| from a point mass. Degree-5 LV cubature and TME-2
are exact for this quadratic; EnKF is Monte Carlo; EKF/SE-UKF miss the area.

  .venv/bin/python experiments/heston_test/levy_area_moment_pareto.py
"""

from __future__ import annotations

import csv
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax import random as jr

jax.config.update("jax_enable_x64", True)

from cd_dynamax_mirror.filters import (
    UKFParams,
    _ekf_predict,
    _enkf_forecast,
    _ukf_predict,
    _ukf_sigmas,
    _ukf_weights,
)
from levy_area_model import exact_second_moment, ito_L, ito_drift, qc_fn
from tme import tme_mean_cov
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict

OUTPUT_DIR = os.path.join(HERE, "output", "levy_area")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DELTA = 0.5
INNER = 64
TRUE_SECOND = exact_second_moment(DELTA)
X0 = jnp.zeros(2, dtype=jnp.float64)
P0 = jnp.zeros((2, 2), dtype=jnp.float64)
CKF = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)
STRAT_DRIFT, STRAT_L = ito_to_stratonovich_dynamics(ito_drift, ito_L)


def _second_cloud(samples, weights=None) -> float:
    x = np.asarray(samples, dtype=np.float64)
    if weights is None:
        w = np.full(x.shape[0], 1.0 / x.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)
    return float(np.sum(w * x[:, 1] ** 2))


def _second_mean_cov(m, P) -> float:
    m = np.asarray(m, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    return float(P[1, 1] + m[1] ** 2)


def _time_call(fn, repeats: int = 11):
    out = fn()
    jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def wsc_predict(m: int, k: int):
    ts, paths, weights = precompute_cubature_paths(
        horizon=float(DELTA),
        noise_dim=2,
        integration_dt=float(DELTA / INNER),
        cubature_order=int(k),
        multidim_method="lv",
        lyons_degree=int(m),
    )

    @jit
    def _run():
        return predict(X0, jnp.zeros((2, 2)), ts, paths, weights, STRAT_DRIFT, STRAT_L)

    (_m, _chol, samples, wts), wall = _time_call(_run)
    return _second_cloud(samples, wts), wall, int(paths.shape[0])


def ekf_predict():
    dt0 = jnp.asarray(DELTA / INNER, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(DELTA, dtype=jnp.float64)

    @jit
    def _run():
        return _ekf_predict(
            X0, P0, ito_drift, ito_L, qc_fn, t0, t1, dt0,
            "flowmap", solver="rk4", n_steps=INNER,
        )

    (m_pred, P_pred), wall = _time_call(_run)
    return _second_mean_cov(m_pred, P_pred), wall


def ukf_predict(params: UKFParams):
    n = 2
    lamb = params.alpha**2 * (n + params.kappa) - n
    w_mean, w_cov, W = _ukf_weights(n, params.alpha, params.beta, lamb)
    dt0 = jnp.asarray(DELTA / INNER, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(DELTA, dtype=jnp.float64)

    @jit
    def _run():
        return _ukf_predict(
            X0, P0, ito_drift, ito_L, qc_fn, t0, t1, dt0, n, lamb,
            w_mean, w_cov, W, "flowmap", solver="rk4", n_steps=INNER,
        )

    (m_pred, P_pred), wall = _time_call(_run)
    return _second_mean_cov(m_pred, P_pred), wall


def tme_predict(order: int):
    def fn():
        return tme_mean_cov(np.zeros(2), DELTA, int(order), ito_drift, ito_L)

    (m, P), wall = _time_call(lambda: jax.block_until_ready(fn()))
    return _second_mean_cov(m, P), wall


def seukf_predict(n_modes: int):
    from predictive_nll_stepwise import _kl_wdot

    n_x, noise_dim = 2, 2
    n_z = n_modes * noise_dim
    n_aug = n_x + n_z
    lamb = CKF.alpha**2 * (n_aug + CKF.kappa) - n_aug
    w_mean, _w_cov, _ = _ukf_weights(n_aug, CKF.alpha, CKF.beta, lamb)
    horizon = jnp.asarray(DELTA, dtype=jnp.float64)
    n_inner = INNER

    def ode_rhs(t, x, Z):
        wdot = _kl_wdot(t, Z, horizon)
        return STRAT_DRIFT(t, x) + STRAT_L(t, x) @ wdot

    def integrate_one(x0, Z):
        def body(x, i):
            ti = (i.astype(x.dtype)) * (horizon / n_inner)
            t1 = ti + horizon / n_inner
            dtp = horizon / n_inner
            t_mid = ti + 0.5 * dtp
            k1 = ode_rhs(ti, x, Z)
            k2 = ode_rhs(t_mid, x + 0.5 * dtp * k1, Z)
            k3 = ode_rhs(t_mid, x + 0.5 * dtp * k2, Z)
            k4 = ode_rhs(t1, x + dtp * k3, Z)
            return x + (dtp / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

        x1, _ = jax.lax.scan(body, x0, jnp.arange(n_inner))
        return x1

    @jit
    def _run():
        m_aug = jnp.zeros(n_aug, dtype=jnp.float64)
        P_aug = jnp.eye(n_aug, dtype=jnp.float64).at[:n_x, :n_x].set(0.0)
        sigmas = _ukf_sigmas(m_aug, P_aug, n_aug, lamb)
        X0s = sigmas[:, :n_x]
        Z = sigmas[:, n_x:].reshape(sigmas.shape[0], n_modes, noise_dim)
        Xf = vmap(integrate_one)(X0s, Z)
        return Xf, w_mean

    (Xf, w), wall = _time_call(_run)
    return _second_cloud(Xf, w), wall


def make_enkf(n_particles: int):
    parts0 = jnp.repeat(X0[None, :], n_particles, axis=0)
    dt0 = jnp.asarray(DELTA / INNER, dtype=jnp.float64)

    @jit
    def _run(key):
        return _enkf_forecast(
            key, parts0, ito_drift, ito_L, qc_fn, 0.0, float(DELTA),
            float(dt0), 2, 1.0, solver="rk4", n_steps=INNER, sde_form="ito",
        )

    return _run


def _row(*, family, label, budget, second, wall, err_std=0.0, n_seeds=1):
    err = abs(second - TRUE_SECOND)
    return {
        "delta": DELTA,
        "family": family,
        "label": label,
        "budget": budget,
        "second": second,
        "abs_err": err,
        "abs_err_std": err_std,
        "time_s": wall,
        "n_seeds": n_seeds,
        "true_second": TRUE_SECOND,
    }


def main() -> None:
    print(
        f"Lévy-area second-moment Pareto  δ={DELTA:g}  "
        f"true E[(x²)²]={TRUE_SECOND:.8f}  inner={INNER}\n"
    )
    rows = []

    for order in (2, 3):
        second, wall = tme_predict(order)
        rows.append(_row(family="TME", label=f"TME-{order}", budget=order, second=second, wall=wall))
        print(f"  TME-{order:d}   E2={second:.16g}  err={abs(second-TRUE_SECOND):.3e}  t={1e3*wall:7.3f}ms")

    second, wall = ekf_predict()
    rows.append(_row(family="CD-EKF", label="EKF", budget=0, second=second, wall=wall))
    print(f"  EKF     E2={second:.16g}  err={abs(second-TRUE_SECOND):.3e}  t={1e3*wall:7.3f}ms")

    second, wall = ukf_predict(CKF)
    rows.append(_row(family="CD-CKF", label="CKF", budget=0, second=second, wall=wall))
    print(f"  CKF     E2={second:.16g}  err={abs(second-TRUE_SECOND):.3e}  t={1e3*wall:7.3f}ms")

    second, wall = seukf_predict(8)
    rows.append(_row(family="SE-UKF", label="N8", budget=8, second=second, wall=wall))
    print(f"  SE-UKF N=8  E2={second:.16g}  err={abs(second-TRUE_SECOND):.3e}  t={1e3*wall:7.3f}ms")

    for m, k in ((3, 1), (3, 2), (3, 3), (5, 1), (5, 2)):
        second, wall, n_paths = wsc_predict(m, k)
        rows.append(
            _row(family=f"WSC-KF-m{m}", label=f"m{m}-k{k}", budget=n_paths, second=second, wall=wall)
        )
        print(
            f"  WSC m{m}-k{k}  paths={n_paths:4d}  E2={second:.16g}  "
            f"err={abs(second-TRUE_SECOND):.3e}  t={1e3*wall:7.3f}ms"
        )

    for n, n_seed in ((25, 80), (50, 80), (100, 80), (500, 48), (2000, 32)):
        run = make_enkf(n)
        warmup = run(jr.key(0))
        jax.block_until_ready(warmup)
        times = []
        for _ in range(9):
            t0 = time.perf_counter()
            jax.block_until_ready(run(jr.key(1)))
            times.append(time.perf_counter() - t0)
        wall = float(np.median(times))
        seconds = []
        for s in range(n_seed):
            parts = run(jr.key(10_000 + 17 * n + s))
            jax.block_until_ready(parts)
            seconds.append(_second_cloud(parts))
        seconds = np.asarray(seconds, dtype=np.float64)
        abs_errs = np.abs(seconds - TRUE_SECOND)
        second = float(np.mean(seconds))
        mae = float(np.mean(abs_errs))
        mae_std = float(np.std(abs_errs, ddof=1))
        rows.append(
            _row(
                family="CD-EnKF",
                label=f"EnKF-{n}",
                budget=n,
                second=second,
                wall=wall,
                err_std=mae_std,
                n_seeds=n_seed,
            )
        )
        rows[-1]["abs_err"] = mae
        print(
            f"  EnKF n={n:<4d}  E2={second:.6g}±{np.std(seconds, ddof=1):.3g}  "
            f"MAE={mae:.3e}±{mae_std / np.sqrt(n_seed):.2e}  "
            f"t={1e3 * wall:7.3f}ms  seeds={n_seed}"
        )

    path = os.path.join(OUTPUT_DIR, "levy_area_moment_pareto.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
