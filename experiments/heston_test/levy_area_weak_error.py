"""One-step weak error of E[cos(ω x²)] and E[(x²)²] vs Δ on an area SDE.

Dirac initial law at the model's x0 (``area_models.py``). Default model
``levy_area``: dx² = x¹ dW² from 0, exact E[cos] = [cosh(ωΔ)]^{-1/2}.

CD-CKF cubatures the state and evaluates L at the mean. From a Dirac
law every node is at 0, so x²≡0 and the error is |1−exact| ∼ O(Δ²),
the same as WSC m=3 (no Lévy area). That is not a one-step IT-1.5 of
the driving Brownian motion; m=5 cubatures I_{(j,k)} and is O(Δ³).
With ``--model sin_area`` (dx² = sin(x¹) dW² from (1, 0)) every level of
the expected signature contributes, so the rates O(Δ²) and O(Δ³) are sharp.

  .venv/bin/python experiments/heston_test/levy_area_weak_error.py
  .venv/bin/python experiments/heston_test/levy_area_weak_error.py --model sin_area
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from numpy.polynomial.hermite import hermgauss

jax.config.update("jax_enable_x64", True)

from area_models import MODELS, model_from_cli
from cd_dynamax_mirror.filters import UKFParams, _ekf_predict, _ukf_predict, _ukf_weights
from tme import tme_mean_cov
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict

MODEL = model_from_cli()
OUTPUT_DIR = os.path.join(HERE, "output", MODEL.name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OMEGA = 3.0
INNER = 128
X0 = jnp.asarray(MODEL.x0, dtype=jnp.float64)
P0 = jnp.zeros((2, 2), dtype=jnp.float64)
CKF = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)
STRAT_DRIFT, STRAT_L = ito_to_stratonovich_dynamics(MODEL.ito_drift, MODEL.ito_L)


def _cloud_stats(samples, weights=None) -> tuple[float, float]:
    """``(E[cos(ω x²)], E[(x²)²])`` of a weighted point cloud."""
    x = np.asarray(samples, dtype=np.float64)
    if weights is None:
        w = np.full(x.shape[0], 1.0 / x.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)
    return float(np.sum(w * np.cos(OMEGA * x[:, 1]))), float(np.sum(w * x[:, 1] ** 2))


def _gaussian_stats(m, P) -> tuple[float, float]:
    """``(E[cos(ω x²)], E[(x²)²])`` of a Gaussian with mean ``m`` and covariance ``P``."""
    mu = float(np.asarray(m)[1])
    var = float(np.asarray(P)[1, 1])
    second = var + mu**2
    if var <= 0.0:
        return float(np.cos(OMEGA * mu)), second
    z, wz = hermgauss(24)
    wz = wz / np.sqrt(np.pi)
    x2 = mu + np.sqrt(2.0 * var) * z
    return float(np.sum(wz * np.cos(OMEGA * x2))), second


def wsc_stats(delta: float, m: int, k: int = 1) -> tuple[float, float]:
    ts, paths, weights = precompute_cubature_paths(
        horizon=float(delta),
        noise_dim=2,
        integration_dt=float(delta / INNER),
        cubature_order=int(k),
        multidim_method="lv",
        lyons_degree=int(m),
    )

    @jit
    def _run():
        return predict(X0, P0, ts, paths, weights, STRAT_DRIFT, STRAT_L)

    _mean, _chol, samples, wts = _run()
    jax.block_until_ready(samples)
    return _cloud_stats(samples, wts)


def ckf_stats(delta: float) -> tuple[float, float]:
    n = 2
    lamb = CKF.alpha**2 * (n + CKF.kappa) - n
    w_mean, w_cov, W = _ukf_weights(n, CKF.alpha, CKF.beta, lamb)
    dt0 = jnp.asarray(delta / INNER, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(delta, dtype=jnp.float64)

    @jit
    def _run():
        return _ukf_predict(
            X0, P0, MODEL.ito_drift, MODEL.ito_L, MODEL.qc_fn, t0, t1, dt0, n, lamb,
            w_mean, w_cov, W, "flowmap", solver="rk4", n_steps=INNER,
        )

    m_pred, P_pred = _run()
    jax.block_until_ready(m_pred)
    return _gaussian_stats(m_pred, P_pred)


def ekf_stats(delta: float) -> tuple[float, float]:
    dt0 = jnp.asarray(delta / INNER, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(delta, dtype=jnp.float64)

    @jit
    def _run():
        return _ekf_predict(
            X0, P0, MODEL.ito_drift, MODEL.ito_L, MODEL.qc_fn, t0, t1, dt0,
            "flowmap", solver="rk4", n_steps=INNER,
        )

    m_pred, P_pred = _run()
    jax.block_until_ready(m_pred)
    return _gaussian_stats(m_pred, P_pred)


def tme_stats(delta: float, order: int) -> tuple[float, float]:
    m, P = tme_mean_cov(
        np.asarray(MODEL.x0), float(delta), int(order), MODEL.ito_drift, MODEL.ito_L
    )
    return _gaussian_stats(m, P)


def _fit_slope(deltas, errs, dmin, dmax, floor=1e-16):
    d = np.asarray(deltas, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    mask = (e > floor) & (d >= dmin) & (d <= dmax)
    if int(np.sum(mask)) < 2:
        return float("nan")
    return float(np.polyfit(np.log(d[mask]), np.log(e[mask]), 1)[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), default=MODEL.name)
    p.add_argument("--delta-min", type=float, default=0.10)
    p.add_argument("--delta-max", type=float, default=0.80)
    p.add_argument("--n-deltas", type=int, default=12)
    p.add_argument("--fit-dmin", type=float, default=0.10)
    p.add_argument("--fit-dmax", type=float, default=0.50)
    args = p.parse_args()

    deltas = np.logspace(
        np.log10(args.delta_min), np.log10(args.delta_max), int(args.n_deltas)
    )
    print(
        f"{MODEL.name} weak error  x0={MODEL.x0}  ω={OMEGA:g}  inner={INNER}  "
        f"δ ∈ [{args.delta_min:g}, {args.delta_max:g}] ({len(deltas)} pts)"
    )

    # JIT warmup
    _ = wsc_stats(float(deltas[0]), 3, 1)
    _ = wsc_stats(float(deltas[0]), 5, 1)
    _ = ckf_stats(float(deltas[0]))
    _ = ekf_stats(float(deltas[0]))

    rows = []
    for d in deltas:
        true = MODEL.exact_char(float(d), OMEGA)
        true2 = MODEL.exact_second_moment(float(d))
        stats = {
            "ekf": ekf_stats(float(d)),
            "ckf": ckf_stats(float(d)),
            "tme2": tme_stats(float(d), 2),
            "wsckf_m3": wsc_stats(float(d), 3, 1),
            "wsckf_m5": wsc_stats(float(d), 5, 1),
        }
        err = {key: abs(v[0] - true) for key, v in stats.items()}
        err2 = {key: abs(v[1] - true2) for key, v in stats.items()}
        rows.append(
            {
                "delta": float(d),
                "omega": OMEGA,
                "true_char": true,
                "analytic_miss": abs(1.0 - true),
                "ekf_err": err["ekf"],
                "ckf_err": err["ckf"],
                "tme2_err": err["tme2"],
                "wsckf_err_m3": err["wsckf_m3"],
                "wsckf_err_m5": err["wsckf_m5"],
                "true_second": true2,
                "ekf_err2": err2["ekf"],
                "ckf_err2": err2["ckf"],
                "tme2_err2": err2["tme2"],
                "wsckf_err2_m3": err2["wsckf_m3"],
                "wsckf_err2_m5": err2["wsckf_m5"],
            }
        )
        print(
            f"  δ={d:.4g}  exact={true:.6f}  "
            f"CKF={err['ckf']:.3e}  m3={err['wsckf_m3']:.3e}  m5={err['wsckf_m5']:.3e}  "
            f"TME-2={err['tme2']:.3e}  |  E[(x²)²] m3={err2['wsckf_m3']:.3e}  "
            f"m5={err2['wsckf_m5']:.3e}"
        )

    dlt = np.array([r["delta"] for r in rows])
    print("\nFitted log-log slopes "
          f"(δ ∈ [{args.fit_dmin:g}, {args.fit_dmax:g}]), E[cos] / E[(x²)²]:")
    for key, lab in (
        ("ckf", "CD-CKF"),
        ("ekf", "CD-EKF"),
        ("wsckf_m3", "WSC m=3"),
        ("wsckf_m5", "WSC m=5"),
        ("tme2", "TME-2"),
    ):
        k1 = f"{key}_err" if not key.startswith("wsckf") else f"wsckf_err_{key[-2:]}"
        k2 = f"{key}_err2" if not key.startswith("wsckf") else f"wsckf_err2_{key[-2:]}"
        s1 = _fit_slope(dlt, [r[k1] for r in rows], args.fit_dmin, args.fit_dmax)
        s2 = _fit_slope(dlt, [r[k2] for r in rows], args.fit_dmin, args.fit_dmax)
        print(f"  {lab:<14s}  {s1:.2f} / {s2:.2f}")

    path = os.path.join(OUTPUT_DIR, f"{MODEL.name}_weak_error.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
