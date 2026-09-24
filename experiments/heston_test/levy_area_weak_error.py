"""Weak error of E[cos(ω x²)] vs Δ on the Lévy-area SDE.

Dirac initial law. Exact value [cosh(ωΔ)]^{-1/2}.

CD-CKF cubatures the state and evaluates L at the mean. From a Dirac
law every node is at 0, so x²≡0 and the error is |1−exact| ∼ O(Δ²),
the same as WSC m=3 (no Lévy area). That is not a one-step IT-1.5 of
the driving Brownian motion; m=5 cubatures I_{(j,k)} and is O(Δ³).

  .venv/bin/python experiments/heston_test/levy_area_weak_error.py
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

from cd_dynamax_mirror.filters import UKFParams, _ekf_predict, _ukf_predict, _ukf_weights
from levy_area_model import exact_char, ito_L, ito_drift, qc_fn
from tme import tme_mean_cov
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict

OUTPUT_DIR = os.path.join(HERE, "output", "levy_area")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OMEGA = 3.0
INNER = 128
X0 = jnp.zeros(2, dtype=jnp.float64)
P0 = jnp.zeros((2, 2), dtype=jnp.float64)
CKF = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)
STRAT_DRIFT, STRAT_L = ito_to_stratonovich_dynamics(ito_drift, ito_L)


def _ecos_cloud(samples, weights=None) -> float:
    x = np.asarray(samples, dtype=np.float64)
    if weights is None:
        w = np.full(x.shape[0], 1.0 / x.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)
    return float(np.sum(w * np.cos(OMEGA * x[:, 1])))


def _ecos_mean_cov(m, P) -> float:
    mu = float(np.asarray(m)[1])
    var = float(np.asarray(P)[1, 1])
    if var <= 0.0:
        return float(np.cos(OMEGA * mu))
    z, wz = hermgauss(24)
    wz = wz / np.sqrt(np.pi)
    x2 = mu + np.sqrt(2.0 * var) * z
    return float(np.sum(wz * np.cos(OMEGA * x2)))


def wsc_ecos(delta: float, m: int, k: int = 1) -> float:
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
    return _ecos_cloud(samples, wts)


def ckf_ecos(delta: float) -> float:
    n = 2
    lamb = CKF.alpha**2 * (n + CKF.kappa) - n
    w_mean, w_cov, W = _ukf_weights(n, CKF.alpha, CKF.beta, lamb)
    dt0 = jnp.asarray(delta / INNER, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(delta, dtype=jnp.float64)

    @jit
    def _run():
        return _ukf_predict(
            X0, P0, ito_drift, ito_L, qc_fn, t0, t1, dt0, n, lamb,
            w_mean, w_cov, W, "flowmap", solver="rk4", n_steps=INNER,
        )

    m_pred, P_pred = _run()
    jax.block_until_ready(m_pred)
    return _ecos_mean_cov(m_pred, P_pred)


def ekf_ecos(delta: float) -> float:
    dt0 = jnp.asarray(delta / INNER, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(delta, dtype=jnp.float64)

    @jit
    def _run():
        return _ekf_predict(
            X0, P0, ito_drift, ito_L, qc_fn, t0, t1, dt0,
            "flowmap", solver="rk4", n_steps=INNER,
        )

    m_pred, P_pred = _run()
    jax.block_until_ready(m_pred)
    return _ecos_mean_cov(m_pred, P_pred)


def tme_ecos(delta: float, order: int) -> float:
    m, P = tme_mean_cov(np.zeros(2), float(delta), int(order), ito_drift, ito_L)
    return _ecos_mean_cov(m, P)


def _fit_slope(deltas, errs, dmin, dmax, floor=1e-16):
    d = np.asarray(deltas, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    mask = (e > floor) & (d >= dmin) & (d <= dmax)
    if int(np.sum(mask)) < 2:
        return float("nan")
    return float(np.polyfit(np.log(d[mask]), np.log(e[mask]), 1)[0])


def main() -> None:
    p = argparse.ArgumentParser()
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
        f"Lévy-area weak error  ω={OMEGA:g}  inner={INNER}  "
        f"δ ∈ [{args.delta_min:g}, {args.delta_max:g}] ({len(deltas)} pts)"
    )

    # JIT warmup
    _ = wsc_ecos(float(deltas[0]), 3, 1)
    _ = wsc_ecos(float(deltas[0]), 5, 1)
    _ = ckf_ecos(float(deltas[0]))
    _ = ekf_ecos(float(deltas[0]))

    rows = []
    for d in deltas:
        true = exact_char(float(d), OMEGA)
        miss = abs(1.0 - true)
        ekf = abs(ekf_ecos(float(d)) - true)
        ckf = abs(ckf_ecos(float(d)) - true)
        tme2 = abs(tme_ecos(float(d), 2) - true)
        m3 = abs(wsc_ecos(float(d), 3, 1) - true)
        m5 = abs(wsc_ecos(float(d), 5, 1) - true)
        rows.append(
            {
                "delta": float(d),
                "omega": OMEGA,
                "true_char": true,
                "analytic_miss": miss,
                "ekf_err": ekf,
                "ckf_err": ckf,
                "tme2_err": tme2,
                "wsckf_err_m3": m3,
                "wsckf_err_m5": m5,
            }
        )
        print(
            f"  δ={d:.4g}  exact={true:.6f}  "
            f"CKF={ckf:.3e}  m3={m3:.3e}  m5={m5:.3e}  TME-2={tme2:.3e}"
        )

    dlt = np.array([r["delta"] for r in rows])
    print("\nFitted log-log slopes "
          f"(δ ∈ [{args.fit_dmin:g}, {args.fit_dmax:g}]):")
    for key, lab in (
        ("ckf_err", "CD-CKF"),
        ("ekf_err", "CD-EKF"),
        ("wsckf_err_m3", "WSC m=3"),
        ("wsckf_err_m5", "WSC m=5"),
        ("tme2_err", "TME-2"),
        ("analytic_miss", "|1 − exact|"),
    ):
        sl = _fit_slope(
            dlt, [r[key] for r in rows], args.fit_dmin, args.fit_dmax
        )
        print(f"  {lab:<14s}  {sl:.2f}")

    path = os.path.join(OUTPUT_DIR, "levy_area_weak_error.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
