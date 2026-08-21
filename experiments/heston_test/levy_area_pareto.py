"""Lévy-area match vs wall-clock Pareto.

Isolating Itô SDE (Brownian dim 2), point-mass start at 0:

    dx¹ = dW¹
    dx² = x¹ dW²

Exact: E[x²]=0, E[(x²)²] = δ²/2  (needs Lévy area / joint rough path).

Y-axis: |Ê[(x²)²] − δ²/2|
X-axis: one-step forecast wall-clock [ms]

Run:
  JAX_PLATFORMS=cpu uv run --extra experiments python \\
    experiments/heston_test/levy_area_pareto.py
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax import jit

jax.config.update("jax_enable_x64", True)

from cd_dynamax_mirror.filters import (
    UKFParams,
    _ekf_predict,
    _enkf_forecast,
    _ukf_predict,
    _ukf_weights,
)
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict

OUTPUT_DIR = os.path.join(HERE, "output", "levy_area")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DELTA = 0.25
INNER = 64
X0 = jnp.zeros(2, dtype=jnp.float64)
P0 = jnp.zeros((2, 2), dtype=jnp.float64)
TRUE_SECOND = 0.5 * DELTA**2
CKF = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)


def ito_drift(t, x):
    del t
    return jnp.zeros_like(x)


def diffusion(t, x):
    del t
    return jnp.array([[1.0, 0.0], [0.0, x[0]]], dtype=x.dtype)


STRAT_DRIFT, STRAT_L = ito_to_stratonovich_dynamics(ito_drift, diffusion)
QC = lambda t, x: jnp.eye(2)


def _second_moment_from_mean_cov(m, P):
    m = np.asarray(m, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    return float(P[1, 1] + m[1] ** 2)


def _second_moment_from_cloud(samples, weights=None):
    x = np.asarray(samples, dtype=np.float64)
    if weights is None:
        weights = np.full(x.shape[0], 1.0 / x.shape[0])
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    return float(np.sum(w * x[:, 1] ** 2))


def _time_call(fn, repeats: int = 9):
    out = fn()
    jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def wsc_predict(method: str, m: int, k: int):
    ts, paths, weights = precompute_cubature_paths(
        horizon=float(DELTA),
        noise_dim=2,
        integration_dt=float(DELTA / INNER),
        cubature_order=int(k),
        multidim_method=method,
        lyons_degree=int(m),
    )

    @jit
    def _run():
        return predict(X0, jnp.linalg.cholesky(P0 + 1e-30 * jnp.eye(2)), ts, paths, weights, STRAT_DRIFT, STRAT_L)

    # P0=0 → use zero chol
    @jit
    def _run0():
        return predict(X0, jnp.zeros((2, 2)), ts, paths, weights, STRAT_DRIFT, STRAT_L)

    (m_pred, chol, samples, wts), wall = _time_call(_run0)
    P = np.asarray(chol @ chol.T)
    second = _second_moment_from_cloud(samples, wts)
    return second, wall, int(paths.shape[0])


def ekf_predict(n_inner: int):
    dt0 = jnp.asarray(DELTA / n_inner, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(DELTA, dtype=jnp.float64)

    @jit
    def _run():
        return _ekf_predict(
            X0, P0, ito_drift, diffusion, QC, t0, t1,
            dt0, "flowmap", solver="rk4", n_steps=n_inner,
        )

    (m_pred, P_pred), wall = _time_call(_run)
    return _second_moment_from_mean_cov(m_pred, P_pred), wall


def ukf_predict(n_inner: int, params: UKFParams):
    n = 2
    lamb = params.alpha**2 * (n + params.kappa) - n
    w_mean, w_cov, W = _ukf_weights(n, params.alpha, params.beta, lamb)
    dt0 = jnp.asarray(DELTA / n_inner, dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    t1 = jnp.asarray(DELTA, dtype=jnp.float64)

    @jit
    def _run():
        return _ukf_predict(
            X0, P0, ito_drift, diffusion, QC, t0, t1,
            dt0, n, lamb, w_mean, w_cov, W,
            "flowmap", solver="rk4", n_steps=n_inner,
        )

    (m_pred, P_pred), wall = _time_call(_run)
    return _second_moment_from_mean_cov(m_pred, P_pred), wall


def particle_predict(n_particles: int, seed: int = 0):
    parts0 = jnp.repeat(X0[None, :], n_particles, axis=0)
    dt0 = jnp.asarray(DELTA / INNER, dtype=jnp.float64)

    @jit
    def _run(key):
        return _enkf_forecast(
            key, parts0, ito_drift, diffusion, QC, 0.0, float(DELTA),
            float(DELTA / INNER), 2, 1.0, solver="rk4", n_steps=INNER, sde_form="ito",
        )

    key = jax.random.key(seed)

    def fn():
        return _run(key)

    parts, wall = _time_call(fn)
    return _second_moment_from_cloud(parts), wall


def main():
    rows = []
    print(f"Lévy-area Pareto  δ={DELTA:g}  true E[(x²)²]={TRUE_SECOND:.6g}\n")

    for method in ("lv", "product"):
        for m in (3, 5):
            for k in (1, 2, 3):
                try:
                    second, wall, n_paths = wsc_predict(method, m, k)
                except Exception as e:
                    print(f"  skip WSC {method} m={m} k={k}: {e}")
                    continue
                err = abs(second - TRUE_SECOND)
                lab = f"WSC-{method} m={m} k={k}"
                rows.append(
                    {
                        "family": f"WSC-{method}",
                        "label": lab,
                        "m": m,
                        "k": k,
                        "budget": n_paths,
                        "second": second,
                        "abs_err": err,
                        "time_s": wall,
                    }
                )
                print(f"  {lab:28s}  paths={n_paths:4d}  E2={second:.6g}  err={err:.3e}  t={1e3*wall:7.2f}ms")

    for name, n_inner in (("EKF x1", 1), ("EKF x8", 8), ("EKF x32", 32)):
        second, wall = ekf_predict(n_inner)
        err = abs(second - TRUE_SECOND)
        rows.append(
            {
                "family": "CD-EKF",
                "label": name,
                "m": 0,
                "k": n_inner,
                "budget": n_inner,
                "second": second,
                "abs_err": err,
                "time_s": wall,
            }
        )
        print(f"  {name:28s}  E2={second:.6g}  err={err:.3e}  t={1e3*wall:7.2f}ms")

    for name, params, n_inner in (
        ("UKF x8", UKFParams(), 8),
        ("CKF x1", CKF, 1),
        ("CKF x8", CKF, 8),
        ("CKF x32", CKF, 32),
    ):
        second, wall = ukf_predict(n_inner, params)
        err = abs(second - TRUE_SECOND)
        fam = "CD-UKF" if name.startswith("UKF") else "CD-CKF"
        rows.append(
            {
                "family": fam,
                "label": name,
                "m": 0,
                "k": n_inner,
                "budget": n_inner,
                "second": second,
                "abs_err": err,
                "time_s": wall,
            }
        )
        print(f"  {name:28s}  E2={second:.6g}  err={err:.3e}  t={1e3*wall:7.2f}ms")

    for n in (100, 500, 2000, 8000):
        # average a few seeds for MC second moment
        seconds, walls = [], []
        for s in range(5):
            sec, wall = particle_predict(n, seed=10 + s)
            seconds.append(sec)
            walls.append(wall)
        second = float(np.mean(seconds))
        wall = float(np.median(walls))
        err = abs(second - TRUE_SECOND)
        # Also report std across seeds as uncertainty (not plotted as main y)
        rows.append(
            {
                "family": "EnKF/PF-Euler",
                "label": f"Euler n={n}",
                "m": 0,
                "k": n,
                "budget": n,
                "second": second,
                "abs_err": err,
                "time_s": wall,
                "err_seed_std": float(np.std(seconds, ddof=1)),
            }
        )
        print(
            f"  {'Euler n='+str(n):28s}  E2={second:.6g}±{np.std(seconds,ddof=1):.3g}  "
            f"err={err:.3e}  t={1e3*wall:7.2f}ms"
        )

    csv_path = os.path.join(OUTPUT_DIR, "levy_area_pareto.csv")
    with open(csv_path, "w", newline="") as f:
        fields = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # plot
    styles = {
        "WSC-lv": ("#9B5DE5", "D-", "WSC-KF LV"),
        "WSC-product": ("#6A4C93", "s--", "WSC-KF product"),
        "CD-EKF": ("#2E86AB", "^", "CD-EKF"),
        "CD-UKF": ("#90BE6D", "v", "CD-UKF"),
        "CD-CKF": ("#4EA5D9", "P", "CD-CKF"),
        "EnKF/PF-Euler": ("#E09F3E", "o-", "Euler–Maruyama cloud"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for fam, (color, fmt, disp) in styles.items():
        pts = [r for r in rows if r["family"] == fam]
        if not pts:
            continue
        pts = sorted(pts, key=lambda r: r["time_s"])
        xs = [1e3 * r["time_s"] for r in pts]
        ys = [max(r["abs_err"], 1e-18) for r in pts]
        ax.loglog(xs, ys, fmt, color=color, label=disp, markersize=6, lw=1.4)
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(
                r["label"].replace("WSC-lv ", "").replace("WSC-product ", ""),
                (x, y),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6.5,
                color=color,
            )

    ax.axhline(TRUE_SECOND * 1e-3, color="0.7", ls=":", lw=1.0, label=r"$10^{-3}\times\delta^2/2$")
    ax.set_xlabel("One-step forecast wall-clock [ms]")
    ax.set_ylabel(r"$|\widehat{\mathbb{E}}[(x^2)^2]-\delta^2/2|$")
    ax.set_title(
        rf"Lévy-area second moment  ($\delta={DELTA:g}$, true $\delta^2/2={TRUE_SECOND:.4g}$)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    png = os.path.join(OUTPUT_DIR, "levy_area_pareto.png")
    fig.savefig(png, dpi=160)
    plt.close(fig)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
