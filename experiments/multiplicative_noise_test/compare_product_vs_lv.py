"""Compare 1D-product vs explicit multi-D Lyons–Victoir paths.

Product cubature and LV multi-D agree on many additive / commuting problems.
They separate when the flow feels **Lévy area** / non-commuting noise fields.

Minimal isolating SDE (Itô), state in R², Brownian dimension 2:

    dx¹ = dW¹
    dx² = x¹ dW²

Then x¹_δ = W¹_δ and
    x²_δ = ∫₀^δ W¹_s dW²_s,
so E[x²]=0 and E[(x²)²]=∫₀^δ E[(W¹_s)²] ds = δ²/2.
The integrand depends on the joint rough path (W¹, W²), not just the
marginals — so product-of-1D paths and LV multi-D (with controlled A₁₂)
can disagree.

Experiment: point-mass predict (P₀=0), sweep δ, report |Ê[(x²)²] − δ²/2|
for product vs LV at m∈{3,5}.

Run:
  uv run --extra experiments python \\
      experiments/multiplicative_noise_test/compare_product_vs_lv.py
"""

from __future__ import annotations

import csv
import os
import sys

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
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (EXPERIMENTS_DIR, ROOT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, predict
from lyons_victoir_paths import precompute_lv_paths

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INNER_STEPS = 64
X0 = np.array([0.0, 0.0])
NOISE_DIM = 2


def ito_drift(t, x):
    del t
    return jnp.zeros_like(x)


def diffusion(t, x):
    """L(x) = [[1, 0], [0, x¹]] — second column does not commute with the first."""
    del t
    return jnp.array([[1.0, 0.0], [0.0, x[0]]], dtype=x.dtype)


def qc_fn(t, x):
    del t, x
    return jnp.eye(2)


STRAT_DRIFT, STRAT_L = ito_to_stratonovich_dynamics(ito_drift, diffusion)


def exact_second_moment_x2(delta: float) -> float:
    """E[(x²_δ)²] = δ²/2 for the Itô equation above with x(0)=0."""
    return 0.5 * float(delta) ** 2


def predict_moment(delta: float, builder: str, lyons_degree: int) -> tuple[float, int]:
    """Return predicted E[(x²)²] ≈ (weighted second moment of cloud) and path count."""
    integration_dt = float(delta) / INNER_STEPS
    if builder == "product":
        ts, paths, weights = precompute_cubature_paths(
            horizon=float(delta),
            noise_dim=NOISE_DIM,
            integration_dt=integration_dt,
            cubature_order=1,  # single LV step; fair vs multi-D (also one step)
            multidim_method="product",
            lyons_degree=lyons_degree,
        )
    elif builder == "lv":
        ts, paths, weights = precompute_lv_paths(
            horizon=float(delta),
            noise_dim=NOISE_DIM,
            integration_dt=integration_dt,
            lyons_degree=lyons_degree,  # type: ignore[arg-type]
        )
    else:
        raise ValueError(builder)

    mean = jnp.asarray(X0, dtype=jnp.float64)
    chol = jnp.zeros((2, 2), dtype=jnp.float64)
    # predict returns collapsed moments; for P0=0 the cloud is pure path terminals.
    m_pred, chol_pred, samples, w = predict(
        mean,
        chol,
        jnp.asarray(ts),
        jnp.asarray(paths),
        jnp.asarray(weights),
        STRAT_DRIFT,
        STRAT_L,
    )
    # Second moment of x² from the weighted terminal cloud (exact for the cubature
    # pushforward; equivalent to m²+P for this coordinate after collapse).
    samples = np.asarray(samples)
    w = np.asarray(w)
    x2 = samples[:, 1]
    second = float(np.sum(w * x2**2))
    # Sanity: also from moments
    m = np.asarray(m_pred)
    P = np.asarray(chol_pred @ chol_pred.T)
    second_mom = float(P[1, 1] + m[1] ** 2)
    del second  # prefer moment form (same up to numerics)
    return second_mom, int(np.asarray(paths).shape[0])


def fit_slope(deltas, errs, dmax=0.4, floor=1e-14):
    d = np.asarray(deltas, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    mask = (d <= dmax) & (e > floor)
    if mask.sum() < 2:
        mask = e > floor
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(d[mask]), np.log(e[mask]), 1)[0])


def main():
    deltas = np.logspace(np.log10(0.05), np.log10(1.0), 10)
    degrees = (3, 5)
    builders = ("product", "lv")

    series = {(b, m): [] for b in builders for m in degrees}
    n_paths = {}

    print("Lévy-area SDE: dx¹=dW¹, dx²=x¹ dW²  |  score |Ê[(x²)²]−δ²/2|")
    print("-" * 72)
    for d in deltas:
        truth = exact_second_moment_x2(d)
        row = [f"δ={d:.3f}  truth={truth:.3e}"]
        for b in builders:
            for m in degrees:
                est, npth = predict_moment(d, b, m)
                err = abs(est - truth)
                series[(b, m)].append(err)
                n_paths[(b, m)] = npth
                row.append(f"{b[:3]}-m{m}:{err:.2e}(P={npth})")
        print("  ".join(row))

    for key in series:
        series[key] = np.asarray(series[key])

    print("-" * 72)
    print(f"{'builder':10s} {'m':>3s} {'P':>5s} {'slope':>8s}  (theory (m+1)/2)")
    for b in builders:
        for m in degrees:
            s = fit_slope(deltas, series[(b, m)])
            print(
                f"{b:10s} {m:3d} {n_paths[(b, m)]:5d} {s:8.2f}  "
                f"(theory {0.5 * (m + 1):.1f})"
            )

    # plot
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    styles = {
        ("product", 3): ("o--", "#6A4C93"),
        ("product", 5): ("o--", "#9B5DE5"),
        ("lv", 3): ("s-", "#2E86AB"),
        ("lv", 5): ("s-", "#F24236"),
    }
    labels = {
        ("product", 3): rf"product $m$=3 (P={n_paths[('product', 3)]})",
        ("product", 5): rf"product $m$=5 (P={n_paths[('product', 5)]})",
        ("lv", 3): rf"LV multi-D $m$=3 (P={n_paths[('lv', 3)]})",
        ("lv", 5): rf"LV multi-D $m$=5 (P={n_paths[('lv', 5)]})",
    }
    floor = 1e-16
    for key, (fmt, color) in styles.items():
        ys = np.maximum(series[key], floor)
        s = fit_slope(deltas, series[key])
        ax.loglog(
            deltas,
            ys,
            fmt,
            color=color,
            lw=1.8,
            ms=5,
            label=f"{labels[key]}  fit={s:.2f}",
        )
    # guides
    anchor = max(series[("lv", 3)][-1], floor)
    xs = np.asarray(deltas)
    for slope, col in ((2.0, "0.45"), (3.0, "0.55")):
        c = anchor / (xs[-1] ** slope)
        ax.loglog(xs, c * xs**slope, "--", color=col, alpha=0.5, lw=1.1,
                  label=rf"$\propto\delta^{{{int(slope)}}}$")

    ax.set_xlabel(r"assimilation interval $\delta$")
    ax.set_ylabel(r"$|\widehat{\mathbb{E}}[(x^{2})^2]-\delta^{2}/2|$")
    ax.set_title(
        r"Product 1D vs LV multi-D on Lévy-area SDE "
        r"($dx^{1}=dW^{1},\; dx^{2}=x^{1}dW^{2}$)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    png = os.path.join(OUTPUT_DIR, "compare_product_vs_lv.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()

    csv_path = os.path.join(OUTPUT_DIR, "compare_product_vs_lv.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["delta", "exact_E_x2_sq"]
        for b in builders:
            for m in degrees:
                header.append(f"err_{b}_m{m}")
                header.append(f"n_paths_{b}_m{m}")
        w.writerow(header)
        for i, d in enumerate(deltas):
            row = [float(d), exact_second_moment_x2(d)]
            for b in builders:
                for m in degrees:
                    row.append(series[(b, m)][i])
                    row.append(n_paths[(b, m)])
            w.writerow(row)

    print(f"\nWrote {png}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
