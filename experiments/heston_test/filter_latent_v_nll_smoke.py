"""Smoke: filter Pareto with latent-v predictive NLL (option 1).

Same twin / obs y=X+ε, but also scores the one-step predictive cloud on the
true latent variance:

    log p(v_k | y_{1:k-1}) ≈ log Σ_j w_j N(v_k; v_j, h²)

(Gaussian filters: N(v; m_v, P_vv).)

Quick run:
  JAX_PLATFORMS=cpu uv run --extra experiments python \\
    experiments/heston_test/filter_latent_v_nll_smoke.py
"""

from __future__ import annotations

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
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, lax, vmap
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

from cd_dynamax_mirror.filters import (
    UKFParams,
    _ekf_predict,
    _ekf_update,
    _enkf_analysis,
    _enkf_forecast,
    _rk4_n_steps,
    _systematic_resample,
    _ukf_predict,
    _ukf_sigmas,
    _ukf_update,
    _ukf_weights,
    gaussian_logpdf,
    psd,
)
from heston_model import HestonParams, init_estimate, make_heston_dynamics, simulate_heston
from predictive_nll_stepwise import mixture_obs_logpdf
from wsckf.filter.filtering import predict, update
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths, lv_path_count

OUTPUT_DIR = os.path.join(HERE, "output", "filter_nll")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRESS = HestonParams(xi=1.0, rho=-0.9, meas_std=0.02)
CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)
SPATIAL_NODES = 4
V_BW2 = 1e-4  # kernel bandwidth² for latent-v mixture score


def mixture_v_logpdf(v_true, samples, weights, bw2: float = V_BW2):
    """log Σ_j w_j N(v_true; v_j, bw2)."""
    weights = jnp.asarray(weights, dtype=jnp.float64)
    weights = weights / jnp.sum(weights)
    vj = samples[:, 1]
    r = v_true - vj
    log_phi = -0.5 * (jnp.log(2.0 * jnp.pi * bw2) + (r * r) / bw2)
    return logsumexp(jnp.log(weights) + log_phi)


def _scan(body, init, n_obs: int):
    def step(carry, k):
        carry, obs_ll, v_ll, mean_k = body(carry, k)
        return carry, (obs_ll, v_ll, mean_k)

    _, (obs_lls, v_lls, means) = lax.scan(step, init, jnp.arange(n_obs))
    return (
        -jnp.mean(obs_lls),
        -jnp.mean(v_lls),
        means,
    )


def run_wsc(drift, L_fn, h, R, chol_R, ys, true_v, m0, chol_P0, dt, dt0, m, k):
    strat_drift, strat_L = ito_to_stratonovich_dynamics(drift, L_fn)
    ts, path_values, path_weights = precompute_cubature_paths(
        horizon=float(dt),
        noise_dim=2,
        integration_dt=float(dt0),
        cubature_order=int(k),
        multidim_method="lv",
        lyons_degree=int(m),
    )

    def body(carry, step_k):
        mean, chol = carry
        pred_mean, pred_chol, samples, weights = predict(
            mean, chol, ts, path_values, path_weights, strat_drift, strat_L
        )
        ll_obs = mixture_obs_logpdf(ys[step_k], samples, weights, h, R)
        ll_v = mixture_v_logpdf(true_v[step_k], samples, weights)
        mean, chol, _, _, _ = update(pred_mean, samples, weights, h, chol_R, ys[step_k])
        return (mean, chol), ll_obs, ll_v, mean

    return _scan(body, (m0, chol_P0), ys.shape[0])


def run_ekf(drift, L_fn, Qc_fn, h, R, ys, true_v, m0, P0, dt, dt0):
    n_inner = _rk4_n_steps(dt, dt0)

    def body(carry, step_k):
        m, P = carry
        t0 = step_k * dt
        m_pred, P_pred = _ekf_predict(
            m, P, drift, L_fn, Qc_fn, t0, t0 + dt, dt0,
            "flowmap", solver="rk4", n_steps=n_inner,
        )
        H = jax.jacfwd(h)(m_pred)
        S = psd(H @ P_pred @ H.T + R)
        ll_obs = gaussian_logpdf(ys[step_k], h(m_pred), S)
        ll_v = gaussian_logpdf(
            jnp.asarray([true_v[step_k]]),
            jnp.asarray([m_pred[1]]),
            jnp.asarray([[P_pred[1, 1]]]),
        )
        m, P = _ekf_update(m_pred, P_pred, h, R, ys[step_k], 1)
        return (m, P), ll_obs, ll_v, m

    return _scan(body, (m0, P0), ys.shape[0])


def run_ukf(drift, L_fn, Qc_fn, h, R, ys, true_v, m0, P0, dt, dt0, ukf_params):
    n = m0.shape[0]
    alpha, beta, kappa = ukf_params.alpha, ukf_params.beta, ukf_params.kappa
    lamb = alpha**2 * (n + kappa) - n
    w_mean, w_cov, W_matrix = _ukf_weights(n, alpha, beta, lamb)
    n_inner = _rk4_n_steps(dt, dt0)

    def body(carry, step_k):
        m, P = carry
        t0 = step_k * dt
        m_pred, P_pred = _ukf_predict(
            m, P, drift, L_fn, Qc_fn, t0, t0 + dt, dt0, n, lamb,
            w_mean, w_cov, W_matrix, "flowmap", solver="rk4", n_steps=n_inner,
        )
        sigmas = _ukf_sigmas(m_pred, P_pred, n, lamb)
        ll_obs = mixture_obs_logpdf(ys[step_k], sigmas, w_mean, h, R)
        ll_v = mixture_v_logpdf(true_v[step_k], sigmas, w_mean)
        _, m, P, _, _ = _ukf_update(
            m_pred, P_pred, h, R, ys[step_k], n, lamb, w_mean, w_cov, 1
        )
        return (m, P), ll_obs, ll_v, m

    return _scan(body, (m0, P0), ys.shape[0])


def run_particles(
    drift, L_fn, Qc_fn, h, R, ys, true_v, m0, P0, dt, dt0, key, n_particles, *, enkf: bool
):
    n_inner = _rk4_n_steps(dt, dt0)
    key, init_key = jr.split(key)
    particles = jr.multivariate_normal(init_key, mean=m0, cov=psd(P0), shape=(n_particles,))
    w_eq = jnp.full(n_particles, 1.0 / n_particles)

    def body(carry, step_k):
        parts, key = carry
        key, k_pred, k_upd = jr.split(key, 3)
        t0 = step_k * dt
        x_pred = _enkf_forecast(
            k_pred, parts, drift, L_fn, Qc_fn, t0, t0 + dt, dt0,
            2, 1.0, solver="rk4", n_steps=n_inner, sde_form="ito",
        )
        ll_obs = mixture_obs_logpdf(ys[step_k], x_pred, w_eq, h, R)
        ll_v = mixture_v_logpdf(true_v[step_k], x_pred, w_eq)
        if enkf:
            x_filt, _, _ = _enkf_analysis(k_upd, x_pred, h, R, ys[step_k], True, 0.0)
            mean = jnp.mean(x_filt, axis=0)
            return (x_filt, key), ll_obs, ll_v, mean
        log_w = vmap(lambda x: gaussian_logpdf(ys[step_k], h(x), R))(x_pred)
        w = jax.nn.softmax(log_w)
        idx = _systematic_resample(k_upd, w)
        mean = jnp.sum(w[:, None] * x_pred, axis=0)
        return (x_pred[idx], key), ll_obs, ll_v, mean

    return _scan(body, (particles, key), ys.shape[0])


def _timed(fn, repeats=2):
    out = fn()
    jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    obs_nll, v_nll, means = out
    return float(obs_nll), float(v_nll), float(np.median(times))


def _simulate(p, n_obs, delta, seed):
    horizon = n_obs * delta
    n_fine = max(n_obs * 8, int(round(horizon / (delta / 4))))
    obs_stride = max(1, n_fine // n_obs)
    true_fine, ys_np, _ = simulate_heston(
        horizon=horizon, n_fine=n_fine, p=p, seed=seed, obs_stride=obs_stride
    )
    true_obs = np.asarray(true_fine, dtype=np.float64)[obs_stride::obs_stride][:n_obs]
    return true_obs, jnp.asarray(ys_np), jnp.asarray(true_obs[:, 1])


def main():
    seeds = [100, 101, 102]
    n_obs, delta, dt0 = 16, 0.05, 0.05 / 128
    p = STRESS
    rows = []

    configs = []
    for m, k in ((3, 1), (3, 2), (5, 1), (5, 2)):
        bud = SPATIAL_NODES * lv_path_count(2, k, lyons_degree=m)
        configs.append(("WSC", f"m{m}-k{k}", m, k, bud))
    configs += [
        ("EKF", "EKF", None, None, 0),
        ("CKF", "CKF", None, None, 0),
        ("EnKF", "EnKF-100", 100, None, 100),
        ("EnKF", "EnKF-500", 500, None, 500),
        ("PF", "PF-100", 100, None, 100),
        ("PF", "PF-500", 500, None, 500),
    ]

    print(
        f"Latent-v smoke: seeds={seeds}, N={n_obs}, Δ={delta}, bw2={V_BW2:g}\n"
        f"{'family':8s} {'label':10s} {'obsNLL':8s} {'vNLL':8s} {'t_ms':8s}"
    )

    for seed in seeds:
        true_obs, ys, true_v = _simulate(p, n_obs, delta, seed)
        m0_np, P0_np = init_estimate(
            p, true_state=np.asarray(true_obs[0], dtype=np.float64), cov_scale=1.0
        )
        m0 = jnp.asarray(m0_np)
        P0 = jnp.asarray(P0_np)
        chol_P0 = jnp.linalg.cholesky(P0)
        drift, L_fn, Qc_fn, h, R = make_heston_dynamics(p)
        chol_R = jnp.linalg.cholesky(R)
        key = jr.key(seed)

        for fam, lab, a, b, bud in configs:
            if fam == "WSC":
                fn = jit(
                    lambda m=a, k=b: run_wsc(
                        drift, L_fn, h, R, chol_R, ys, true_v, m0, chol_P0, delta, dt0, m, k
                    )
                )
            elif fam == "EKF":
                fn = jit(lambda: run_ekf(drift, L_fn, Qc_fn, h, R, ys, true_v, m0, P0, delta, dt0))
            elif fam == "CKF":
                fn = jit(
                    lambda: run_ukf(
                        drift, L_fn, Qc_fn, h, R, ys, true_v, m0, P0, delta, dt0, CKF_PARAMS
                    )
                )
            elif fam == "EnKF":
                key, sub = jr.split(key)
                fn = jit(
                    lambda k=sub, n=a: run_particles(
                        drift, L_fn, Qc_fn, h, R, ys, true_v, m0, P0, delta, dt0, k, n, enkf=True
                    )
                )
            else:
                key, sub = jr.split(key)
                fn = jit(
                    lambda k=sub, n=a: run_particles(
                        drift, L_fn, Qc_fn, h, R, ys, true_v, m0, P0, delta, dt0, k, n, enkf=False
                    )
                )

            obs_nll, v_nll, wall = _timed(fn, repeats=2)
            rows.append(
                {
                    "seed": seed,
                    "family": fam,
                    "label": lab,
                    "budget": bud,
                    "obs_nll": obs_nll,
                    "v_nll": v_nll,
                    "time_s": wall,
                }
            )
            print(f"{fam:8s} {lab:10s} {obs_nll:8.3f} {v_nll:8.3f} {1e3 * wall:8.1f}")

    # aggregate
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["label"], r["budget"])].append(r)

    summary = []
    print("\n=== mean ± std over seeds ===")
    print(f"{'label':12s} {'obsNLL':14s} {'vNLL':14s} {'t_ms':8s}")
    for (fam, lab, bud), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][2])):
        obs = np.array([r["obs_nll"] for r in rs])
        vn = np.array([r["v_nll"] for r in rs])
        ts = np.array([r["time_s"] for r in rs])
        summary.append(
            {
                "family": fam,
                "label": lab,
                "budget": bud,
                "obs_mean": float(obs.mean()),
                "obs_std": float(obs.std(ddof=1)) if len(obs) > 1 else 0.0,
                "v_mean": float(vn.mean()),
                "v_std": float(vn.std(ddof=1)) if len(vn) > 1 else 0.0,
                "t_mean": float(ts.mean()),
            }
        )
        print(
            f"{lab:12s} {obs.mean():7.3f}±{obs.std(ddof=1):.3f}  "
            f"{vn.mean():7.3f}±{vn.std(ddof=1):.3f}  {1e3 * ts.mean():7.1f}"
        )

    # plot v-NLL Pareto
    colors = {
        "WSC": "#9B5DE5",
        "EKF": "#2E86AB",
        "CKF": "#4EA5D9",
        "EnKF": "#E09F3E",
        "PF": "#6A4C93",
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for fam in ("WSC", "EnKF", "PF", "EKF", "CKF"):
        pts = [s for s in summary if s["family"] == fam]
        if not pts:
            continue
        pts = sorted(pts, key=lambda s: s["t_mean"])
        xs = [1e3 * s["t_mean"] for s in pts]
        ys = [s["v_mean"] for s in pts]
        yerr = [s["v_std"] for s in pts]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o-" if len(pts) > 1 else "o",
            color=colors[fam],
            label=fam,
            markersize=5,
            capsize=2,
        )
        for s, x, y in zip(pts, xs, ys):
            ax.annotate(s["label"], (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock [ms]")
    ax.set_ylabel(r"Latent predictive NLL of $v$ [nats/step]")
    ax.set_title(rf"Smoke: latent-$v$ score ($N={n_obs}$, stress, {len(seeds)} seeds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "filter_latent_v_nll_smoke.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
