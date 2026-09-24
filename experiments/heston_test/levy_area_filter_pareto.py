r"""Sequential filter Pareto on the Lévy-area SDE with cosine observations.

State is the mixed iterated integral \(x^2=\int W^1\,dW^2\); observations are
\(y=\cos(ω x^2)+ε\). Mixture predictive NLL of \(y\) is the proper score of the
path law pushed through a nonlinear functional. RMSE of latent \(x^2\) is
reported but is a weaker metric (cosine is many-to-one).

TME predict uses Zhao moment maps; predictive NLL evaluates \(\cos\) on CKF
sigma points of that Gaussian (fair nonlinear observation of an exact-moment
Gaussian, not EKF linearization at 0).

  .venv/bin/python experiments/heston_test/levy_area_filter_pareto.py
  .venv/bin/python experiments/heston_test/levy_area_filter_pareto.py --quick
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass

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

from cd_dynamax_mirror.filters import UKFParams, _ekf_update, _ukf_sigmas, _ukf_weights
from levy_area_model import (
    LevyAreaParams,
    init_estimate,
    make_levy_area_dynamics,
    simulate_levy_area,
)
from mfts_test.plot_style import (
    DOUBLE_COLUMN_WIDTH,
    METHOD_COLORS,
    save_publication_figure,
    style_curve_axis,
)
from predictive_nll_stepwise import (
    _scan_nlls,
    _spd,
    mixture_obs_logpdf,
    run_ekf_predictive_nll,
    run_enkf_predictive_nll,
    run_pf_predictive_nll,
    run_seukf_predictive_nll,
    run_ukf_predictive_nll,
    run_wsc_predictive_nll,
)
from tme import assemble_tme_jax, cached_eval_fn
from wsckf.library import lv_path_count

OUTPUT_DIR = os.path.join(HERE, "output", "levy_area")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPATIAL_NODES = 4
CKF_PARAMS = UKFParams(alpha=1.0, beta=0.0, kappa=0.0)


@dataclass(frozen=True)
class WSCKFConfig:
    family: str
    label: str
    lyons_degree: int
    k: int
    budget: int


def default_wsc_configs() -> list[WSCKFConfig]:
    configs = []
    for m, ks in ((3, (1, 2, 3)), (5, (1, 2))):
        for k in ks:
            wiener = lv_path_count(2, k, lyons_degree=m)
            configs.append(
                WSCKFConfig(
                    family=f"WSC-KF-m{m}",
                    label=f"m{m}-k{k}",
                    lyons_degree=m,
                    k=k,
                    budget=SPATIAL_NODES * wiener,
                )
            )
    return configs


def run_tme_utobs_predictive_nll(*, drift, L_fn, h, R, ys, m0, P0, dt, order, ukf_params):
    """TME mean/cov predict; mixture NLL of h on CKF sigmas of that Gaussian."""
    n = m0.shape[0]
    alpha, beta, kappa = ukf_params.alpha, ukf_params.beta, ukf_params.kappa
    lamb = alpha**2 * (n + kappa) - n
    w_mean, w_cov, _W = _ukf_weights(n, alpha, beta, lamb)
    eval_all = cached_eval_fn(drift, L_fn, int(order))

    def moments_at(x):
        return assemble_tme_jax(eval_all(x), float(dt), int(order))

    def body(carry, step_k):
        m, P = carry
        X = _ukf_sigmas(m, P, n, lamb)
        a_i, S_i = jax.vmap(moments_at)(X)
        S_i = jax.vmap(_spd)(S_i)
        m_pred = w_mean @ a_i
        second = jax.vmap(lambda a, S: S + jnp.outer(a, a))(a_i, S_i)
        P_pred = _spd(jnp.tensordot(w_mean, second, axes=1) - jnp.outer(m_pred, m_pred))
        sigmas = _ukf_sigmas(m_pred, P_pred, n, lamb)
        ll_pred = mixture_obs_logpdf(ys[step_k], sigmas, w_mean, h, R)
        m, P = _ekf_update(m_pred, P_pred, h, R, ys[step_k], 1)
        return (m, P), ll_pred, ll_pred, m

    return _scan_nlls(body, (m0, P0), ys.shape[0])


def _parse_ints(s: str) -> list[int]:
    out, seen = [], set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    xs = np.asarray([r[key] for r in rows], dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan"), float("nan")
    if xs.size == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1))


def _simulate_track(p: LevyAreaParams, *, n_obs: int, delta: float, seed: int):
    horizon = n_obs * delta
    steps_per_obs = 64
    n_fine = n_obs * steps_per_obs
    true_fine, ys_np, _ = simulate_levy_area(
        horizon=horizon, n_fine=n_fine, p=p, seed=seed, obs_stride=steps_per_obs
    )
    true_obs = np.asarray(true_fine, dtype=np.float64)[steps_per_obs::steps_per_obs][:n_obs]
    return true_obs, jnp.asarray(ys_np)


def _stepwise_call(fn, repeats: int = 2):
    out = fn()
    jax.block_until_ready(out)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - t0)
    pred_nll, filt_nll, means = out
    return float(pred_nll), float(filt_nll), np.asarray(means), float(np.median(samples))


COLORS = {
    "WSC-KF-m3": "#9B5DE5",
    "WSC-KF-m5": "#7B2CBF",
    "TME": "#CC79A7",
    "SE-UKF": "#00897B",
    "CD-EnKF": METHOD_COLORS["CD-EnKF"],
    "PF": "#C1121F",
    "CD-EKF": METHOD_COLORS["CD-EKF"],
    "CD-UKF": METHOD_COLORS["CD-UKF"],
    "CD-CKF": METHOD_COLORS["CD-CKF"],
}


def _summarize(raw_rows: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(r["family"], r["label"], int(r["budget"])) for r in raw_rows})
    for fam, lab, bud in keys:
        group = [r for r in raw_rows if r["family"] == fam and r["label"] == lab and int(r["budget"]) == bud]
        nll_m, nll_s = _mean_std(group, "nll_per_obs")
        pred_m, pred_s = _mean_std(group, "pred_nll_per_obs")
        rmse_m, rmse_s = _mean_std(group, "rmse")
        t_m, t_s = _mean_std(group, "time_s")
        summary.append(
            {
                "n_obs": int(group[0]["n_obs"]),
                "delta": float(group[0]["delta"]),
                "family": fam,
                "label": lab,
                "budget": bud,
                "n_seeds": len(group),
                "nll_per_obs_mean": nll_m,
                "nll_per_obs_std": nll_s,
                "pred_nll_per_obs_mean": pred_m,
                "pred_nll_per_obs_std": pred_s,
                "rmse_mean": rmse_m,
                "rmse_std": rmse_s,
                "time_s_mean": t_m,
                "time_s_std": t_s,
            }
        )
    return summary


def _plot_pareto(
    summary: list[dict],
    *,
    y_key: str,
    y_std_key: str,
    ylabel: str,
    outfile: str,
    y_hi: float | None = None,
):
    fig, ax = plt.subplots(figsize=(0.55 * DOUBLE_COLUMN_WIDTH, 2.9))
    families = []
    for r in summary:
        if r["family"] not in families:
            families.append(r["family"])
    y_lo = min(float(r[y_key]) for r in summary) - 0.08
    park_y = None if y_hi is None else y_hi - 0.12 * (y_hi - y_lo)
    for fam in families:
        pts = [r for r in summary if r["family"] == fam]
        pts = sorted(pts, key=lambda r: r["time_s_mean"])
        color = COLORS.get(fam, "#333333")
        xs = np.array([1e3 * r["time_s_mean"] for r in pts])
        ys = np.array([r[y_key] for r in pts])
        keep = np.ones(len(pts), dtype=bool) if y_hi is None else ys <= y_hi
        if np.any(keep):
            ax.plot(
                xs[keep], ys[keep], "-o", color=color, markerfacecolor="white",
                markersize=5.0, lw=1.25, label=fam, zorder=4,
            )
            for r, x, y in zip([p for p, k in zip(pts, keep) if k], xs[keep], ys[keep]):
                ax.annotate(
                    r["label"], (x, y), textcoords="offset points", xytext=(3, 3),
                    fontsize=5.5, color=color,
                )
        if y_hi is not None and np.any(~keep):
            ax.plot(
                xs[~keep], np.full((~keep).sum(), park_y), linestyle="none",
                marker="o", color=color, markerfacecolor="white", markersize=5.0,
                zorder=5, clip_on=False,
            )
    ax.set_xlabel("Wall-clock / track [ms]")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if y_hi is not None:
        ax.set_ylim(y_lo, y_hi)
    style_curve_axis(ax, grid_axis="both")
    ax.legend(fontsize=6.0, loc="best", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.97)
    path = os.path.join(OUTPUT_DIR, outfile)
    save_publication_figure(fig, path)
    plt.close(fig)
    print(f"Wrote {path}")


def run_benchmark(args) -> tuple[list[dict], list[dict]]:
    p = LevyAreaParams(omega=float(args.omega), meas_std=float(args.meas_std))
    seeds = _parse_ints(args.seeds)
    enkf_list = _parse_ints(args.enkf)
    pf_list = _parse_ints(args.pf)
    tme_list = _parse_ints(args.tme)
    seukf_list = _parse_ints(args.seukf)
    wsc_m = set(_parse_ints(args.wsc_m))
    wsc_configs = [c for c in default_wsc_configs() if c.lyons_degree in wsc_m]
    if args.wsc_max_budget > 0:
        wsc_configs = [c for c in wsc_configs if c.budget <= args.wsc_max_budget]

    dt = float(args.delta)
    dt0 = dt / float(args.inner_div)
    n_obs = int(args.n_obs)
    drift, L_fn, Qc_fn, h_obs, R = make_levy_area_dynamics(p)
    chol_R = jnp.linalg.cholesky(R)

    print(
        f"Lévy-area filter Pareto: ω={p.omega:g}, meas_std={p.meas_std:g}, "
        f"N={n_obs}, Δ={dt:g}, seeds={seeds}, WSC={[c.label for c in wsc_configs]}, "
        f"TME={tme_list}, SE-UKF={seukf_list}, EnKF={enkf_list}, PF={pf_list}",
        flush=True,
    )

    wsc_jits = {
        (cfg.lyons_degree, cfg.k): jit(
            lambda ys, m0, chol_P0, ly=cfg.lyons_degree, kk=cfg.k: run_wsc_predictive_nll(
                drift=drift, L_fn=L_fn, h=h_obs, R=R, chol_R=chol_R,
                ys=ys, m0=m0, chol_P0=chol_P0, dt=dt, dt0=dt0,
                lyons_degree=ly, k=kk,
            )
        )
        for cfg in wsc_configs
    }
    ekf_jit = jit(
        lambda ys, m0, P0: run_ekf_predictive_nll(
            drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
            ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0,
        )
    )
    ukf_jit = jit(
        lambda ys, m0, P0: run_ukf_predictive_nll(
            drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
            ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0, ukf_params=UKFParams(),
        )
    )
    ckf_jit = jit(
        lambda ys, m0, P0: run_ukf_predictive_nll(
            drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
            ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0, ukf_params=CKF_PARAMS,
        )
    )
    tme_jits = {
        order: jit(
            lambda ys, m0, P0, o=order: run_tme_utobs_predictive_nll(
                drift=drift, L_fn=L_fn, h=h_obs, R=R, ys=ys, m0=m0, P0=P0,
                dt=dt, order=o, ukf_params=CKF_PARAMS,
            )
        )
        for order in tme_list
    }
    seukf_jits = {
        n_modes: jit(
            lambda ys, m0, P0, n=n_modes: run_seukf_predictive_nll(
                drift=drift, L_fn=L_fn, h=h_obs, R=R, ys=ys, m0=m0, P0=P0,
                dt=dt, dt0=dt0, n_modes=n, ukf_params=CKF_PARAMS,
            )
        )
        for n_modes in seukf_list
    }
    enkf_jits = {
        n_part: jit(
            lambda ys, m0, P0, k, n=n_part: run_enkf_predictive_nll(
                drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
                ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0, key=k, n_particles=n,
            )
        )
        for n_part in enkf_list
    }
    pf_jits = {
        n_part: jit(
            lambda ys, m0, P0, k, n=n_part: run_pf_predictive_nll(
                drift=drift, L_fn=L_fn, Qc_fn=Qc_fn, h=h_obs, R=R,
                ys=ys, m0=m0, P0=P0, dt=dt, dt0=dt0, key=k, n_particles=n,
            )
        )
        for n_part in pf_list
    }

    raw_rows: list[dict] = []
    n_timing = int(args.n_timing)

    def record(family, label, budget, pred_nll, filt_nll, means, wall_s, seed, true_obs):
        rmse = float(np.sqrt(np.mean((np.asarray(means)[:, 1] - true_obs[:, 1]) ** 2)))
        raw_rows.append(
            {
                "seed": seed,
                "n_obs": n_obs,
                "delta": dt,
                "family": family,
                "label": label,
                "budget": budget,
                "pred_nll_per_obs": pred_nll,
                "nll_per_obs": filt_nll,
                "rmse": rmse,
                "time_s": wall_s,
            }
        )
        print(
            f"  {family:12s} {label:12s}  predNLL={pred_nll:8.4f}  "
            f"RMSE(x²)={rmse:7.4f}  time={1e3*wall_s:8.1f} ms",
            flush=True,
        )

    for seed_i, seed in enumerate(seeds, start=1):
        print(f"\n=== seed {seed}  [{seed_i}/{len(seeds)}] ===", flush=True)
        true_obs, ys = _simulate_track(p, n_obs=n_obs, delta=dt, seed=seed)
        m0_np, P0_np = init_estimate(p)
        m0 = jnp.asarray(m0_np)
        P0 = jnp.asarray(P0_np)
        chol_P0 = jnp.linalg.cholesky(P0)

        for cfg in wsc_configs:
            pred, filt, means, wall = _stepwise_call(
                lambda c=cfg: wsc_jits[(c.lyons_degree, c.k)](ys, m0, chol_P0),
                repeats=n_timing,
            )
            record(cfg.family, cfg.label, cfg.budget, pred, filt, means, wall, seed, true_obs)
        if not args.skip_ekf:
            pred, filt, means, wall = _stepwise_call(lambda: ekf_jit(ys, m0, P0), repeats=n_timing)
            record("CD-EKF", "EKF", 0, pred, filt, means, wall, seed, true_obs)
        if not args.skip_ukf:
            pred, filt, means, wall = _stepwise_call(lambda: ukf_jit(ys, m0, P0), repeats=n_timing)
            record("CD-UKF", "UKF", 0, pred, filt, means, wall, seed, true_obs)
        if not args.skip_ckf:
            pred, filt, means, wall = _stepwise_call(lambda: ckf_jit(ys, m0, P0), repeats=n_timing)
            record("CD-CKF", "CKF", 0, pred, filt, means, wall, seed, true_obs)
        for order in tme_list:
            pred, filt, means, wall = _stepwise_call(
                lambda o=order: tme_jits[o](ys, m0, P0), repeats=n_timing
            )
            record("TME", f"TME-{order}", int(order), pred, filt, means, wall, seed, true_obs)
        for n_modes in seukf_list:
            pred, filt, means, wall = _stepwise_call(
                lambda n=n_modes: seukf_jits[n](ys, m0, P0), repeats=n_timing
            )
            record("SE-UKF", f"N{n_modes}", int(n_modes), pred, filt, means, wall, seed, true_obs)
        for n_part in enkf_list:
            key = jax.random.key(int(seed) + 17_000 + n_part)
            pred, filt, means, wall = _stepwise_call(
                lambda n=n_part, k=key: enkf_jits[n](ys, m0, P0, k), repeats=n_timing
            )
            record("CD-EnKF", f"EnKF-{n_part}", n_part, pred, filt, means, wall, seed, true_obs)
        for n_part in pf_list:
            key = jax.random.key(int(seed) + 29_000 + n_part)
            pred, filt, means, wall = _stepwise_call(
                lambda n=n_part, k=key: pf_jits[n](ys, m0, P0, k), repeats=n_timing
            )
            record("PF", f"PF-{n_part}", n_part, pred, filt, means, wall, seed, true_obs)

    return raw_rows, _summarize(raw_rows)


def _write_csvs(raw_rows, summary):
    raw_path = os.path.join(OUTPUT_DIR, "levy_area_filter_pareto_raw.csv")
    sum_path = os.path.join(OUTPUT_DIR, "levy_area_filter_pareto.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"Wrote {raw_path}\nWrote {sum_path}")


def _read_raw_csv() -> list[dict]:
    path = os.path.join(OUTPUT_DIR, "levy_area_filter_pareto_raw.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _coerce_raw_row(r: dict) -> dict:
    out = dict(r)
    out["seed"] = int(out["seed"])
    out["n_obs"] = int(float(out["n_obs"]))
    out["delta"] = float(out["delta"])
    out["budget"] = int(float(out["budget"]))
    for k in ("nll_per_obs", "pred_nll_per_obs", "rmse", "time_s"):
        out[k] = float(out[k])
    return out


def _merge_raw(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    def key(r):
        return (int(r["seed"]), r["family"], r["label"], int(r["budget"]))

    merged = {key(r): _coerce_raw_row(r) for r in existing}
    for r in new_rows:
        merged[key(r)] = r
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Lévy-area cosine-observation filter Pareto")
    parser.add_argument("--seeds", type=str, default=",".join(str(i) for i in range(100, 120)))
    parser.add_argument("--n-obs", type=int, default=8, dest="n_obs")
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--inner-div", type=int, default=32, dest="inner_div")
    parser.add_argument("--omega", type=float, default=3.0)
    parser.add_argument("--meas-std", type=float, default=0.25, dest="meas_std")
    parser.add_argument("--enkf", type=str, default="100,500")
    parser.add_argument("--pf", type=str, default="100,500")
    parser.add_argument("--tme", type=str, default="2,3")
    parser.add_argument("--seukf", type=str, default="8")
    parser.add_argument("--wsc-m", type=str, default="3,5", dest="wsc_m")
    parser.add_argument("--wsc-max-budget", type=int, default=0, dest="wsc_max_budget")
    parser.add_argument("--n-timing", type=int, default=2, dest="n_timing")
    parser.add_argument("--skip-ekf", action="store_true")
    parser.add_argument("--skip-ukf", action="store_true")
    parser.add_argument("--skip-ckf", action="store_true")
    parser.add_argument("--skip-wsc", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge into existing raw CSV (replace matching seed/family/label/budget)",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = "100,101,102"
        args.enkf = "100"
        args.pf = "100"
        args.seukf = "4"
        args.wsc_m = "3,5"
        args.wsc_max_budget = 80
        args.n_obs = 4
        args.n_timing = 1
    if args.skip_wsc:
        args.wsc_m = ""
    raw, summary = run_benchmark(args)
    if args.append:
        existing = _read_raw_csv()
        if existing:
            raw = _merge_raw(existing, raw)
            summary = _summarize(raw)
            print(f"Append-merged into existing raw (now {len(raw)} rows)", flush=True)
    _write_csvs(raw, summary)
    _plot_pareto(
        summary,
        y_key="pred_nll_per_obs_mean",
        y_std_key="pred_nll_per_obs_std",
        ylabel="Predictive NLL [nats / obs.]",
        outfile="levy_area_filter_pred_nll_pareto.png",
        y_hi=3.2,
    )
    _plot_pareto(
        summary,
        y_key="rmse_mean",
        y_std_key="rmse_std",
        ylabel=r"RMSE ($x^2$)",
        outfile="levy_area_filter_rmse_pareto.png",
    )


if __name__ == "__main__":
    main()
