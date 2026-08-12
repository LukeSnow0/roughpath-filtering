"""Per-subject PF-MLE of (p3, σ_G); write params JSON only (no bakeoff / stress plots).

Mirrors the EnKF-MLE protocol in ``run_cohort_calibrated.py`` but fits with a
bootstrap PF. Default output:
  output/cohort_calibrated_params_dt20_pf1000.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
for pth in (HERE, EXPERIMENTS_DIR, ROOT_DIR):
    sys.path.insert(0, pth)

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, value_and_grad

jax.config.update("jax_enable_x64", True)

from cd_dynamax_mirror.filters import run_bootstrap_pf
from azt1d_data import list_subjects, load_azt1d_series
from glucose_model import GlucoseParams, init_estimate, make_glucose_dynamics

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Cohort PF-MLE of (p3, σ_G)")
    ap.add_argument("--subjects", type=str, default="")
    ap.add_argument("--n-obs", type=int, default=2016, dest="n_obs")
    ap.add_argument("--dt-assim", type=float, default=20.0, dest="dt_assim")
    ap.add_argument("--holdout-frac", type=float, default=0.3, dest="holdout_frac")
    ap.add_argument("--mle-iters", type=int, default=25, dest="mle_iters")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--p3-init", type=float, default=5e-4, dest="p3_init")
    ap.add_argument("--sigma-g-init", type=float, default=1.5, dest="sigma_g_init")
    ap.add_argument("--dt0", type=float, default=1.0)
    ap.add_argument("--pf-particles", type=int, default=1000, dest="pf_particles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meas-std", type=float, default=8.0, dest="meas_std")
    ap.add_argument(
        "--out-json",
        type=str,
        default="",
        help="default: cohort_calibrated_params_dt{dt}_pf{N}.json",
    )
    ap.add_argument(
        "--compare-json",
        type=str,
        default=os.path.join(OUTPUT_DIR, "cohort_calibrated_params_dt20_fixed.json"),
        help="optional EnKF params JSON for side-by-side σ_G comparison",
    )
    args = ap.parse_args()

    base_dt = 5.0
    dt_assim = float(args.dt_assim)
    stride = int(round(dt_assim / base_dt))
    if abs(stride * base_dt - dt_assim) > 1e-6:
        raise SystemExit("--dt-assim must be multiple of 5")

    subjects = (
        _parse_ints(args.subjects) if args.subjects.strip() else list_subjects()
    )
    n_part = int(args.pf_particles)
    out_json = args.out_json.strip()
    if not out_json:
        tag = "" if abs(dt_assim - base_dt) < 1e-9 else f"_dt{int(round(dt_assim))}"
        out_json = os.path.join(OUTPUT_DIR, f"cohort_calibrated_params{tag}_pf{n_part}.json")

    enkf_ref = {}
    if args.compare_json and os.path.isfile(args.compare_json):
        with open(args.compare_json) as f:
            blob = json.load(f)
        enkf_ref = {
            int(r["subject"]): (float(r["p3"]), float(r["sigma_g"]))
            for r in blob.get("params", [])
        }

    print(
        f"PF-MLE cohort  subjects={subjects}  Δt={dt_assim:g}  "
        f"PF-{n_part}  iters={args.mle_iters}  → {out_json}"
    )

    P = GlucoseParams(meas_std=float(args.meas_std))
    all_params: list[dict] = []
    failures: list[str] = []

    for sid in subjects:
        t_sub = time.perf_counter()
        try:
            series = load_azt1d_series(subject=sid, max_obs=int(args.n_obs), dt_min=base_dt)
            idx = np.arange(0, int(series.cgm.shape[0]), stride)
            cgm = np.asarray(series.cgm, dtype=np.float64)[idx]
            n_obs = int(cgm.shape[0])
            n_held = max(1, int(round(n_obs * args.holdout_frac)))
            n_train = n_obs - n_held
            if n_train < 20:
                raise RuntimeError(f"train too short ({n_train})")
            YS_TRAIN = jnp.asarray(cgm[:n_train].reshape(-1, 1))
            m0_np, P0_np = init_estimate(cgm[:n_train], P)
            M0 = jnp.asarray(m0_np)
            P0 = jnp.asarray(P0_np)
            _, _, _, OBS_FN, R = make_glucose_dynamics(
                P,
                t_min=series.t_min,
                insulin_u_per_min=series.insulin_u_per_min,
                carb_g_per_min=series.carb_g_per_min,
            )
            dt = float(dt_assim)
            dt0 = float(args.dt0)

            def _dyn(p3, sigma_g):
                return make_glucose_dynamics(
                    P,
                    t_min=series.t_min,
                    insulin_u_per_min=series.insulin_u_per_min,
                    carb_g_per_min=series.carb_g_per_min,
                    p3=p3,
                    sigma_g=sigma_g,
                )

            key_ad = jr.PRNGKey(args.seed + 17 * sid)

            def pf_ll_train(p3, sigma_g, key):
                drift, L, Qc, _, _ = _dyn(p3, sigma_g)
                return run_bootstrap_pf(
                    drift, OBS_FN, L, Qc, R, M0, P0, YS_TRAIN, dt, key,
                    n_particles=n_part, noise_dim=2, dt0=dt0, solver="rk4",
                ).cum_loglik[-1]

            vg = jit(
                value_and_grad(
                    lambda th: pf_ll_train(jnp.exp(th[0]), jnp.exp(th[1]), key_ad)
                )
            )
            th = np.array([np.log(args.p3_init), np.log(args.sigma_g_init)], dtype=float)
            b1, b2, eps = 0.9, 0.999, 1e-8
            m = np.zeros(2)
            v = np.zeros(2)
            ll_last = float("nan")
            for t in range(1, args.mle_iters + 1):
                val, g = vg(jnp.asarray(th))
                ll = float(val)
                g = np.asarray(g, dtype=float)
                if not np.isfinite(ll) or not np.all(np.isfinite(g)):
                    print(f"  S{sid:02d} stop at iter {t}: non-finite ll/grad")
                    break
                ll_last = ll
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                mhat = m / (1 - b1**t)
                vhat = v / (1 - b2**t)
                th = th + args.lr * mhat / (np.sqrt(vhat) + eps)
            p3_hat = float(np.exp(th[0]))
            sg_hat = float(np.exp(th[1]))
            row = {
                "subject": sid,
                "p3": p3_hat,
                "sigma_g": sg_hat,
                "dt_assim": dt_assim,
                "n_obs": n_obs,
                "n_train": n_train,
                "n_held": n_held,
                "span_days": float(series.span_days),
                "train_ll": ll_last,
                "mle_engine": f"PF-{n_part}",
            }
            all_params.append(row)
            extra = ""
            if sid in enkf_ref:
                e_p3, e_sg = enkf_ref[sid]
                extra = f"  (EnKF σ_G={e_sg:.4f}, ratio PF/EnKF={sg_hat / e_sg:.3f})"
            print(
                f"  S{sid:02d} PF-{n_part}: p3={p3_hat:.3e}  σ_G={sg_hat:.4f}  "
                f"L={ll_last:.1f}  ({time.perf_counter() - t_sub:.1f}s){extra}"
            )
        except Exception as e:
            msg = f"subject {sid}: {type(e).__name__}: {e}"
            print(f"  FAIL {msg}")
            failures.append(msg)

    if not all_params:
        raise SystemExit("no subjects succeeded")

    payload = {
        "dt_assim": dt_assim,
        "mle_engine": f"PF-{n_part}",
        "mle_iters": int(args.mle_iters),
        "pf_particles": n_part,
        "params": all_params,
        "failures": failures,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")

    sgs = np.array([r["sigma_g"] for r in all_params], dtype=float)
    print(
        f"\nPF-{n_part} σ̂_G (N={len(sgs)}): "
        f"mean={sgs.mean():.4f}  median={np.median(sgs):.4f}  "
        f"min={sgs.min():.4f}  max={sgs.max():.4f}  std={sgs.std(ddof=1):.4f}"
    )
    if enkf_ref:
        ratios = []
        pairs = []
        for r in all_params:
            sid = int(r["subject"])
            if sid not in enkf_ref:
                continue
            e_sg = enkf_ref[sid][1]
            pairs.append((e_sg, float(r["sigma_g"])))
            ratios.append(float(r["sigma_g"]) / e_sg)
        if pairs:
            e = np.array([p[0] for p in pairs])
            p = np.array([p[1] for p in pairs])
            print(
                f"EnKF σ̂_G (same subjects): mean={e.mean():.4f}  median={np.median(e):.4f}"
            )
            print(
                f"PF/EnKF σ_G ratio: mean={np.mean(ratios):.3f}  "
                f"median={np.median(ratios):.3f}  "
                f"min={np.min(ratios):.3f}  max={np.max(ratios):.3f}"
            )


if __name__ == "__main__":
    main()
