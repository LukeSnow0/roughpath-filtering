"""Peak-RSS probe for one pathwise gradient at a given assimilation interval.

Run in a fresh process so ``ru_maxrss`` is not polluted by prior configs:

  python experiments/mfts_test/grad_memory_probe.py \\
      --method CRF-ckpt --delta-t 30 --dt0 0.25 --n-obs 16

Methods: CRF-ckpt (checkpointed ODE adjoint), CRF-scan, CD-EKF, CD-UKF.
Prints a single line: ``method delta_t_s n_steps peak_rss_mb``
"""

from __future__ import annotations

import argparse
import os
import resource
import sys

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENTS_DIR, ".."))
sys.path[:0] = [HERE, EXPERIMENTS_DIR, ROOT_DIR]

import jax
import jax.numpy as jnp
from jax import jit, random, value_and_grad

jax.config.update("jax_enable_x64", True)

from crf import build_filter
from crf.library import default_recombined_num_paths
from cuthbert import filter as run_filter
from experiments.cd_dynamax_mirror.filters import UKFParams, run_cd_ekf, run_cd_ukf
from mfts_model import (
    MFTSParams,
    bar_F_schedule_jax,
    init_estimate,
    make_mfts_dynamics,
    simulate_mfts,
)
from typing import NamedTuple


class StepInputs(NamedTuple):
    t_idx: jnp.ndarray
    lambda_dist: jnp.ndarray


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        required=True,
        choices=("CRF-ckpt", "CRF-scan", "CD-EKF", "CD-UKF"),
    )
    ap.add_argument("--delta-t", type=float, required=True, help="Assimilation interval [s]")
    ap.add_argument("--dt0", type=float, default=0.25, help="Fixed RK4 step [s]")
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--cubature-order", type=int, default=3)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.05)
    args = ap.parse_args()

    dt = float(args.delta_t)
    dt0 = float(args.dt0)
    n_obs = int(args.n_obs)
    n_steps = max(2, int(round(dt / dt0)))
    # Match the requested interval with an integer RK4 grid.
    dt0 = dt / n_steps
    horizon_min = (n_obs * dt) / 60.0

    p = MFTSParams()
    F1 = F2 = 5.0
    sigma = 5.0
    true_traj, ys_np, R = simulate_mfts(
        horizon_min=horizon_min,
        n_obs=n_obs,
        n_substeps_truth=max(200, n_steps * 4),
        p=p,
        lambda_sim=0.1,
        sigma_sim=sigma,
        F1=F1,
        F2=F2,
        seed=args.seed,
    )
    ys = jnp.asarray(ys_np)
    x0, P0 = init_estimate(p, F1, F2, true_state=true_traj[0])
    m0 = jnp.asarray(x0, dtype=jnp.float64)
    P0_j = jnp.asarray(P0, dtype=jnp.float64)
    chol_P0 = jnp.linalg.cholesky(P0_j)
    chol_R = jnp.linalg.cholesky(R)
    bar = lambda t_min: bar_F_schedule_jax(t_min)

    def dynamics(lam):
        return make_mfts_dynamics(
            p,
            lambda_dist=lam,
            sigma_dist=sigma,
            bar_F3=bar,
            bar_F4=bar,
            F1=F1,
            F2=F2,
        )

    drift0, L0, _, obs_fn, _ = dynamics(args.lam)
    alpha = jnp.log(jnp.asarray(args.lam, dtype=jnp.float64))

    if args.method.startswith("CRF"):
        ckpt = args.method == "CRF-ckpt"
        num_paths = default_recombined_num_paths(2, args.cubature_order)

        def get_init(mi: StepInputs):
            return m0, chol_P0

        def get_dyn(mi: StepInputs):
            t0_abs = (mi.t_idx - 1) * dt
            drift, L, _, _, _ = dynamics(mi.lambda_dist)

            def drift_abs(t, x):
                return drift(t0_abs + t, x)

            def diff_abs(t, x):
                return L(t0_abs + t, x)

            return drift_abs, diff_abs

        def get_obs(mi: StepInputs):
            return obs_fn, chol_R, ys[mi.t_idx - 1]

        filt = build_filter(
            get_init_params=get_init,
            get_dynamics=get_dyn,
            get_observations=get_obs,
            noise_dim=2,
            dt=dt,
            integration_dt=dt0,
            cubature_order=args.cubature_order,
            num_paths=num_paths,
            checkpoint_steps=ckpt,
            sde_form="ito",
        )
        run = jit(run_filter, static_argnames=["filter_obj"])

        def ll(a):
            mi = StepInputs(
                t_idx=jnp.arange(n_obs + 1),
                lambda_dist=jnp.full((n_obs + 1,), jnp.exp(a)),
            )
            return run(filt, mi).log_normalizing_constant[-1]

        vg = jit(value_and_grad(ll))
    elif args.method == "CD-EKF":
        def ll(a):
            lam = jnp.exp(a)
            drift, L, Qc, _, _ = dynamics(lam)
            res = run_cd_ekf(
                drift, obs_fn, L, Qc, R, m0, P0_j, ys, dt, dt0,
                state_order="first", solver="rk4",
            )
            return res.cum_loglik[-1]

        vg = jit(value_and_grad(ll))
    else:  # CD-UKF
        def ll(a):
            lam = jnp.exp(a)
            drift, L, Qc, _, _ = dynamics(lam)
            res = run_cd_ukf(
                drift, obs_fn, L, Qc, R, m0, P0_j, ys, dt, dt0, UKFParams(),
                state_order="first", solver="rk4",
            )
            return res.cum_loglik[-1]

        vg = jit(value_and_grad(ll))

    # Compile + execute; peak RSS covers AD tape for this config in a fresh process.
    jax.block_until_ready(vg(alpha))
    for _ in range(2):
        jax.block_until_ready(vg(alpha))

    print(f"{args.method} {dt:.6g} {n_steps} {peak_rss_mb():.4f}")


if __name__ == "__main__":
    main()
