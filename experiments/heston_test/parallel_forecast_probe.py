"""Run one Heston forecast and print median wall time [s] (for parallel scaling probes).

Invoked as a subprocess with OMP/MKL threads set before JAX import.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from heston_model import HestonParams, make_heston_dynamics
from non_gaussian_flow_eval import (
    _predict_cd_ckf_pointmass,
    _predict_particle_forecast_pointmass,
    _prepare_wsckf_predictor,
)

DELTA = 0.05
INTEGRATION_DT = DELTA / 256
N_INNER = 256
X0 = np.array([0.0, 0.04], dtype=np.float64)
STRESS = HestonParams(xi=1.0, rho=-0.9, meas_std=0.02, v0=float(X0[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("wsc", "enkf", "ckf"), required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p = STRESS
    if args.method == "wsc":
        runner = _prepare_wsckf_predictor(
            p, X0, DELTA, lyons_degree=5, k=2, integration_dt=INTEGRATION_DT
        )
        fn = runner
    elif args.method == "enkf":

        def fn():
            parts, _ = _predict_particle_forecast_pointmass(
                p, X0, DELTA, n_inner=512, n_particles=2000, seed=args.seed
            )
            return parts

    else:

        def fn():
            return _predict_cd_ckf_pointmass(p, X0, DELTA, n_inner=N_INNER * 8)

    out = fn()
    jax.block_until_ready(out[0] if args.method == "ckf" else out)
    samples = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out[0] if args.method == "ckf" else out)
        samples.append(time.perf_counter() - t0)
    print(float(np.median(samples)))


if __name__ == "__main__":
    main()
