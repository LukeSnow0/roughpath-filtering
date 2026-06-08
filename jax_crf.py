#!/usr/bin/env python3
"""
Backward-compatible re-exports. Implementation lives in ``wiener_cubature.py``.
"""

from wiener_cubature import (
    JaxCubatureRecursiveFilter,
    PathSampling,
    demo_autodiff_sigma,
    integrate_stratonovich_cde,
    make_benchmark_dynamics,
    make_highdim_dynamics,
    make_l63_dynamics,
    rk4_stratonovich_step,
    spatial_cubature_nodes,
    weighted_mean_cov,
)

__all__ = [
    "PathSampling",
    "spatial_cubature_nodes",
    "weighted_mean_cov",
    "rk4_stratonovich_step",
    "integrate_stratonovich_cde",
    "JaxCubatureRecursiveFilter",
    "make_benchmark_dynamics",
    "make_highdim_dynamics",
    "make_l63_dynamics",
    "demo_autodiff_sigma",
]

if __name__ == "__main__":
    from wiener_cubature import demo_autodiff_sigma, build_cubature_driver

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    x0 = jnp.array([1.0, 0.5])
    ts_np = np.linspace(0.0, 0.1, 11)
    paths, _ = build_cubature_driver(ts_np, noise_dim=2, k=2, multidim_method="product")
    val, grad = demo_autodiff_sigma(x0, jnp.asarray(ts_np), jnp.asarray(paths[0]), sigma_init=0.05)
    print(f"Autodiff (nonzero cubature path): x_T[0]={val:.6f}, d/dσ={grad:.6f}")
