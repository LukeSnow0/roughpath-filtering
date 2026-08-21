"""Kusuoka–Lyons–Victoir particle filter (Crisan & Ortiz-Latorre, 2013).

Implements paper §5 TBBA and:
  - §6.1 forecast-only (cubature-weight thinning; no likelihood) via
    ``klv_pf_forecast`` / ``klv_pf_forecast_from_particles``
  - bootstrap-style sequential filter with KLV proposal via
    ``run_klv_predictive_nll`` (mixture predictive obs NLL + likelihood resample)
"""

from __future__ import annotations

import time

import numpy as np
from jax import numpy as jnp
from scipy.special import logsumexp

from heston_model import HestonParams, make_heston_dynamics
from wsckf.filter.filtering import _propagate
from wsckf.library import ito_to_stratonovich_dynamics, precompute_cubature_paths


def _frac(x: float) -> float:
    return float(x - np.floor(x))


def tbba(
    n_particles: int,
    gamma_hat_z: int,
    gamma_z: float,
    lambdas: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate ``gamma_hat_z`` particles among children with probs ``lambdas``.

    Paper Algorithm TBBA(N, γ̂_z, γ_z, {γ_i}). Returns integer counts summing
    to ``gamma_hat_z``.
    """
    lambdas = np.asarray(lambdas, dtype=np.float64)
    lambdas = lambdas / np.sum(lambdas)
    k = int(lambdas.shape[0])
    out = np.zeros(k, dtype=np.int64)
    if gamma_hat_z <= 0 or gamma_z <= 0.0 or n_particles <= 0:
        return out

    n = float(n_particles)
    kappa1 = n * float(gamma_z)
    kappa2 = int(gamma_hat_z)

    for i in range(k - 1):
        target = n * float(gamma_z) * float(lambdas[i])
        f_target = _frac(target)
        f_rest = _frac(kappa1 - target)
        floor_t = int(np.floor(target))
        floor_k1 = int(np.floor(kappa1))
        u = float(rng.random())

        if f_target + f_rest < 1.0 - 1e-15:
            # Case 1
            f_k1 = _frac(kappa1)
            if f_k1 <= 1e-15:
                hat = floor_t
            elif u < 1.0 - (f_target / f_k1):
                hat = floor_t
            else:
                hat = floor_t + (kappa2 - floor_k1)
        else:
            # Case 2
            f_k1 = _frac(kappa1)
            if f_k1 >= 1.0 - 1e-15:
                hat = floor_t + 1
            elif u < 1.0 - ((1.0 - f_target) / (1.0 - f_k1)):
                hat = floor_t + 1
            else:
                hat = floor_t + (kappa2 - floor_k1)

        hat = int(max(0, min(hat, kappa2)))
        out[i] = hat
        kappa1 -= target
        kappa2 -= hat

    out[k - 1] = int(max(0, kappa2))
    # Repair floating drift so counts sum exactly.
    diff = int(gamma_hat_z) - int(np.sum(out))
    if diff != 0:
        out[k - 1] = int(max(0, out[k - 1] + diff))
    return out


def _gaussian_logpdf(y: np.ndarray, mean: np.ndarray, R: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 0 or R.size == 1:
        var = float(np.reshape(R, ()))
        r = float(y[0] - mean[0])
        return -0.5 * (np.log(2.0 * np.pi * var) + r * r / var)
    chol = np.linalg.cholesky(R)
    resid = np.linalg.solve(chol, y - mean)
    return float(
        -0.5 * resid.size * np.log(2.0 * np.pi)
        - np.sum(np.log(np.diag(chol)))
        - 0.5 * float(resid @ resid)
    )


def _precompute_klv_paths(
    *,
    delta: float,
    n_macro: int,
    lyons_degree: int,
    cubature_order: int,
    integration_dt: float | None,
    p: HestonParams,
):
    n_macro = max(1, int(n_macro))
    step = float(delta) / float(n_macro)
    if integration_dt is None:
        integration_dt = step / 32.0
    integration_dt = min(float(integration_dt), step)

    drift, L_fn, _, _, _ = make_heston_dynamics(p)
    strat_drift, strat_L = ito_to_stratonovich_dynamics(drift, L_fn)
    ts, path_values, path_weights = precompute_cubature_paths(
        horizon=step,
        noise_dim=2,
        integration_dt=integration_dt,
        cubature_order=int(cubature_order),
        multidim_method="lv",
        lyons_degree=int(lyons_degree),
    )
    ts_j = jnp.asarray(ts)
    paths_j = jnp.asarray(path_values)
    lambdas = np.asarray(path_weights, dtype=np.float64)
    lambdas = lambdas / np.sum(lambdas)
    return n_macro, ts_j, paths_j, lambdas, strat_drift, strat_L


def _expand_counts(
    positions: np.ndarray, counts: np.ndarray, n_particles: int, x_fallback: np.ndarray
):
    samples = np.repeat(positions, counts, axis=0)
    if samples.shape[0] != n_particles:
        if samples.shape[0] > n_particles:
            samples = samples[:n_particles]
        elif samples.shape[0] < n_particles and samples.shape[0] > 0:
            pad = n_particles - samples.shape[0]
            samples = np.concatenate(
                [samples, np.repeat(samples[-1:], pad, axis=0)], axis=0
            )
        else:
            samples = np.repeat(
                np.asarray(x_fallback).reshape(1, -1), n_particles, axis=0
            )
    weights = np.full(n_particles, 1.0 / float(n_particles), dtype=np.float64)
    return samples, weights


def _klv_tbba_propagate(
    *,
    positions: np.ndarray,
    counts: np.ndarray,
    lam_prods: np.ndarray,
    n_particles: int,
    n_macro: int,
    ts_j,
    paths_j,
    lambdas: np.ndarray,
    strat_drift,
    strat_L,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate KLV+TBBA macro-steps; returns (positions, counts, lam_prods)."""
    n_paths = int(lambdas.shape[0])
    for _ in range(n_macro):
        active = np.where(counts > 0)[0]
        positions = positions[active]
        counts = counts[active]
        lam_prods = lam_prods[active]
        m_sites = int(positions.shape[0])
        if m_sites == 0:
            break

        children = np.asarray(
            _propagate(
                jnp.asarray(positions),
                paths_j,
                ts_j,
                strat_drift,
                strat_L,
            )
        )  # (M, P, d)

        new_pos: list[np.ndarray] = []
        new_counts: list[int] = []
        new_lams: list[float] = []
        for i in range(m_sites):
            child_counts = tbba(
                n_particles,
                int(counts[i]),
                float(lam_prods[i]),
                lambdas,
                rng,
            )
            for j in range(n_paths):
                cj = int(child_counts[j])
                if cj <= 0:
                    continue
                new_pos.append(np.asarray(children[i, j], dtype=np.float64))
                new_counts.append(cj)
                new_lams.append(float(lam_prods[i]) * float(lambdas[j]))

        if not new_pos:
            break
        positions = np.stack(new_pos, axis=0)
        counts = np.asarray(new_counts, dtype=np.int64)
        lam_prods = np.asarray(new_lams, dtype=np.float64)

        total = int(np.sum(counts))
        if total != n_particles and counts.size > 0:
            counts[-1] = int(max(0, counts[-1] + (n_particles - total)))

    return positions, counts, lam_prods


def klv_pf_forecast_from_particles(
    *,
    particles: np.ndarray,
    p: HestonParams,
    delta: float,
    n_macro: int = 8,
    lyons_degree: int = 5,
    cubature_order: int = 1,
    integration_dt: float | None = None,
    seed: int = 0,
    path_cache: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast an equal-weight particle cloud one assimilation interval with KLV+TBBA."""
    particles = np.asarray(particles, dtype=np.float64)
    n_particles = int(particles.shape[0])
    rng = np.random.default_rng(seed)

    if path_cache is None:
        n_macro, ts_j, paths_j, lambdas, strat_drift, strat_L = _precompute_klv_paths(
            delta=delta,
            n_macro=n_macro,
            lyons_degree=lyons_degree,
            cubature_order=cubature_order,
            integration_dt=integration_dt,
            p=p,
        )
    else:
        n_macro, ts_j, paths_j, lambdas, strat_drift, strat_L = path_cache

    # Collapse exact duplicates to sites with multiplicity (speeds TBBA).
    rounded = np.round(particles, decimals=10)
    _, inv = np.unique(rounded, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=int(inv.max()) + 1).astype(np.int64)
    positions = np.zeros((counts.shape[0], particles.shape[1]), dtype=np.float64)
    for i, idx in enumerate(inv):
        positions[idx] = particles[i]
    lam_prods = counts.astype(np.float64) / float(n_particles)

    positions, counts, lam_prods = _klv_tbba_propagate(
        positions=positions,
        counts=counts,
        lam_prods=lam_prods,
        n_particles=n_particles,
        n_macro=n_macro,
        ts_j=ts_j,
        paths_j=paths_j,
        lambdas=lambdas,
        strat_drift=strat_drift,
        strat_L=strat_L,
        rng=rng,
    )
    return _expand_counts(positions, counts, n_particles, particles[0])


def klv_pf_forecast(
    *,
    x0: np.ndarray,
    p: HestonParams,
    delta: float,
    n_particles: int,
    n_macro: int = 8,
    lyons_degree: int = 5,
    cubature_order: int = 1,
    integration_dt: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-step signal forecast via iterated KLV + TBBA from a point mass."""
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    particles = np.repeat(x0.reshape(1, -1), int(n_particles), axis=0)
    return klv_pf_forecast_from_particles(
        particles=particles,
        p=p,
        delta=delta,
        n_macro=n_macro,
        lyons_degree=lyons_degree,
        cubature_order=cubature_order,
        integration_dt=integration_dt,
        seed=seed,
    )


def run_klv_predictive_nll(
    *,
    p: HestonParams,
    h,
    R,
    ys,
    m0,
    P0,
    dt: float,
    n_particles: int,
    seed: int,
    n_macro: int = 8,
    lyons_degree: int = 5,
    cubature_order: int = 1,
    integration_dt: float | None = None,
) -> tuple[float, float, np.ndarray, float]:
    """Sequential KLV-PF with mixture predictive obs NLL (numpy / not JAX-jitted).

    Returns ``(pred_nll, filt_nll, means, wall_s)`` averaged over observations.
    """
    ys = np.asarray(ys, dtype=np.float64)
    n_obs = int(ys.shape[0])
    n_particles = int(n_particles)
    R = np.asarray(R, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)
    P0 = np.asarray(P0, dtype=np.float64)

    path_cache = _precompute_klv_paths(
        delta=dt,
        n_macro=n_macro,
        lyons_degree=lyons_degree,
        cubature_order=cubature_order,
        integration_dt=integration_dt,
        p=p,
    )

    rng = np.random.default_rng(seed)
    particles = rng.multivariate_normal(m0, P0, size=n_particles)

    def h_np(x):
        out = h(jnp.asarray(x))
        return np.asarray(out, dtype=np.float64).reshape(-1)

    t0 = time.perf_counter()
    pred_lls = np.zeros(n_obs)
    filt_lls = np.zeros(n_obs)
    means = np.zeros((n_obs, m0.shape[0]), dtype=np.float64)

    for k in range(n_obs):
        x_pred, _w_eq = klv_pf_forecast_from_particles(
            particles=particles,
            p=p,
            delta=dt,
            n_macro=n_macro,
            lyons_degree=lyons_degree,
            cubature_order=cubature_order,
            integration_dt=integration_dt,
            seed=seed + 10007 * (k + 1),
            path_cache=path_cache,
        )
        yk = ys[k]
        log_like = np.array(
            [_gaussian_logpdf(yk, h_np(x_pred[i]), R) for i in range(n_particles)],
            dtype=np.float64,
        )
        # Equal-weight mixture predictive density (= bootstrap evidence increment).
        pred_lls[k] = float(logsumexp(log_like) - np.log(n_particles))
        filt_lls[k] = pred_lls[k]
        w = np.exp(log_like - logsumexp(log_like))
        means[k] = np.sum(w[:, None] * x_pred, axis=0)
        # Systematic resample to equal weights.
        u0 = (rng.random() + np.arange(n_particles)) / float(n_particles)
        cdf = np.cumsum(w)
        idx = np.searchsorted(cdf, u0, side="right")
        idx = np.clip(idx, 0, n_particles - 1)
        particles = x_pred[idx]

    wall = time.perf_counter() - t0
    pred_nll = float(-np.mean(pred_lls))
    filt_nll = float(-np.mean(filt_lls))
    return pred_nll, filt_nll, means, wall
