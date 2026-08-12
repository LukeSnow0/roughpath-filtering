"""Johansson-style quadruple-tank model driven by InTaSet pump commands.

State ``x = [h1, h2, h3, h4]`` (tank levels [cm]). Observations are identity
(sensor heights) with diagonal ``R``. Process noise is 2-D pump-flow noise
(so WSC-KF product ladders match the simulated MFTS ``d=2`` experiments).

Geometry defaults reuse Nielsen/Azam scale as a working prior until
``calibrate_open_loop.py`` writes fitted parameters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class IntaSetParams:
    gamma1: float = 0.45
    gamma2: float = 0.40
    g: float = 981.0
    A: tuple[float, float, float, float] = (380.1327, 380.1327, 380.1327, 380.1327)
    a_out: tuple[float, float, float, float] = (1.2272, 1.2272, 1.2272, 1.2272)
    # Volumetric flow [cm^3/s] at 100% pump command.
    k_pump: float = 120.0
    meas_std: float = 0.15  # cm
    h_eps: float = 1e-3
    # Additive sensor offsets subtracted from raw heights before filtering.
    height_bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    sigma_u: float = 15.0  # default pump-flow process noise [cm^3/s]


def params_to_dict(p: IntaSetParams) -> dict[str, Any]:
    d = asdict(p)
    d["A"] = list(p.A)
    d["a_out"] = list(p.a_out)
    d["height_bias"] = list(p.height_bias)
    return d


def params_from_dict(d: dict[str, Any]) -> IntaSetParams:
    base = IntaSetParams()
    kwargs: dict[str, Any] = {}
    for field in (
        "gamma1",
        "gamma2",
        "g",
        "k_pump",
        "meas_std",
        "h_eps",
        "sigma_u",
    ):
        if field in d:
            kwargs[field] = float(d[field])
    for field in ("A", "a_out", "height_bias"):
        if field in d:
            kwargs[field] = tuple(float(x) for x in d[field])
    return replace(base, **kwargs)


def save_params(path: Path | str, p: IntaSetParams, *, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": params_to_dict(p), "meta": meta or {}}
    path.write_text(json.dumps(payload, indent=2))


def load_params(path: Path | str) -> tuple[IntaSetParams, dict]:
    payload = json.loads(Path(path).read_text())
    if "params" in payload:
        return params_from_dict(payload["params"]), dict(payload.get("meta") or {})
    return params_from_dict(payload), {}


def apply_height_bias(heights: np.ndarray, p: IntaSetParams) -> np.ndarray:
    bias = np.asarray(p.height_bias, dtype=np.float64)
    return np.asarray(heights, dtype=np.float64) - bias


def make_intaset_dynamics(
    p: IntaSetParams,
    *,
    t_sec: np.ndarray | jnp.ndarray,
    pumps: np.ndarray | jnp.ndarray,
    sigma_u: float,
):
    """Return drift, diffusion L, Qc, obs map, R for continuous-discrete filters.

    ``pumps`` is shape ``(T, 2)`` with commands in ``[0, 100]``, aligned to
    ``t_sec`` (seconds from window start).
    """
    A = jnp.asarray(p.A, dtype=jnp.float64)
    a = jnp.asarray(p.a_out, dtype=jnp.float64)
    g1, g2 = p.gamma1, p.gamma2
    eps = p.h_eps
    k = p.k_pump
    t_grid = jnp.asarray(t_sec, dtype=jnp.float64)
    u_grid = jnp.asarray(pumps, dtype=jnp.float64)
    # Pad endpoints so jnp.interp is well-defined slightly past the last sample.
    t_lo = t_grid[0]
    t_hi = t_grid[-1]

    def _pump_frac(t: float) -> jnp.ndarray:
        tt = jnp.clip(t, t_lo, t_hi)
        u1 = jnp.interp(tt, t_grid, u_grid[:, 0])
        u2 = jnp.interp(tt, t_grid, u_grid[:, 1])
        return jnp.array([u1, u2]) / 100.0

    def _outflow(h: jnp.ndarray) -> jnp.ndarray:
        return a * jnp.sqrt(2.0 * p.g * jnp.maximum(h, eps))

    def drift_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        q = _outflow(h)
        u1, u2 = _pump_frac(t)
        f1 = k * u1
        f2 = k * u2
        dh1 = (-q[0] + q[2] + g1 * f1) / A[0]
        dh2 = (-q[1] + q[3] + g2 * f2) / A[1]
        dh3 = (-q[2] + (1.0 - g2) * f2) / A[2]
        dh4 = (-q[3] + (1.0 - g1) * f1) / A[3]
        return jnp.array([dh1, dh2, dh3, dh4])

    def diffusion_coefficient_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        # Columns: pump-1 / pump-2 flow noise [cm^3/s], mapped into dh/dt.
        s = sigma_u
        return jnp.array(
            [
                [g1 * s / A[0], 0.0],
                [0.0, g2 * s / A[1]],
                [0.0, (1.0 - g2) * s / A[2]],
                [(1.0 - g1) * s / A[3], 0.0],
            ]
        )

    def diffusion_cov_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(2)

    def obs_fn(x: jnp.ndarray) -> jnp.ndarray:
        return x

    R = jnp.diag(jnp.full(4, p.meas_std**2))
    return drift_fn, diffusion_coefficient_fn, diffusion_cov_fn, obs_fn, R


def init_estimate(
    heights0: np.ndarray,
    p: IntaSetParams,
    *,
    cov_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.maximum(np.asarray(heights0, dtype=np.float64), p.h_eps)
    P0 = (np.eye(4, dtype=np.float64) * (p.meas_std**2 + 0.25)) * cov_scale
    return x0, P0


def mape_pct(
    estimates: np.ndarray,
    truth: np.ndarray,
    state_idx: slice = slice(0, 4),
    *,
    eps: float = 1e-2,
) -> float:
    err = estimates[:, state_idx] - truth[:, state_idx]
    denom = np.maximum(np.abs(truth[:, state_idx]), eps)
    return float(100.0 * np.mean(np.abs(err / denom)))


def prepare_filter_grids(
    traj,
    *,
    delta_t_min: float,
    horizon_min: float | None = None,
    height_bias: np.ndarray | tuple | None = None,
):
    """Build fine height/pump grids and subsampled observations.

    Fine grid uses the recording's native sample times (≈0.1 s). Observations
    are every ``delta_t_min`` minutes. Returns arrays aligned so that
    ``ys[k]`` is the observation at fine index ``(k+1)*stride - 1``, matching
    the simulated MFTS experiment convention.
    """
    dt_native = float(np.median(np.diff(traj.t_sec)))
    if horizon_min is None:
        horizon_sec = float(traj.t_sec[-1] - traj.t_sec[0])
    else:
        horizon_sec = float(horizon_min) * 60.0

    # Resample onto a uniform fine grid at native dt for clean striding.
    t_fine = np.arange(0.0, horizon_sec + 0.5 * dt_native, dt_native, dtype=np.float64)
    if t_fine[-1] > horizon_sec + 1e-9:
        t_fine = t_fine[t_fine <= horizon_sec + 1e-12]
    if not np.isclose(t_fine[-1], horizon_sec, atol=1e-6):
        t_fine = np.append(t_fine, horizon_sec)

    h_fine = np.column_stack(
        [np.interp(t_fine, traj.t_sec, traj.heights[:, i]) for i in range(4)]
    )
    if height_bias is not None:
        h_fine = h_fine - np.asarray(height_bias, dtype=np.float64)
    u_fine = np.column_stack(
        [np.interp(t_fine, traj.t_sec, traj.pumps[:, i]) for i in range(2)]
    )

    dt_fine_sec = float(np.median(np.diff(t_fine)))
    delta_sec = float(delta_t_min) * 60.0
    stride = int(round(delta_sec / dt_fine_sec))
    if stride < 1 or not np.isclose(stride * dt_fine_sec, delta_sec, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Δt={delta_t_min} min is not an integer multiple of native "
            f"dt={dt_fine_sec:g}s (stride={stride})"
        )
    n_fine = t_fine.shape[0] - 1  # intervals; heights at 0..n_fine
    if n_fine % stride != 0:
        # Trim fine grid so Δt tiles the horizon.
        n_fine = (n_fine // stride) * stride
        t_fine = t_fine[: n_fine + 1]
        h_fine = h_fine[: n_fine + 1]
        u_fine = u_fine[: n_fine + 1]

    n_obs = n_fine // stride
    # Observations at the end of each assimilation interval.
    ys = h_fine[stride::stride]
    assert ys.shape[0] == n_obs

    return {
        "t_fine_sec": t_fine,
        "h_fine": h_fine,
        "u_fine": u_fine,
        "ys": ys,
        "stride": stride,
        "n_obs": n_obs,
        "n_fine": n_fine,
        "dt_fine_sec": float(np.median(np.diff(t_fine))),
        "dt_obs_sec": stride * float(np.median(np.diff(t_fine))),
        "dt_obs_min": delta_t_min,
    }


def _drift_np(
    h: np.ndarray,
    u_frac: np.ndarray,
    *,
    A: np.ndarray,
    a: np.ndarray,
    g1: float,
    g2: float,
    k_pump: float,
    g: float,
    h_eps: float,
) -> np.ndarray:
    q = a * np.sqrt(2.0 * g * np.maximum(h, h_eps))
    f1 = k_pump * u_frac[0]
    f2 = k_pump * u_frac[1]
    return np.array(
        [
            (-q[0] + q[2] + g1 * f1) / A[0],
            (-q[1] + q[3] + g2 * f2) / A[1],
            (-q[2] + (1.0 - g2) * f2) / A[2],
            (-q[3] + (1.0 - g1) * f1) / A[3],
        ],
        dtype=np.float64,
    )


def simulate_open_loop(
    p: IntaSetParams,
    t_sec: np.ndarray,
    pumps: np.ndarray,
    h0: np.ndarray,
) -> np.ndarray:
    """Integrate Johansson ODE with known pumps; return heights on ``t_sec``."""
    t = np.asarray(t_sec, dtype=np.float64)
    u = np.asarray(pumps, dtype=np.float64)
    A = np.asarray(p.A, dtype=np.float64)
    a = np.asarray(p.a_out, dtype=np.float64)
    h = np.maximum(np.asarray(h0, dtype=np.float64), p.h_eps)
    out = np.zeros((t.shape[0], 4), dtype=np.float64)
    out[0] = h
    for i in range(t.shape[0] - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0:
            out[i + 1] = h
            continue
        u0 = u[i] / 100.0
        u1 = u[i + 1] / 100.0

        def f(tt, hh, uu=u0):
            # Piecewise-constant pump over the step (left value).
            return _drift_np(
                hh,
                uu,
                A=A,
                a=a,
                g1=p.gamma1,
                g2=p.gamma2,
                k_pump=p.k_pump,
                g=p.g,
                h_eps=p.h_eps,
            )

        k1 = f(t[i], h, u0)
        k2 = f(t[i] + 0.5 * dt, h + 0.5 * dt * k1, 0.5 * (u0 + u1))
        k3 = f(t[i] + 0.5 * dt, h + 0.5 * dt * k2, 0.5 * (u0 + u1))
        k4 = f(t[i] + dt, h + dt * k3, u1)
        h = np.maximum(h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), p.h_eps)
        out[i + 1] = h
    return out
