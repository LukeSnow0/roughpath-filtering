r"""Isolating Lévy-area / iterated-integral SDE for filter comparisons.

Itô dynamics (noise fields do not commute):

    dx¹ = dW¹
    dx² = x¹ dW²

From \(x_0=0\), \(x^1_δ=W^1_δ\) and \(x^2_δ=\int_0^δ W^1_s\,dW^2_s\). The second
coordinate *is* a mixed iterated integral: \(\mathbb{E}[(x^2)^2]=δ^2/2\) needs
Lévy area / the joint rough path, and \(x^2\) is not Gaussian (excess kurtosis).

TME-2 is exact for the first two moments of \(x\) (the generator terminates on
quadratics). It is *not* a cubature of the path law, so a nonlinear functional
of \(x^2\) still sees the signature gap.

Default observation is the characteristic function of that coordinate:

    y_k = cos(ω x²_{t_k}) + ε_k,    ε ~ N(0, R)

A Gaussian with the exact moments of \(x^2\) predicts \(\mathbb{E}[y]=e^{-ω^2 δ^2/4}\);
degree-3 Wiener cubature from a point mass leaves \(x^2\equiv 0\) and predicts
\(\mathbb{E}[y]=1\).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LevyAreaParams:
    omega: float = 3.0
    meas_std: float = 0.25
    x0: tuple[float, float] = (0.0, 0.0)
    prior_std: float = 1e-4  # near point-mass; cholesky-stable


def ito_drift(t, x):
    del t
    return jnp.zeros_like(x)


def ito_L(t, x):
    del t
    return jnp.array([[1.0, 0.0], [0.0, x[0]]], dtype=x.dtype)


def qc_fn(t, x):
    del t, x
    return jnp.eye(2)


def make_levy_area_dynamics(p: LevyAreaParams):
    omega = float(p.omega)

    def obs_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos(omega * x[1:2])

    R = jnp.array([[p.meas_std**2]], dtype=jnp.float64)
    return ito_drift, ito_L, qc_fn, obs_fn, R


def exact_second_moment(delta: float) -> float:
    r"""\(\mathbb{E}[(x^2_δ)^2] = δ^2/2\) from \(x_0=0\)."""
    return 0.5 * float(delta) ** 2


def gaussian_char(delta: float, omega: float) -> float:
    r"""\(\mathbb{E}[\cos(\omega Z)]\) for \(Z\sim\mathcal{N}(0, δ^2/2)\)."""
    return float(np.exp(-0.5 * (omega**2) * exact_second_moment(delta)))


def exact_char(delta: float, omega: float) -> float:
    r"""\(\mathbb{E}[\cos(\omega x^2_δ)]=[\cosh(\omega\delta)]^{-1/2}\).

    Given \(W^1\), the Itô integral \(x^2=\int W^1\,dW^2\) is
    \(\mathcal{N}(0,\int (W^1)^2)\,ds\), so the cosine is the Cameron–Martin
    Laplace transform of Brownian quadratic time.
    """
    return float(np.cosh(float(omega) * float(delta)) ** -0.5)


def simulate_levy_area(
    *,
    horizon: float,
    n_fine: int,
    p: LevyAreaParams,
    seed: int,
    obs_stride: int,
):
    r"""Itô Euler–Maruyama twin; observations of \(\cos(\omega x^2)\) at ``obs_stride``."""
    rng = np.random.default_rng(int(seed))
    dt = float(horizon) / int(n_fine)
    sdt = np.sqrt(dt)
    x = np.array(p.x0, dtype=np.float64)
    xs = [x.copy()]
    dW = rng.normal(0.0, sdt, size=(int(n_fine), 2))
    for i in range(int(n_fine)):
        x = x + np.array([dW[i, 0], x[0] * dW[i, 1]])
        xs.append(x.copy())
    true_fine = np.stack(xs, axis=0)
    obs_idx = np.arange(obs_stride, true_fine.shape[0], obs_stride)
    x2 = true_fine[obs_idx, 1]
    ys = np.cos(p.omega * x2)[:, None] + rng.normal(
        0.0, p.meas_std, size=(obs_idx.shape[0], 1)
    )
    times = np.linspace(0.0, float(horizon), int(n_fine) + 1)
    return true_fine, ys.astype(np.float64), times


def init_estimate(p: LevyAreaParams) -> tuple[np.ndarray, np.ndarray]:
    m0 = np.array(p.x0, dtype=np.float64)
    s = float(p.prior_std)
    P0 = (s * s) * np.eye(2, dtype=np.float64)
    return m0, P0
