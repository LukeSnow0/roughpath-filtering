r"""Two-dimensional iterated-integral SDEs for the one-step and filtering benchmarks.

Itô dynamics with Brownian dimension 2 and non-commuting noise fields:

    dx¹ = dW¹
    dx² = g(x¹) dW²

The Itô and Stratonovich forms coincide: the correction ½ Σ_j (∂G_{·j}) G_{·j}
vanishes because g depends only on x¹, whose noise is W¹.

* ``levy_area``: g(x) = x from (0, 0), observed through y = cos(ω x²) + ε. Here
  x² = ∫ W¹ dW² is exactly a mixed iterated integral: the vector fields are
  affine and the stochastic Taylor expansion terminates (``levy_area_model.py``).
* ``sin_area``: g = sin from (1, 0), observed through y = sin(ω x²) + ε. The
  fields e₁, sin(x¹)e₂ generate a non-nilpotent Lie algebra, so every level of
  the expected signature enters the moments of x² and the degree-m weak rate
  O(δ^{(m+1)/2}) is sharp. Starting at x¹ = 1 avoids the degenerate point
  sin(0) = 0 (where every first-order Itô step predicts x² ≡ 0) and x¹ = π/4
  (where cos(2x¹) = 0 removes the leading Euler error term).

Scripts pick a model with ``--model {levy_area,sin_area}`` via
:func:`model_from_cli`; ``levy_area`` is the default, so existing commands keep
their outputs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

import levy_area_model


def _sin_second_moment(delta: float, a: float) -> float:
    r"""\(\mathbb{E}[(x^2_δ)^2] = \int_0^δ \mathbb{E}[\sin^2(a + W_s)]\,ds
    = δ/2 - \cos(2a)(1 - e^{-2δ})/4\)."""
    delta = float(delta)
    return 0.5 * delta - 0.25 * np.cos(2.0 * a) * (1.0 - np.exp(-2.0 * delta))


def _sin_char(delta: float, omega: float, a: float, n_modes: int = 40) -> float:
    r"""\(\mathbb{E}[\cos(ω x^2_δ)]\) for g = sin, from \(x_0 = (a, 0)\).

    Given W¹, x² is Gaussian with variance \(\int_0^δ \sin^2(a + W_s)\,ds\), so
    \(\mathbb{E}[\cos(ω x^2)] = \mathbb{E}[\exp(-\tfrac{ω^2}{2}\int_0^δ \sin^2(a + W_s)\,ds)]\).
    By Feynman–Kac this is u(δ, a) with \(u_t = \tfrac12 u_{xx} - \tfrac{ω^2}{2}\sin^2(x)\,u\),
    u(0, ·) = 1. The potential has period π, so the PDE is solved exactly in the
    Fourier basis \(e^{2ikx}\), \(|k| \le\) ``n_modes``, by a matrix exponential.
    """
    k = np.arange(-n_modes, n_modes + 1)
    size = k.size
    # ½∂²ₓ e^{2ikx} = -2k² e^{2ikx};  sin²x = ½ - ¼(e^{2ix} + e^{-2ix}).
    gen = np.diag(-2.0 * k**2 - omega**2 / 4.0) + (omega**2 / 8.0) * (
        np.eye(size, k=1) + np.eye(size, k=-1)
    )
    coef = expm(float(delta) * gen)[:, n_modes]
    return float(np.real(np.sum(coef * np.exp(2j * k * a))))


@dataclass(frozen=True)
class AreaSDE:
    """dx¹ = dW¹, dx² = g(x¹) dW² started from the point mass ``x0``."""

    name: str
    x0: tuple[float, float]
    g: Callable  # JAX-traceable diffusion coefficient of x² as a function of x¹
    g_np: Callable
    obs: Callable  # observation y = obs(ω x²) + ε (JAX)
    obs_np: Callable
    obs_label: str
    second_moment_fn: Callable[[float], float]
    char_fn: Callable[[float, float], float]

    def ito_drift(self, t, x):
        del t
        return jnp.zeros_like(x)

    def ito_L(self, t, x):
        del t
        return jnp.array([[1.0, 0.0], [0.0, self.g(x[0])]], dtype=x.dtype)

    @staticmethod
    def qc_fn(t, x):
        del t, x
        return jnp.eye(2)

    def exact_second_moment(self, delta: float) -> float:
        r"""\(\mathbb{E}[(x^2_δ)^2]\) from ``x0``."""
        return float(self.second_moment_fn(delta))

    def exact_char(self, delta: float, omega: float) -> float:
        r"""\(\mathbb{E}[\cos(ω x^2_δ)]\) from ``x0``."""
        return float(self.char_fn(delta, omega))

    def gaussian_char(self, delta: float, omega: float) -> float:
        r"""\(\mathbb{E}[\cos(ω Z)]\) for a Gaussian Z with the exact mean 0 and variance of x²."""
        return float(np.exp(-0.5 * omega**2 * self.exact_second_moment(delta)))

    def make_dynamics(self, omega: float, meas_std: float):
        """``(drift, L, Qc, obs_fn, R)`` for the filters, observing ``obs(ω x²)``."""

        def obs_fn(x):
            return self.obs(omega * x[1:2])

        R = jnp.array([[meas_std**2]], dtype=jnp.float64)
        return self.ito_drift, self.ito_L, self.qc_fn, obs_fn, R

    def init_estimate(self, prior_std: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
        """Near point-mass prior at ``x0`` (Cholesky-stable)."""
        return np.array(self.x0, dtype=np.float64), prior_std**2 * np.eye(2)

    def simulate(
        self,
        *,
        horizon: float,
        n_fine: int,
        omega: float,
        meas_std: float,
        seed: int,
        obs_stride: int,
    ):
        """Itô Euler–Maruyama twin with observations every ``obs_stride`` fine steps."""
        rng = np.random.default_rng(int(seed))
        dW = rng.normal(0.0, np.sqrt(float(horizon) / int(n_fine)), size=(int(n_fine), 2))
        x1 = np.cumsum(np.concatenate([[self.x0[0]], dW[:, 0]]))
        x2 = np.cumsum(np.concatenate([[self.x0[1]], self.g_np(x1[:-1]) * dW[:, 1]]))
        true_fine = np.stack([x1, x2], axis=1)
        obs_idx = np.arange(obs_stride, true_fine.shape[0], obs_stride)
        ys = self.obs_np(omega * true_fine[obs_idx, 1])[:, None] + rng.normal(
            0.0, meas_std, size=(obs_idx.shape[0], 1)
        )
        times = np.linspace(0.0, float(horizon), int(n_fine) + 1)
        return true_fine, ys.astype(np.float64), times


LEVY_AREA = AreaSDE(
    name="levy_area",
    x0=(0.0, 0.0),
    g=lambda x: x,
    g_np=lambda x: x,
    obs=jnp.cos,
    obs_np=np.cos,
    obs_label="cos",
    second_moment_fn=levy_area_model.exact_second_moment,
    char_fn=levy_area_model.exact_char,
)

_SIN_A = 1.0
SIN_AREA = AreaSDE(
    name="sin_area",
    x0=(_SIN_A, 0.0),
    g=jnp.sin,
    g_np=np.sin,
    obs=jnp.sin,
    obs_np=np.sin,
    obs_label="sin",
    second_moment_fn=lambda delta: _sin_second_moment(delta, _SIN_A),
    char_fn=lambda delta, omega: _sin_char(delta, omega, _SIN_A),
)

MODELS = {m.name: m for m in (LEVY_AREA, SIN_AREA)}


def model_from_cli(argv: list[str] | None = None) -> AreaSDE:
    """Read ``--model`` from the command line, ignoring every other argument."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=sorted(MODELS), default=LEVY_AREA.name)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return MODELS[args.model]
