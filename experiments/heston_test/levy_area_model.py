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

The benchmark scripts use this model through ``area_models.LEVY_AREA``, which
also provides the simulator and filter dynamics.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def ito_drift(t, x):
    del t
    return jnp.zeros_like(x)


def ito_L(t, x):
    del t
    return jnp.array([[1.0, 0.0], [0.0, x[0]]], dtype=x.dtype)


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
