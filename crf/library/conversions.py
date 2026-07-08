"""SDE form conversions for use with the Stratonovich CRF kernels.

The CRF predict step integrates a Stratonovich SDE,

    dx = f_S(t, x) dt + G(t, x) o dW.

Many models are specified in Ito form instead,

    dx = f_I(t, x) dt + G(t, x) dW,

with the same diffusion matrix ``G``. The drifts are related by

    f_S = f_I - 0.5 * sum_j (G_{:,j} · ∇_x G_{:,j}),

where ``G_{:,j}`` is the j-th column of the diffusion matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import jax
from jax import numpy as jnp

from crf.filter.filtering import DiffusionFn, DriftFn

if TYPE_CHECKING:
    from crf.filter.types import GetCRFDynamics


def ito_to_stratonovich_drift(
    ito_drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
) -> DriftFn:
    """Convert an Ito drift to the equivalent Stratonovich drift.

    Args:
        ito_drift_fn: Ito drift ``(t, x) -> (x_dim,)``.
        diffusion_fn: Diffusion matrix ``(t, x) -> (x_dim, noise_dim)`` shared
            by both Ito and Stratonovich forms.

    Returns:
        Stratonovich drift function with the same signature.
    """

    def stratonovich_drift(t, x):
        x = jnp.asarray(x)
        f_ito = ito_drift_fn(t, x)
        g = diffusion_fn(t, x)
        x_dim, noise_dim = g.shape

        def flat_diffusion(x_):
            return diffusion_fn(t, x_).reshape(-1)

        jacobian = jax.jacfwd(flat_diffusion)(x).reshape(x_dim, noise_dim, x_dim)
        correction = 0.5 * jnp.einsum("ijk,kj->i", jacobian, g)
        return f_ito - correction

    return stratonovich_drift


def ito_to_stratonovich_dynamics(
    ito_drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
) -> Tuple[DriftFn, DiffusionFn]:
    """Convert Ito dynamics ``(f_I, G)`` to Stratonovich ``(f_S, G)``.

    The diffusion matrix is unchanged; only the drift is corrected.
    """
    return ito_to_stratonovich_drift(ito_drift_fn, diffusion_fn), diffusion_fn


def wrap_ito_dynamics(get_dynamics: GetCRFDynamics) -> GetCRFDynamics:
    """Wrap a ``get_dynamics`` callback that returns Ito ``(drift, diffusion)``.

    Returns a callback suitable for :func:`crf.build_filter`, whose drift is
    the Stratonovich correction of the supplied Ito drift.
    """

    def get_stratonovich_dynamics(model_inputs):
        ito_drift, diffusion = get_dynamics(model_inputs)
        return ito_to_stratonovich_dynamics(ito_drift, diffusion)

    return get_stratonovich_dynamics
