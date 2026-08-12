"""Provides types for WSC-KF callback functions."""

from typing import Protocol

from wsckf.filter.filtering import DiffusionFn, DriftFn, ObservationFn
from cuthbertlib.types import Array, ArrayTreeLike


class GetInitParams(Protocol):
    """Protocol for defining the initial Gaussian distribution."""

    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[Array, Array]:
        """Get initial parameters ``(m0, chol_P0)`` from model inputs.

        Args:
            model_inputs: Model inputs.

        Returns:
            Tuple ``(m0, chol_P0)`` with the initial mean and the Cholesky factor
            of the initial covariance.
        """
        ...


class GetWSCKFDynamics(Protocol):
    """Protocol for getting the Stratonovich SDE dynamics of the latent state."""

    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[DriftFn, DiffusionFn]:
        """Get drift and diffusion functions from model inputs.

        The latent state evolves as ``dx = drift(t, x) dt + diffusion(t, x) o dW``
        over one assimilation interval (relative time ``[0, dt]``).

        Args:
            model_inputs: Model inputs.

        Returns:
            Tuple ``(drift_fn, diffusion_fn)`` where ``drift_fn(t, x) -> (x_dim,)``
            and ``diffusion_fn(t, x) -> (x_dim, noise_dim)``.
        """
        ...


class GetWSCKFObservations(Protocol):
    """Protocol for getting observation function, chol_R, and observation vector.

    i.e. for state ``x_t``, ``y_t = h(x_t) + v_t`` with ``v_t ~ N(0, chol_R chol_R^T)``,
    return ``(h, chol_R, y)``.
    """

    def __call__(
        self, model_inputs: ArrayTreeLike
    ) -> tuple[ObservationFn, Array, Array]:
        """Get observation function, observation noise Cholesky factor, and observation.

        Args:
            model_inputs: Model inputs.

        Returns:
            Tuple ``(h, chol_R, y)`` with the observation function, the Cholesky
            factor of the observation noise covariance, and the observation vector.
        """
        ...
