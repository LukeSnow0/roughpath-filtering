"""Implements the high-level Cubature Recursive Filter (CRF).

The CRF maintains a Gaussian filtering distribution (mean + Cholesky covariance).
Its predict step propagates spatial cubature nodes along a small set of
deterministic Wiener-space cubature paths through the nonlinear Stratonovich SDE
(no Jacobian, no Monte-Carlo), and its update is a square-root sigma-point Kalman
update. See `crf.library` for the underlying kernels.

References:
    Lyons & Victoir (2004), *Cubature on Wiener space*.
    Litterer & Lyons (2012), *High order recombination ... cubature on Wiener space*.
"""

from functools import partial
from typing import Literal, NamedTuple, Optional, Tuple

from jax import eval_shape, tree
from jax import numpy as jnp

from crf.filter.types import GetCRFDynamics, GetCRFObservations, GetInitParams
from cuthbert.inference import Filter
from cuthbert.utils import dummy_tree_like
from crf.library import predict, update, precompute_cubature_paths
from crf.library.conversions import wrap_ito_dynamics
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class CRFState(NamedTuple):
    """Cubature Recursive Filter state."""

    mean: Array
    chol_cov: Array
    model_inputs: ArrayTree
    log_normalizing_constant: ScalarArray

    @property
    def cov(self) -> Array:
        """Filtering covariance from its (generalised) Cholesky factor."""
        return self.chol_cov @ jnp.swapaxes(self.chol_cov, -1, -2)


def build_filter(
    get_init_params: GetInitParams,
    get_dynamics: GetCRFDynamics,
    get_observations: GetCRFObservations,
    noise_dim: int,
    dt: float,
    integration_dt: float = 0.01,
    cubature_order: int = 2,
    cubature_gamma: float = 0.6,
    num_paths: Optional[int] = None,
    multidim_paths: Literal["product", "recombined", "auto"] = "auto",
    seed: int = 0,
    candidate_factor: float = 8.0,
    recombination_degree: int = 3,
    sde_form: Literal["stratonovich", "ito"] = "stratonovich",
    checkpoint_steps: bool = False,
    lyons_degree: int = 5,
    wiener_paths: Optional[Tuple[Array, Array, Array]] = None,
) -> Filter:
    """Builds a Cubature Recursive Filter object.

    The Wiener cubature paths depend only on ``dt``, ``noise_dim`` and the
    cubature settings, so they are precomputed once here (host-side) and closed
    over by the filtering kernels, entering the traced computation as constants.
    This assumes a fixed assimilation interval ``dt``.

    Args:
        get_init_params: Function to get ``(m0, chol_P0)`` from model inputs.
        get_dynamics: Function to get ``(drift_fn, diffusion_fn)`` from model
            inputs. If ``sde_form="ito"``, the drift is interpreted in Ito form
            and converted to Stratonovich automatically. If
            ``sde_form="stratonovich"`` (default), the drift is used as-is.
        get_observations: Function to get ``(observation_fn, chol_R, y)`` from model inputs.
        noise_dim: Dimension of the Brownian driver.
        dt: Length of one assimilation interval.
        integration_dt: Inner integration step size for the Stratonovich solver.
        cubature_order: Cubature order ``k`` (number of macro segments).
        cubature_gamma: Macro-segment time-grid concentration exponent.
        num_paths: Number of Wiener paths (meaning depends on ``multidim_paths``).
            If ``None``, a method-specific default is used.
        multidim_paths: Multidimensional path construction:
            ``"auto"`` (default: ``product`` when ``(3^k)^d <= 256``, else
            ``recombined``),
            ``"product"`` (tensor-product Lyons), or
            ``"recombined"`` (moment-matched sparse measure).
        seed: Seed for Rademacher candidate generation.
        candidate_factor: Candidate-pool scaling for recombination.
        recombination_degree: Target moment degree for recombination (not Lyons ``m``).
        sde_form: Whether ``get_dynamics`` supplies Ito or Stratonovich drift.
            The CRF always integrates in Stratonovich form; Ito drifts are
            corrected internally. Constant diffusion needs no correction.
        checkpoint_steps: If ``True``, rematerialize each inner RK4 step under
            reverse-mode AD. Frozen cubature Wiener paths make the predict step a
            pure ODE, so this enables memory-efficient ODE adjoints through the
            filter likelihood (see :func:`crf.library.integrate_stratonovich_cde`).
        lyons_degree: Explicit Lyons–Victoir degree ``m ∈ {3, 5}``. Default
            ``5`` matches the repo's standard three-path 1D formula.
        wiener_paths: Optional precomputed ``(ts, path_values, path_weights)``
            arrays. When set, library cubature construction is skipped (used for
            experiment-side path builders such as explicit multi-D LV).

    Returns:
        Filter object for the CRF.

    Raises:
        ValueError: If ``noise_dim`` or ``dt`` is not positive, or ``sde_form``
            is not ``"stratonovich"`` or ``"ito"``.
    """
    if noise_dim < 1:
        raise ValueError("noise_dim must be at least 1.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if sde_form not in ("stratonovich", "ito"):
        raise ValueError('sde_form must be "stratonovich" or "ito".')

    if sde_form == "ito":
        get_dynamics = wrap_ito_dynamics(get_dynamics)

    if wiener_paths is not None:
        ts, path_values, path_weights = wiener_paths
        ts = jnp.asarray(ts)
        path_values = jnp.asarray(path_values)
        path_weights = jnp.asarray(path_weights)
    else:
        if multidim_paths not in ("auto", "product", "recombined"):
            raise ValueError(
                'multidim_paths must be "auto", "product", or "recombined".'
            )

        ts, path_values, path_weights = precompute_cubature_paths(
            horizon=dt,
            noise_dim=noise_dim,
            integration_dt=integration_dt,
            cubature_order=cubature_order,
            cubature_gamma=cubature_gamma,
            num_paths=num_paths,
            multidim_method=multidim_paths,
            seed=seed,
            candidate_factor=candidate_factor,
            recombination_degree=recombination_degree,
            lyons_degree=lyons_degree,
        )

    return Filter(
        init_prepare=partial(init_prepare, get_init_params=get_init_params),
        filter_prepare=partial(filter_prepare, get_init_params=get_init_params),
        filter_combine=partial(
            filter_combine,
            get_dynamics=get_dynamics,
            get_observations=get_observations,
            ts=ts,
            path_values=path_values,
            path_weights=path_weights,
            checkpoint_steps=checkpoint_steps,
        ),
        associative=False,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    key: KeyArray | None = None,
) -> CRFState:
    """Prepare the initial state for the CRF.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get ``(m0, chol_P0)`` from model inputs.
        key: JAX random key - not used (the CRF is deterministic).

    Returns:
        Initial CRF state.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    m0, chol_P0 = get_init_params(model_inputs)

    return CRFState(
        mean=m0,
        chol_cov=chol_P0,
        model_inputs=model_inputs,
        log_normalizing_constant=jnp.array(0.0),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    key: KeyArray | None = None,
) -> CRFState:
    """Prepare a state for a CRF step.

    Just passes through the model inputs with a dummy mean and covariance of the
    correct shape (inferred from ``get_init_params``).

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get ``(m0, chol_P0)`` from model inputs.
            Only used to infer the shape of the mean and covariance.
        key: JAX random key - not used (the CRF is deterministic).

    Returns:
        Prepared CRF state with dummy mean and covariance.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    dummy_mean_struct = eval_shape(lambda mi: get_init_params(mi)[0], model_inputs)
    dummy_mean = dummy_tree_like(dummy_mean_struct)
    dummy_chol_cov = dummy_tree_like(
        jnp.empty(dummy_mean.shape + dummy_mean.shape[-1:], dtype=dummy_mean.dtype)
    )

    return CRFState(
        mean=dummy_mean,
        chol_cov=dummy_chol_cov,
        model_inputs=model_inputs,
        log_normalizing_constant=jnp.array(0.0),
    )


def filter_combine(
    state_1: CRFState,
    state_2: CRFState,
    get_dynamics: GetCRFDynamics,
    get_observations: GetCRFObservations,
    ts: Array,
    path_values: Array,
    path_weights: Array,
    checkpoint_steps: bool = False,
) -> CRFState:
    """Combine the previous CRF state with the prepared state for the current step.

    Implements the CRF predict (cubature propagation) + update (square-root
    sigma-point Kalman) cycle.

    Args:
        state_1: CRF state from the previous time step.
        state_2: CRF state prepared for the current step (only ``model_inputs`` used).
        get_dynamics: Function to get ``(drift_fn, diffusion_fn)`` from model inputs.
        get_observations: Function to get ``(observation_fn, chol_R, y)`` from model inputs.
        ts: Integration time grid for one assimilation interval.
        path_values: Wiener cubature paths.
        path_weights: Wiener cubature weights.
        checkpoint_steps: Rematerialize each inner RK4 step under reverse-mode AD.

    Returns:
        Updated CRF state.
    """
    drift_fn, diffusion_fn = get_dynamics(state_2.model_inputs)
    predicted_mean, predicted_chol_cov, samples, weights = predict(
        state_1.mean,
        state_1.chol_cov,
        ts,
        path_values,
        path_weights,
        drift_fn,
        diffusion_fn,
        checkpoint_steps=checkpoint_steps,
    )

    observation_fn, chol_R, y = get_observations(state_2.model_inputs)
    updated_mean, updated_chol_cov, log_likelihood = update(
        predicted_mean,
        samples,
        weights,
        observation_fn,
        chol_R,
        y,
    )

    return CRFState(
        mean=updated_mean,
        chol_cov=updated_chol_cov,
        model_inputs=state_2.model_inputs,
        log_normalizing_constant=state_1.log_normalizing_constant + log_likelihood,
    )
