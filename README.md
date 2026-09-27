# Wiener Space Cubature Kalman Filter

A **Wiener Space Cubature Kalman Filter (WSC-KF)** for continuous-discrete nonlinear state
estimation: a deterministic, Gaussian filtering method that propagates spatial
cubature nodes along Wiener-space cubature paths through a Stratonovich SDE,
without Jacobians or Monte Carlo sampling.

The installable library lives in [`wsckf/`](wsckf/). The [`experiments/`](experiments/)
directory contains benchmark and real-data scripts that compare the WSC-KF against
standard continuous-discrete estimators (CD-EKF, CD-UKF, CD-EnKF, bootstrap PF)
on shared observation sequences, reporting tracking accuracy, marginal
log-likelihood, and runtime.

For the mathematical motivation, see [EXPOSITION.md](EXPOSITION.md).

## Installation

Python **3.13+** is required (see `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments]"
```

The `[experiments]` extra adds **matplotlib** (and **openpyxl** for some
real-data loaders). For library use only:

```bash
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra experiments
```

Optional extras:

| Extra | Purpose |
| --- | --- |
| `experiments` | Plotting + spreadsheet deps for benchmark scripts |
| `dev` | `pytest` (also a uv dependency group, so `uv sync` installs it by default) |
| `metal` | Apple Silicon JAX MPS backend (`jax-mps`) |

**Apple Silicon / `metal` extra:** with `jax-mps` installed, JAX picks the MPS
backend by default, and MPS rejects float64 ("MLX does not support float64").
Set `JAX_PLATFORMS=cpu` when running the tests, examples or experiments.

**Dependencies:** [cuthbert](https://pypi.org/project/cuthbert/) and
**cuthbertlib** provide the generic filtering interface (`cuthbert.inference.Filter`,
`cuthbert.filter`); WSC-KF predict/update kernels live in this repo under
`wsckf`. Both are installed automatically by `pip install -e .`.
This project currently pins `cuthbert` / `cuthbertlib` to `>=0.0.12,<0.0.14`
(API change in `0.0.14`).

To hack on cuthbert locally, you can create optional symlinks in the repo root
(not tracked by git):

```bash
ln -s /path/to/cuthbert/cuthbert cuthbert
ln -s /path/to/cuthbert/cuthbertlib cuthbertlib
pip install -e /path/to/cuthbert   # editable install of the cuthbert project
```

The filter expects **float64**. The examples, the experiment scripts and
`tests/conftest.py` all call `jax.config.update("jax_enable_x64", True)`. Do the
same at the top of your own code, before creating any arrays (float32 inputs
fail inside the integrator's `lax.scan`), or set `JAX_ENABLE_X64=1` in the
environment.

Exact dependency versions are locked in [`uv.lock`](uv.lock).
Tests live under [`tests/`](tests/):

```bash
pip install -e ".[dev]"
pytest                        # JAX_PLATFORMS=cpu pytest if jax-mps is installed
```

## Using the library

Define three model callbacks, build a filter, and run it with cuthbert:

```python
import jax
import jax.numpy as jnp
from cuthbert import filter as run_filter
from jax import jit

from wsckf import build_filter

jax.config.update("jax_enable_x64", True)  # float64; set before creating any arrays

T = ys.shape[0]                   # number of observations
model_inputs = jnp.arange(T + 1)  # T + 1 entries; entry 0 feeds get_init_params

def get_init_params(model_inputs):
    return m0, chol_P0  # mean and Cholesky factor of initial covariance

def get_dynamics(model_inputs):
    # Stratonovich SDE on [0, dt]:
    # drift_fn(t, x) -> (x_dim,), diffusion_fn(t, x) -> (x_dim, noise_dim)
    return drift_fn, diffusion_fn

def get_observations(model_inputs):
    return observation_fn, chol_R, ys[model_inputs - 1]

filter_obj = build_filter(
    get_init_params,
    get_dynamics,
    get_observations,
    noise_dim=2,
    dt=0.1,
)

states = jit(run_filter, static_argnames=["filter_obj"])(filter_obj, model_inputs)
# `states` is a single WSCKFState whose fields are stacked over time
# (T + 1 entries, t0 first), so index the fields, not the state.
print(states.mean[-1], states.log_normalizing_constant[-1])
```

The Wiener paths are precomputed once in `build_filter` for the given `dt`, so a
filter assumes a fixed assimilation interval.

### Public API

| Import | Purpose |
| --- | --- |
| `wsckf.build_filter` | Build a `cuthbert.inference.Filter` for your model |
| `wsckf.WSCKFState` | Filter state: `mean`, `chol_cov`, `log_normalizing_constant`, … |
| `wsckf.GetInitParams`, `GetWSCKFDynamics`, `GetWSCKFObservations` | Callback type contracts |
| `wsckf.library.predict`, `update`, `precompute_cubature_paths`, … | Lower-level kernels (advanced) |
| `wsckf.library.ito_to_stratonovich_dynamics`, `wrap_ito_dynamics`, `ito_to_stratonovich_drift` | Ito → Stratonovich drift conversion |
| `wsckf.library.select_multidim_paths`, `lv_path_count`, `product_path_count`, `default_recombined_num_paths` | Path-method heuristics and path counts (advanced) |

The WSC-KF integrates a **Stratonovich** SDE internally. Pass ``sde_form="ito"`` if
your ``get_dynamics`` callback is written in Ito form. The drift correction is
then applied automatically inside `wsckf.build_filter`:

```python
filter_obj = build_filter(
    get_init_params,
    get_dynamics,       # Ito drift + diffusion, as you would write on paper
    get_observations,
    noise_dim=1,
    dt=0.1,
    sde_form="ito",
)
```

For Stratonovich models (or constant diffusion, where Ito = Stratonovich), omit
``sde_form`` or use the default ``sde_form="stratonovich"``.

Low-level helpers ``ito_to_stratonovich_dynamics`` and ``wrap_ito_dynamics`` remain
available in ``wsckf.library`` for advanced use. See
[`examples/ito_ou_1d.py`](examples/ito_ou_1d.py) and
[`examples/linear_gaussian_1d.py`](examples/linear_gaussian_1d.py).

See [`wsckf/filter/README.md`](wsckf/filter/README.md) and
[`wsckf/library/README.md`](wsckf/library/README.md) for detail.

**Runnable example:** a self-contained 1D Ornstein–Uhlenbeck script lives in
[`examples/linear_gaussian_1d.py`](examples/linear_gaussian_1d.py):

```bash
pip install -e .
python examples/linear_gaussian_1d.py
```

The WSC-KF maintains a Gaussian posterior (mean + Cholesky covariance). Its predict
step propagates spatial cubature nodes along deterministic Wiener-space cubature
paths; the update is a square-root sigma-point Kalman correction. In
`build_filter`, `cubature_order` (*k*, default 2) is the number of
macro-segments and `lyons_degree` (*m* ∈ {3, 5}, default 5) is the
Lyons–Victoir degree. Wiener path construction defaults to
``multidim_paths="auto"``:

- **`auto`** (default): for ``d=2`` with ``lyons_degree=5``, explicit multi-D
  Lyons–Victoir (``lv``) when it needs at most 512 paths, else ``recombined``;
  ``recombined`` for ``d≥3`` with ``lyons_degree=5``; ``product`` for ``d>1``
  with ``lyons_degree=3``. 1D always uses the classical 1D formulae (``3^k``
  paths for m=5, ``2^k`` for m=3).
- **`lv`** — explicit Lyons–Victoir multi-D paths (Lyons & Victoir 2004,
  Theorem 5.9): a fixed path with the required log-signature, scaled by each
  Gauss–Hermite node and paired with its coordinate reversal
  (``(2·3^d)^k`` paths). Degree 5 is available for ``d ≤ 2``; degree 3 is the
  Stroud rule for any ``d``.
- **`product`** — tensor product of 1D Lyons paths (no mixed area tensors)
- **`recombined`** — recursive KLV expansion with high-order recombination
  (Snow & Krishnamurthy, 2025); path budget defaults via
  ``default_recombined_num_paths`` when ``num_paths=None``

To override, pass `multidim_paths="lv" | "product" | "recombined"` (and
`num_paths` for `recombined`). `select_multidim_paths`, `lv_path_count` and
`product_path_count` in `wsckf.library` report what `auto` would pick and how
many paths each method uses.

## Experiments

After `pip install -e ".[experiments]"` (or `uv sync --extra experiments`),
run benchmarks from the repo root. The scripts are standalone (not a package):
each adds the repo root and `experiments/` to `sys.path` itself, then imports
the library (`wsckf`) and the shared baselines (`cd_dynamax_mirror`).

### Core synthetic benchmarks

```bash
python experiments/mfts_test/run_comparison.py
python experiments/l63_test/run_comparison.py
```

#### Modified Four-Tank System (MFTS)

State `x = [m₁, m₂, m₃, m₄, F₃, F₄]ᵀ`: four tank masses [g] (ρ = 1) plus two
Ornstein–Uhlenbeck disturbance flows [cm³/s], driven by known pump inputs
`F₁, F₂`. The benchmark follows Nielsen et al. (2022), Sec. IV
([arXiv:2205.02730](https://arxiv.org/abs/2205.02730)).

```
dx = f(t, x) dt + L dW,     L ∈ ℝ^{6×2}
y_k = [h₁, h₂, h₃, h₄]ᵀ + v_k,   hᵢ = βᵢ·mᵢ,   v_k ~ N(0, meas_std²·I₄)
```

Truth simulation steps the nominal disturbance flow `F̄` through
100 → 200 → 300 cm³/s over a 30-minute horizon (120 observations); estimators
assume a constant `F̄ = 150`. Metrics follow Nielsen eq. (51): **MAPEx** on tank
masses and **MAPEd** on disturbances.

```bash
python experiments/mfts_test/run_comparison.py
python experiments/mfts_test/run_comparison.py --seed 42 --cubature-order 3
```

| Flag | Description |
| --- | --- |
| `--seed` | RNG seed for truth simulation and stochastic filters (default 100) |
| `--cubature-order` | WSC-KF Lyons macro-segment count *k* (default 3) |
| `--solver` | Integrator for the CD baselines: `rk4` (default, matches WSC-KF) or `diffrax` |

Outputs: `experiments/mfts_test/output/` (tank heights, disturbance flows, MAPE
bars, cumulative log-likelihood, runtime). Related scripts also include Pareto
budget / interval sweeps and an `intaset_real/` real-data subfolder. MFTS
plotting scripts write both vector PDF and 450-dpi PNG versions at final 90 mm
single-column or 190 mm full-width manuscript sizes.

#### Lorenz 63 (L63)

Chaotic-attractor tracking from a single noisy observation channel
(`x₁` only), in the spirit of cd-dynamax `l63_mech_x1`:

```
dx = f(x) dt + q dW,   f(x) = [ σ(x₂−x₁),  ρx₁−x₂−x₁x₃,  −βx₃+x₁x₂ ]
y_k = x₁ + v_k,        v_k ~ N(0, obs_std²)
```

with σ = 10, ρ = 28, β = 8/3, q = 1, `obs_std` = 0.3 and assimilation
interval 0.35. The WSC-KF uses `multidim_paths="recombined"` here (`d = 3`).

```bash
python experiments/l63_test/run_comparison.py
python experiments/l63_test/run_comparison.py --cubature-order 3 --solver diffrax
```

| Flag | Description |
| --- | --- |
| `--solver` | Integrator for the CD baselines: `rk4` (default, matches WSC-KF) or `diffrax` |
| `--cubature-order` | WSC-KF Lyons macro-segment count *k* (default 2) |
| `--num-paths` | Post-recombination path budget (default `default_recombined_num_paths(3, k)`) |

Outputs: `experiments/l63_test/output/` (state trajectories, per-component RMSE,
cumulative log-likelihood, runtime). Also see `run_pareto_budget_smoke.py` and
`run_rmse_vs_interval.py`.

#### Heston (stochastic volatility)

Simulated twin with noisy log-price observations and latent variance
(Brownian dim `d = 2`). Main entry:

```bash
uv run --extra experiments python experiments/heston_test/run_pareto_budget.py
```

Outputs under `experiments/heston_test/output/` (budget vs RMSE Pareto curves
and CSVs). Overlay helper: `plot_pareto_L_overlay.py`.

`heston_test/` also holds the Lévy-area SDE experiments (`levy_area_*.py`) and
the weak-error scripts. Their archived CSVs and manuscript figures live in
[`experiments/numerics_section/`](experiments/numerics_section/); see that
README to rebuild.

### Path-construction diagnostics

[`experiments/multiplicative_noise_test/`](experiments/multiplicative_noise_test/)
isolates cases where product cubature and multi-D Lyons–Victoir paths disagree
(non-commuting noise / Lévy area), plus weak-error and diffusion-gap scripts:

```bash
uv run --extra experiments python \
    experiments/multiplicative_noise_test/compare_product_vs_lv.py
```

### Real-data experiments

| Directory | Domain | Notes |
| --- | --- | --- |
| [`experiments/AZT1D/`](experiments/AZT1D/) | T1D CGM + insulin pump | Bergman-style Ito SDE; see [`AZT1D/README.md`](experiments/AZT1D/README.md) |
| [`experiments/movement_test/`](experiments/movement_test/) | Dabbling-duck GPS | OU-velocity SDE with speed-dependent diffusion; see [`movement_test/README.md`](experiments/movement_test/README.md) |

Data folders for these experiments are typically gitignored; follow the
per-experiment READMEs for download and layout.

### Presentation figures

[`experiments/paths_display/`](experiments/paths_display/) — cubature-path and
algorithm figures for talks / exposition.

## Methods compared

| Method | Description |
| --- | --- |
| **CD-EKF** | Continuous-discrete extended Kalman filter |
| **CD-UKF** | Continuous-discrete additive-noise unscented Kalman filter (α = √3, β = 2, κ = 1) |
| **CD-EnKF** | Perturbed-observation ensemble Kalman filter |
| **PF** | Bootstrap particle filter |
| **WSC-KF** | Wiener Space Cubature Kalman Filter (this work) |

Baseline filters live in [`experiments/cd_dynamax_mirror/filters.py`](experiments/cd_dynamax_mirror/filters.py):
faithful, dependency-light reimplementations of the
[hd-UQ/cd_dynamax](https://github.com/hd-UQ/cd_dynamax) continuous-discrete
nonlinear-Gaussian filters that avoid its TensorFlow Probability dependency.
Each takes a `solver` argument. `solver="diffrax"` (the function default)
matches cd_dynamax: Dopri5 for ODEs, Heun + VirtualBrownianTree for SDEs.
`solver="rk4"` uses fixed-step RK4 to match the WSC-KF integrator for fair
wall-clock timing. The MFTS and L63 comparison scripts default to
`--solver rk4`.

The baselines treat models as **Itô** (EnKF and PF default to `sde_form="ito"`,
and the EKF/UKF moment ODEs are Itô). The WSC-KF defaults to Stratonovich, so
pass `sde_form="ito"` to `build_filter` when comparing on state-dependent
diffusion. With constant diffusion the two forms coincide.

On MFTS, the CD-EKF/CD-UKF use a **moment-ODE** predict (`state_order="first"`);
on L63 they use a **flow-map** predict (`state_order="flowmap"`), because the
moment-ODE variants are unstable on that chaotic, sparsely observed problem.

## Repository layout

```
cubature-filtering/
├── wsckf/                         # Installable library
│   ├── __init__.py              # Public API: build_filter, WSCKFState, types
│   ├── library/                 # Wiener cubature paths + Ito/Stratonovich helpers
│   │   ├── cubature_paths.py
│   │   └── conversions.py
│   └── filter/                  # High-level WSC-KF + predict/update kernels
│       ├── cubature_filter.py
│       ├── filtering.py
│       └── types.py
├── examples/                    # Minimal usage scripts
│   ├── linear_gaussian_1d.py
│   └── ito_ou_1d.py
├── experiments/                 # Benchmarks and real-data studies
│   ├── cd_dynamax_mirror/       # CD-EKF, CD-UKF, CD-EnKF, bootstrap PF
│   ├── mfts_test/               # Nielsen four-tank benchmark
│   ├── l63_test/                # Lorenz-63 tracking benchmark
│   ├── heston_test/             # Heston SV twin, Lévy-area SDE numerics
│   ├── numerics_section/        # Archived CSVs + manuscript figures
│   ├── multiplicative_noise_test/  # Product vs LV path diagnostics
│   ├── AZT1D/                   # Real CGM / insulin filtering
│   ├── movement_test/           # Real GPS movement filtering
│   └── paths_display/           # Presentation figures
├── tests/                       # pytest suite
├── EXPOSITION.md                # Mathematical overview
├── LICENSE                      # MIT
├── pyproject.toml
└── uv.lock
```

## License

This project is licensed under the [MIT License](LICENSE).

## References

Nielsen, M. K., et al. *State Estimation Methods for Continuous-Discrete Nonlinear
Systems Involving Stochastic Differential Equations.* 2022.
[arXiv:2205.02730](https://arxiv.org/abs/2205.02730).

Lyons, T., & Victoir, N. *Cubature on Wiener space.* 2004.

Arasaratnam, I., & Haykin, S. *Cubature Kalman Filters.* 2009.
