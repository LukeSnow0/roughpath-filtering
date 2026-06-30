# Cubature Recursive Filter

A **Cubature Recursive Filter (CRF)** for continuous-discrete nonlinear state
estimation: a deterministic, Gaussian filtering method that propagates spatial
cubature nodes along Wiener-space cubature paths through a Stratonovich SDE,
without Jacobians or Monte Carlo sampling.

The installable library lives in [`crf/`](crf/). The [`experiments/`](experiments/)
directory contains benchmark scripts that compare the CRF against standard
continuous-discrete estimators (CD-EKF, CD-UKF, CD-EnKF, bootstrap PF) on shared
observation sequences, reporting tracking accuracy, marginal log-likelihood, and
runtime.

For the mathematical motivation, see [EXPOSITION.md](EXPOSITION.md).

## Installation

Python **3.13+** is required (see `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments]"
```

The `[experiments]` extra adds **matplotlib** for benchmark plots. For library
use only:

```bash
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra experiments
```

**Dependencies:** [cuthbert](https://pypi.org/project/cuthbert/) and
**cuthbertlib** provide the generic filtering interface (`cuthbert.inference.Filter`,
`cuthbert.filter`); CRF predict/update kernels live in this repo under
`crf.library`.

For local development against a sibling cuthbert checkout, optional symlinks at
`cuthbert/` and `cuthbertlib/` may point to that source tree. They are not
required for a normal install from PyPI or `pip install -e .`.

JAX 64-bit precision is enabled inside the experiment scripts
(`jax.config.update("jax_enable_x64", True)`). You can also set
`JAX_ENABLE_X64=1` in the environment.

A pinned dependency tree is also available in [`requirements.txt`](requirements.txt).

## Using the library

Define three model callbacks, build a filter, and run it with cuthbert:

```python
import jax.numpy as jnp
from cuthbert import filter as run_filter

from crf import (
    CRFState,
    GetCRFDynamics,
    GetCRFObservations,
    GetInitParams,
    build_filter,
)

def get_init_params(model_inputs):
    return m0, chol_P0  # mean and Cholesky factor of initial covariance

def get_dynamics(model_inputs):
    return drift_fn, diffusion_fn  # Stratonovich SDE on [0, dt]

def get_observations(model_inputs):
    return observation_fn, chol_R, y

filter_obj = build_filter(
    get_init_params,
    get_dynamics,
    get_observations,
    noise_dim=2,
    dt=0.1,
    multidim_paths="recombined",
)

states = run_filter(filter_obj, model_inputs)
final: CRFState = states[-1]
print(final.mean, final.log_normalizing_constant)
```

### Public API

| Import | Purpose |
| --- | --- |
| `crf.build_filter` | Build a `cuthbert.inference.Filter` for your model |
| `crf.CRFState` | Filter state: `mean`, `chol_cov`, `log_normalizing_constant`, … |
| `crf.GetInitParams`, `GetCRFDynamics`, `GetCRFObservations` | Callback type contracts |
| `crf.library.predict`, `update`, `precompute_cubature_paths`, … | Lower-level kernels (advanced) |

See [`crf/filter/README.md`](crf/filter/README.md) and
[`crf/library/README.md`](crf/library/README.md) for detail.

The CRF maintains a Gaussian posterior (mean + Cholesky covariance). Its predict
step propagates spatial cubature nodes along deterministic Wiener-space cubature
paths; the update is a square-root sigma-point Kalman correction. Multidimensional
path construction is selected via `multidim_paths`:

- **`product`** — tensor-product Lyons paths (exact to degree 3)
- **`rademacher`** — cheap, scalable candidate paths
- **`recombined`** — recursive KLV expansion with high-order recombination
  (Snow & Krishnamurthy, 2025)

## Experiments

After `pip install -e ".[experiments]"`, run benchmarks from the repo root:

```bash
python experiments/mfts_test/run_comparison.py
python experiments/l63_test/run_comparison.py
python experiments/scalable_ring_test/run_comparison.py
```

Experiment code imports the library (`crf`) and shared baselines
(`experiments.cd_dynamax_mirror`); no `sys.path` setup is required.

### Modified Four-Tank System (MFTS)

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
| `--cubature-order` | CRF Lyons macro-segment count *k* (default 3) |

Outputs: `experiments/mfts_test/output/` (tank heights, disturbance flows, MAPE
bars, cumulative log-likelihood, runtime).

### Lorenz 63 (L63)

Chaotic-attractor tracking from a single noisy observation channel
(`x₁` only), in the spirit of cd-dynamax `l63_mech_x1`:

```
dx = f(x) dt + σ dW,   f(x) = [ σ(x₂−x₁),  ρx₁−x₂−x₁x₃,  −βx₃+x₁x₂ ]
y_k = x₁ + v_k,        v_k ~ N(0, obs_noise_std²)
```

```bash
python experiments/l63_test/run_comparison.py
```

Outputs: `experiments/l63_test/output/` (state trajectories, per-component RMSE,
cumulative log-likelihood, runtime). Settings are fixed in the script; edit
there to explore variants.

### Scalable ring lattice

Ring-coupled nonlinear lattice with independently configurable state, Brownian,
and observation dimensions. The CRF uses **recombined** Wiener cubature paths;
baselines are CD-EKF, CD-UKF, CD-EnKF, and a bootstrap PF.

```bash
python experiments/scalable_ring_test/run_comparison.py
python experiments/scalable_ring_test/run_comparison.py --state-dim 24 --noise-dim 6 --obs-dim 8
python experiments/scalable_ring_test/run_scaling_sweep.py
python experiments/scalable_ring_test/run_scaling_sweep.py --noise-dims 1,2,3,4,5,6
```

Outputs: `experiments/scalable_ring_test/output/` and
`experiments/scalable_ring_test/output/scaling/` (CSV/JSON sweeps).

### Presentation plots

Wiener-path and filtering cartoons for talks:

```bash
pip install plotly   # not a core dependency
python experiments/paths_display/paths_display.py
```

Outputs: `experiments/paths_display/output/`.

## Methods compared

| Method | Description |
| --- | --- |
| **CD-EKF** | Continuous-discrete extended Kalman filter (Joseph form) |
| **CD-UKF** | Continuous-discrete unscented Kalman filter (stable UT) |
| **CD-EnKF** | Perturbed-observation ensemble Kalman filter |
| **PF** | Bootstrap particle filter |
| **CRF** | Cubature Recursive Filter (this work) |

Baseline filters live in [`experiments/cd_dynamax_mirror/filters.py`](experiments/cd_dynamax_mirror/filters.py):
faithful, dependency-light reimplementations of the
[hd-UQ/cd_dynamax](https://github.com/hd-UQ/cd_dynamax) continuous-discrete
nonlinear-Gaussian filters, using **diffrax** for SDE integration instead of
TensorFlow Probability.

On MFTS, the CD-EKF/CD-UKF use a **moment-ODE** predict (`state_order="first"`);
on L63 they use a **flow-map** predict (`state_order="flowmap"`), because the
moment-ODE variants are unstable on that chaotic, sparsely observed problem.

## Repository layout

```
cubature-filtering/
├── crf/                         # Installable library
│   ├── __init__.py              # Public API: build_filter, CRFState, types
│   ├── library/                 # Wiener cubature paths + predict/update kernels
│   │   ├── cubature_paths.py
│   │   └── filtering.py
│   └── filter/                  # High-level CRF wired into cuthbert's Filter API
│       ├── cubature_filter.py
│       └── types.py
├── experiments/                 # Benchmarks (installed in editable dev mode)
│   ├── cd_dynamax_mirror/       # CD-EKF, CD-UKF, CD-EnKF, bootstrap PF
│   ├── mfts_test/               # Nielsen four-tank benchmark
│   ├── l63_test/                # Lorenz-63 tracking benchmark
│   ├── scalable_ring_test/      # Ring lattice + scaling sweeps
│   └── paths_display/           # path figures
├── cuthbert/                    # Optional symlink → local cuthbert checkout
├── cuthbertlib/                 # Optional symlink → local cuthbertlib checkout
├── EXPOSITION.md                # Mathematical overview
├── pyproject.toml
└── requirements.txt
```

## References

Nielsen, K., et al. *State Estimation Methods for Continuous-Discrete Nonlinear
Systems Involving Stochastic Differential Equations.* 2022.
[arXiv:2205.02730](https://arxiv.org/abs/2205.02730).

Lyons, T., & Victoir, N. *Cubature on Wiener space.* 2004.

Arasaratnam, S., & Haykin, S. *Cubature Kalman Filters.* 2009.
