# Cubature Recursive Filter

A **Cubature Recursive Filter (CRF)** for continuous-discrete nonlinear state
estimation: a deterministic, Gaussian filtering method that propagates spatial
cubature nodes along Wiener-space cubature paths through a Stratonovich SDE,
without Jacobians or Monte Carlo sampling.

The installable library lives in [`crf/`](crf/). The [`experiments/`](experiments/)
directory contains benchmark and real-data scripts that compare the CRF against
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
| `dev` | `pytest` |
| `metal` | Apple Silicon JAX MPS backend (`jax-mps`) |

**Dependencies:** [cuthbert](https://pypi.org/project/cuthbert/) and
**cuthbertlib** provide the generic filtering interface (`cuthbert.inference.Filter`,
`cuthbert.filter`); CRF predict/update kernels live in this repo under
`crf`. Both are installed automatically by `pip install -e .`.
This project currently pins `cuthbert` / `cuthbertlib` to `>=0.0.12,<0.0.14`
(API change in `0.0.14`).

To hack on cuthbert locally, you can create optional symlinks in the repo root
(not tracked by git):

```bash
ln -s /path/to/cuthbert/cuthbert cuthbert
ln -s /path/to/cuthbert/cuthbertlib cuthbertlib
pip install -e /path/to/cuthbert   # editable install of the cuthbert project
```

JAX 64-bit precision is enabled inside the experiment scripts
(`jax.config.update("jax_enable_x64", True)`). You can also set
`JAX_ENABLE_X64=1` in the environment.

A pinned dependency tree is also available in [`requirements.txt`](requirements.txt).
Tests live under [`tests/`](tests/):

```bash
pip install -e ".[dev]"
pytest
```

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
| `crf.library.ito_to_stratonovich_dynamics`, `wrap_ito_dynamics` | Ito → Stratonovich drift conversion |
| `crf.library.select_multidim_paths`, `product_path_count` | Path-method heuristics (advanced) |

The CRF integrates a **Stratonovich** SDE internally. Pass ``sde_form="ito"`` if
your ``get_dynamics`` callback is written in Ito form — the drift correction is
applied automatically inside :func:`crf.build_filter`:

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
available in ``crf.library`` for advanced use. See
[`examples/ito_ou_1d.py`](examples/ito_ou_1d.py) and
[`examples/linear_gaussian_1d.py`](examples/linear_gaussian_1d.py).

See [`crf/filter/README.md`](crf/filter/README.md) and
[`crf/library/README.md`](crf/library/README.md) for detail.

**Runnable example:** a self-contained 1D Ornstein–Uhlenbeck script lives in
[`examples/linear_gaussian_1d.py`](examples/linear_gaussian_1d.py):

```bash
pip install -e .
python examples/linear_gaussian_1d.py
```

The CRF maintains a Gaussian posterior (mean + Cholesky covariance). Its predict
step propagates spatial cubature nodes along deterministic Wiener-space cubature
paths; the update is a square-root sigma-point Kalman correction. Wiener path
construction defaults to ``multidim_paths="auto"``:

- **`auto`** (default) — ``product`` when ``(3^k)^d <= 256``, else ``recombined``
- **`product`** — tensor-product Lyons paths (exact to degree 3)
- **`recombined`** — recursive KLV expansion with high-order recombination
  (Snow & Krishnamurthy, 2025); path budget defaults via
  ``default_recombined_num_paths`` when ``num_paths=None``

Override explicitly when benchmarking, e.g.
``from crf.library import select_multidim_paths, product_path_count``.

## Experiments

After `pip install -e ".[experiments]"` (or `uv sync --extra experiments`),
run benchmarks from the repo root. Experiment code imports the library (`crf`)
and shared baselines (`experiments.cd_dynamax_mirror`).

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
| `--cubature-order` | CRF Lyons macro-segment count *k* (default 3) |

Outputs: `experiments/mfts_test/output/` (tank heights, disturbance flows, MAPE
bars, cumulative log-likelihood, runtime). Related scripts also include Pareto
budget / interval sweeps and an `intaset_real/` real-data subfolder.

#### Lorenz 63 (L63)

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
there to explore variants. Also see `run_pareto_budget_smoke.py` and
`run_rmse_vs_interval.py`.

#### Heston (stochastic volatility)

Simulated twin with noisy log-price observations and latent variance
(Brownian dim `d = 2`). Main entry:

```bash
uv run --extra experiments python experiments/heston_test/run_pareto_budget.py
```

Outputs under `experiments/heston_test/output/` (budget vs RMSE Pareto curves
and CSVs). Overlay helper: `plot_pareto_L_overlay.py`.

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
│   ├── library/                 # Wiener cubature paths + Ito/Stratonovich helpers
│   │   ├── cubature_paths.py
│   │   └── conversions.py
│   └── filter/                  # High-level CRF + predict/update kernels
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
│   ├── heston_test/             # Heston SV twin / Pareto budget
│   ├── multiplicative_noise_test/  # Product vs LV path diagnostics
│   ├── AZT1D/                   # Real CGM / insulin filtering
│   ├── movement_test/           # Real GPS movement filtering
│   └── paths_display/           # Presentation figures
├── tests/                       # pytest suite
├── EXPOSITION.md                # Mathematical overview
├── pyproject.toml
├── uv.lock
└── requirements.txt
```

## References

Nielsen, K., et al. *State Estimation Methods for Continuous-Discrete Nonlinear
Systems Involving Stochastic Differential Equations.* 2022.
[arXiv:2205.02730](https://arxiv.org/abs/2205.02730).

Lyons, T., & Victoir, N. *Cubature on Wiener space.* 2004.

Arasaratnam, S., & Haykin, S. *Cubature Kalman Filters.* 2009.
