# Cubature Recursive Filter Benchmarks

Benchmarks of continuous-discrete nonlinear filters built around a **Cubature
Recursive Filter (CRF)**: a deterministic, Gaussian filtering method that
propagates spatial cubature nodes along Wiener-space cubature paths through a
Stratonovich SDE, without Jacobians or Monte Carlo sampling.

Two experiments compare the CRF against standard continuous-discrete estimators
(CD-EKF, CD-UKF, CD-EnKF, bootstrap PF) on shared observation sequences,
reporting tracking accuracy, marginal log-likelihood, and runtime.

For the mathematical motivation, see [EXPOSITION.md](EXPOSITION.md).

## Experiments

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

Outputs are written to `experiments/mfts_test/output/` (tank heights,
disturbance flows, MAPE bars, cumulative log-likelihood, runtime).

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

Outputs are written to `experiments/l63_test/output/` (state trajectories, per-component
RMSE, cumulative log-likelihood, runtime). Settings (horizon, noise scales,
ensemble sizes) are fixed in the script; edit there to explore variants.

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

## Installation

Python **3.13+** is required (see `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The experiments depend on **[cuthbert](https://pypi.org/project/cuthbert/)** and
**cuthbertlib** for the generic filtering interface and CRF kernels. These are
listed in `requirements.txt`. For local development, this repo also provides
symlinks at `cuthbert/` and `cuthbertlib/` pointing to a sibling checkout of the
cuthbert source tree.

JAX 64-bit precision is enabled inside the experiment scripts
(`jax.config.update("jax_enable_x64", True)`). You can also set
`JAX_ENABLE_X64=1` in the environment.

## Repository layout

```
cubature-filtering/
├── crf/
│   ├── library/          # Wiener cubature paths + predict/update kernels
│   │   ├── cubature_paths.py
│   │   ├── filtering.py
│   │   └── paths_display.py
│   └── filter/           # High-level CRF wired into cuthbert's Filter API
│       └── cubature_filter.py
├── experiments/
│   ├── cd_dynamax_mirror/   # CD-EKF, CD-UKF, CD-EnKF, bootstrap PF
│   ├── mfts_test/           # Nielsen four-tank benchmark
│   │   ├── mfts_model.py
│   │   └── run_comparison.py
│   └── l63_test/            # Lorenz-63 tracking benchmark
│       └── run_comparison.py
├── cuthbert/             # Symlink → local cuthbert checkout (optional)
├── cuthbertlib/          # Symlink → local cuthbertlib checkout (optional)
├── EXPOSITION.md         # Mathematical overview of cubature filtering
├── pyproject.toml
└── requirements.txt
```

### Core library

- **`crf.library`** — Host-side Wiener cubature path construction and JAX
  kernels for the CRF predict/update steps. See
  [`crf/library/README.md`](crf/library/README.md).
- **`crf.filter.cubature_filter`** — Builds a
  [`cuthbert.inference.Filter`](cuthbert/) object from model callbacks
  (`get_init_params`, `get_dynamics`, `get_observations`). See
  [`crf/filter/README.md`](crf/filter/README.md).

The CRF maintains a Gaussian posterior (mean + Cholesky covariance). Its predict
step propagates spatial cubature nodes along a small set of deterministic
Wiener-space cubature paths; the update is a square-root sigma-point Kalman
correction. Two multidimensional path constructions are available via
`multidim_paths`:

- **`product`** — tensor-product Lyons paths (exact to degree 3)
- **`recombined`** — recursive KLV expansion with high-order recombination
  (Snow & Krishnamurthy, 2025)

## References

Nielsen, K., et al. *State Estimation Methods for Continuous-Discrete Nonlinear
Systems Involving Stochastic Differential Equations.* 2022.
[arXiv:2205.02730](https://arxiv.org/abs/2205.02730).

Lyons, T., & Victoir, N. *Cubature on Wiener space.* 2004.

Arasaratnam, S., & Haykin, S. *Cubature Kalman Filters.* 2009.
