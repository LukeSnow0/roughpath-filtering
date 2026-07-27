# Movement filter-MLE: dabbling-duck GPS

Brief summary of the real-data experiment in `experiments/movement_test/`.

## Data

**Source.** Osborne et al. (2024), Dryad [doi:10.5061/dryad.1ns1rn93p](https://doi.org/10.5061/dryad.1ns1rn93p) (CC0): hourly GPS for mallards, green-winged teal, and American wigeon in Arkansas.

**Scale.** ~49k hourly fixes, **121 individuals**, 3 species; tracks from tens to ~2400 fixes. Loader picks one bird’s longest contiguous exact-hourly run, converts UTM `Final_X/Y` → mean-centred km.

**Default track.** `MALL 825 M` (mallard ♂), typically **N = 600** (~25 days) for MLE; extent ~95 × 115 km. Velocity is **latent**; only noisy position is observed (~30 m GPS sd in the model).

Download: see [`data/README.md`](data/README.md).

## Model

Continuous–discrete **OU-velocity** SDE with weak home-range spring and **speed-dependent diffusion**:

\[
\begin{aligned}
dx &= v_x\,dt, &
dy &= v_y\,dt, \\
dv_x &= \bigl(-\beta_v v_x - \omega^2(x-c_x)\bigr)\,dt + g(v)\,dW_1, \\
dv_y &= \bigl(-\beta_v v_y - \omega^2(y-c_y)\bigr)\,dt + g(v)\,dW_2,
\end{aligned}
\]

\[
g(v)=\sigma_0\sqrt{1+\|v\|^2/v_c^2},\qquad
z_k=(x,y)+\varepsilon_k,\ \varepsilon\sim\mathcal N(0,R).
\]

**Interpretable parameters** (joint MLE target):

| Symbol | Meaning | Link to SDE |
|--------|---------|-------------|
| \(\tau_v=1/\beta_v\) | velocity autocorrelation time (h) | mean-reversion |
| \(\sigma_v=\sigma_0/\sqrt{2\beta_v}\) | asymptotic per-axis velocity scale (km/h) under frozen \(g\equiv\sigma_0\) | \(\sigma_0=\sigma_v\sqrt{2/\tau_v}\) |

Fixed in fits: home-range \(\omega^2\), speed scale \(v_c\), measurement \(R\). Idealized process + **real** telemetry (not a twin experiment).

## Filtering & MLE

Maximize each filter’s marginal log-likelihood of the GPS series over \((\tau_v,\sigma_v)\) (Adam on \((\log\tau_v,\log\sigma_v)\); optional 2D grid).

| Filter | Forecast | Role in this study |
|--------|----------|--------------------|
| **CRF** | Deterministic Wiener cubature through the SDE | Primary; pathwise AD via checkpointed ODE adjoint |
| **CD-EnKF** | Particles through the SDE (RK4 or Diffrax) | Agrees with CRF on \(\hat\theta\); stochastic / ensemble likelihood |
| **CD-EKF** | Gaussian moment ODEs | Different optimum (closure bias under multiplicative \(g(v)\)) |
| CD-UKF / bootstrap PF | — | UKF often NaNs; PF degenerates on long tracks (not in joint harness) |

**Fair timing baseline:** match integrators — CRF vs **RK4** EnKF (similar wall time). **Diffrax** EnKF is ~10× slower; that is a deployment cost, not the scientific comparison.

## Main findings (N = 600, `MALL 825 M`)

- **CRF ≈ EnKF:** \(\hat\tau_v\sim 0.75\,\mathrm{h}\), \(\hat\sigma_v\sim 2.5\)–\(2.8\,\mathrm{km/h}\) (same high-likelihood ridge).
- **EKF differs:** shorter \(\tau_v\), larger \(\sigma_v\) / implied \(\sigma_0\) — moment closure mishandles state-dependent diffusion.
- No ground-truth \(\theta\); agreement of SDE-propagating filters is the claim, not a validated biological constant. Rank accuracy further via simulation recovery and held-out predictive loglik.

## Run

```bash
uv run --extra experiments python experiments/movement_test/run_movement_mle.py
uv run --extra experiments python experiments/movement_test/run_movement_mle.py \
    --n-obs 600 --mle-iters 40 --grid 10
uv run --extra experiments python experiments/movement_test/run_movement_mle.py \
    --timing-only --solver rk4   # or diffrax
```

## Outputs

`output/movement_loglik.png` (2D \(L(\tau_v,\sigma_v)\)), `movement_mle_trace.png`, `movement_track.png`, `movement_timing*.png`, `movement_summary.csv`.
