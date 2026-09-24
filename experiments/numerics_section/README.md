# Numerics figures

Archived CSVs and publication figures for the Lévy-area SDE and Heston
weak-error experiments. The scripts that produce the CSVs live under
[`experiments/heston_test/`](../heston_test/).

```
experiments/numerics_section/
  assemble_figures.py
  data/
    levy_area_weak_error.csv
    levy_area_moment_pareto.csv
    levy_area_char_pareto.csv
    levy_area_filter_pareto.csv
    heston_weak_error.csv
    heston_filter_pareto.csv      # archived; not plotted by assemble_figures.py
  figures/
    fig_levy_area_weak_error.{png,pdf}
    fig_levy_area_split.{png,pdf}
    fig_levy_area_moment.{png,pdf}
    fig_levy_area_char.{png,pdf}
    fig_levy_area_filter.{png,pdf}
    fig_levy_area_weak_error_local.{png,pdf}
    fig_heston_weak_error.{png,pdf}
```

## Figures

**Lévy-area weak error** (`fig_levy_area_weak_error`). One-step error of
`E[cos(ω x²)]` from a Dirac law (`P0=0`, ω=3) versus forecast interval.
The exact value is `[cosh(ωΔ)]^{-1/2}`. CD-CKF and WSC `m=3` coincide:
both miss Lévy area and predict 1. WSC `m=5` cubatures the mixed
integrals and converges at a higher order. CD-CKF is not a one-step
IT-1.5 scheme of the driving noise, so it does not inherit the `m=5`
rate. The driving SDE is globally Lipschitz.

**Polynomial moment** (`fig_levy_area_moment`). One-step
`|E[(x²)²] − Δ²/2|`. TME-2 and WSC `m=5` are exact; EKF, CKF, SE-UKF,
and WSC `m=3` miss the area; EnKF is limited by Monte Carlo.

**Sequential cosine NLL** (`fig_levy_area_split`). Mixture predictive NLL
of `y = cos(ω x²) + ε` with `R^{1/2}=0.25`. EKF/CKF/SE-UKF sit near 9.2
nats/obs; TME-2 near 4.2; WSC interpolates toward EnKF/PF as the path
budget grows.

**Supporting plots.** `fig_levy_area_char` is the one-step characteristic
function error versus wall-clock. `fig_levy_area_filter` is the same
sequential NLL series as the split figure. `fig_heston_weak_error` is
Heston covariance weak error versus `Δ`. `heston_filter_pareto.csv` is
kept for reference and is not assembled into a figure here.

## Rebuild

```bash
.venv/bin/python experiments/heston_test/levy_area_weak_error.py
.venv/bin/python experiments/heston_test/levy_area_moment_pareto.py
.venv/bin/python experiments/heston_test/levy_area_char_pareto.py
.venv/bin/python experiments/heston_test/levy_area_filter_pareto.py
.venv/bin/python experiments/heston_test/weak_error_delta_scaling.py

cp experiments/heston_test/output/levy_area/levy_area_weak_error.csv \
   experiments/numerics_section/data/levy_area_weak_error.csv
cp experiments/heston_test/output/levy_area/levy_area_moment_pareto.csv \
   experiments/numerics_section/data/levy_area_moment_pareto.csv
cp experiments/heston_test/output/levy_area/levy_area_char_pareto.csv \
   experiments/numerics_section/data/levy_area_char_pareto.csv
cp experiments/heston_test/output/levy_area/levy_area_filter_pareto.csv \
   experiments/numerics_section/data/levy_area_filter_pareto.csv
cp experiments/heston_test/output/weak_error_delta_scaling_sv.csv \
   experiments/numerics_section/data/heston_weak_error.csv

.venv/bin/python experiments/numerics_section/assemble_figures.py
```
