# Numerics figures

Archived CSVs and publication figures for the Lévy-area SDE, the sin(x¹)
area SDE, and the Heston weak-error experiments. The scripts that produce the
CSVs live under [`experiments/heston_test/`](../heston_test/); the two area SDEs
are defined in `experiments/heston_test/area_models.py` and selected with
`--model {levy_area,sin_area}`.

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
    sin_area_weak_error.csv
    sin_area_moment_pareto.csv
    sin_area_char_pareto.csv
    sin_area_filter_pareto.csv
  figures/
    fig_levy_area_weak_error.{png,pdf}
    fig_levy_area_split.{png,pdf}
    fig_levy_area_moment.{png,pdf}
    fig_levy_area_char.{png,pdf}
    fig_levy_area_filter.{png,pdf}
    fig_levy_area_weak_error_local.{png,pdf}
    fig_heston_weak_error.{png,pdf}
    fig_sin_area_weak_error.{png,pdf}
    fig_sin_area_filter.{png,pdf}
    tab_sin_area_one_step.tex
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

**sin(x¹) area SDE.** dx¹ = dW¹, dx² = sin(x¹) dW² from the point mass
(1, 0). The vector fields generate a non-nilpotent Lie algebra, so every level
of the expected signature enters the moments of x² and the degree-m rates are
sharp; on the Lévy-area SDE the Taylor expansion terminates, which hides the
τ³ term. Exact references: E[(x²)²] = τ/2 − cos(2)(1 − e^{−2τ})/4, and
E[cos(ω x²)] from the Feynman–Kac PDE solved in a Fourier basis.

- `fig_sin_area_weak_error`: one-step errors of (a) `E[cos(ω x²)]` and
  (b) `E[(x²)²]` versus τ (ω=3). Legend slopes are fitted for τ ≤ 0.2.
  CD-EKF and CD-CKF coincide from a point mass; in (b) WSC m=3 also coincides
  with them.
- `fig_sin_area_filter`: mixture predictive NLL of `y = sin(ω x²) + ε`
  (R^{1/2}=0.25, K=8, Δ=0.5, 60 tracks) versus wall-clock. The odd observation
  breaks the x² → −x² symmetry, so latent RMSE is also informative (CSV only).
- `tab_sin_area_one_step.tex`: one-step errors at τ = 0.5 (Table 1 analogue);
  EnKF entries are mean ± standard error over seeds.

## Rebuild

```bash
.venv/bin/python experiments/heston_test/levy_area_weak_error.py
.venv/bin/python experiments/heston_test/levy_area_moment_pareto.py
.venv/bin/python experiments/heston_test/levy_area_char_pareto.py
.venv/bin/python experiments/heston_test/levy_area_filter_pareto.py \
    --seeds $(seq -s, 100 159) --enkf 5,10,25,50,100,500 --pf 5,10,25,50,100,500
.venv/bin/python experiments/heston_test/weak_error_delta_scaling.py

.venv/bin/python experiments/heston_test/levy_area_weak_error.py --model sin_area \
    --delta-min 0.02 --delta-max 0.8 --n-deltas 16 --fit-dmin 0.02 --fit-dmax 0.2
.venv/bin/python experiments/heston_test/levy_area_moment_pareto.py --model sin_area
.venv/bin/python experiments/heston_test/levy_area_char_pareto.py --model sin_area
.venv/bin/python experiments/heston_test/levy_area_filter_pareto.py --model sin_area \
    --seeds $(seq -s, 100 159) --enkf 5,10,25,50,100,500 --pf 5,10,25,50,100,500

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
for f in weak_error moment_pareto char_pareto filter_pareto; do
  cp experiments/heston_test/output/sin_area/sin_area_$f.csv \
     experiments/numerics_section/data/sin_area_$f.csv
done

.venv/bin/python experiments/numerics_section/assemble_figures.py
```
