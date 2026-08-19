# AZT1D continuous–discrete glucose filtering

Real-data experiment on the [AZT1D](https://data.mendeley.com/datasets/gk9m674wcx/1)
(T1D CGM + insulin pump) release.

## Data

Place the Mendeley folder at `experiments/AZT1D/AZT1D 2025/` (gitignored).
CSVs live under `AZT1D 2025/CGM Records/Subject {k}/`.

## Model

Minimal Bergman-style Itô SDE with known insulin / carb inputs from pump logs,
state `[G, X]`, multiplicative `√G` glucose noise, CGM observation of `G`
(`glucose_model.py`). Brownian dim `d = 2`.

## EnKF MLE of \((p_3,\sigma_G)\)

```bash
uv run --extra experiments python experiments/AZT1D/run_glucose_mle.py
uv run --extra experiments python experiments/AZT1D/run_glucose_mle.py \
    --subject 1 --n-obs 2016 --mle-iters 30 --grid 8
```

Outputs under `output/`: `calibrated_params_enkf.json`, `glucose_summary.csv`,
and PDF/PNG pairs for `glucose_loglik`, `glucose_mle_trace`, `glucose_track`,
and `glucose_heldout_compare`.

## Assimilation-interval Pareto (Δt = 5 min)

Fixed calendar span (~7 d) at EnKF-calibrated \(\hat\theta\):

```bash
uv run --extra experiments python experiments/AZT1D/run_pareto_assimilation.py \
    --deltas 5 --seeds 0,1,2 --span-min 10080 \
    --wsckf-max-paths 4 --enkf 25,50,100,500 --pf 25,50,100,500
```

Kept figure: `output/pareto_assimilation_dt5.{pdf,png}` (plus backing
`pareto_assimilation.csv` / `_raw.csv`).

## Cohort (per-subject \(\hat\theta_i\))

Calibrate EnKF \(\hat\theta_i\) on each patient, then bakeoff filters at that
\(\theta_i\); aggregate over subjects:

```bash
uv run --extra experiments python experiments/AZT1D/run_cohort_calibrated.py \
    --n-obs 2016 --mle-iters 25 --wsckf-paths 4 --enkf 25,500,1000 --pf 25,100,500
```

Kept figures: `output/cohort_calibrated_pareto.{pdf,png}` (held-out NLL /
full-record MAPE),
`output/cohort_calibrated_crps.{pdf,png}` (mixture CRPS for PF), plus
`cohort_calibrated_{summary,per_subject,params}.*`.

For Δt = 20 min (every 4th 5-min sample; re-MLE + bakeoff):

```bash
uv run --extra experiments python experiments/AZT1D/run_cohort_calibrated.py \
    --dt-assim 20 --n-obs 2016 --mle-iters 25 --wsckf-paths 4 --enkf 25,500,1000 --pf 25,100,500
```

Writes `cohort_calibrated_crps_dt20.{pdf,png}` (also refreshes the unsuffixed pair;
the Δt=5 pair is kept with the `cohort_calibrated_crps_dt5` stem).

## Filter tracks (CGM overlay)

```bash
uv run --extra experiments python experiments/AZT1D/run_filter_tracks.py
uv run --extra experiments python experiments/AZT1D/run_filter_tracks.py \
    --subject 1 --enkf 500 --pf 500
```

Writes `output/filter_tracks.{pdf,png}` (full-series WSC-KF ±1σ + held-out
multi-filter overlay).

All manuscript figures use final-size 90 mm (single-column) or 190 mm
(double-column) canvases, vector PDF, and 450 dpi PNG.
