# WCKF Empirical Validation — Implementation Spec

## Purpose

This document specifies two experiments that empirically validate the theoretical
predictions of the Wiener Cubature Kalman Filter (WCKF) consistency theorems. The
goal is **not** a generic accuracy comparison — it is to demonstrate that the
*specific rates and crossover conditions predicted by theory* are observed in
practice. Each experiment below is designed to test one falsifiable prediction.

This spec is self-contained: it restates the model, the filters to implement, the
theoretical quantities being tested, and the exact plots required. No other
document should be needed to implement this.

---

## 1. Background: what is being tested

### 1.1 Model

Continuous-discrete state-space model:

```
dx_t = f(t, x_t) dt + G(t, x_t) dW_t          (process, Ito SDE)
y_k  = h(t_k, x_tk) + v_k,  v_k ~ N(0, R(t_k))  (discrete observation)
```

Use the Modified Four-Tank System (MFTS) already implemented for the main paper
(masses m1..m4, disturbance inflows F3, F4; see mass-balance equations and
disturbance SDE already in the codebase). For Experiment 2 (crossover), a
**state-dependent diffusion variant** is additionally required — see Section 3.

### 1.2 Filters to compare

Three filters must be implemented (or already exist in the repo — reuse if
present):

1. **CD-CKF** — continuous-discrete cubature Kalman filter, using the
   Ito-Taylor-1.5 discretization with the mean-frozen diffusion-coupling term
   (the standard published CD-CKF time update).
2. **WCKF** — Wiener Cubature Kalman Filter (Algorithm 1 in the main paper):
   `2*dx` spatial cubature nodes x `N_eta` Wiener-cubature paths, tensor-product
   propagated, aggregated via a single weighted second moment.
3. *(Optional, useful sanity check)* **CD-EKF** — for a baseline showing the
   qualitative underestimation phenomenon is not WCKF/CD-CKF-specific.

### 1.3 Exact quantities under test

Let `dx` = state dimension, `m` = Wiener-cubature degree used by WCKF (3 or 5),
`N_eta` = number of Wiener-cubature paths, `Delta_t` = observation interval,
`P_{k-1}` = prior covariance entering a single time-update step.

**Theorem 2 predicts** (local, single time-update step):

```
|| P_{k|k-1}^WCKF - P_{k|k-1}^true || <= 2*dx*C2*Delta_t^((m+1)/2) + C3*||P_{k-1}||^2
```

i.e. for fixed prior spread, the WCKF's error in a single time-update step should
shrink as `Delta_t -> 0` at rate `(m+1)/2` in log-log slope, **until** the second
(spatial-cubature, degree-3-ceiling) term dominates and the curve flattens.

**Corollary 1 predicts** a crossover condition: the CD-CKF's mean-freezing error
exceeds the WCKF's total error whenever

```
c * H * tr(P_{k-1})  >  2*dx*C2*Delta_t^((m+1)/2) + 2*C3*||P_{k-1}||^2
```

where `H` bounds the curvature (operator norm of the appropriate derivative
tensor) of `Sigma(x) := Cov(x_k | x_{k-1}=x)` near the prior mean, and `c > 0`
is a Jensen-gap constant. In practice `H` is nonzero only when the diffusion
coefficient `G(t,x)` is state-dependent (multiplicative noise); it is
**identically zero** for the additive-noise MFTS model as currently specified,
so Experiment 2 requires the state-dependent variant below.

---

## 2. Experiment 1 — Convergence rate validation

**Prediction being tested:** WCKF's per-step covariance error decays at rate
`(m+1)/2` in `Delta_t`, until the spatial-cubature error floor (independent of
`Delta_t`) is reached.

### 2.1 Setup

- Use the (additive-noise) MFTS model exactly as in the main paper — no
  modification needed for this experiment, since Theorem 2 does not require
  state-dependent diffusion.
- Fix a single prior `(x_hat_{k-1}, P_{k-1})` — e.g. take it from a representative
  point mid-trajectory of an existing filter run, or construct a synthetic prior
  with `P_{k-1}` set to a realistic magnitude for this system.
- Sweep `Delta_t` over a geometric grid, e.g.
  `Delta_t in {60, 30, 15, 7.5, 3.75, 1.875, 0.9375} s` (7 points spanning ~2
  orders of magnitude; adjust bounds so the smallest value is well below the
  system's fastest time constant and the largest is comparable to existing
  `Delta_t = 15s` baseline).
- For each `Delta_t`, compute a high-accuracy **ground-truth** predictive
  covariance `P_{k|k-1}^true` by Monte Carlo: simulate `N_mc >= 20000`
  independent SDE paths from `x_{k-1} ~ N(x_hat_{k-1}, P_{k-1})` forward by
  `Delta_t` using a fine Euler-Maruyama or higher-order SDE integrator (small
  internal step size, e.g. `Delta_t / 200`), and compute the empirical mean and
  covariance of the terminal states. This is the reference "truth" against which
  both filters are scored — **do not** compare WCKF against CD-CKF directly in
  this experiment; compare both against this Monte Carlo ground truth.
- For each `Delta_t`, run:
  - WCKF's time update (degree `m=3` and, if implemented, `m=5`) from the same
    fixed prior, producing `P_{k|k-1}^WCKF(Delta_t)`.
  - CD-CKF's time update from the same fixed prior, producing
    `P_{k|k-1}^CDCKF(Delta_t)`, for comparison/context (not the main claim of
    this experiment, but useful to show alongside).
- Compute error metric: `err(Delta_t) = || P_{k|k-1}^WCKF(Delta_t) -
  P_{k|k-1}^true(Delta_t) ||_F` (Frobenius norm), and identically for CD-CKF.

### 2.2 Required plot

**Log-log plot**: x-axis `Delta_t`, y-axis `err(Delta_t)`, one line per method
(WCKF degree 3, WCKF degree 5 if available, CD-CKF).

- Overlay reference slope lines of slope `(m+1)/2` (i.e. slope 2 for m=3, slope 3
  for m=5) anchored at the largest-`Delta_t` data point, so the reader can
  visually compare the empirical slope to the theoretical prediction.
- Fit an empirical log-log slope (simple linear regression on `log(err)` vs.
  `log(Delta_t)` over the small-`Delta_t` portion of the curve, before any
  visible flattening) and report the fitted slope numerically in the plot
  legend or a caption-ready text output (e.g. `fitted_slope = 1.97, predicted =
  2.0`).
- Expect and **explicitly show** the curve flattening at small `Delta_t` once the
  `C3*||P_{k-1}||^2` floor dominates — this is a predicted feature, not a bug;
  annotate it if visible (e.g. a text label "spatial-cubature floor" with an
  arrow, or a horizontal dashed line at the estimated floor level).

### 2.3 Output artifacts

- `experiment1_convergence.png` (or `.pdf`) — the log-log plot described above.
- `experiment1_results.csv` — columns: `Delta_t, method, degree_m, err_frobenius,
  N_eta_used`.
- Console/log output reporting the fitted slope with 95% CI, and the predicted
  slope, for each method/degree combination.

---

## 3. Experiment 2 — Crossover condition validation

**Prediction being tested:** the WCKF outperforms CD-CKF specifically when
`c*H*tr(P_{k-1})` exceeds the sum of WCKF's own error terms (Corollary 1's
inequality) — and the two methods perform comparably otherwise.

### 3.1 Required model modification (important — do this first)

The current MFTS disturbance model has **additive** diffusion:

```
dFj = lambda*(Fbar_j(t) - Fj) dt + sigma_F dW^(j-2)_t,   j = 3,4
```

This has `H = 0` identically (Sigma(x) does not depend on x), so Corollary 1's
predicted advantage is vacuous under the current model — **Experiment 2 cannot
be run on the existing additive model.** Implement a state-dependent diffusion
variant, e.g.:

```
dFj = lambda*(Fbar_j(t) - Fj) dt + sigma_F * sqrt(max(Fj, eps)) dW^(j-2)_t,  j = 3,4
```

(a CIR-type / square-root diffusion, physically defensible as inflow-rate noise
scaling with inflow magnitude — flag this choice explicitly as a modeling
assumption in any resulting write-up, since it changes the model beyond what the
main paper's MFTS uses). This introduces genuine state-dependence in the
diffusion coefficient, hence nonzero curvature `H` in `Sigma(x)`.

If a physically-motivated multiplicative-noise variant is preferred instead,
substitute it here, but confirm analytically or numerically that the resulting
`Sigma(x)` has nonzero curvature near typical operating points before proceeding
— e.g. numerically estimate `H` via finite differences on `Sigma` (see 3.3).

### 3.2 Setup — sweep design

Corollary 1's inequality has three tunable knobs on the left side
(`H, tr(P_{k-1})`, folded into one product) and the right side depends on
`Delta_t, N_eta, m` (WCKF's own error budget). Design the sweep to move the
**left-hand side** while holding the right-hand side roughly fixed, so that
crossing the predicted boundary is attributable to the intended mechanism:

- Fix `Delta_t` and WCKF's cubature degree `m` and `N_eta` at reasonable values
  (e.g. `Delta_t = 15s`, `m = 3`, matching the main paper's baseline).
- Sweep **one or both** of:
  - `sigma_F` (diffusion intensity — directly scales `H`), over e.g.
    `sigma_F in {0.1, 0.5, 1, 2, 5, 10} * (baseline sigma_F)`.
  - Prior spread `tr(P_{k-1})` — either via the measurement noise level
    `sigma_h` (larger `sigma_h` -> larger posterior/prior uncertainty
    propagates forward) swept over e.g. `{1, 2, 4, 8, 16} cm`, or by directly
    inflating a synthetic `P_{k-1}` by a scalar factor for a controlled,
    single-step version of this test (recommended for a cleaner, less
    confounded first pass — see 3.4 "controlled single-step variant").
- For each sweep point, compute:
  - `H_hat`: numerically estimate the curvature bound of `Sigma(x)` via finite
    differences (perturb `x` along a few representative directions spanning
    `P_{k-1}`'s principal axes, evaluate `Sigma` via a short high-accuracy Monte
    Carlo or fine SDE integration at each perturbed point, and estimate the
    second difference).
  - `tr(P_{k-1})`: read directly from the filter/synthetic prior.
  - The predicted crossover quantity: `LHS = H_hat * tr(P_{k-1})` vs.
    `RHS = 2*dx*C2*Delta_t^((m+1)/2) + 2*C3*||P_{k-1}||^2` (C2, C3 need not be
    known exactly for this plot — report `LHS` and the empirically-observed
    error gap side by side; the qualitative comparison, "does the gap open when
    LHS grows," is the point, not an exact numerical verification of the
    inequality with derived constants).
  - The **empirical** error gap: run both WCKF and CD-CKF against Monte Carlo
    ground truth (as in Experiment 1) at this sweep point, and record
    `gap = ||P^CDCKF - P^true|| - ||P^WCKF - P^true||`.

### 3.3 Required plots

**Plot A (primary):** x-axis = sweep parameter (`sigma_F` or `sigma_h`, whichever
is varied — if both are varied, use two subplots or a 2D heatmap of the gap),
y-axis = `gap` (as defined above). Expect `gap` to increase (WCKF advantage
grows) as the sweep parameter increases. Add a horizontal reference line at
`gap = 0` (crossover point) to show clearly where WCKF and CD-CKF trade places,
if they do.

**Plot B (mechanism check, secondary but valuable):** scatter or line plot of
`gap` (y-axis) against the estimated `LHS = H_hat * tr(P_{k-1})` (x-axis) across
all sweep points — if the theory's mechanism is correct, this should show a
clear positive/monotonic relationship, which is a much sharper confirmation than
Plot A alone (it directly tests the *mechanism*, not just the outcome).

### 3.4 Controlled single-step variant (recommended first pass)

Before running the full closed-loop filtering sweep (which confounds many time
steps together), first validate on **one isolated time-update step**, mirroring
Experiment 1's design:

- Fix a single synthetic `(x_hat_{k-1}, P_{k-1})`.
- Sweep `sigma_F` only (holding `P_{k-1}` and `Delta_t` fixed).
- Compute `gap` for this single step, against Monte Carlo ground truth, exactly
  as above.
- This isolates the diffusion-curvature mechanism cleanly, without confounding
  from covariance accumulation across a full filter run. Report this as Plot A'
  alongside (or instead of, if time-constrained) the full closed-loop version.

### 3.5 Output artifacts

- `experiment2_crossover_sweep.png` — Plot A (or A').
- `experiment2_mechanism_check.png` — Plot B.
- `experiment2_results.csv` — columns: `sweep_param_name, sweep_param_value,
  H_hat, tr_P, LHS_estimate, err_WCKF, err_CDCKF, gap`.

---

## 4. Implementation notes / suggested structure

```
experiments/
  common/
    mfts_model.py          # existing additive MFTS model (reuse from main repo)
    mfts_model_multiplicative.py   # new: state-dependent diffusion variant (Sec 3.1)
    ground_truth_mc.py     # Monte Carlo ground-truth predictive moment estimator
    filters/
      cd_ekf.py, cd_ckf.py, wckf.py   # reuse existing implementations if present
  experiment1_convergence.py
  experiment2_crossover.py
  results/
    experiment1_results.csv, experiment1_convergence.png
    experiment2_results.csv, experiment2_crossover_sweep.png,
    experiment2_mechanism_check.png
```

- `ground_truth_mc.py` is shared infrastructure for both experiments — implement
  it once as: given `(x_hat, P, Delta_t, model)`, return `(mean_hat, cov_hat)`
  from `N_mc` fine-stepped SDE simulations. This is the single most important
  piece of new code, since both experiments' correctness depends on it being a
  genuinely accurate reference (use a small internal step size and confirm
  convergence of the MC estimate itself, e.g. by checking that `N_mc = 20000`
  and `N_mc = 40000` give visually indistinguishable moments before trusting it
  as ground truth).
- Random seeds: fix and record seeds for all Monte Carlo ground-truth runs, for
  reproducibility.
- If wall-clock cost of `N_mc >= 20000` fine-stepped simulations per sweep point
  is prohibitive across the full sweep grids in both experiments, it is
  acceptable to reduce `N_mc` for exploratory runs and increase it only for the
  final reported figures — flag this in code comments so it isn't silently left
  at a low-fidelity setting in the final results.

## 5. Definition of success

- **Experiment 1** succeeds if the fitted log-log slope for WCKF is within a
  reasonable tolerance (e.g. +/- 20%) of the predicted `(m+1)/2`, over the
  `Delta_t` range before the predicted floor is reached, and the floor itself is
  visible (curve flattens) at small `Delta_t`.
- **Experiment 2** succeeds if `gap` is monotonically (or clearly trending)
  increasing in the sweep parameter, and if Plot B shows a positive relationship
  between `gap` and the estimated `LHS = H_hat * tr(P_{k-1})`, providing direct
  evidence for the Jensen-gap mechanism claimed in Corollary 1, rather than
  merely an unexplained accuracy difference.
