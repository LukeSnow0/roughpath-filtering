# Wiener Space Cubature Kalman Filter

The core atomic functions can be found in [`wsckf.library`](../library/README.md) and [`wsckf.filter.filtering`](filtering.py). The high-level filter is in [`wsckf.filter.cubature_filter`](cubature_filter.py).

The WSC-KF represents the filtering distribution as **Gaussian** (a mean and a Cholesky covariance factor), like the EKF and EnKF. Unlike the EKF, it never linearizes the dynamics; unlike the EnKF/particle filter, it does not use Monte-Carlo samples.

**Predict.** From the current Gaussian, deterministic spatial cubature nodes are placed and each is propagated through the nonlinear Stratonovich SDE along a small set of deterministic **Wiener-space cubature paths** $\omega^{(j)}(t)$ that match the moments of the Brownian driver. The propagated weighted cloud is collapsed back to a mean and (square-root) covariance. The Wiener paths depend only on the assimilation interval, the Brownian dimension, and the cubature order, so they are precomputed once at build time and treated as static constants.

**Update.** A statistical-linearization (sigma-point) Kalman update is applied in square-root form: a single triangularization of the pre-array yields the innovation Cholesky factor, the Kalman gain, and the updated state Cholesky factor simultaneously.

Two multidimensional path constructions are available via `multidim_paths`: `lv` (explicit Lyons–Victoir multi-D formulae, including Lévy area for degree 5; degree 5 is implemented for `d ≤ 2`), `product` (tensor-product of 1D Lyons paths), and `recombined` (a large candidate pool reduced to a sparse moment-matched measure). Default `auto` selects `lv` for degree 5 with `d = 2` when the support is small enough, and `recombined` for `d ≥ 3`.

