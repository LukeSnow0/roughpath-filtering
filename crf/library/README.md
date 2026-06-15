# Cubature Recursive Filter (CRF)

This sub-repository provides modular functions for the Cubature Recursive Filter.

The core functions are:

- `predict`: Propagate spatial cubature nodes along deterministic Wiener-space
  cubature paths through the nonlinear Stratonovich SDE, then collapse the
  resulting weighted cloud back to Gaussian moments.
- `update`: Update the Gaussian moments with an observation using a
  statistical-linearization (sigma-point) Kalman update in square-root form.

Supporting routines (host-side NumPy) in `cubature_paths.py` build the Wiener
cubature paths and weights. These depend only on the assimilation interval, the
Brownian dimension and the cubature order, so they are precomputed once and
treated as static constants by the JAX kernels.

The CRF treats the filtering distribution as Gaussian (like the EKF/EnKF) but,
unlike the EKF, requires no Jacobian of the dynamics, and unlike the EnKF/PF,
uses a small set of *deterministic* cubature paths rather than Monte-Carlo
samples. Two multidimensional path constructions are available: `product`
(tensor-product Lyons paths, exact to degree 3)and `recombined` (recursive KLV cubature expansion with high-order
recombination/RMP after each macro-segment; Algorithm 1 of Snow &
Krishnamurthy, 2025).