## Cubature Filtering  - Mathematical Overview

Here we introduce the mathematics behind the *cubature filter*: a deterministic, efficient and expressive replacement to standard filtering methods such as the KF, EKF, UKF, EnKF, PF. This mathematical overview should motivate the advantages and well-posedness of this technique. 

# Filtering Abstraction

We begin with a *model* for the data-generating process, specified by a stochastic differential equation in Stratonovich form:
$$ dx_{t,x_0} = \sum_{i=1}^d V_i(x_{t,x_0})\circ dB_t^i $$
