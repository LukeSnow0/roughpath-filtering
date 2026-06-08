## Cubature Filtering  - Mathematical Overview

Here we introduce the mathematics behind the *cubature filter*: a deterministic, efficient and expressive replacement to standard filtering methods such as the KF, EKF, UKF, EnKF, PF. This mathematical overview should motivate the advantages and well-posedness of this technique. 

### Filtering Abstraction

We begin with a *model* for the data-generating process, specified by a stochastic differential equation in Stratonovich form:

$$
dx_{t} = \sum_{i=0}^d V_i(t, x_{t})\circ dB_t^i, \quad x_0 \sim p_0 \tag{1}
$$

where $dB_t^0 = dt$ by convention, $x_{t} \in \mathbb{R}^n$, $V_0: \mathbb{R}^n \to \mathbb{R}^n$ is the drift vector field, and $\{V_i\}_{i=1}^d$ are vector fields specifying the diffusion matrix. Here $d$ is the dimension of the driving Brownian motion $B_t$. 

We let $\mathcal{F}(y,t,dt) = \mathbb{E}\[x_{t+dt} | x_t = y\]$ be the time-inhomogeneous Markov semigroup defined by the dynamics (1). 


