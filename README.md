## Cubature Filtering  - Mathematical Overview

Here we introduce the mathematics behind the *cubature filter*: a deterministic, efficient and expressive replacement to standard filtering methods such as the KF, EKF, UKF, EnKF, PF. This mathematical overview should motivate the advantages and well-posedness of this technique. 

### Filtering Abstraction

We begin with a *model* for the data-generating process, specified by a stochastic differential equation in Stratonovich form:

$$
dx_{t} = \sum_{i=0}^d V_i(x_{t})\circ dB_t^i, \quad x_0 \sim p_0 
$$

where $dB_t^0 = dt$ by convention, $x_{t} \in \mathbb{R}^n$, $V_0: \mathbb{R}^n \to \mathbb{R}^n$ is the drift vector field, and $\{V_i\}_{i=1}^d$ are vector fields specifying the diffusion matrix. Here $d$ is the dimension of the driving Brownian motion $B_t$. Note that time-dependent vector fields are naturally modeled here by simply augmenting the state space to $\hat{x}_t = (t,x_t)$.

We let $\mathcal{F}(y,dt) = \mathbb{E}\[x_{dt} | x_0 = y\]$ be the Markov semigroup defined by the SDE dynamics. Approximation of this semigroup through stochastic simulation of the SDE dynamics will form the basis for the time-update step in each filter implementation. The key advantage of the cubature filter is an alternative efficient computational approach for approximating this semigroup, through *deterministic* ODE evaluations along carefully specified Wiener-space cubature paths and weights which approximate the statistical behavior of the driving Brownian motion. We now present these details. 

### Wiener Space Cubature

Recall that the aim of Wiener-space cubature is to obtain an efficient deterministic approximation of the semigroup $\mathcal{F}(y,dt) = \mathbb{E}\[x_{dt} | x_0 = y\]$. To begin our derivation of such an approximation, let us fist consider the stochastic Taylor expansion. We begin by writing the following solution form of the Stratonovich SDE dynamics: for smooth bounded functions $f:\mathbb{R}^n \to \mathbb{R}$,

$$
f(x_t) = f(x_0) + \int_0^t \sum_{i=0}^d V_if(x_s)\circ dB_s^i
$$

Let us also define an index set $\mathcal{A}_m = \\{(i_1,\dots,i_k)\in\\{0,\dots,d\\}^k, \, k + card\\{j, i_j = 0\\}\leq m\\}, \, m\in\mathbb{N}$. Then, the solution above can be expanded in the following weak stochastic Taylor formula (see Kloeden \& Platen (1992))

$$
f(x_t) = \sum_{(i_1,\dots,i_k)\in \mathcal{A}_m} V_{i_1}\cdots V_{i_k} f(x) \int_{0 < t_1 < \dots < t_k < t} \circ dB_{t_1}^{i_1} \cdots \circ d B_{t_k}^{i_k} \,+ \,R_m(t,x,f)
$$

Here $V_i$ act as differential operators, i.e., $V_i f(x) = \sum_{l=1}^n V_i^l(x)\partial_{x_l}f(x)$. The remainder process $R_m$ satisfies, for $C$ depending only on $d$ and $m$, 

$$
\sup_{x\in\mathbb{R}^n} \sqrt{\mathbb{E}(R_m(t,x,f)^2)} \leq C t^{(m+1)/2} \sup_{(i_1,\dots,i_k) \in \mathcal{A}_{m+2} backslash \mathcal{A}_m} \|V_{i_1} \cdots V_{i_k} f\|_{\infty}
$$


