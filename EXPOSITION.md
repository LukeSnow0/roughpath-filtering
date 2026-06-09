# Cubature Filtering  - Mathematical Overview

Here we introduce the mathematics behind the *cubature filter*: a deterministic, efficient and expressive replacement to standard filtering methods such as the KF, EKF, UKF, EnKF, PF. This mathematical overview should motivate the advantages and well-posedness of this technique. 

## Filtering Abstraction

We begin with a *model* for the data-generating process, specified by a stochastic differential equation in Stratonovich form:

$$
dx_{t} = V_0(x_t)dt + \sum_{i=1}^d V_i(x_{t})\circ dB_t^i, \quad x_0 \sim p_0 
$$

where $x_{t} \in \mathbb{R}^n$, $V_0: \mathbb{R}^n \to \mathbb{R}^n$ is the drift vector field, and $\{V_i\}_{i=1}^d$ are vector fields specifying the diffusion matrix. Here $d$ is the dimension of the driving Brownian motion $B_t$. Note that time-dependent vector fields are naturally modeled here by simply augmenting the state space to $\hat{x}_t = (t,x_t)$.

We let $\mathcal{F}(y,\Delta_t) = \mathbb{E}\[x_{\Delta_t} | x_0 = y\]$ be the Markov semigroup defined by the SDE dynamics. Approximation of this semigroup through stochastic simulation of the SDE dynamics will form the basis for the time-update step in each filter implementation. The key advantage of the cubature filter is an alternative efficient computational approach for approximating this semigroup, through *deterministic* ODE evaluations along carefully specified Wiener-space cubature paths and weights which approximate the statistical behavior of the driving Brownian motion. We now present these details. 

## Wiener Space Cubature

Recall that the aim of Wiener-space cubature is to obtain an efficient deterministic approximation of the semigroup $\mathcal{F}(y,\Delta_t) = \mathbb{E}\[x_{\Delta_t} | x_0 = y\]$. To begin our derivation of such an approximation, let us fist consider the stochastic Taylor expansion. We begin by writing the following solution form of the Stratonovich SDE dynamics: for smooth bounded functions $f:\mathbb{R}^n \to \mathbb{R}$,

$$
f(x_t) = f(x_0) + \int_0^t \sum_{i=0}^d V_if(x_s)\circ dB_s^i
$$

Let us also define an index set $\mathcal{A}_m = \\{(i_1,\dots,i_k)\in\\{0,\dots,d\\}^k, k + card\\{j, i_j = 0\\}\leq m\\}, m\in\mathbb{N}$. Then, the solution above can be expanded in the following weak stochastic Taylor formula (see Kloeden \& Platen (1992))

$$
f(x_t) = \sum_{(i_1,\dots,i_k)\in \mathcal{A}_m} V_{i_1}\cdots V_{i_k} f(x) \int_{0 < t_1 < \dots < t_k < t} \circ dB_{t_1}^{i_1} \cdots \circ d B_{t_k}^{i_k} \,+ \,R_m(t,x,f)
$$

Here $V_i$ act as differential operators, i.e., $V_i f(x) = \sum_{l=1}^n V_i^l(x)\partial_{x_l}f(x)$. The remainder process $R_m$ satisfies, for $C$ depending only on $d$ and $m$, 

$$
\sup_{x\in\mathbb{R}^n} \sqrt{\mathbb{E}(R_m(t,x,f)^2)} \leq C t^{(m+1)/2} \sup_{(i_1,\dots,i_k) \in \mathcal{A}_{m+2} / \mathcal{A}_m} |V_{i_1} \cdots V_{i_k} f|_{\infty}
$$

Thus, in order to get a high-order approximation of the expectation  

$$
\mathbb{E}[f(x_{\Delta_t})] = \sum_{(i_1,\dots,i_k)\in \mathcal{A}_m} V_{i_1}\cdots V_{i_k} f(x) \mathbb{E} \biggl\[ \int_{0 < t_1 < \dots < t_k < t} \circ dB_{t_1}^{i_1} \cdots \circ d B_{t_k}^{i_k} \biggr\] + \mathcal{O}({\Delta_t}^{(m+1)/2})
$$

we may approximate the "probabilistic core" of the solution, given by 

$$
\mathbb{E} \biggl\[ \int_{0 < t_1 < \dots < t_k < t} \circ dB_{t_1}^{i_1} \cdots \circ d B_{t_k}^{i_k} \biggr\]. 
$$

This is precisely the object which Wiener-space cubature reproduces, through carefully chosen deterministic paths and weights.

### Cubature Paths and Weights

Let $m$ be a natural number. We say that the paths 

$$
w_1,\dots,w_n \in C_{0,bv}^0([0,T],\mathbb{R}^d)
$$

and weights $\lambda_1,\dots,\lambda_n > 0$ define a *cubature formula on Wiener space* of degree $m$ at time $T$, if for all $(i_1,\dots,i_k) \in \mathcal{A}_m$,

$$
\mathbb{E} \biggl\[ \int_{0 < t_1 < \dots < t_k < t} \circ dB_{t_1}^{i_1} \cdots \circ d B_{t_k}^{i_k} \biggr\] = \sum_{j=1}^n \lambda_j \int_{0 < t_1 < \dots < t_k < t} dw_j^{i_1}(t_1) \cdots dw_j^{i_k}(t_k).
$$

*Existence* of cubature paths and weights is given in the following

**Theorem (Lyons, Victoir 2004)** Let $m$ be a natural number and $T$ be a positive real number. Then, there exist $n$ paths $w_1,\dots,w_n \in C_{0,bv}^0([0,T]; \mathbb{R}^d)$ and $n$ positive weights $\lambda_1, \dots, \lambda_n$, with $n \leq card\\{\mathcal{A}_m\\}$, defining a cubature formula on Wiener space of degree $m$ at time $T$.

Furthermore, Lyons and Victoir provide *explicit* constructions of Wiener-space cubature paths and weights for orders $m=3,5$. 

### Cubature-based Filtering Semigroup Approximation

These paths and weights *replicate the statistics of the driving Brownian motion*, through expectations of iterated integrals up to a certain order. Practically, they allow us to accurately approximate the SDE solution by solving a *system of ODEs*. Specifically, let 

$$
\\{w_i,\lambda_i\\}_{i=1}^n
$$

define a cubature formula on Wiener space of degree $m$ at time $\Delta_t$, and denote by $\Phi_{\Delta_t,x_0}(w)$ the solution at time $\Delta_t$ of the ODE

$$
dy_t = V_0(y_t)dt + \sum_{i=1}^d V_i(y_t)dw_i(t), \quad y_0 = x_0
$$

Then, we have the following filtering semigroup approximation guarantee:

$$ 
\sup_{x\in\mathbb{R}^n}\biggl\|\mathbb{E}\biggl\(f(x_{\Delta_t} - \sum_{i=1}^n \lambda_i f(\Phi_{\Delta_t,x_0}(w_i))\biggr\)\biggr\| \leq C \Delta_t^{(m+1)/2} \sup_{(i_1,\dots,i_k) \in \mathcal{A}_{m+2} / \mathcal{A}_m} |V_{i_1} \cdots V_{i_k} f|_{\infty}
$$

Thus, we may efficiently approximate the statistic $\mathbb{E}\[f(x_{\Delta_t})\]$ by solving a system of deterministic ODEs along vector fields $\\{V_i\\}_{i=0}^d$, driven by these carefully chosen cubature paths. 


## Cubature Filter

Here we exploit these Wiener space cubature developments in a concrete continuous-discrete filtering algorithm. We model the data generation process by augmenting the Stratonovich SDE with a discrete-time indexed obervation map, as follows

$$
dx_{t} = V_0(x_t)dt + \sum_{i=1}^d V_i(x_{t})\circ dB_t^i, \quad x_0 \sim p_0, \quad Y_k = h(X_{t_k}) + V_k, V_k \sim \mathcal{N}(0,R)
$$

### Posterior Approximation
At filtering time $t_k$, we approximate the posterior process law $\pi_k(dx)$ by a Gaussian moment approximation:

$$
\pi_k(dx) := \mathcal{L}(X_{t_k} | Y_{1:k}) \approx \mathcal{N}(\hat{x}_k, P_k)
$$

Unlike a particle filter or ensemble Kalman filter, we do *not* maintain a weighted empirical distributions over particles after the update step. We only keep 

$$
\hat{x}_k = \mathbb{E}[X_{t_k} | Y_{1:k}], \quad P_k = \text{Cov}(X_{t_k} | Y_{1:k}).
$$

### Spatial Cubature Approximation
This step utilizes the *spatial* cubature approximation of (Haykin, et. al. 2009), which is a deterministic third-order aproximation of finite-dimensional Gaussian distributions. This will be combined with our *Wiener space cubature* evaluation, which approximates the infinite-dimensional path-space Wiener measure in the forward particle propagation step. The spatial cubature approximates the Gaussian $\mathcal{N}(\hat{x}_k, P_k)$ by points 

$$
\zeta_i^+ = \hat{x}_k + \sqrt{n}U^T e_i, \quad \zeta_i^{-} = \hat{x}_k - \sqrt{n}U^T e_i, \quad i=1,\dots,n,
$$

where $P_k = UU^T$ is the Cholesky factorization. Assigning each point a uniform weight $\alpha_i^{\pm} = \frac{1}{2n}$, the current Gaussian posterior is approximated by 

$$
\mathcal{N}(\hat{x}_k, P_k) \approx \sum_{i=1}^{2n} \alpha_i \delta_{\zeta_i}
$$

### Deterministic Cubature Propagation
For each spatial cubature note $\zeta_i$ and each Wiener cubature path $w_j$, we solve the *controlled ODE*

$$
X_{t_{k+1}}^{i,j} = \Phi_k^{w_j}(\zeta_i) = \zeta_i + \int_{t_k}^{t_{k+1}} V_0(X_t^{i,j})dt + \int_{t_k}^{t_{k+1}} \sum_{i=1}^d V_i(X_t^{i,j})\dot{w}_j(t)
$$

Then, predicted mean and covariance are computed as

$$
\hat{x}_{k+1 | k} = \sum_{i,j} \alpha_i \lambda_j X_{t_{k+1}}^{i,j}, \quad P_{k+1|k} = \sum_{i,j} \alpha_i \lambda_j \biggl\(X_{t_{k+1}}^{i,j} - \hat{x}_{k+1|k} \biggr\)\biggl\(X_{t_{k+1}}^{i,j} - \hat{x}_{k+1|k}\biggr\)^T
$$

### Measurement Update

For the update step we use a Kalman-style Gaussian projection, using the propagated cubature points to approximate observation moments. Specifically, the predicted observation mean is 

$$
\hat{z}_{k+1 | k} := \sum_{i,j} w_{ij}z_{ij}, \quad z_{ij} := h(X_{t_{k+1}}^{i,j})
$$

The innovation covariance is 

$$
P_{zz} = \sum_{i,j}w_{ij}(z_{ij} - \hat{z}_{k+1|k})(z_{ij} - \hat{z}_{k+1|k})^T + R
$$

and the state-observation cross-covariance is 

$$
P_{xz} = \sum_{i,j}w_{ij}(X_{t_{k+1}}^{i,j} - \hat{x}_{k+1|k})(z_{ij} - \hat{z}_{k+1|k})^T.
$$

Then, we form the Kalman gain as 

$$
K = P\_{xz}P_{zz}^{-1}
$$
 
and update based on observation $y_{k+1}$ as 

$$
\hat{x}\_{k+1} = \hat{x}\_{k+1|k} + K(y_{k+1} - \hat{z}\_{k+1|k}),  \quad P_{k+1} = P_{k+1|k} - KP_{zz} K^T
$$








