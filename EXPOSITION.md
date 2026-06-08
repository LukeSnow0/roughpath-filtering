# Cubature Filtering  - Mathematical Overview

Here we introduce the mathematics behind the *cubature filter*: a deterministic, efficient and expressive replacement to standard filtering methods such as the KF, EKF, UKF, EnKF, PF. This mathematical overview should motivate the advantages and well-posedness of this technique. 

## Filtering Abstraction

We begin with a *model* for the data-generating process, specified by a stochastic differential equation in Stratonovich form:

$$
dx_{t} = \sum_{i=0}^d V_i(x_{t})\circ dB_t^i, \quad x_0 \sim p_0 
$$

where $dB_t^0 = dt$ by convention, $x_{t} \in \mathbb{R}^n$, $V_0: \mathbb{R}^n \to \mathbb{R}^n$ is the drift vector field, and $\{V_i\}_{i=1}^d$ are vector fields specifying the diffusion matrix. Here $d$ is the dimension of the driving Brownian motion $B_t$. Note that time-dependent vector fields are naturally modeled here by simply augmenting the state space to $\hat{x}_t = (t,x_t)$.

We let $\mathcal{F}(y,dt) = \mathbb{E}\[x_{dt} | x_0 = y\]$ be the Markov semigroup defined by the SDE dynamics. Approximation of this semigroup through stochastic simulation of the SDE dynamics will form the basis for the time-update step in each filter implementation. The key advantage of the cubature filter is an alternative efficient computational approach for approximating this semigroup, through *deterministic* ODE evaluations along carefully specified Wiener-space cubature paths and weights which approximate the statistical behavior of the driving Brownian motion. We now present these details. 

## Wiener Space Cubature

Recall that the aim of Wiener-space cubature is to obtain an efficient deterministic approximation of the semigroup $\mathcal{F}(y,dt) = \mathbb{E}\[x_{dt} | x_0 = y\]$. To begin our derivation of such an approximation, let us fist consider the stochastic Taylor expansion. We begin by writing the following solution form of the Stratonovich SDE dynamics: for smooth bounded functions $f:\mathbb{R}^n \to \mathbb{R}$,

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
\mathbb{E}[f(x_{dt})] = \sum_{(i_1,\dots,i_k)\in \mathcal{A}_m} V_{i_1}\cdots V_{i_k} f(x) \mathbb{E} \biggl\[ \int_{0 < t_1 < \dots < t_k < t} \circ dB_{t_1}^{i_1} \cdots \circ d B_{t_k}^{i_k} \biggr\] + \mathcal{O}({dt}^{(m+1)/2})
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

and weights $lambda_1,\dots,\lambda_n > 0$ define a *cubature formula on Wiener space* of degree $m$ at time $T$, if for all $(i_1,\dots,i_k) \in \mathcal{A}_m$,

$$
\mathbb{E} \biggl\[ \int_{0 < t_1 < \dots < t_k < t} \circ dB_{t_1}^{i_1} \cdots \circ d B_{t_k}^{i_k} \biggr\] = \sum_{j=1}^n \lambda_j \int_{0 < t_1 < \dots < t_k < t} dw_j^{i_1}(t_1) \cdots dw_j^{i_k}(t_k).
$$

*Existence* of cubature paths and weights is given in the following

**Theorem (Lyons, Victoir 2004)** Let $m$ be a natural number and $T$ be a positive real number. Then, there exist $n$ paths $w_1,\dots,w_n \in C_{0,bv}^0([0,T]; \mathbb{R}^d)$ and $n$ positive weights $\lambda_1, \dots, \lambda_n$, with $n \leq card\\{\mathcal{A}_m\\}$, defining a cubature formula on Wiener space of degree $m$ at time $T$.

Furthermore, Lyons and Victoir provide *explicit* constructions of Wiener-space cubature paths and weights for orders $m=3,5$. 

These paths and weights *replicate the statistics of the driving Brownian motion*, through expectations of iterated integrals up to a certain order. Practically, they allow us to accurately approximate the SDE solution by solving a *system of ODEs*. Specifically, let 

$$
\\{w_i,\lambda_i\\}_{i=1}^n
$$

define a cubature formula on Wiener space of degree $m$ at time $dt$, and denote by $\Phi_{dt,x_0}(w)$ the solution at time $dt$ of the ODE

$$
dy_t = \sum_{i=0}^d V_i(y_t)dw_i(t), \quad y_0 = x_0
$$

Then, we have the following approximation guarantee:

$$ 
\sup_{x\in\mathbb{R}^n}\biggl\|\mathbb{E}\biggl\(f(x_{dt} - \sum_{i=1}^n \lambda_i f(\Phi_{dt,x_0}(w_i))\biggr\)\biggr\| \leq C dt^{(m+1)/2} \sup_{(i_1,\dots,i_k) \in \mathcal{A}_{m+2} / \mathcal{A}_m} |V_{i_1} \cdots V_{i_k} f|_{\infty}
$$

Thus, we may efficiently approximate the statistic $\mathbb{E}\[f(x_{dt})\]$ by solving a system of deterministic ODEs along vector fields $\\{V_i\\}_{i=0}^d$, driven by these carefully chosen cubature paths. 




