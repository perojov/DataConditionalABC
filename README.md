# datacondabc

## Overview

datacondbac is a Python package for conducting Bayesian inference for Stochastic Differential Equations through Approximate Bayesian Computation - Sequential Monte Carlo (ABC-SMC). The package features both ABC-SMC with forward simulation using Euler-Maruyama, and ABC-SMC with data-conditional simulation as introduced in our [arXiv paper](https://arxiv.org/abs/2310.10329). It utilizes partially exchangeable networks for the estimation of summary statistics. 

## Installation

Download this folder and run `pip install .`.

## Overview

Consider the time-homogeneous diffusion $(X_t)_{t \geq 0}$ satisfying the SDE
$${\rm d}X_t = {\mu}({X}_t, \boldsymbol{\theta}) {\rm d}t + {\sigma}({X}_t, \boldsymbol{\theta}) {\rm d} {B}_t, \quad X_0 = x_0.$$
where the drift $\mu(X_t, \boldsymbol{\theta})$ and diffusion $\sigma(X_t, \boldsymbol{\theta})$ are known in parametric form. The goal is to perform Bayesian inference on the parameter $\boldsymbol{\theta}$, given discrete observations $X^{\rm o} = (X_0, X_1, ..., X_n)$ of the SDE at the time instants $0 < t_1 < t_2 < ..., t_n$. Due to the analytical intractability of the likelihood function for general SDE models, one can instead target the ABC posterior distribution

$$\pi_\epsilon(\boldsymbol{\theta} \ \vert \ S(X^{\rm o}))  \propto \pi(\boldsymbol{\theta}) \int 1( \Vert S(X) - S(X^{\rm o}) \Vert \leq \epsilon) p(S(X) \ \vert \ \boldsymbol{\theta}) {\rm d}X,$$

where $||\cdot||$ is an appropriate distance metric, $S(X)$ is a low-dimensional summarization of the trajectory $X$, $1( \Vert S(X) - S(X^{\rm o}) \Vert \leq \epsilon)$ is the indicator function, and $\epsilon>0$ a tolerance value determining the accuracy of the approximation.  

`datacondabc` facilitates ABC posterior inference using the ABC-SMC algorithm with:
* Fixed summary statistics function and forward simulation.
* Fixed summary statistics function and data-conditional simulation.
* Continually improving summary statistics function and forward simulation.
* Continually improving summary statistics function and data-conditional simulation.

## Files

### models.py
Implements functions for the drift and diffusion of a few known SDE models: Ornstein--Uhlenbeck, Cox-Ingersoll-Ross, Chan-Karolyi–Longstaff–Sanders, nonlinear drift, Schlogl and Lotka--Volterra.

### nnets.py
Implements the partially exchangeable neural network and the data module with pytorch-lightning.

### samplers.py
Implements the ABC-SMC samplers for the four different approaches: abcsmc_dataconditional and abcsmc_forward. 

### utilities.py
Implements functions that are frequently used in the package. It also includes numba jitted reimplementations of some standard functions like weighted random sampling and the LogSumExp function.

### simulators/exactsim.py
Implements two numba-jitted functions for sampling from the exact transition densities for the Ornstein--Uhlenbeck and Cox--Ingersoll--Ross model. Useful for when an exact observation is required for testing purposes. 

### simulators/approxsim.py
Implements three numba-jitted functions for simulating trajectories: 1) the Euler-Maruyama method, 2) the lookahead sequential-importance-sampling method, and 3) a standard backward simulation particle smoother. 

## References
Petar Jovanovski, Andrew Golightly and Umberto Picchini: Towards Data-Conditional Simulation for ABC Inference in Stochastic Differential Equations. [arxiv:2310.10329](https://arxiv.org/abs/2310.10329), 2023
