# DataConditionalABC

## Overview

DataConditionalABC is a Python package for conducting Bayesian inference in Stochastic Differential Equations through Approximate Bayesian Computation - Sequential Monte Carlo (ABC-SMC). The package features both ABC-SMC with forward simulation using Euler-Maruyama, and ABC-SMC with data-conditional simulation as introduced in our [arXiv paper](https://arxiv.org/abs/2310.10329). It utilizes partially exchangeable networks for the estimation of summary statistics. 

## Installation

To install run: 
```bash
pip install datacondabc
```

## Overview

Explain when one benefits of using data-conditional simulation...

## Files

### utilities.py
Implements functions that are frequently used in the package. It also includes numba jitted reimplementations of some standard functions like weighted random sampling and the LogSumExp function.

### simulators/exactsim.py
Implements two numba-jitted functions for sampling from the exact transition densities for the Ornstein--Uhlenbeck and Cox--Ingersoll--Ross model. Useful for when an exact observation is required for testing purposes. 

### simulators/approxsim.py
Implements three numba-jitted functions for simulating trajectories: 1) the Euler-Maruyama method, 2) the lookahead sequential-importance-sampling method, and 3) a standard backward simulation particle smoother. 

