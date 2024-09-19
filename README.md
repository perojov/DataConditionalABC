# datacondabc

## Overview

datacondbac is a Python package for conducting Bayesian inference for Stochastic Differential Equations through Approximate Bayesian Computation - Sequential Monte Carlo (ABC-SMC). The package features both ABC-SMC with forward simulation using Euler-Maruyama, and ABC-SMC with data-conditional simulation as introduced in our [arXiv paper](https://arxiv.org/abs/2310.10329) (accepted in Bayesian Analysis). It utilizes partially exchangeable networks for the estimation of summary statistics. 

## Installation

Download this folder and run `pip install .`.

## Overview

`datacondabc` facilitates ABC posterior inference using the ABC-SMC algorithm with:
* Fixed summary statistics function and forward simulation.
* Fixed summary statistics function and data-conditional simulation.
* Continually improving summary statistics function and forward simulation.
* Continually improving summary statistics function and data-conditional simulation.

## Running the code.
The following script reproduces the Lotka--Volterra example.

```python
# Load data.
obs = np.load("obs.npy")
n = obs.shape[1] - 1

# Configure the SDE integrator.
A, dt, x0 = 100, 1, obs[0, :, 0]
forward = partial(finescale_em_2d, x0, n, A, dt)

# Number of particles for the data-conditional simulator.
P = 30

# Uniform prior bounds.
prior_bounds = np.array([[0.0, 1.0], [0.00, 0.05], [0.0, 1]])

# Run ABC-SMC.
out = abcsmc_dataconditional(
    obs=obs,
    prior_bounds=prior_bounds,
    lookahead=lookahead_sis_2d,
    backward=smoother_2d,
    forward=forward,
    npart_sim=P,
    nsubint=A,
    dt=dt,
)

with open("dc_inference_result.pkl", "wb") as f:
    pickle.dump(out, f)
```
## Files

- **`models.py`**  
  Implements functions for the drift and diffusion of a few known SDE models: Ornstein--Uhlenbeck, Cox-Ingersoll-Ross, Chan-Karolyi–Longstaff–Sanders, nonlinear drift, Schlogl, and Lotka--Volterra.

- **`nnets.py`**  
  Implements the partially exchangeable neural network and the data module with `pytorch-lightning`.

- **`samplers.py`**  
  Implements the ABC-SMC samplers for the four different approaches: `abcsmc_dataconditional` and `abcsmc_forward`.

- **`utilities.py`**  
  Implements functions that are frequently used in the package. It also includes `numba`-jitted reimplementations of some standard functions like weighted random sampling and the `LogSumExp` function.

- **`simulators/exactsim.py`**  
  Implements two `numba`-jitted functions for sampling from the exact transition densities for the Ornstein--Uhlenbeck and Cox--Ingersoll--Ross models. Useful for when an exact observation is required for testing purposes.

- **`simulators/approxsim.py`**  
  Implements `numba`-jitted functions for simulating trajectories: 
  1) the Euler-Maruyama method, 
  2) the lookahead sequential-importance-sampling method, 
  3) a standard backward simulation particle smoother.


## References
Petar Jovanovski, Andrew Golightly and Umberto Picchini: Towards Data-Conditional Simulation for ABC Inference in Stochastic Differential Equations. [arxiv:2310.10329](https://arxiv.org/abs/2310.10329), 2023
