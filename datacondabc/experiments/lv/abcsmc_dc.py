import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import pickle
import numpy as np
from functools import partial
from datacondabc.samplers import abcsmc_dataconditional
from datacondabc.simulators.approxsim import (
    finescale_em_2d,
    lookahead_sis_2d,
    smoother_2d,
)

# Load data.
obs = np.load("obs.npy")
n = obs.shape[1] - 1

# Configuration for SDE integrator.
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
