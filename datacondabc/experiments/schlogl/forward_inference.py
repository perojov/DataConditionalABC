import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
import sys
import torch
import pickle
import numpy as np
from functools import partial
from datacondabc.samplers import dynamic_forward_abcsmc
from datacondabc.simulators.approxsim import finescale_em
from datacondabc.models import schlogl
from datacondabc.nnets import PENCNN

# True parameter.
k1, k2, k4 = 3e-7, 1e-4, 3.5
theta = np.array([k1, k2, k4])
param_dim = len(theta)

# Configuration for SDE solver.
A, dt, n, x0, d = 100, 1, 50, 249, 1
simulator = partial(finescale_em, schlogl, x0, n, A, dt)

# Prior.
prior_bounds = np.array([[1.6e-7, 1e-6], [0.0, 5e-4], [1, 5]])

# Quantile for determining the ABC thresholds.
q = 50

# ABC-SMC configuration.
niter = 50
npart = 10000
xo = np.load("obs.npy")
init_size = 2000000
out = dynamic_forward_abcsmc(xo, npart, prior_bounds, q, niter, init_size, simulator)

with open('f_run_timemeasure_' + str(int(sys.argv[1])) + '.pkl', 'wb') as f:
    pickle.dump(out, f)
