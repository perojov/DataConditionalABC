import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from functools import partial

# Model and simulator.
from approxsimulators import finescale_em
from utilities import network_pretrainer

import numpy as np
from approxsimulators import finescale_em
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save", type=str, default="False")
parser.add_argument("--nsamples", type=int, default=20000)
args = parser.parse_args()
model_type = args.model
savedata = True if args.save == "True" else False
nsamples = args.nsamples

# Configuration.
x0 = 0.1
n = 100
A = 10
dt = 0.1

# Select model and prior.
if model_type == "ckls":
    prior_bounds = np.array([[0, 40], [0, 10], [0, 2], [0, 1]])
    fname = "40_10_2_1"
    from models import ckls as model
elif model_type == "ou":
    prior_bounds = np.array([[0, 30], [0, 10], [0, 2]])
    fname = "30_10_2"
    from models import ou as model
elif model_type == "cir":
    prior_bounds = np.array([[0, 30], [0, 10], [0, 3]])
    fname = "30_10_3"
    from models import cir as model

# Instantiate model.
simulator = partial(finescale_em, x0, n, A, dt, model)

# Train network.
network_pretrainer(
    n=n,
    prior_bounds=prior_bounds,
    nsamples=nsamples,
    simulator=simulator,
    fname="NN/" + model_type.upper() + "/" + fname,
    savedata=savedata,
)
