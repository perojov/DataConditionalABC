import argparse
import numpy as np
from functools import partial

# Model and simulator.
from simulators.approxsim import finescale_em
from utilities import network_pretrainer


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save", type=str, default="True")
parser.add_argument("--nsamples", type=int, default=50000)
parser.add_argument("--init", type=str, default="0.1")
parser.add_argument("--n", type=str, default="100")
parser.add_argument("--A", type=str, default="10")
parser.add_argument("--dt", type=str, default="0.1")
args = parser.parse_args()
model_type = args.model
savedata = True if args.save == "True" else False
nsamples = args.nsamples

# Configuration.
x0 = float(args.init)
n = int(args.n)
A = int(args.A)
dt = float(args.dt)

# Select model and prior.
if model_type == "ckls":
    prior_bounds = np.array([[0, 40], [0, 10], [0, 4], [0, 1]])
    fname = "40_10_4_1"
    from models import ckls as model
elif model_type == "ou":
    prior_bounds = np.array([[0, 30], [0, 10], [0, 2]])
    fname = "30_10_2"
    from models import ou as model
elif model_type == "cir":
    prior_bounds = np.array([[0, 20], [0, 10], [0, 3]])
    fname = "30_10_3"
    from models import cir as model
elif model_type == "nonlin":
    prior_bounds = np.array([[0, 30], [0, 10], [0, 2]])
    fname = "30_10_2"
    from models import nonlinear as model
elif model_type == "nonlinckls":
    prior_bounds = np.array([[0, 20], [0, 5], [0, 3], [0, 1]])
    fname = "20_5_3_1"
    from models import nonlinear_ckls as model

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
