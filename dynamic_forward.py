import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numba as nb

nb.set_num_threads(8)
import torch
import numpy as np
from functools import partial
from approxsimulators import finescale_em
import argparse
from neuralnetwork import MarkovExchangeableNeuralNetwork
from samplers import dynamic_forward_abcsmc

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="CKLS")
parser.add_argument("--save", type=str, default="False")
parser.add_argument("--nsamples", type=int, default=20000)
parser.add_argument("--id", type=str, default="")

args = parser.parse_args()
model_type = args.model.upper()
savedata = True if args.save == "True" else False
nsamples = args.nsamples
fid = args.id

# Select model and prior.
if model_type == "CKLS":
    xo = np.load("CKLS/data.npy")
    prior_bounds = np.array([[0, 40], [0, 10], [0, 2], [0, 1]])
    fname = "40_10_2_1"
    from models import ckls as model
elif model_type == "OU":
    xo = np.load("OU/data.npy")
    prior_bounds = np.array([[0, 30], [0, 20], [0, 2]])
    fname = "30_20_2"
    from models import ou as model
elif model_type == "CIR":
    xo = np.load("CIR/data.npy")
    prior_bounds = np.array([[0, 30], [0, 10], [0, 2]])
    fname = "30_10_2"
    from models import cir as model
elif model_type == "NONLIN":
    xo = np.load("NONLIN/data.npy")
    prior_bounds = np.array([[0, 30], [0, 10], [0, 2]])
    fname = "30_10_2"
    from models import nonlinear as model

# Configuration.
dt = 0.1
A = 10

# Quantile for determining the ABC thresholds.
q = 50

# ABC-SMC configuration.
niter = 10
npart = 10000

# Neural network.
nnet_name = "NN/" + model_type + "/" + fname + "_nn_model_" + str(nsamples) + ".ckpt"

net = MarkovExchangeableNeuralNetwork().load_from_checkpoint(nnet_name)

data_name = "NN/" + model_type + "/" + fname
seq_init_paths = torch.load(data_name + "_init_paths_" + str(nsamples))
seq_init_params = torch.load(data_name + "_init_params_" + str(nsamples))

# Simulator.
simulator = partial(finescale_em, xo[0], len(xo) - 1, A, dt, model)

# Run sampler.
particles, weights, timing = dynamic_forward_abcsmc(
    obs=xo,
    npart=npart,
    prior_bounds=prior_bounds,
    q=q,
    niter=niter,
    net=net,
    init_paths=seq_init_paths,
    init_params=seq_init_params,
    simulator=simulator,
)
np.save(model_type + "/" + "Forward/particles" + str(fid) + ".npy", particles)
np.save(model_type + "/" + "Forward/weights" + str(fid) + ".npy", weights)
np.save(model_type + "/" + "Forward/timing" + str(fid) + ".npy", np.array(timing))
