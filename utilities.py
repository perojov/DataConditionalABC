import time
import torch
import numba as nb
import numpy as np
import ot
from typing import Callable
from math import log, sqrt, exp, pi
from random import uniform
import pytorch_lightning as pl
from neuralnetwork import PENDataModule
from neuralnetwork import MarkovExchangeableNeuralNetwork
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.stats import multivariate_normal as mvn
import warnings
from pytorch_lightning.tuner.tuning import Tuner

warnings.filterwarnings("ignore")
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def instantiate_model(model_type):
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

    return xo, prior_bounds, fname, model


def network_pretrainer(
    n: int,
    prior_bounds: np.ndarray,
    nsamples: int,
    simulator: Callable,
    fname: str,
    savedata=False,
):
    """Network trainer.

    Args:
        n (int): Observation length.
        prior_bounds (np.ndarray): Bounds for the uniform priors.
        nsamples (int): Number of samples for training.
        simulator (Callable): Numerical discretization method.
        fname (str): Path to store the model, e.g NN/CIR/30_20_3.
        savedata (bool, optional): Whether to store the data
        that is used for training. Defaults to False.
    """
    nparams = len(prior_bounds)
    train_paths = np.zeros((nsamples, n + 1))
    train_params = np.zeros((nsamples, nparams))

    # Generate parameters and paths.
    for i in range(nsamples):
        for j in range(nparams):
            train_params[i, j] = log(
                np.random.uniform(prior_bounds[j, 0], prior_bounds[j, 1])
            )
        train_paths[i] = simulator(np.exp(train_params[i]))

    # Create data module for PyTorch Lightning.
    train_paths = torch.Tensor(train_paths)
    train_params = torch.Tensor(train_params)
    tsize = int(nsamples * 0.8)
    data_module = PENDataModule(
        train_paths=train_paths[:tsize],
        train_params=train_params[:tsize],
        val_paths=train_paths[tsize:],
        val_params=train_params[tsize:],
    )

    # Neural network.
    net = MarkovExchangeableNeuralNetwork(nparams=nparams)

    # Setup trainer.
    early_stopping = EarlyStopping(monitor="val_loss", patience=100)
    trainer = pl.Trainer(
        enable_model_summary=None,
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        callbacks=[early_stopping],
    )
    trainer.fit(net, datamodule=data_module)
    trainer.save_checkpoint(fname + "_nn_model_" + str(nsamples) + ".ckpt")

    if savedata:
        torch.save(train_paths, fname + "_init_paths_" + str(nsamples))
        torch.save(train_params, fname + "_init_params_" + str(nsamples))


def prepare_dataset_and_train(
    net,
    nn_params,
    nn_paths,
    npart,
    pre_size,
    pre_tsize,
    tsize,
    pre_vsize,
    vsize,
    t,
    n,
    param_dim,
):
    """Append the new dataset for training, whilst keeping track of what is training and
    what is validation.

    Args:
        nn_params (torch.Tensor): Log parameter values.
        nn_paths (torch.Tensor): Simulated trajectories.
        pre_size (int): Size of the pretraining dataset.
        pre_tsize (int): Training size of the pretraining dataset.
        tsize (int): Training size of the newly computed dataset.
        pre_vsize (int): Training size of the pretraining dataset.
        vsize (_type_): Validation size of the newly computed dataset.
        t (int): Iteration.
        n (int): Trajectory length.
        param_dim (int): Dimension of the parameter.
    """
    # Prepare new dataset.
    train_paths = torch.zeros(pre_tsize + tsize * (t + 1), n)
    train_params = torch.zeros(pre_tsize + tsize * (t + 1), param_dim)
    val_paths = torch.zeros(pre_vsize + vsize * (t + 1), n)
    val_params = torch.zeros(pre_vsize + vsize * (t + 1), param_dim)

    # Pretrained data first.
    train_paths[:pre_tsize] = nn_paths[:pre_tsize]
    train_params[:pre_tsize] = nn_params[:pre_tsize]
    val_paths[:pre_vsize] = nn_paths[pre_tsize:pre_size]
    val_params[:pre_vsize] = nn_params[pre_tsize:pre_size]

    # Sampled paths.
    for k in range(t + 1):

        # Training chunks.
        tstart, tend = pre_size + k * npart, pre_size + k * npart + tsize
        train_paths[pre_tsize + k * tsize : pre_tsize + (k + 1) * tsize] = nn_paths[
            tstart:tend
        ]
        train_params[pre_tsize + k * tsize : pre_tsize + (k + 1) * tsize] = nn_params[
            tstart:tend
        ]

        # Validation chunks.
        vstart, vend = pre_size + k * npart + tsize, pre_size + (k + 1) * npart
        val_paths[pre_vsize + k * vsize : pre_vsize + (k + 1) * vsize] = nn_paths[
            vstart:vend
        ]
        val_params[pre_vsize + k * vsize : pre_vsize + (k + 1) * vsize] = nn_params[
            vstart:vend
        ]
    # Setup trainer and fit the neural network.
    datamodule = PENDataModule(train_paths, train_params, val_paths, val_params)
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=100)],
    )

    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(net, mode="power")
    trainer.fit(net, datamodule=datamodule)


@nb.njit(parallel=True)
def parallel_bs(
    lsis_x: np.ndarray,
    lsis_w: np.ndarray,
    thetas: np.ndarray,
    dt: float,
    backward: Callable,
    model: Callable,
):
    """Sample P backward paths for M particles in parallel.

    Args:
        lsis_x (np.ndarray): Lookahead SIS trajectories.
        lsis_w (np.ndarray): Lookahead SIS weights.
        thetas (np.ndarray): Array of parameters.
        dt (float): Timestep.
        backward (Callable): Backward simulation smoother.
        model (Callable): SDE model.

    Returns:
        np.ndarray: Backward paths.
    """
    npart, obs_len, npart_sim = lsis_x.shape
    bs_x = np.zeros((npart, npart_sim, obs_len))
    for j in nb.prange(npart_sim):
        for i in range(npart):
            bs_x[i, j] = backward(
                dt=dt,
                x=lsis_x[i],
                w=lsis_w[i],
                model=model,
                theta=thetas[i],
            )
    return bs_x


def compute_sl_ratios(
    lsis_x: torch.Tensor, bs_x: torch.Tensor, summaries: np.ndarray, net: Callable
):
    """Computes the ratio of the likelihoods of the summary statistics.

    Args:
        sl_lsis_paths (np.ndarray): Trajectories from the forward model.
        sl_lsis_weights (np.ndarray): Weights of the trajectories.
        summaries (np.ndarray): Exponential of the log summary statistics.
        thetas (np.ndarray): Array of parameters.

    Returns:
        np.ndarray: Log ratios for every summary statistic.
    """
    npart = len(lsis_x)

    # Synthetic likelihood ratios.
    sl = np.zeros(npart)

    # Compute the synthetic likelihoods per particle
    # and summary.
    condition_numbers = np.zeros(npart)
    for i in range(npart):

        # The covariance matrix is amenable to singularity therefore
        # we need to keep track of whether this happens. If it does,
        # the particle and summary are given a very low weight.
        error = False

        # Forward and backward summaries.
        fs, bs = net(lsis_x[i].T), net(bs_x[i])

        # Forward mean and covariance.
        mu_f, cov_f = fs.mean(axis=0), np.cov(fs.T)

        # Backward mean and covariance.
        mu_b, cov_b = bs.mean(axis=0), np.cov(bs.T)

        # p(s | theta) = N(s | mu, cov).
        try:
            f_sl = mvn(mu_f, cov_f).logpdf(summaries[i])
        except np.linalg.LinAlgError:
            error = True

        # p(s | theta, obs) = N(s | mu~, cov~).
        try:
            b_sl = mvn(mu_b, cov_b).logpdf(summaries[i])
        except np.linalg.LinAlgError:
            error = True

        condition_numbers[i] = np.linalg.cond(cov_b)
        # Compute the SL ratio in log space and store if
        # LinAlgError was not raised nor if it is NaN nor
        # if it's nearly singular.
        if error:
            sl[i] = -1000
        else:
            sl[i] = f_sl - b_sl

    return sl, condition_numbers


@nb.njit
def norm_logpdf(x, mu, sigma):
    """
    Implementation of the log pdf of a Gaussian
    distribution, in tune with numba.
    :param x: Point to evaluate.
    :param mu: Mean.
    :param sigma: Standard deviation
    :return:
    """
    sigsq = sigma * sigma
    return -log(sqrt(2 * pi * sigsq)) - ((x - mu) * (x - mu)) / (2 * sigsq)


@nb.njit
def norm_logpdf_vect(x, mu, sigma):
    """
    Implementation of the log pdf of a Gaussian
    distribution, in tune with numba.
    :param x: Point to evaluate.
    :param mu: Mean.
    :param sigma: Standard deviation
    :return:
    """
    sigsq = sigma * sigma
    return -np.log(sqrt(2 * pi * sigsq)) - ((x - mu) * (x - mu)) / (2 * sigsq)


@nb.njit
def elerian(x_s, x_t, dt, model, theta):
    """Elerian transition density of the Milstein scheme.

    Args:
        x_s (float): Starting state.
        x_t (float): Ending state.
        s (float): Time of starting state.
        dt (float): Timestep.
        theta (np.ndarray): Parameter.

    Returns:
        float: Probability.
    """

    # Drift, diffusion, derivative of diffusion.
    mu, sig, sig_x = model(x_s, theta)

    # When sig_x is zero, the Milstein scheme becomes
    # the Euler scheme and therefore the transition
    # density is the induced Gaussian.
    if sig_x == 0:
        # Euler transition density.
        return norm_logpdf(x_t, x_s + mu * dt, sig * sqrt(dt))
    A = sig * sig_x * dt * 0.5
    B = -0.5 * sig / sig_x + x_s + mu * dt - A
    z = (x_t - B) / A
    C = 1.0 / (sig_x**2 * dt)
    scz = sqrt(C * z)
    cpz = -0.5 * (C + z)
    ch = exp(scz + cpz) + exp(-scz + cpz)
    return log((1 / sqrt(z)) * ch / (2 * abs(A) * sqrt(2 * np.pi)))


@nb.njit
def logsumexp(x: np.ndarray) -> float:
    """
    Stable computation of the log sum of
    logs.
    :param x: Array of logs.
    :return: Logarithm of the summed logs.
    """
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


@nb.njit
def fast_mvnorm_pdf(x, mean, cov):
    """Evaluates the multivariate Gaussian pdf
    with given mean and covariance at x.

    Args:
        x (np.ndarray): Array
        mean (np.ndarray): Mean of the MVN.
        cov (np.ndarray): Covariance of the MVN.

    Returns:
        float: MVN evaluated at x.
    """
    vals, vecs = np.linalg.eigh(cov)
    logdet = np.sum(np.log(vals))
    valsinv = 1.0 / vals
    U = vecs * np.sqrt(valsinv)
    dim = len(vals)
    dev = x - mean
    maha = np.square(np.dot(dev, U)).sum()
    log2pi = np.log(2 * np.pi)
    return exp(-0.5 * (dim * log2pi + maha + logdet))


# @nb.njit
def fast_mvnorm_logpdf(x, mean, cov):
    vals, vecs = np.linalg.eigh(cov)
    logdet = np.sum(np.log(vals))
    valsinv = 1.0 / vals
    U = vecs * np.sqrt(valsinv)
    dim = len(vals)
    dev = x - mean
    maha = np.square(np.dot(dev, U)).sum()
    log2pi = np.log(2 * np.pi)
    return -0.5 * (dim * log2pi + maha + logdet)


@nb.njit
def mvnorm_pdf(x, mu, cov_det, cov_inv):
    """Hardcoded pdf of the multivarite Gaussian.
    The determinant and the inverse of the covariance are
    precomputed for efficiency.

    Args:
        x (np.ndarray): Parameter value
        mu (np.ndarray): Mean
        cov_det (np.float): Determinant of the covariance
        cov_inv (np.ndarray): Inverse of covariance.

    Returns:
        np.float: PDF evaluated in x.
    """
    d = len(x)
    return exp(-(1 / 2) * (x - mu) @ cov_inv @ (x - mu)) / sqrt((2 * pi) ** d * cov_det)


@nb.njit(parallel=True)
def compute_param_ratio(
    t: int,
    prior_bounds: np.ndarray,
    particles: np.ndarray,
    weights: np.ndarray,
    cov_det: np.ndarray,
    cov_inv: np.ndarray,
):
    M, d = particles[t].shape

    # Product of uniform priors.
    prior_prob = 1
    for j in range(d):
        diff = abs(prior_bounds[j, 1] - prior_bounds[j, 0])
        prior_prob *= 1 / diff
    logprior_prob = log(prior_prob)

    #
    for i in nb.prange(M):
        proposal_prob = 0
        for j in range(M):
            kernel_weight = mvnorm_pdf(
                particles[t, i],
                particles[t - 1, j],
                cov_det,
                cov_inv,
            )
            proposal_prob += kernel_weight * weights[t - 1, j]
        weights[t, i] = logprior_prob - log(proposal_prob)


@nb.njit(parallel=True)
def compute_unifweights(
    t: int,
    prior_bounds: np.ndarray,
    particles: np.ndarray,
    weights: np.ndarray,
    scale: np.ndarray,
):
    M, d = particles[t].shape

    prior_prob = 1
    for j in range(d):
        diff = abs(prior_bounds[j, 1] - prior_bounds[j, 0])
        prior_prob *= 1 / diff
    logprior_prob = log(prior_prob)

    # sig = np.zeros(d)
    # for k in range(d):
    # sig[k] = (1 / 2) * (maxes[k] - mins[k])

    # For each particle
    for i in nb.prange(M):

        # Store the mixture sum.
        proposal_prob = 0
        for j in range(M):

            # Per component weight.
            kernel_weight = 1.0
            for k in range(d):
                kernel_weight *= 1 / (2 * scale[k])

            proposal_prob += kernel_weight * weights[t - 1, j]
        weights[t, i] = logprior_prob - log(proposal_prob)


@nb.njit
def uniform_proposal(
    particles: np.ndarray,
    weights: np.ndarray,
    scale: np.ndarray,
    prior_bounds: np.ndarray,
):
    """Randomly pick a particle and perturb it by a Uniform.

    Args:
        particles (np.ndarray): Array of particles.
        weights (np.ndarray): Weights of particles.
        prior_bounds (np.ndarray): Array of prior bounds.

    Returns:
        np.ndarray: Parameter proposal.
    """

    # Store theta.
    d = len(prior_bounds)
    theta = np.zeros(d)

    while True:
        # Check whether theta is within bounds.
        flag = True

        # Sample particle.
        part = particles[random_choice(weights)]

        # Sample from kernel.
        for j in range(d):
            # Uniformly perturb each component.
            theta[j] = uniform(part[j] - scale[j], part[j] + scale[j])

            # Check if it is within bounds.
            if theta[j] < prior_bounds[j, 0] or theta[j] > prior_bounds[j, 1]:
                flag = False
                break
        if flag:
            return theta


@nb.njit
def gaussian_proposal(
    particles: np.ndarray,
    weights: np.ndarray,
    chol: np.ndarray,
    prior_bounds: np.ndarray,
):
    """Randomly pick a particle and perturb it by a Gaussian

    Args:
        particles (np.ndarray): Array of particles.
        weights (np.ndarray): Weights of particles.
        chol (np.ndarray): Cholesky decomposition of a covariance matrix.
        prior_bounds (np.ndarray): Array of prior bounds.

    Returns:
        np.ndarray: Parameter proposal.
    """
    d = len(prior_bounds)
    while True:
        # Resample and perturb.
        theta = chol @ np.random.randn(d) + particles[random_choice(weights)]
        # Check whether theta is within bounds.
        flag = True
        for i in range(d):
            if theta[i] < prior_bounds[i, 0] or theta[i] > prior_bounds[i, 1]:
                flag = False
                break
        if flag:
            return theta


@nb.njit
def prior_proposal(prior_bounds: np.ndarray):
    """Sample from a prior distribution

    Args:
        prior_bounds (np.ndarray): Array of prior bounds.

    Returns:
        np.ndarray: Parameter proposal.
    """
    d = len(prior_bounds)
    theta = np.zeros(d)
    for i in range(d):
        theta[i] = np.random.uniform(prior_bounds[i, 0], prior_bounds[i, 1])
    return theta


@nb.njit
def random_choice(w: np.ndarray):
    """Weighted random sample

    Args:
        w (np.ndarray): Array of weights.

    Returns:
        np.int: Sampled index.
    """
    return np.searchsorted(np.cumsum(w), np.random.random())


def distance(
    so: torch.Tensor,
    ss: torch.Tensor,
    mad: np.ndarray,
) -> torch.Tensor:
    """Weighted Euclidean distance.

    Args:
        so (torch.Tensor): Observed summary.
        ss (torch.Tensor): Simulated summary.
        mad (np.ndarray): Median absolute deviations.

    Returns:
        torch.Tensor: _description_
    """
    d = len(mad)
    dist = 0
    for i in range(d):
        scaled = (ss[i] - so[i]) / mad[i]
        dist += scaled**2
    return sqrt(dist)


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def calc_emd(ref_data_set, data_set, p=2):
    n = ref_data_set.shape[0]
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    if p == 2:
        M = ot.dist(ref_data_set, data_set)
        return np.sqrt(ot.emd2(a, b, M))
    elif p == 1:
        M = ot.dist(ref_data_set, data_set, metric="euclidean")
        return ot.emd2(a, b, M)
