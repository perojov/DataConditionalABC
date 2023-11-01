import torch
import numba as nb
import numpy as np
import ot
from typing import Callable
from math import log, sqrt, exp, pi
import pytorch_lightning as pl
from datacondabc.nnets import PENDataModule, MarkovExchangeableNeuralNetwork
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


@nb.njit
def norm_logpdf(x: float, mu: float, sigma: float):
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
def mvnorm_pdf(x: np.ndarray, mu: np.ndarray, cov_det: np.ndarray, cov_inv: np.ndarray):
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


@nb.njit
def logsumexp(x: np.ndarray) -> float:
    """
    Stable computation of the log sum of logs.
    :param x: Array of logs.
    :return: Logarithm of the summed logs.
    """
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


@nb.njit
def random_choice(w: np.ndarray):
    """Weighted random sample.

    Args:
        w (np.ndarray): Array of weights.

    Returns:
        np.int: Sampled index.
    """
    return np.searchsorted(np.cumsum(w), np.random.random())


@nb.njit(parallel=True)
def compute_param_ratio(
    t: int,
    prior_bounds: np.ndarray,
    particles: np.ndarray,
    weights: np.ndarray,
    cov_det: np.ndarray,
    cov_inv: np.ndarray,
):
    """Compute the ratio of the prior and the proposal distribution
    for all particles at once.

    Args:
        t (int): Round id.
        prior_bounds (np.ndarray): Bounds of the uniform prior distribution.
        particles (np.ndarray): Parameter values.
        weights (np.ndarray): Parameter weights.
        cov_det (np.ndarray): Determinant of the ABC-SMC covariance.
        cov_inv (np.ndarray): Inverse of the ABC-SMC covariance.
    """
    M, d = particles[t].shape

    # Product of uniform priors.
    prior_prob = 1
    for j in range(d):
        diff = abs(prior_bounds[j, 1] - prior_bounds[j, 0])
        prior_prob *= 1 / diff
    logprior_prob = log(prior_prob)

    # Gaussian mixture probability.
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
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stopping],
    )
    trainer.fit(net, datamodule=data_module)
    trainer.save_checkpoint(fname + "_nn_model_" + str(nsamples) + ".ckpt")

    if savedata:
        torch.save(train_paths, fname + "_init_paths_" + str(nsamples))
        torch.save(train_params, fname + "_init_params_" + str(nsamples))


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
