import torch
import numba as nb
import numpy as np
import ot
from tqdm import tqdm
from typing import Callable
from math import log, sqrt, exp, pi
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from datacondabc.nnets import PENDataModule
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
def mvnorm_logpdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray):
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

    # Determinant of a 2x2 matrix.
    cov_det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]

    # Inverse of a 2x2 matrix.
    cov_inv = np.zeros((2, 2))
    cov_inv[0, 0] = cov[1, 1]
    cov_inv[0, 1] = -cov[0, 1]
    cov_inv[1, 0] = -cov[1, 0]
    cov_inv[1, 1] = cov[0, 0]
    cov_inv *= 1 / cov_det
    return (
        -(d / 2) * log(2 * pi)
        - (1 / 2) * log(cov_det)
        - (1 / 2) * (x - mu) @ cov_inv @ (x - mu)
    )


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


@nb.njit
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
    for i in range(M):
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
) -> torch.Tensor:
    """Weighted Euclidean distance.

    Args:
        so (torch.Tensor): Observed summary.
        ss (torch.Tensor): Simulated summary.
        mad (np.ndarray): Median absolute deviations.

    Returns:
        torch.Tensor: _description_
    """
    d = len(so)
    dist = 0
    for i in range(d):
        scaled = ss[i] - so[i]
        dist += scaled**2
    return sqrt(dist)


def distance_mad(
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
