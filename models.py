import numpy as np
import numba as nb
from math import sqrt
from math import log


@nb.njit
def ou(x: float, theta: np.ndarray):
    """Drift and diffusion of the Ornstein--Uhlenbeck model.

    Args:
        x (float): State
        theta (np.ndarray): Parameter

    Returns:
        (float, float): Drift and diffusion.
    """
    a, b, s = theta
    return b * (a - x), s


@nb.njit
def cir(x: float, theta: np.ndarray):
    """Drift and diffusion of the Cox--Ingersoll--Ross model.

    Args:
        x (float): State
        theta (np.ndarray): Parameter

    Returns:
        (float, float): Drift and diffusion.
    """
    a, b, s = theta
    return b * (a - x), s * sqrt(x)


@nb.njit
def ckls(x: float, theta: np.ndarray):
    """Drift and diffusion of the CKLS model.

    Args:
        x (float): State
        theta (np.ndarray): Parameter

    Returns:
        (float, float): Drift and diffusion.
    """
    a, b, s, g = theta
    return b * (a - x), s * x**g


@nb.njit
def nonlinear_cir(x: float, theta: np.ndarray):
    """Drift and diffusion of the Cox--Ingersoll--Ross model.

    Args:
        x (float): State
        theta (np.ndarray): Parameter

    Returns:
        (float, float): Drift and diffusion.
    """
    a, b, s = theta
    return b * (a - x) + sqrt(x), s * sqrt(x)


@nb.njit
def nonlinear_ckls(x: float, theta: np.ndarray):
    """Drift and diffusion of the Cox--Ingersoll--Ross model.

    Args:
        x (float): State
        theta (np.ndarray): Parameter

    Returns:
        (float, float): Drift and diffusion.
    """
    a, b, s, g = theta
    return b * (a - x) + log(x**10), s * x**g
