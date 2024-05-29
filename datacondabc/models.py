import numpy as np
import numba as nb
from math import sqrt, log


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
def nonlinear(x: float, theta: np.ndarray):
    """Drift and diffusion of the Cox--Ingersoll--Ross model.

    Args:                                                                                        x (float): State
        theta (np.ndarray): Parameter

    Returns:
        (float, float): Drift and diffusion.
    """
    a, b, s = theta
    return b * (a - x) + sqrt(x), s * sqrt(x)


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
def schlogl(x: float, theta: np.ndarray):
    """Drift and diffusion of the Schlogl model.

    Args:
        x (float): State.
        theta (np.ndarray): Parameter.

    Returns:
        (float, float, float, float): Propensities.
    """
    k1, k2, k4 = theta
    k3 = 1e-3
    A = 1e5
    B = 2e5
    a1 = 0.0 if x <= 1.0 else A * k1 * x * (x - 1) / 2
    a2 = 0.0 if x <= 2.0 else k2 * x * (x - 1) * (x - 2) / 6
    a3 = k3 * B
    a4 = 0.0 if x <= 0 else k4 * x
    return a1 - a2 + a3 - a4, sqrt(a1 + a2 + a3 + a4)


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


@nb.njit
def lv(x: float, theta: np.ndarray):
    """Drift and diffusion of the Lotka-Volterra model.

    Args:
        x (float): State.
        theta (np.ndarray): Parameter.

    Returns:
        (float, float, float, float): Propensities.
    """
    k1, k2, k3 = theta
    x1, x2 = x
    mu = np.array([k1 * x1 - k2 * x1 * x2, k2 * x1 * x2 - k3 * x2])
    cov11 = k1 * x1 + k2 * x1 * x2
    cov12 = -k2 * x1 * x2
    cov21 = -k2 * x1 * x2
    cov22 = k2 * x1 * x2 + k3 * x2
    cov = np.array([[cov11, cov12], [cov21, cov22]])

    # Square root of a 2x2 matrix formula
    I = np.eye(2)
    tau = cov11 + cov22
    delta = cov11 * cov22 - cov12 * cov21
    s = sqrt(delta)
    t = sqrt(tau + 2 * s)
    return mu, cov, (1 / t) * (cov + s * I)
