import numpy as np
import numba as nb
from math import exp, sqrt
from random import normalvariate


@nb.njit
def ou(x0: float, n: int, dt: float, theta: np.ndarray):
    """Exact sample from the Ornstein-Uhlenbeck model.

    Args:
        x0 (float): Initial state.
        n (int): Number of observations (without x0).
        dt (float): Timestep
        theta (np.ndarray): Parameter.

    Returns:
        np.ndarray: Ornstein-Uhlenbeck trajectory.
    """

    # Extract parameters.
    a, b, s = theta

    # Allocate space for the forward trajectory.
    x = np.zeros(n + 1)
    x[0] = x0

    # Incrementally sample from the transition density.
    for i in range(1, n + 1):
        mu = a + (x[i - 1] - a) * exp(-b * dt)
        var = (s**2) * (1 - exp(-2 * b * dt)) / (2 * b)
        x[i] = normalvariate(mu, sqrt(var))
    return x


@nb.njit
def cir(x0: float, n: int, dt: float, theta: np.ndarray):
    """Exact sample from the Cox-Ingersoll-Ross model.

    Args:
        x0 (float): Initial state.
        n (int): Number of observations (without x0).
        dt (float): Timestep
        theta (np.ndarray): Parameter.

    Returns:
        np.ndarray: Cox-Ingersoll-Ross trajectory.
    """
    # Extract parameters.
    a, b, s = theta

    # Allocate space for the forward trajectory.
    x = np.zeros(n + 1)
    x[0] = x0

    # Incrementally sample from the transition density.
    for i in range(1, n + 1):
        c = 2 * b / ((1 - exp(-b * dt)) * s**2)
        ncp = 2 * c * x[i - 1] * exp(-b * dt)
        df = 4 * a * b / s**2
        x[i] = np.random.noncentral_chisquare(df, nonc=ncp) / (2 * c)

    return x
