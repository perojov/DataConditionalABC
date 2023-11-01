import numpy as np
import numba as nb
from math import sqrt
from typing import Callable
from random import normalvariate
from datacondabc.utilities import norm_logpdf, logsumexp, random_choice


@nb.njit
def finescale_em(
    x0: float, n: int, A: int, dt: float, model: Callable, theta: np.ndarray
):
    """Euler-Maruyama method.

    Args:
        x0 (float): Initial state.
        n (int): Number of observations (without x0).
        A (int): Number of subintervals.
        dt (float): Timestep.
        theta (np.ndarray): Parameter.
    """
    # New timestep.
    dt /= A

    # Precompute the root of the timestep.
    sq_dt = sqrt(dt)

    # Allocate space for the forward trajectory.
    x = np.zeros(n * A + 1)
    x[0] = x0

    # Euler-Maruyama.
    for i in range(1, n * A + 1):
        mu, sig = model(x[i - 1], theta)
        x[i] = abs(x[i - 1] + mu * dt + sig * sq_dt * normalvariate(0, 1))
    return x[::A]


@nb.njit(parallel=True)
def lookahead_sis(
    xo: np.ndarray, A: int, P: int, dt: float, model: Callable, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential-Importance-Sampling with a lookahead.

    Args:
        xo (np.ndarray): Observation.
        A (int): Number of subintervals.
        P (int): Number of particles.
        dt (float): Timestep of the coarse grid.
        model (Callable): Drift and diffusion of the model.
        theta (np.ndarray): Parameter.

    Returns:
        (np.ndarray, np.ndarray): Weighted particle system.
    """
    # Timesteps
    dt /= A
    sq_dt = sqrt(dt)

    # Particle system.
    n = len(xo) - 1
    x = np.zeros((n + 1, P))
    w = np.zeros((n + 1, P))
    x[0] = xo[0]
    w[0] = 1 / P

    # Particle parallelism.
    for p in nb.prange(P):
        # Initialize trajectory.
        path = xo[0]

        for i in range(n):
            # Forward propagate using Euler-Maruyama.
            # up to the second to last point.
            for _ in range(A - 1):
                mu, sig = model(path, theta)
                path = abs(path + mu * dt + sig * sq_dt * normalvariate(0, 1))

            # Weigh the observational sample
            # as p(x^o_{i + 1} | path).
            mu, sig = model(path, theta)
            w[i + 1, p] = norm_logpdf(xo[i + 1], path + mu * dt, sig * sq_dt)

            # Propagate one more step and store.
            path = abs(path + mu * dt + sig * sq_dt * normalvariate(0, 1))
            x[i + 1, p] = path
    return x, w


@nb.njit
def smoother(
    dt: float,
    x: np.ndarray,
    w: np.ndarray,
    model: Callable,
    theta: np.ndarray,
) -> np.ndarray:
    """Backward simulation particle smoothing.

    Args:
        dt (float): Timestep of the coarse grid.
        x (np.ndarray): Particles.
        w (np.ndarray): Particle weights.
        theta (np.ndarray): Parameter.

    Returns:
        np.ndarray: Backward trajectory.
    """
    # Shape of the particle array.
    n, P = x.shape

    # Take away the initial point.
    n -= 1

    # Backward path.
    xback = np.zeros(n + 1)
    xback[0] = x[0, 0]
    wback = np.zeros(P)
    sq_dt = sqrt(dt)

    # Sample the endpoint.
    xback[-1] = x[-1, random_choice(np.exp(w[-1] - logsumexp(w[-1])))]

    for k in range(1, n):
        # Reverse index.
        curr_id = n - k

        for p in range(P):
            # Mean and variance.
            mu, sig = model(x[curr_id, p], theta)
            wback[p] = (
                norm_logpdf(xback[curr_id + 1], x[curr_id, p] + mu * dt, sig * sq_dt)
                + w[curr_id, p]
            )

        randi = random_choice(np.exp(wback - logsumexp(wback)))
        xback[curr_id] = x[curr_id, randi]

    return xback
