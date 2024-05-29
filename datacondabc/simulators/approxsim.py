import numpy as np
import numba as nb
from math import sqrt, exp, log
from typing import Callable
from random import normalvariate
from datacondabc.utilities import norm_logpdf, logsumexp, random_choice, mvnorm_logpdf
import time


@nb.njit
def finescale_em(
    model: Callable, x0: float, n: int, A: int, dt: float, theta: np.ndarray
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
    x = np.zeros((1, 1, n * A + 1))
    x[0, 0, 0] = x0

    # Euler-Maruyama.
    for i in range(1, n * A + 1):
        mu, sig = model(x[0, 0, i - 1], theta)
        x[0, 0, i] = max(
            x[0, 0, i - 1] + mu * dt + sig * sq_dt * normalvariate(0, 1), 1e-6
        )
    return x[:, :, ::A]


@nb.njit
def finescale_em_schlogl(x0: float, n: int, nsubint: int, dt: float, theta: np.ndarray):
    """Euler-Maruyama method.

    Args:
        x0 (float): Initial state.
        n (int): Number of observations (without x0).
        A (int): Number of subintervals.
        dt (float): Timestep.
        theta (np.ndarray): Parameter.
    """
    # New timestep.
    dt /= nsubint

    # Precompute the root of the timestep.
    sq_dt = sqrt(dt)

    # Allocate space for the forward trajectory.
    x = np.zeros(n * nsubint + 1)
    x[0] = x0

    k1, k2, k4 = theta
    k3 = 1e-3
    A = 1e5
    B = 2e5

    # Euler-Maruyama.
    for i in range(1, n * nsubint + 1):
        a1 = A * k1 * x[i - 1] * (x[i - 1] - 1) / 2
        a2 = k2 * x[i - 1] * (x[i - 1] - 1) * (x[i - 1] - 2) / 6
        a3 = k3 * B
        a4 = k4 * x[i - 1]
        mu = a1 - a2 + a3 - a4
        sig = sqrt(a1 + a2 + a3 + a4)
        x[i] = max(x[i - 1] + mu * dt + sig * sq_dt * normalvariate(0, 1), 1e-6)
    return x[::nsubint]


@nb.njit
def finescale_em_2d(
    x0: np.ndarray,
    n: int,
    A: int,
    dt: float,
    theta: np.ndarray,
):
    """Euler-Maruyama method for a 2D SDE.

    Args:
        x0 (np.ndarray): Initial state.
        n (int): Number of observations (without x0).
        A (int): Number of subintervals.
        dt (float): Timestep.
        model (Callable): Returns the mean and covariance.
        theta (np.ndarray): Parameter.
    """
    # Adjust timestep for fine-scale simulation
    dt /= A

    # Precompute the square root of the timestep
    sq_dt = np.sqrt(dt)

    # Allocate space for the forward trajectory
    x = np.zeros((1, 2, n * A + 1))
    x[0, :, 0] = x0
    k1, k2, k3 = theta
    mu = np.zeros(2)
    cov = np.zeros((2, 2))
    I = np.eye(2)

    # Euler-Maruyama simulation
    for i in range(1, n * A + 1):
        x1, x2 = x[0, :, i - 1]
        mu[0] = k1 * x1 - k2 * x1 * x2
        mu[1] = k2 * x1 * x2 - k3 * x2
        cov[0, 0] = k1 * x1 + k2 * x1 * x2
        cov[0, 1] = -k2 * x1 * x2
        cov[1, 0] = -k2 * x1 * x2
        cov[1, 1] = k2 * x1 * x2 + k3 * x2
        # Square root of a 2x2 matrix formula
        tau = cov[0, 0] + cov[1, 1]
        delta = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        s = sqrt(delta)
        t = sqrt(tau + 2 * s)
        # Generate a 2D normal random vector
        dw = np.random.normal(0, 1, 2)
        # Update the state; for 2D, you can directly multiply component-wise
        x[0, :, i] = x[0, :, i - 1] + mu * dt + ((1 / t) * (cov + s * I)) @ dw * sq_dt
        x[0, 0, i] = max(x[0, 0, i], 1e-6)
        x[0, 1, i] = max(x[0, 1, i], 1e-6)
    # Downsample the fine-scale path to the coarser time scale
    return x[:, :, ::A]


@nb.njit(parallel=True)
def lookahead_sis(
    xo: np.ndarray, A: int, P: int, dt: float, model: Callable, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential-Importance-Sampling with lookahead.

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


@nb.njit(parallel=True)
def lookahead_sis_2d(
    xo: np.ndarray,
    A: int,
    P: int,
    dt: float,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential-Importance-Sampling with lookahead.

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
    _, d, n = xo.shape
    k1, k2, k3 = theta
    n -= 1

    # Timesteps
    dt /= A
    sq_dt = sqrt(dt)

    # Particle system.
    w = np.zeros((P, n + 1))
    w[:, 0] = 1 / P
    x = np.zeros((P, d, n + 1))
    x[:, 0, 0] = xo[0, 0, 0]
    x[:, 1, 0] = xo[0, 1, 0]
    I = np.eye(2)

    # Particle parallelism.
    for p in nb.prange(P):
        mu = np.zeros(2)
        cov = np.zeros((2, 2))
        path = xo[0, :, 0]

        for i in range(n):
            for _ in range(A):

                # Mean and covariance.
                mu[0] = k1 * path[0] - k2 * path[0] * path[1]
                mu[1] = k2 * path[0] * path[1] - k3 * path[1]
                cov[0, 0] = k1 * path[0] + k2 * path[0] * path[1]
                cov[0, 1] = -k2 * path[0] * path[1]
                cov[1, 0] = -k2 * path[0] * path[1]
                cov[1, 1] = k2 * path[0] * path[1] + k3 * path[1]

                # Square root of a 2x2 matrix formula
                tau = cov[0, 0] + cov[1, 1]
                delta = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
                s = sqrt(delta)
                t = sqrt(tau + 2 * s)

                # Store second to last point.
                x[p, :, i + 1] = path

                # Propagate.
                dw = np.random.normal(0, 1, 2)
                path = path + mu * dt + ((1 / t) * (cov + s * I)) @ dw * sq_dt

                path[0] = max(path[0], 1e-6)
                path[1] = max(path[1], 1e-6)

            # Weight wrt second to last state.
            w[p, i + 1] = mvnorm_logpdf(
                xo[0, :, i + 1], x[p, :, i + 1] + mu * dt, cov * 0.5
            )
            x[p, :, i + 1] = path

    return x, w


@nb.njit
def smoother_2d(
    dt: float,
    x: np.ndarray,
    w: np.ndarray,
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
    P, d, n = x.shape
    k1, k2, k3 = theta

    # Take away the initial point.
    n -= 1

    # Backward path.
    xback = np.zeros((1, d, n + 1))
    xback[0, :, 0] = x[0, :, 0]
    wback = np.zeros(P)

    # Sample the endpoint.
    pid = random_choice(np.exp(w[:, -1] - logsumexp(w[:, -1])))
    xback[0, :, -1] = x[pid, :, -1]
    mu = np.zeros(2)
    cov = np.zeros((2, 2))
    for k in range(1, n):
        # Reverse index.
        curr_id = n - k

        for p in range(P):
            # Mean and variance.
            x1, x2 = x[p, :, curr_id]
            mu[0] = k1 * x1 - k2 * x1 * x2
            mu[1] = k2 * x1 * x2 - k3 * x2
            cov[0, 0] = k1 * x1 + k2 * x1 * x2
            cov[0, 1] = -k2 * x1 * x2
            cov[1, 0] = -k2 * x1 * x2
            cov[1, 1] = k2 * x1 * x2 + k3 * x2
            wback[p] = (
                mvnorm_logpdf(
                    xback[0, :, curr_id + 1], x[p, :, curr_id] + mu * dt, cov * dt
                )
                + w[p, curr_id]
            )

        randi = random_choice(np.exp(wback - logsumexp(wback)))
        xback[0, :, curr_id] = x[randi, :, curr_id]

    return xback


@nb.njit(parallel=True)
def lookahead_sis_schlogl(
    xo: np.ndarray,
    A: int,
    P: int,
    dt: float,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential-Importance-Sampling with lookahead.

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
    _, d, n = xo.shape
    k1, k2, k4 = theta
    k3 = 1e-3
    C = 1e5
    B = 2e5
    n -= 1

    # Timesteps
    dt /= A
    sq_dt = sqrt(dt)

    # Particle system.
    w = np.zeros((P, n + 1))
    w[:, 0] = 1 / P
    x = np.zeros((P, d, n + 1))
    x[0, 0, 0] = xo[0, 0, 0]

    # Particle parallelism.
    for p in nb.prange(P):
        path = xo[0, 0, 0]
        mu = 0.0
        var = 0.0

        for i in range(n):
            for _ in range(A):
                # Mean and variance.
                a1 = C * k1 * path * (path - 1) / 2
                a2 = k2 * path * (path - 1) * (path - 2) / 6
                a3 = k3 * B
                a4 = k4 * path

                # a1 = 0.0 if path <= 1.0 else A * k1 * path * (path - 1) / 2
                # a2 = 0.0 if path <= 2.0 else k2 * path * (path - 1) * (path - 2) / 6
                # a3 = k3 * B
                # a4 = 0.0 if path <= 0 else k4 * path

                mu = a1 - a2 + a3 - a4
                var = a1 + a2 + a3 + a4

                # Store second to last point.
                x[p, 0, i + 1] = path

                # Propagate.
                dw = np.random.normal(0, 1)
                path = max(path + mu * dt + sqrt(var) * dw * sq_dt, 1e-6)

            # Weight wrt second to last state.
            w[p, i + 1] = norm_logpdf(
                xo[0, 0, i + 1], x[p, 0, i + 1] + mu * dt, sqrt(var * 0.01)
            )
            x[p, 0, i + 1] = path

    return x, w


@nb.njit
def smoother_schlogl(
    dt: float,
    x: np.ndarray,
    w: np.ndarray,
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
    P, d, n = x.shape
    k1, k2, k4 = theta
    k3 = 1e-3
    A = 1e5
    B = 2e5

    # Take away the initial point.
    n -= 1

    # Backward path.
    xback = np.zeros((1, d, n + 1))
    xback[0, 0, 0] = x[0, 0, 0]
    wback = np.zeros(P)

    # Sample the endpoint.
    pid = random_choice(np.exp(w[:, -1] - logsumexp(w[:, -1])))
    xback[0, 0, -1] = x[pid, 0, -1]
    for k in range(1, n):
        # Reverse index.
        curr_id = n - k

        for p in range(P):
            # Mean and variance.
            x1 = x[p, 0, curr_id]
            a1 = A * k1 * x1 * (x1 - 1) / 2
            a2 = k2 * x1 * (x1 - 1) * (x1 - 2) / 6
            a3 = k3 * B
            a4 = k4 * x1
            mu = a1 - a2 + a3 - a4
            var = a1 + a2 + a3 + a4
            wback[p] = (
                norm_logpdf(
                    xback[0, 0, curr_id + 1], x[p, 0, curr_id] + mu * dt, sqrt(var * dt)
                )
                + w[p, curr_id]
            )

        randi = random_choice(np.exp(wback - logsumexp(wback)))
        xback[0, 0, curr_id] = x[randi, 0, curr_id]

    return xback


@nb.njit(parallel=True)
def lookahead_sis_cov_pos(
    model: Callable,
    cov_pos: int,
    xo: np.ndarray,
    A: int,
    P: int,
    dt: float,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential-Importance-Sampling with lookahead.

    Args:
        xo (np.ndarray): Observation.
        A (int): Number of subintervals.
        P (int): Number of particles.
        cov_pos (int): Position where to compute the cov.
        dt (float): Timestep of the coarse grid.
        model (Callable): Drift and diffusion of the model.
        theta (np.ndarray): Parameter.

    Returns:
        (np.ndarray, np.ndarray): Weighted particle system.
    """
    # Timesteps
    h = dt / A
    ts = h * cov_pos
    sq_h = sqrt(h)
    sq_ts = sqrt(ts)

    # Particle system.
    n = xo.shape[2] - 1
    x = np.zeros((P, 1, n + 1))
    w = np.zeros((P, n + 1))
    x[:, 0, 0] = xo[0, 0, 0]
    w[:, 0] = 1 / P

    # Particle parallelism.
    for p in nb.prange(P):
        # Initialize trajectory.
        path = xo[0, 0, 0]

        for i in range(n):
            # Forward propagate using Euler-Maruyama.
            for _ in range(A - cov_pos):
                mu, sig = model(path, theta)
                path = max(path + mu * h + sig * sq_h * normalvariate(0, 1), 1e-6)

            # Weigh the observational sample
            # as p(x^o_{i + 1} | path).
            mu, sig = model(path, theta)
            w[p, i + 1] = norm_logpdf(xo[0, 0, i + 1], path + mu * ts, sig * sq_ts)

            # Propagate a few more step and store.
            for _ in range(cov_pos):
                mu, sig = model(path, theta)
                path = max(path + mu * h + sig * sq_h * normalvariate(0, 1), 1e-6)
            x[p, 0, i + 1] = path
    return x, w


@nb.njit
def smoother(
    dt: float,
    x: np.ndarray,
    w: np.ndarray,
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
    P, d, n = x.shape
    k1, k2, k4 = theta
    k3 = 1e-3
    A = 1e5
    B = 2e5

    # Take away the initial point.
    n -= 1

    # Backward path.
    xback = np.zeros((1, d, n + 1))
    xback[0, 0, 0] = x[0, 0, 0]
    wback = np.zeros(P)

    # Sample the endpoint.
    pid = random_choice(np.exp(w[:, -1] - logsumexp(w[:, -1])))
    xback[0, 0, -1] = x[pid, 0, -1]

    for k in range(1, n):
        # Reverse index.
        curr_id = n - k

        for p in range(P):
            # Mean and variance.
            x1 = x[p, 0, curr_id]
            a1 = A * k1 * x1 * (x1 - 1) / 2
            a2 = k2 * x1 * (x1 - 1) * (x1 - 2) / 6
            a3 = k3 * B
            a4 = k4 * x1
            mu = a1 - a2 + a3 - a4
            var = a1 + a2 + a3 + a4

            wback[p] = (
                norm_logpdf(
                    xback[0, 0, curr_id + 1], x[p, 0, curr_id] + mu * dt, sqrt(var * dt)
                )
                + w[p, curr_id]
            )

        randi = random_choice(np.exp(wback - logsumexp(wback)))
        xback[0, 0, curr_id] = x[randi, 0, curr_id]
    return xback
