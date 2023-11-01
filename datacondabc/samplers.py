import time
import torch
import numba as nb
import numpy as np
from math import exp, sqrt, log
from scipy.special import ive
from functools import partial
from typing import Callable
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn
import pytorch_lightning as pl
from datacondabc.nnets import PENDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datacondabc.utilities import (
    norm_logpdf,
    prior_proposal,
    gaussian_proposal,
    distance,
    compute_param_ratio,
    logsumexp,
)


def amortized_forward_abcsmc(
    obs: np.ndarray,
    npart: int,
    prior_bounds: np.ndarray,
    q: float,
    niter: int,
    net: Callable,
    simulator: Callable,
):
    """ABC-SMC with a pretrained network.

    Args:
        obs (np.ndarray): Observation
        npart (int): Number of particles.
        prior_bounds (np.ndarray): Bounds of the uniform prior.
        q (float): Percentile for choosing the thresholds.
        niter (int): Number of iterations.
        net (Callable): Sufficient statistics estimator.
        simulator (Callable): _description_
    """

    def _summarize(x: torch.Tensor):
        """Inner function to summarize a given path.

        Args:
            x (torch.Tensor): Sample path.

        Returns:
            torch.Tensor: S(x).
        """
        with torch.no_grad():
            return net(x)

    param_dim = len(prior_bounds)

    # Particle system and global covariance matrix..
    particles = np.zeros((niter, npart, param_dim))
    weights = np.zeros((niter, npart))
    abc_cov = np.zeros((param_dim, param_dim))

    # Thresholds for each iteration.
    thresholds = np.zeros(niter)
    thresholds[0] = np.inf
    weights[0] = 1 / npart
    mad = np.ones(param_dim)

    # Accepted summary statistics and all generated summaries.
    accepted_summaries = torch.zeros(npart, param_dim)
    obssummary = _summarize(torch.Tensor(obs))
    timing = np.zeros(niter)

    # Run ABC-SMC.
    for t in range(niter):
        # Instantiate the parameter proposal density.
        if t == 0:
            proposal = partial(prior_proposal, prior_bounds)
        else:
            proposal = partial(
                gaussian_proposal,
                particles[t - 1],
                weights[t - 1],
                np.linalg.cholesky(abc_cov),
                prior_bounds,
            )

        # Iterate over particles.
        total_sim = 0
        all_summaries = []
        start = time.time()
        for i in range(npart):
            if i % 1000 == 0:
                print("Particle progress", i)

            # Accept/reject.
            while True:
                total_sim += 1

                # Propose parameter.
                particle = proposal()

                # Sample trajectory and summarize.
                simsummary = _summarize(torch.Tensor(simulator(particle)))

                # Store the summary (to be used for the adaptive distance).
                all_summaries.append(simsummary)

                # Accept / reject step.
                if distance(obssummary, simsummary, mad) <= thresholds[t]:
                    # Store the particle.
                    particles[t, i] = particle

                    # Store the *accepted* summary (to compute the adapted distance).
                    accepted_summaries[i] = simsummary

                    # Particle is accepted, go to sample the next one.
                    break

        # Once all particles have been sampled, compute the weights.
        if t > 0:
            compute_param_ratio(
                t=t,
                prior_bounds=prior_bounds,
                particles=particles,
                weights=weights,
                cov_det=np.linalg.det(abc_cov),
                cov_inv=np.linalg.inv(abc_cov),
            )
        timing[t] = time.time() - start

        # Normalize weights and compute covariance matrix for the next round.
        weights[t] = np.exp(weights[t] - logsumexp(weights[t]))
        abc_cov = 2.0 * np.cov(particles[t].T, aweights=weights[t])

        if t == niter - 1:
            break

        # Compute the median absolute deviations.
        all_summaries = torch.stack(all_summaries)
        for i in range(param_dim):
            median = torch.median(all_summaries[:, i])
            mad[i] = torch.median(torch.abs(all_summaries[:, i] - median))

        # Compute the distances and pick the qth percentile.
        distances = np.zeros(npart)
        for i in range(npart):
            distances[i] = distance(obssummary, accepted_summaries[i], mad)
        new_threshold = np.percentile(distances, q)

        if new_threshold <= thresholds[t]:
            thresholds[t + 1] = new_threshold
        else:
            thresholds[t + 1] = thresholds[t] * 0.9

        # Compute the acceptance rate and stop if it's below 1.5%.
        acceptance_rate = (npart / total_sim) * 100
        print(
            "Round",
            t + 1,
            "\nThreshold",
            round(thresholds[t], 2),
            "\nAcceptance rate is",
            round(acceptance_rate, 2),
        )
        if t > 2 and acceptance_rate < 1.5:
            print("Ending ABC-SMC..")
            print("Round is ", t)
            break
    print("Saving until", t)
    return (
        particles[: t + 1],
        weights[: t + 1],
        timing[: t + 1],
    )


def amortized_backward_abcsmc(
    obs: np.ndarray,
    npart: int,
    npart_sim: int,
    nsubint: int,
    dt: float,
    prior_bounds: np.ndarray,
    q: float,
    niter: int,
    net,
    model: Callable,
    lookahead,
    backward,
):
    obs_len = len(obs)

    def _s(x: torch.Tensor):
        """Inner function to summarize a given path.

        Args:
            x (torch.Tensor): Sample path.

        Returns:
            torch.Tensor: S(x).
        """
        with torch.no_grad():
            return net(x)

    @nb.njit(parallel=True)
    def _parallel_bs(lsis_x: np.ndarray, lsis_w: np.ndarray, thetas: np.ndarray):
        """Sample P backward paths for M particles in parallel.

        Args:
            lsis_x (np.ndarray): Lookahead SIS trajectories.
            lsis_w (np.ndarray): Lookahead SIS weights.
            thetas (np.ndarray): Array of parameters.

        Returns:
            np.ndarray: Backward paths.
        """
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

    def _compute_sl_ratios(
        lsis_x: np.ndarray,
        lsis_w: np.ndarray,
        summaries: np.ndarray,
        thetas: np.ndarray,
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
        # Synthetic likelihood ratios.
        sl = np.zeros(npart)

        # Compute backward paths for the SL approximation and tensorize the forward paths.
        bs_x = torch.Tensor(
            _parallel_bs(
                lsis_x=lsis_x,
                lsis_w=lsis_w,
                thetas=thetas,
            )
        )
        lsis_x = torch.Tensor(lsis_x)

        # Compute the synthetic likelihoods per particle
        # and summary.
        condition_numbers = np.zeros(npart)
        for i in range(npart):
            # The covariance matrix is amenable to singularity therefore
            # we need to keep track of whether this happens. If it does,
            # the particle and summary are given a very low weight.
            error = False

            # Forward and backward summaries.
            fs, bs = _s(lsis_x[i].T), _s(bs_x[i])

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

    param_dim = len(prior_bounds)

    # Particle system and covariance matrix.
    particles = np.zeros((niter, npart, param_dim))
    weights = np.zeros((niter, npart))
    abc_cov = np.zeros((param_dim, param_dim))

    # Distances and thresholds for each iteration.
    thresholds = np.zeros(niter)
    thresholds[0] = np.inf
    mad = np.ones(param_dim)

    # Accepted summary statistics and all generated summaries.
    accepted_summaries = torch.zeros(npart, param_dim)
    obssummary = _s(torch.Tensor(obs))
    timing = np.zeros(niter)

    # Lookahead SIS paths and weights.
    lsis_x = np.zeros((npart, len(obs), npart_sim))
    lsis_w = np.zeros((npart, len(obs), npart_sim))

    # Run ABC-SMC.
    for t in range(niter):
        # Instantiate the parameter proposal density.
        if t == 0:
            proposal = partial(prior_proposal, prior_bounds)
        else:
            proposal = partial(
                gaussian_proposal,
                particles[t - 1],
                weights[t - 1],
                np.linalg.cholesky(abc_cov),
                prior_bounds,
            )

        # Iterate over particles.
        total_sim = 0
        all_summaries = []
        start = time.time()
        print(thresholds[t])
        for i in range(npart):
            if i % 100 == 0:
                print("Particle progress", i)

            # Accept/reject.
            while True:
                total_sim += 1

                # Propose parameter.
                particle = proposal()

                # Lookahead SIS.
                x, w = lookahead(
                    xo=obs,
                    A=nsubint,
                    P=npart_sim,
                    dt=dt,
                    model=model,
                    theta=particle,
                )

                # Sample trajectory and summarize.
                simsummary = _s(
                    torch.Tensor(backward(dt=dt, x=x, w=w, model=model, theta=particle))
                )

                # Store the summary (to be used for the adaptive distance).
                all_summaries.append(simsummary)

                # Accept / reject step.
                if distance(obssummary, simsummary, mad) < thresholds[t]:
                    # Store the particle.
                    particles[t, i] = particle

                    # Store the *accepted* summary (to compute the adapted distance).
                    accepted_summaries[i] = simsummary

                    # Store the forward particle system to compute the
                    # ratio of synthetic likelihoods.
                    lsis_x[i] = x
                    lsis_w[i] = w

                    # Particle is accepted, go to sample the next one.
                    break

        # Compute the synthetic likelihood ratios.
        synlik_ratios, condition_numbers = _compute_sl_ratios(
            lsis_x=lsis_x,
            lsis_w=lsis_w,
            summaries=accepted_summaries,
            thetas=particles[t],
        )

        # Once all particles have been sampled, compute the weights.
        if t > 0:
            compute_param_ratio(
                t=t,
                prior_bounds=prior_bounds,
                particles=particles,
                weights=weights,
                cov_det=np.linalg.det(abc_cov),
                cov_inv=np.linalg.inv(abc_cov),
            )

        timing[t] = time.time() - start

        # Cutoff.
        synlik_ratios[np.where(synlik_ratios >= 0)[0]] = 0
        for i in range(npart):
            if condition_numbers[i] >= 1000:
                synlik_ratios[i] = -1000

        # Add to weights and normalize.
        weights[t] += synlik_ratios
        weights[t] = np.exp(weights[t] - logsumexp(weights[t]))
        abc_cov = 2.0 * np.cov(particles[t].T, aweights=weights[t])

        if t == niter - 1:
            break

        all_summaries = torch.stack(all_summaries)
        for i in range(param_dim):
            median = torch.median(all_summaries[:, i])
            mad[i] = torch.median(torch.abs(all_summaries[:, i] - median))

        # Compute the distances and pick the qth percentile.
        if t == 0:
            ids = np.where(synlik_ratios != -1000)[0]
            dists = []
            for i in ids:
                dists.append(distance(obssummary, accepted_summaries[i], mad))
            new_threshold = np.percentile(dists, q)
        else:
            distances = np.zeros(npart)
            for i in range(npart):
                distances[i] = distance(obssummary, accepted_summaries[i], mad)
            new_threshold = np.percentile(distances, q)

        if new_threshold <= thresholds[t]:
            thresholds[t + 1] = new_threshold
        else:
            thresholds[t + 1] = thresholds[t] * 0.9
        acceptance_rate = (npart / total_sim) * 100
        print(
            "Round",
            t + 1,
            "\nThreshold",
            round(thresholds[t], 2),
            "\nAcceptance rate is",
            round(acceptance_rate, 2),
        )

        if t > 2 and acceptance_rate < 1.5:
            print("Ending ABC-SMC..")
            break
    return (
        particles[: t + 1],
        weights[: t + 1],
        timing[: t + 1],
    )


def dynamic_backward_abcsmc(
    obs: np.ndarray,
    npart: int,
    npart_sim: int,
    nsubint: int,
    dt: float,
    prior_bounds: np.ndarray,
    q: float,
    niter: int,
    init_paths: torch.Tensor,
    init_params: torch.Tensor,
    net: Callable,
    model: Callable,
    lookahead: Callable,
    backward: Callable,
):
    obs_len = len(obs)

    def _s(x: torch.Tensor):
        """Inner function to summarize a given path.

        Args:
            x (torch.Tensor): Sample path.

        Returns:
            torch.Tensor: S(x).
        """
        with torch.no_grad():
            return net(x)

    @nb.njit(parallel=True)
    def _parallel_bs(lsis_x: np.ndarray, lsis_w: np.ndarray, thetas: np.ndarray):
        """Sample P backward paths for M particles in parallel.

        Args:
            lsis_x (np.ndarray): Lookahead SIS trajectories.
            lsis_w (np.ndarray): Lookahead SIS weights.
            thetas (np.ndarray): Array of parameters.

        Returns:
            np.ndarray: Backward paths.
        """
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

    def _compute_sl_ratios(
        lsis_x: np.ndarray,
        lsis_w: np.ndarray,
        summaries: np.ndarray,
        thetas: np.ndarray,
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
        # Synthetic likelihood ratios.
        sl = np.zeros(npart)

        # Compute backward paths for the SL approximation and tensorize the forward paths.
        bs_x = torch.Tensor(
            _parallel_bs(
                lsis_x=lsis_x,
                lsis_w=lsis_w,
                thetas=thetas,
            )
        )
        lsis_x = torch.Tensor(lsis_x)

        # Compute the synthetic likelihoods per particle
        # and summary.
        condition_numbers = np.zeros(npart)
        for i in range(npart):
            # The covariance matrix is amenable to singularity therefore
            # we need to keep track of whether this happens. If it does,
            # the particle and summary are given a very low weight.
            error = False

            # Forward and backward summaries.
            fs, bs = _s(lsis_x[i].T), _s(bs_x[i])

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

    param_dim = len(prior_bounds)
    pre_size = len(init_paths)
    pre_tsize = int(pre_size * 0.8)
    pre_vsize = pre_size - pre_tsize
    tsize = int(npart * 0.8)
    vsize = npart - tsize

    # Particle system and covariance matrix.
    particles = np.zeros((niter, npart, param_dim))
    weights = np.zeros((niter, npart))
    abc_cov = np.zeros((param_dim, param_dim))

    # Distances and thresholds for each iteration.
    thresholds = np.zeros(niter)
    thresholds[0] = np.inf
    mad = np.ones(param_dim)

    # Accepted summary statistics and all generated summaries.
    accepted_summaries = torch.zeros(npart, param_dim)
    obssummaries = torch.zeros(niter, param_dim)
    obssummaries[0] = _s(torch.Tensor(obs))

    # Training data.
    nn_paths = torch.zeros(pre_size + niter * npart, len(obs))
    nn_params = torch.zeros(pre_size + niter * npart, param_dim)
    nn_paths[:pre_size] = init_paths
    nn_params[:pre_size] = init_params

    # Lookahead SIS paths and weights.
    lsis_x = np.zeros((npart, len(obs), npart_sim))
    lsis_w = np.zeros((npart, len(obs), npart_sim))
    timing = np.zeros(niter)
    acc_rates = np.zeros(niter)

    # Run ABC-SMC.
    for t in range(niter):
        # Instantiate the parameter proposal density.
        # This way one can simly write proposal() to
        # propose a parameter.
        if t == 0:
            proposal = partial(prior_proposal, prior_bounds)
        else:
            proposal = partial(
                gaussian_proposal,
                particles[t - 1],
                weights[t - 1],
                np.linalg.cholesky(abc_cov),
                prior_bounds,
            )

        # Iterate over particles.
        total_sim = 0
        all_summaries = []
        start = time.time()
        for i in range(npart):
            if i % 1000 == 0:
                print("Particle progress", i)

            # Accept/reject.
            while True:
                total_sim += 1

                # Propose parameter.
                particle = proposal()

                # Lookahead SIS.
                x, w = lookahead(
                    xo=obs,
                    A=nsubint,
                    P=npart_sim,
                    dt=dt,
                    model=model,
                    theta=particle,
                )

                # Sample trajectory and summarize.
                simsummary = _s(
                    torch.Tensor(backward(dt=dt, x=x, w=w, model=model, theta=particle))
                )

                # Store the summary (to be used for the adaptive distance).
                all_summaries.append(simsummary)

                # Accept if within distance.
                if distance(obssummaries[t], simsummary, mad) < thresholds[t]:
                    # Store the particle.
                    particles[t, i] = particle

                    # Store the *accepted* summary (to compute the adapted distance).
                    accepted_summaries[i] = simsummary

                    # Store the forward particle system to compute the
                    # ratio of synthetic likelihoods.
                    lsis_x[i] = x
                    lsis_w[i] = w

                    # Store
                    nn_params[pre_size + t * npart + i] = torch.Tensor(np.log(particle))
                    dists = np.sqrt(np.sum((x.T - obs) ** 2, axis=1))
                    nn_paths[pre_size + t * npart + i] = torch.Tensor(
                        x[:, np.argmin(dists)]
                    )
                    break

        # Compute the synthetic likelihood ratios.
        synlik_ratios, condition_numbers = _compute_sl_ratios(
            lsis_x=lsis_x,
            lsis_w=lsis_w,
            summaries=accepted_summaries,
            thetas=particles[t],
        )

        # Once all particles have been sampled, compute the weights.
        if t > 0:
            compute_param_ratio(
                t=t,
                prior_bounds=prior_bounds,
                particles=particles,
                weights=weights,
                cov_det=np.linalg.det(abc_cov),
                cov_inv=np.linalg.inv(abc_cov),
            )

        timing[t] = time.time() - start
        print("Computing time", timing[t])
        # Cutoff.
        synlik_ratios[np.where(synlik_ratios >= 0)[0]] = -1000
        for i in range(npart):
            if condition_numbers[i] >= 1000:
                synlik_ratios[i] = -1000

        # Add to weights and
        weights[t] += synlik_ratios
        weights[t] = np.exp(weights[t] - logsumexp(weights[t]))
        abc_cov = 2.0 * np.cov(particles[t].T, aweights=weights[t])

        acceptance_rate = (npart / total_sim) * 100
        acc_rates[t] = acceptance_rate

        if t == niter - 1:
            break

        all_summaries = torch.stack(all_summaries)
        for i in range(param_dim):
            median = torch.median(all_summaries[:, i])
            mad[i] = torch.median(torch.abs(all_summaries[:, i] - median))

        if t == 0:
            ids = np.where(synlik_ratios != -1000)[0]
            dists = []
            for i in ids:
                dists.append(distance(obssummaries[t], accepted_summaries[i], mad))
            thresholds[t + 1] = np.percentile(dists, q)
        else:
            distances = np.zeros(npart)
            for i in range(npart):
                distances[i] = distance(obssummaries[t], accepted_summaries[i], mad)
            thresholds[t + 1] = np.percentile(distances, q)

        if thresholds[t + 1] > thresholds[t]:
            thresholds[t + 1] = thresholds[t] * 0.95

        print(
            "Round",
            t + 1,
            "\nThreshold",
            round(thresholds[t], 2),
            "\nAcceptance rate is",
            round(acceptance_rate, 2),
        )

        if t > 2 and acceptance_rate < 1.5:
            print("Ending ABC-SMC..")
            break

        if t < niter - 1:
            # Prepare new dataset.
            train_paths = torch.zeros(pre_tsize + tsize * (t + 1), len(obs))
            train_params = torch.zeros(pre_tsize + tsize * (t + 1), param_dim)
            val_paths = torch.zeros(pre_vsize + vsize * (t + 1), len(obs))
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
                train_paths[
                    pre_tsize + k * tsize : pre_tsize + (k + 1) * tsize
                ] = nn_paths[tstart:tend]
                train_params[
                    pre_tsize + k * tsize : pre_tsize + (k + 1) * tsize
                ] = nn_params[tstart:tend]

                # Validation chunks.
                vstart, vend = pre_size + k * npart + tsize, pre_size + (k + 1) * npart
                val_paths[
                    pre_vsize + k * vsize : pre_vsize + (k + 1) * vsize
                ] = nn_paths[vstart:vend]
                val_params[
                    pre_vsize + k * vsize : pre_vsize + (k + 1) * vsize
                ] = nn_params[vstart:vend]
            # Setup trainer and fit the neural network.
            datamodule = PENDataModule(train_paths, train_params, val_paths, val_params)
            early_stopping = EarlyStopping(monitor="val_loss", patience=100)
            trainer = pl.Trainer(
                max_epochs=1000,
                accelerator="gpu",
                devices=1,
                callbacks=[early_stopping],
            )

            start = time.time()
            trainer.fit(net, datamodule=datamodule)
            end = time.time()
            print("Training time", end - start)
            timing[t] += end - start

            # Update summary.
            obssummaries[t + 1] = _s(torch.Tensor(obs))
            print("Next summary", torch.exp(obssummaries[t + 1]))

    return particles[: t + 1], weights[: t + 1], timing[: t + 1], acc_rates[: t + 1]


def dynamic_forward_abcsmc(
    obs: np.ndarray,
    npart: int,
    prior_bounds: np.ndarray,
    q: float,
    niter: int,
    init_paths: torch.Tensor,
    init_params: torch.Tensor,
    net: Callable,
    simulator: Callable,
):
    """ABC-SMC with a pretrained network.

    Args:
        obs (np.ndarray): Observation
        npart (int): Number of particles.
        prior_bounds (np.ndarray): Bounds of the uniform prior.
        q (float): Percentile for choosing the thresholds.
        niter (int): Number of iterations.
        net (_type_): Sufficient statistics estimator.
        simulator (_type_): _description_
    """

    def _summarize(x: torch.Tensor):
        """Inner function to summarize a given path.

        Args:
            x (torch.Tensor): Sample path.

        Returns:
            torch.Tensor: S(x).
        """
        with torch.no_grad():
            return net(x)

    param_dim = len(prior_bounds)
    pre_size = len(init_paths)
    pre_tsize = int(pre_size * 0.8)
    pre_vsize = pre_size - pre_tsize
    tsize = int(npart * 0.8)
    vsize = npart - tsize

    # Particle system and covariance matrix.
    particles = np.zeros((niter, npart, param_dim))
    weights = np.zeros((niter, npart))
    abc_cov = np.zeros((param_dim, param_dim))

    # Thresholds for each iteration.
    thresholds = np.zeros(niter)
    thresholds[0] = np.inf
    weights[0] = np.log(1 / npart)
    mad = np.ones(param_dim)

    # Training data.
    nn_paths = torch.zeros(pre_size + niter * npart, len(obs))
    nn_params = torch.zeros(pre_size + niter * npart, param_dim)
    nn_paths[:pre_size] = init_paths
    nn_params[:pre_size] = init_params

    # Accepted summary statistics and all generated summaries.
    accepted_summaries = torch.zeros(npart, param_dim)
    obssummaries = torch.zeros(niter, param_dim)
    obssummaries[0] = _summarize(torch.Tensor(obs))
    timing = np.zeros(niter)
    acc_rates = np.zeros(niter)

    # Run ABC-SMC.
    for t in range(niter):
        # Instantiate the parameter proposal density.
        if t == 0:
            proposal = partial(prior_proposal, prior_bounds)
        else:
            proposal = partial(
                gaussian_proposal,
                particles[t - 1],
                weights[t - 1],
                np.linalg.cholesky(abc_cov),
                prior_bounds,
            )

        # Iterate over particles.
        total_sim = 0
        distances = []
        all_summaries = []
        start = time.time()

        for i in range(npart):
            if i % 100 == 0:
                print("Particle progress", i)
            # Accept/reject.
            while True:
                total_sim += 1
                particle = proposal()
                xtilde = torch.Tensor(simulator(particle))
                simsummary = _summarize(xtilde)
                all_summaries.append(simsummary)
                d = distance(obssummaries[t], simsummary, mad)
                if d <= thresholds[t]:
                    distances.append(d)
                    particles[t, i] = particle
                    accepted_summaries[i] = simsummary
                    nn_params[pre_size + t * npart + i] = torch.Tensor(np.log(particle))
                    nn_paths[pre_size + t * npart + i] = xtilde
                    break

        # Once all particles have been sampled, compute the weights.
        if t > 0:
            compute_param_ratio(
                t=t,
                prior_bounds=prior_bounds,
                particles=particles,
                weights=weights,
                cov_det=np.linalg.det(abc_cov),
                cov_inv=np.linalg.inv(abc_cov),
            )
        timing[t] = time.time() - start

        # Normalize weights and compute covariance matrix for the next round.
        weights[t] = np.exp(weights[t] - logsumexp(weights[t]))
        abc_cov = 2.0 * np.cov(particles[t].T, aweights=weights[t])

        acceptance_rate = (npart / total_sim) * 100
        acc_rates[t] = acceptance_rate

        if t == niter - 1:
            break

        all_summaries = torch.stack(all_summaries)
        for i in range(param_dim):
            median = torch.median(all_summaries[:, i])
            mad[i] = torch.median(torch.abs(all_summaries[:, i] - median))

        distances = np.zeros(npart)
        for i in range(npart):
            distances[i] = distance(obssummaries[t], accepted_summaries[i], mad)

        thresholds[t + 1] = np.percentile(distances, q)
        if thresholds[t + 1] > thresholds[t]:
            thresholds[t + 1] = thresholds[t] * 0.95

        print(
            "Round",
            t + 1,
            "\nThreshold",
            round(thresholds[t], 2),
            "\nAcceptance rate is",
            round(acceptance_rate, 2),
        )
        if t > 2 and acceptance_rate < 1.5:
            print("Ending ABC-SMC..")
            print("Round is ", t)
            break

        if t < niter - 1:
            # Prepare new dataset.
            train_paths = torch.zeros(pre_tsize + tsize * (t + 1), len(obs))
            train_params = torch.zeros(pre_tsize + tsize * (t + 1), param_dim)
            val_paths = torch.zeros(pre_vsize + vsize * (t + 1), len(obs))
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
                train_paths[
                    pre_tsize + k * tsize : pre_tsize + (k + 1) * tsize
                ] = nn_paths[tstart:tend]
                train_params[
                    pre_tsize + k * tsize : pre_tsize + (k + 1) * tsize
                ] = nn_params[tstart:tend]

                # Validation chunks.
                vstart, vend = pre_size + k * npart + tsize, pre_size + (k + 1) * npart
                val_paths[
                    pre_vsize + k * vsize : pre_vsize + (k + 1) * vsize
                ] = nn_paths[vstart:vend]
                val_params[
                    pre_vsize + k * vsize : pre_vsize + (k + 1) * vsize
                ] = nn_params[vstart:vend]
            # Setup trainer and fit the neural network.
            datamodule = PENDataModule(train_paths, train_params, val_paths, val_params)
            early_stopping = EarlyStopping(monitor="val_loss", patience=100)
            trainer = pl.Trainer(
                max_epochs=1000,
                accelerator="gpu",
                devices=1,
                callbacks=[early_stopping],
            )

            start = time.time()
            trainer.fit(net, datamodule=datamodule)
            end = time.time()
            print("Training time", end - start)
            timing[t] += end - start

            # Update summary.
            obssummaries[t + 1] = _summarize(torch.Tensor(obs))
            print("Next summary", torch.exp(obssummaries[t + 1]))

    print("Saving until", t)
    return particles[: t + 1], weights[: t + 1], timing[: t + 1], acc_rates[: t + 1]


def mcmc(theta, prior, dt, xo, iterations=100000):
    """
    def loglik(r, dt, theta):
        b, a, s = theta
        c = (2 * a) / ((1 - np.exp(-a * dt)) * (s**2))
        q = (2 * a * b) / (s**2) - 1
        logpdf = 0
        for t in range(1, len(r)):
            u = c * r[t - 1] * np.exp(-a * dt)
            v = c * r[t]
            z = 2 * np.sqrt(u * v)
            logpdf += (
                np.log(c)
                - (u + v)
                + (q / 2) * np.log(v / u)
                + np.log(ive(q, 2 * np.sqrt(u * v)))
                + z
            )
        return logpdf
    """

    def loglik(x, dt, theta):
        a, b, s = theta
        logpdf = 0
        for t in range(1, len(x)):
            mu = a + (x[t - 1] - a) * exp(-b * dt)
            var = (s**2) * (1 - exp(-2 * b * dt)) / (2 * b)
            logpdf += norm_logpdf(x[t], mu, sqrt(var))
        return logpdf

    p = len(theta)

    eps = 10e-8
    s_d = (2.4**2) / p

    # Covariance for the proposal density.
    cov = np.diag([0.1] * p)

    # Markov chain in which we store the parameter values.
    chain = np.zeros((iterations, p))
    chain[0] = theta

    # The initial likelihood.
    lik = loglik(xo, dt, theta)

    # Start MCMC.
    for t in range(1, iterations):
        # Track progress.
        if t % 1000 == 0:
            print(t)

        # Sample new parameter.
        theta_new = np.random.multivariate_normal(theta, cov)
        lik_new = loglik(xo, dt, theta_new)

        # Compute prior probabilities.
        prior_old = np.sum([prior[k].logpdf(theta[k]) for k in range(p)])
        prior_new = np.sum([prior[k].logpdf(theta_new[k]) for k in range(p)])

        # Metropolis-Hastings ratio.
        alpha = lik_new + prior_new - (lik + prior_old)
        if np.random.uniform(0, 1) < np.exp(alpha):
            theta = theta_new
            lik = lik_new
        chain[t] = theta
        if t > 10000 and t % 1000 == 0:
            # Adaptive covariance MCMC.
            cov = s_d * np.cov(chain[:t].T) + s_d * eps * np.eye(p, p)
    return chain
