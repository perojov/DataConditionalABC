import time
import torch
import numba as nb
import numpy as np
from math import exp, sqrt, log
from scipy.special import ive
from functools import partial
from typing import Callable
import seaborn as sns
from tqdm import tqdm

from scipy.stats import multivariate_normal as mvn
import pytorch_lightning as pl
from neuralnetwork import PENDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from neuralnetwork import MarkovExchangeableNeuralNetwork
from utilities import norm_logpdf
from utilities import prior_proposal
from utilities import gaussian_proposal
from utilities import distance
from utilities import compute_param_ratio
from utilities import logsumexp
from utilities import prepare_dataset_and_train
from utilities import parallel_bs
from utilities import compute_sl_ratios


class ABCSMC:
    def __init__(
        self,
        obs: np.ndarray,
        npart_param: int,
        prior_bounds: np.ndarray,
        niters: int,
        alpha: float,
        net: Callable,
        cont_learn: bool,
        nsubint: int | None,
        dt: float | None,
        em: Callable | None,
        lsis: Callable | None,
        smoother: Callable | None,
        npart_sim: int | None,
        backward_abc: bool,
        model: Callable | None,
        data_name: str,
        model_type: str,
        fid: str,
        nsamples: int,
    ):
        """ABC-SMC.

        Args:
            obs (np.ndarray): Observation
            npart_param (int): Number of particles
            prior_bounds (np.ndarray): Prior bounds
            niters (int): Number of iterations
            alpha (float): alpha quantile for the distances
            net (Callable): Neural network
            cont_learn (bool): Whether to retrain the network
            nsubint (int | None): Number of subintervals for the simulators
            dt (float | None): Observational timestep
            em (Callable | None): Forward simulator
            lsis (Callable | None): Lookahead SIS
            smoother (Callable | None): Particle smoothing
            npart_sim (int | None): Number of particles for the simulator
            backward_abc (bool): Whether to use the data-conditional simulator
            model (Callable | None): Drift and diffusion
            data_name (str): Data path
            model_type (str): For which model is the inference carried for
            fid (str): File identifier (for saving)
            nsamples (int): Number of samples for pretraining
        """
        self.obs = obs
        self.npart_param = npart_param
        self.prior_bounds = prior_bounds
        self.niters = niters
        self.alpha = alpha
        self.net = net
        self.cont_learn = cont_learn
        self.nsubint = nsubint
        self.dt = dt
        self.lsis = lsis
        self.smoother = smoother
        self.npart_sim = npart_sim
        self.backward_abc = backward_abc
        self.model = model
        self.data_name = data_name
        self.model_type = model_type
        self.fid = fid
        self.nsamples = nsamples

        # Parameter dimension. It is assumed
        # that all parameters are unknown.
        self.param_dim = len(prior_bounds)

        # (Parameter) particle system. Different from the
        # particle system of the data-conditional simulator.
        self.particles = np.zeros((niters, npart_param, self.param_dim))
        self.weights = np.zeros((niters, npart_param))
        self.param_cov = np.eye(self.param_dim)

        # Median absolute deviations.
        # This is required to compute the weighted Euclidean distances.
        # Reference: "Adapting the ABC distance function" by Dennis Prangle.
        self.mad = np.ones(self.param_dim)

        # Store the computing time (without training)
        # and the acceptance rates.
        self.total_comp_time = np.zeros(niters)
        self.acceptance_rates = np.zeros(niters)

        # Auxiliary.
        self.summaries = torch.zeros(npart_param, self.param_dim)
        self.obssummary = None
        self.threshold = np.inf
        self.weights[0] = np.log(1 / npart_param)

        if backward_abc:
            # Instantiate Lookahead SIS and the smoother as functions depending
            # only on the parameter.
            self.lsis_partial = partial(lsis, obs, nsubint, npart_sim, dt, model)
            self.smoother_partial = partial(smoother, dt, model)
            self.folder_name = "Backward"
        else:
            self.folder_name = "Forward"
            self.em = partial(em, obs[0], len(obs) - 1, nsubint, dt, model)

    def _distance(self, summary: torch.Tensor):
        """Weighted Euclidean distance.

        Args:
            summary (torch.Tensor): Simulated summary.

        Returns:
            torch.Tensor: _description_
        """
        dist = 0
        for i in range(self.param_dim):
            scaled = (summary[i] - self.obssummary[i]) / self.mad[i]
            dist += scaled**2
        return sqrt(dist)

    def _forward_round(self, t: int):
        """Runs one round using the forward simulator.

        Args:
            t (int): ABC-SMC iteration.

        Returns:
            (list, int): _description_
        """
        total_sim = 0
        offset = self.nsamples + t * self.npart_param
        self.all_summaries = []
        for i in tqdm(range(self.npart_param)):

            # Sample particles until acceptance.
            while True:
                # Simulation count.
                total_sim += 1

                # Sample.
                particle = self.proposal()
                trajectory = torch.Tensor(self.em(particle))
                summary = self.net(trajectory)
                self.all_summaries.append(summary)

                # Compute distance.
                if self._distance(summary) < self.threshold:
                    self.particles[t, i] = particle
                    self.summaries[i] = summary
                    if self.cont_learn:
                        self.nn_params[offset + i] = torch.Tensor(np.log(particle))
                        self.nn_paths[offset + i] = trajectory
                    break
        self.acceptance_rates[t] = self.npart_param / total_sim
        self.all_summaries = torch.stack(self.all_summaries)

    def _backward_round(self, t: int):
        """Runs one round using the data-conditional simulator.

        Args:
            t (int): ABC-SMC iteration.

        Returns:
            (list, int): _description_
        """
        lsis_x = np.zeros((self.npart_param, len(self.obs), self.npart_sim))
        lsis_w = np.zeros((self.npart_param, len(self.obs), self.npart_sim))
        total_sim = 0
        offset = self.nsamples + t * self.npart_param
        self.all_summaries = []
        for i in tqdm(range(self.npart_param)):

            # Sample particles until acceptance.
            while True:
                # Simulation count.
                total_sim += 1

                # Sample.
                particle = self.proposal()
                x, w = self.lsis_partial(particle)
                summary = self.net(torch.Tensor(self.smoother_partial(x, w, particle)))
                self.all_summaries.append(summary)

                # Accept/reject.
                if self._distance(summary) < self.threshold:
                    self.particles[t, i] = particle
                    self.summaries[i] = summary
                    lsis_x[i] = x
                    lsis_w[i] = w
                    if self.cont_learn:
                        self.nn_params[offset + i] = torch.Tensor(np.log(particle))
                        # Pick closest forward trajectory.
                        dists = np.sqrt(np.sum((x.T - self.obs) ** 2, axis=1))
                        self.nn_paths[offset + i] = torch.Tensor(x[:, np.argmin(dists)])
                    break
        self.acceptance_rates[t] = self.npart_param / total_sim
        self.all_summaries = torch.stack(self.all_summaries)
        return lsis_x, lsis_w

    def _compute_new_threshold(self, t: int):
        """Compute the new threshold.

        Args:
            t (int): ABC-SMC iteration.
        """
        for i in range(self.param_dim):
            median = torch.median(self.all_summaries[:, i])
            self.mad[i] = torch.median(torch.abs(self.all_summaries[:, i] - median))

        # For Backward ABC the first iteration
        # contains a lot of information regarding the
        # parameters. We can utilize this by only
        # computing the distances where the weights
        # are non-negligible.
        if t == 0 and self.backward_abc:
            ids = np.where(self.synlik_ratios != -1000)[0]
            distances = []
            for i in ids:
                distances.append(self._distance(self.summaries[i]))
        else:
            # Compute weighted Euclidean distances
            distances = np.zeros(self.npart_param)
            for i in range(self.npart_param):
                distances[i] = self._distance(self.summaries[i])

        # New threshold
        new_threshold = np.percentile(distances, self.alpha)
        if new_threshold >= self.threshold:
            self.threshold *= 0.95
        else:
            self.threshold = new_threshold

    def _instantiate_proposal(self, t: int):
        """Instantiates the proposal distribution.

        Sample from the prior if ABC-SMC is
        at the initial round. Otherwise,
        use the Gaussian proposal with optimal
        covariance.

        Args:
            t (int): ABC-SMC iteration.

        Returns:
            Callable: Proposal distribution.
        """
        if t == 0:
            return partial(prior_proposal, self.prior_bounds)
        return partial(
            gaussian_proposal,
            self.particles[t - 1],
            self.weights[t - 1],
            np.linalg.cholesky(self.param_cov),
            self.prior_bounds,
        )

    def _compute_param_weights(self, t: int):
        """Compute the ratio between the prior and the
        proposal distribution. Calls a numba jitted function.

        Args:
            t (int): ABC-SMC iteration.
        """
        compute_param_ratio(
            t=t,
            prior_bounds=self.prior_bounds,
            particles=self.particles,
            weights=self.weights,
            cov_det=np.linalg.det(self.param_cov),
            cov_inv=np.linalg.inv(self.param_cov),
        )

        if self.backward_abc:
            self.weights[t] += self.synlik_ratios
        self.weights[t] = np.exp(self.weights[t] - logsumexp(self.weights[t]))

    def _compute_synlik_ratios(self, t: int, lsis_x: np.ndarray, lsis_w: np.ndarray):
        bs_x = torch.Tensor(
            parallel_bs(
                lsis_x=lsis_x,
                lsis_w=lsis_w,
                thetas=self.particles[t],
                dt=self.dt,
                backward=self.smoother,
                model=self.model,
            )
        )
        synlik_ratios, condition_numbers = compute_sl_ratios(
            torch.Tensor(lsis_x), bs_x, self.summaries, self.net
        )
        synlik_ratios[np.where(synlik_ratios >= 0)[0]] = -1000
        cntr = 0
        for i in range(self.npart_param):
            if condition_numbers[i] >= 1000:
                synlik_ratios[i] = -1000
                cntr += 1
        print("Condition numbers >= 1000:", cntr)
        return synlik_ratios

    def run(self):
        for t in range(self.niters):
            # Instantiate the parameter proposal distribution.
            self.proposal = self._instantiate_proposal(t)

            # Run ABC-SMC round.
            print("Round %d, epsilon %f" % (t + 1, round(self.threshold, 4)))
            with torch.no_grad():

                # The observational summary will change if continual learning
                # is allowed, otherwise it will stay fixed.
                self.obssummary = self.net(torch.Tensor(self.obs))

                start = time.time()
                if self.backward_abc:
                    lsis_x, lsis_w = self._backward_round(t)
                    self.synlik_ratios = self._compute_synlik_ratios(t, lsis_x, lsis_w)
                else:
                    self._forward_round(t)
            print("Acceptance rate", round(self.acceptance_rates[t], 2))
            if t > 0:
                self._compute_param_weights(t)

            self.total_comp_time[t] = time.time() - start

            # Compute parameter covariance for next iteration.
            self.weights[t] = np.exp(self.weights[t] - logsumexp(self.weights[t]))
            self.param_cov = 2.0 * np.cov(self.particles[t].T, aweights=self.weights[t])

            if t == self.niters - 1:
                break

            # Compute new threshold.
            self._compute_new_threshold(t)

            if t > 2 and self.acceptance_rates[t] * 100 < 1.5:
                print("Ending ABC-SMC..")
                break

            # Train here.
            if self.cont_learn:
                prepare_dataset_and_train(
                    net=self.net,
                    nn_params=self.nn_params,
                    nn_paths=self.nn_paths,
                    npart=self.npart_param,
                    pre_size=self.pre_size,
                )


def forward_abcsmc_round(
    particles: np.ndarray,
    proposal: Callable,
    net: Callable,
    simulator: Callable,
    obssummary: torch.Tensor,
    threshold: float,
    mad: np.ndarray,
    nn_params: torch.Tensor,
    nn_paths: torch.Tensor,
    offset: int,
    continual_learning: bool,
):
    # Number of parameters and the parameter dimension.
    npart, param_dim = particles.shape

    # Store all and the accepted summaries.
    summaries = torch.zeros(npart, param_dim)
    all_summaries = []

    # Count the number of times the simulator is called.
    total_sim = 0
    start = time.time()
    for i in range(npart):
        if i % 100 == 0:
            print("Particle progress", i)

        # Sample particles until acceptance.
        while True:
            # Simulation count.
            total_sim += 1

            # Sample.
            particle = proposal()
            trajectory = torch.Tensor(simulator(particle))
            summary = net(trajectory)
            all_summaries.append(summary)

            # Compute distance.
            if distance(obssummary, summary, mad) < threshold:
                particles[i] = particle
                summaries[i] = summary
                if continual_learning:
                    nn_params[offset + i] = torch.Tensor(np.log(particles[i]))
                    nn_paths[offset + i] = trajectory
                break
    elapsed_time = time.time() - start
    return torch.stack(all_summaries), summaries, elapsed_time, total_sim


def backward_abcsmc_round(
    obs: np.ndarray,
    particles: np.ndarray,
    proposal: Callable,
    net: Callable,
    lsis_x: np.ndarray,
    lsis_w: np.ndarray,
    lsis: Callable,
    smoother: Callable,
    obssummary: torch.Tensor,
    threshold: float,
    mad: np.ndarray,
    nn_params: torch.Tensor,
    nn_paths: torch.Tensor,
    offset: int,
    continual_learning: bool,
):
    # Number of parameters and the parameter dimension.
    npart, param_dim = particles.shape

    # Store all and the accepted summaries.
    summaries = torch.zeros(npart, param_dim)
    all_summaries = []

    # Count the number of times the simulator is called.
    total_sim = 0
    start = time.time()
    for i in range(npart):
        if i % 100 == 0:
            print("Particle progress", i)

        # Sample particles until acceptance.
        while True:
            # Simulation count.
            total_sim += 1

            # Sample.
            particle = proposal()
            x, w = lsis(particle)
            summary = net(torch.Tensor(smoother(particle)))
            all_summaries.append(summary)

            # Compute distance.
            if distance(obssummary, summary, mad) < threshold:
                particles[i] = particle
                summaries[i] = summary
                # Store the forward particle system to compute the ratio of synthetic likelihoods.
                lsis_x[i] = x
                lsis_w[i] = w
                if continual_learning:
                    nn_params[offset + i] = torch.Tensor(np.log(particles[i]))
                    # Pick closest forward trajectory.
                    dists = np.sqrt(np.sum((x.T - obs) ** 2, axis=1))
                    nn_paths[offset + i] = torch.Tensor(x[:, np.argmin(dists)])
                break
    elapsed_time = time.time() - start
    return torch.stack(all_summaries), summaries, elapsed_time, total_sim


def abcsmc(
    obs: np.ndarray,
    npart: int,
    prior_bounds: np.ndarray,
    niter: int,
    continual_learning: bool,
    net: Callable,
    simulator: Callable,
    lsis: Callable | None,
    smoother: Callable | None,
    npart_sim: int,
    nsubint: int,
    alpha: float,
    dt: float,
    model: Callable,
    data_name: str,
    model_type: str,
    backward_abc: bool,
    fid: str,
    nsamples: int,
):
    """ABC-SMC.

    Args:
        obs (np.ndarray): Observation
        npart (int): Number of particles.
        prior_bounds (np.ndarray): Bounds of the uniform prior.
        q (float): Percentile for choosing the thresholds.
        niter (int): Number of iterations.
        net (_type_): Sufficient statistics estimator.
        simulator (_type_): _description_
    """

    def _instantiate_proposal(t: int):
        """Instantiates the proposal distribution.

        Sample from the prior if ABC-SMC is
        at the initial round. Otherwise,
        use the Gaussian proposal with optimal
        covariance.

        Args:
            t (int): ABC-SMC iteration.

        Returns:
            Callable: Proposal distribution
        """
        if t == 0:
            return partial(prior_proposal, prior_bounds)
        return partial(
            gaussian_proposal,
            particles[t - 1],
            weights[t - 1],
            np.linalg.cholesky(param_cov),
            prior_bounds,
        )

    # Parameter dimension. It is assumed
    # that all parameters are unknown.
    param_dim = len(prior_bounds)

    # (Parameter) particle system. Different from the
    # particle system of the data-conditional simulator.
    particles = np.zeros((niter, npart, param_dim))
    weights = np.zeros((niter, npart))

    # Median absolute deviations.
    # This is required to compute the weighted Euclidean distances.
    # Reference: "Adapting the ABC distance function" by Dennis Prangle.
    mad = np.ones(param_dim)

    # Store the computing time (without training)
    # and the acceptance rates.
    total_comp_time = np.zeros(niter)
    acceptance_rates = np.zeros(niter)

    # Allocate space for the particle system of the
    # data-conditional simulator.
    if backward_abc:
        # Lookahead SIS paths and weights.
        lsis_x = np.zeros((npart, len(obs), npart_sim))
        lsis_w = np.zeros((npart, len(obs), npart_sim))

        # Instantiate Lookahead SIS and the smoother as functions depending
        # only on the parameter.
        lsis_partial = partial(lsis, obs, nsubint, npart_sim, dt, model)
        smoother_partial = partial(smoother, dt, model)
        folder_name = "Backward"
    else:
        folder_name = "Forward"

    # Configuration for continual learning.
    if continual_learning:

        # Training and validation sizes for the pretrained
        # PEN, and for the additional datasets.
        init_paths = torch.load(data_name + "_init_paths_" + str(nsamples))
        init_params = torch.load(data_name + "_init_params_" + str(nsamples))
        pre_size = len(init_paths)
        pre_tsize = int(pre_size * 0.8)
        pre_vsize = pre_size - pre_tsize
        tsize = int(npart * 0.8)
        vsize = npart - tsize

        # Allocate space.
        nn_paths = torch.zeros(pre_size + niter * npart, len(obs))
        nn_params = torch.zeros(pre_size + niter * npart, param_dim)
        nn_paths[:pre_size] = init_paths
        nn_params[:pre_size] = init_params

        # Store the time required to train.
        total_train_time = np.zeros(niter)

    # Initialize.
    threshold = np.inf
    weights[0] = 1 / npart

    # Run ABC-SMC.
    for t in range(niter):
        # Instantiate the parameter proposal distribution.
        proposal = _instantiate_proposal(t)

        # Run ABC-SMC round.
        with torch.no_grad():

            # The observational summary will change if continual learning
            # is allowed, otherwise it will stay fixed.
            obssummary = net(torch.Tensor(obs))

            # Run round.
            if backward_abc:
                (
                    all_summaries,
                    summaries,
                    total_comp_time[t],
                    total_sim,
                ) = backward_abcsmc_round(
                    obs=obs,
                    particles=particles[t],
                    proposal=proposal,
                    net=net,
                    lsis_x=lsis_x,
                    lsis_w=lsis_w,
                    lsis=lsis_partial,
                    smoother=smoother_partial,
                    obssummary=obssummary,
                    threshold=threshold,
                    mad=mad,
                    nn_params=nn_params,
                    nn_paths=nn_paths,
                    offset=pre_size + t * npart,
                    continual_learning=continual_learning,
                )

                # Backward simulated paths.
                bs_x = torch.Tensor(
                    parallel_bs(lsis_x=lsis_x, lsis_w=lsis_w, thetas=particles[t])
                )

                # Compute the synthetic likelihood ratios.
                synlik_ratios, condition_numbers = compute_sl_ratios(
                    lsis_x=lsis_x,
                    bs_x=bs_x,
                    summaries=summaries,
                    thetas=particles[t],
                )

                # Cutoff.
                synlik_ratios[np.where(synlik_ratios >= 0)[0]] = -1000
                cntr = 0
                for i in range(npart):
                    if condition_numbers[i] >= 1000:
                        synlik_ratios[i] = -1000
                        cntr += 1
                print(
                    "Near-singular covariances found (>= 1000 condition number)", cntr
                )
            else:
                (
                    all_summaries,
                    summaries,
                    total_comp_time[t],
                    total_sim,
                ) = forward_abcsmc_round(
                    particles=particles[t],
                    proposal=proposal,
                    net=net,
                    simulator=simulator,
                    obssummary=obssummary,
                    threshold=threshold,
                    mad=mad,
                    nn_params=nn_params,
                    nn_paths=nn_paths,
                    offset=pre_size + t * npart,
                    continual_learning=continual_learning,
                )

        # Compute median absolute deviations.
        for i in range(param_dim):
            median = torch.median(all_summaries[:, i])
            mad[i] = torch.median(torch.abs(all_summaries[:, i] - median))

        # Compute weighted Euclidean distances
        distances = np.zeros(npart)
        for i in range(npart):
            distances[i] = distance(obssummary, summaries[i], mad)

        # New threshold
        new_threshold = np.percentile(distances, alpha)
        if new_threshold > threshold:
            new_threshold = threshold * 0.95

        # Compute parameter weights.
        if t > 0:
            start = time.time()
            compute_param_ratio(
                t=t,
                prior_bounds=prior_bounds,
                particles=particles,
                weights=weights,
                cov_det=np.linalg.det(param_cov),
                cov_inv=np.linalg.inv(param_cov),
            )
            total_comp_time[t] += time.time() - start

        np.save(
            model_type + "/" + folder_name + "/comptime" + str(fid) + ".npy",
            total_comp_time[: t + 1],
        )

        # bs_x = torch.Tensor(
        #    parallel_bs(
        #        lsis_x=lsis_x,
        #        lsis_w=lsis_w,
        #        thetas=particles[i],
        #        dt=
        #    )
        # )

        # Normalize weights and compute covariance matrix for the next round.
        weights[t] = np.exp(weights[t] - logsumexp(weights[t]))
        param_cov = 2.0 * np.cov(particles[t].T, aweights=weights[t])

        # Store parameters and weights.
        np.save(
            model_type + "/" + folder_name + "/particles" + str(fid) + ".npy",
            particles[: t + 1],
        )
        np.save(
            model_type + "/" + folder_name + "/weights" + str(fid) + ".npy",
            weights[: t + 1],
        )

        # Print out round statistics.
        acceptance_rate = (npart / total_sim) * 100
        acceptance_rates[t] = acceptance_rate
        np.save(
            model_type + "/" + folder_name + "/accrates" + str(fid) + ".npy",
            acceptance_rates[: t + 1],
        )

        print(
            "Finished round",
            t + 1,
            "\nAcceptance rate",
            round(acceptance_rate, 2),
        )

        if t == niter - 1 or t > 2 and acceptance_rate < 1.5:
            print("Ending ABC-SMC..")
            print("Round is ", t)
            break

        if continual_learning and t < niter - 1:
            start = time.time()
            prepare_dataset_and_train(
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
                len(obs),
                param_dim,
            )
            total_train_time[t] += time.time() - start

            np.save(
                model_type + "/" + folder_name + "/traintime" + str(fid) + ".npy",
                total_train_time[: t + 1],
            )


def forward_abcsmc(
    obs: np.ndarray,
    npart: int,
    prior_bounds: np.ndarray,
    fid: str,
    alpha: float,
    niter: int,
    continual_learning: bool,
    net: Callable,
    simulator: Callable,
    data_name: str,
    model_type: str,
    nsamples: int,
):
    """ABC-SMC with forward simulation.

    Args:
        obs (np.ndarray): Observation
        npart (int): Number of particles.
        prior_bounds (np.ndarray): Bounds of the uniform prior.
        q (float): Percentile for choosing the thresholds.
        niter (int): Number of iterations.
        net (_type_): Sufficient statistics estimator.
        simulator (_type_): _description_
    """

    def _instantiate_proposal(t: int):
        """Instantiates the proposal distribution.

        Sample from the prior if ABC-SMC is
        at the initial round. Otherwise,
        use the Gaussian proposal with optimal
        covariance ala Fillipi.

        Args:
            t (int): ABC-SMC iteration.

        Returns:
            Callable: Proposal distribution
        """
        if t == 0:
            return partial(prior_proposal, prior_bounds)
        return partial(
            gaussian_proposal,
            particles[t - 1],
            weights[t - 1],
            np.linalg.cholesky(param_cov),
            prior_bounds,
        )

    # Parameter dimension.
    param_dim = len(prior_bounds)

    # Particle system.
    particles = np.zeros((niter, npart, param_dim))
    weights = np.zeros((niter, npart))

    # Weighted distance.
    mad = np.ones(param_dim)

    # Computing time.
    total_comp_time = np.zeros(niter)
    acceptance_rates = np.zeros(niter)

    # Configuration for continual learning.
    if continual_learning:

        # Training and validation sizes for the pretrained
        # PEN, and for the additional datasets.
        init_paths = torch.load(data_name + "_init_paths_" + str(nsamples))
        init_params = torch.load(data_name + "_init_params_" + str(nsamples))
        pre_size = len(init_paths)
        pre_tsize = int(pre_size * 0.8)
        pre_vsize = pre_size - pre_tsize
        tsize = int(npart * 0.8)
        vsize = npart - tsize

        # Allocate space.
        nn_paths = torch.zeros(pre_size + niter * npart, len(obs))
        nn_params = torch.zeros(pre_size + niter * npart, param_dim)
        nn_paths[:pre_size] = init_paths
        nn_params[:pre_size] = init_params

        # Store the observed summary statistics.
        total_train_time = np.zeros(niter)

    # Initialize.
    threshold = np.inf
    weights[0] = 1 / npart

    # Run ABC-SMC.
    for t in range(niter):
        # Instantiate the parameter proposal distribution.
        proposal = _instantiate_proposal(t)

        # Run ABC-SMC round.
        with torch.no_grad():

            # The observational summary will change if continual learning
            # is allowed, otherwise it will stay fixed.
            obssummary = net(torch.Tensor(obs))

            # Run round.
            (
                all_summaries,
                summaries,
                total_comp_time[t],
                total_sim,
            ) = forward_abcsmc_round(
                particles=particles[t],
                proposal=proposal,
                net=net,
                simulator=simulator,
                obssummary=obssummary,
                threshold=threshold,
                mad=mad,
                nn_params=nn_params,
                nn_paths=nn_paths,
                offset=pre_size + t * npart,
                continual_learning=continual_learning,
            )

        # Compute median absolute deviations.
        for i in range(param_dim):
            median = torch.median(all_summaries[:, i])
            mad[i] = torch.median(torch.abs(all_summaries[:, i] - median))

        # Compute weighted Euclidean distances
        distances = np.zeros(npart)
        for i in range(npart):
            distances[i] = distance(obssummary, summaries[i], mad)

        # New threshold
        new_threshold = np.percentile(distances, alpha)
        if new_threshold > threshold:
            new_threshold = threshold * 0.95

        # Compute parameter weights.
        if t > 0:
            start = time.time()
            compute_param_ratio(
                t=t,
                prior_bounds=prior_bounds,
                particles=particles,
                weights=weights,
                cov_det=np.linalg.det(param_cov),
                cov_inv=np.linalg.inv(param_cov),
            )
            total_comp_time[t] += time.time() - start
            np.save(
                model_type + "/" + "Forward/comptime" + str(fid) + ".npy",
                total_comp_time[: t + 1],
            )

        # Normalize weights and compute covariance matrix for the next round.
        weights[t] = np.exp(weights[t] - logsumexp(weights[t]))
        param_cov = 2.0 * np.cov(particles[t].T, aweights=weights[t])

        # Store parameters and weights.
        np.save(
            model_type + "/" + "Forward/particles" + str(fid) + ".npy",
            particles[: t + 1],
        )
        np.save(
            model_type + "/" + "Forward/weights" + str(fid) + ".npy", weights[: t + 1]
        )

        # Print out round statistics.
        acceptance_rate = (npart / total_sim) * 100
        acceptance_rates[t] = acceptance_rate
        np.save(
            model_type + "/" + "Forward/accrates" + str(fid) + ".npy",
            acceptance_rates[: t + 1],
        )

        print(
            "Finished round",
            t + 1,
            "\nAcceptance rate",
            round(acceptance_rate, 2),
        )

        if t == niter - 1 or t > 2 and acceptance_rate < 1.5:
            print("Ending ABC-SMC..")
            print("Round is ", t)
            break

        if continual_learning and t < niter - 1:
            train_paths, train_params, val_paths, val_params = prepare_dataset(
                nn_params,
                nn_paths,
                npart,
                pre_size,
                pre_tsize,
                tsize,
                pre_vsize,
                vsize,
                t,
                len(obs),
                param_dim,
            )

            # Setup trainer and fit the neural network.
            datamodule = PENDataModule(train_paths, train_params, val_paths, val_params)
            trainer = pl.Trainer(
                max_epochs=1000,
                accelerator="gpu",
                devices=1,
                log_every_n_steps=1,
                callbacks=[EarlyStopping(monitor="val_loss", patience=100)],
            )

            start = time.time()
            trainer.fit(net, datamodule=datamodule)
            total_train_time[t] += time.time() - start
            np.save(
                model_type + "/" + "Forward/traintime" + str(fid) + ".npy",
                total_train_time[: t + 1],
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
        synlik_ratios[np.where(synlik_ratios >= 0)[0]] = -1000
        for i in range(npart):
            if condition_numbers[i] >= 10000:
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
        cntr = 0
        for i in range(npart):
            if condition_numbers[i] >= 1000:
                synlik_ratios[i] = -1000
                cntr += 1
        print("FOUND >= 1000", cntr)

        # Add to weights and
        weights[t] += synlik_ratios
        weights[t] = np.exp(weights[t] - logsumexp(weights[t]))
        abc_cov = 2.0 * np.cov(particles[t].T, aweights=weights[t])

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
            trainer = pl.Trainer(
                max_epochs=1000,
                accelerator="gpu",
                devices=1,
                log_every_n_steps=1,
                callbacks=[EarlyStopping(monitor="val_loss", patience=100)],
            )

            start = time.time()
            trainer.fit(net, datamodule=datamodule)
            end = time.time()
            print("Training time", end - start)
            timing[t] += end - start

            # Update summary.
            obssummaries[t + 1] = _s(torch.Tensor(obs))
            print("Next summary", torch.exp(obssummaries[t + 1]))

    return particles[: t + 1], weights[: t + 1], timing[: t + 1]


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
            trainer = pl.Trainer(
                max_epochs=1000,
                accelerator="gpu",
                devices=1,
                log_every_n_steps=1,
                callbacks=[EarlyStopping(monitor="val_loss", patience=100)],
            )

            start = time.time()
            trainer.fit(net, datamodule=datamodule)
            end = time.time()
            print("Training time", end - start)
            timing[t] += end - start

            # Update summary.
            obssummaries[t + 1] = _s(torch.Tensor(obs))
            print("Next summary", torch.exp(obssummaries[t + 1]))

    print("Saving until", t)
    return particles[: t + 1], weights[: t + 1], timing[: t + 1]


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
