import time
import torch
import numba as nb
import numpy as np
from math import exp, sqrt, log
from scipy.special import ive
from torch.quantization import quantize_dynamic
from functools import partial
from typing import Callable
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn
import pytorch_lightning as pl
from datacondabc.nnets import PENDataModule
from datacondabc.utilities import mvnorm_logpdf
import torch.multiprocessing as mp
from datacondabc.nnets import PENCNN
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from datacondabc.utilities import (
    norm_logpdf,
    prior_proposal,
    gaussian_proposal,
    distance,
    compute_param_ratio,
    logsumexp,
)


def abcsmc_dataconditional(
    obs: np.ndarray,
    prior_bounds: np.ndarray,
    lookahead: Callable,
    backward: Callable,
    forward: Callable,
    npart: int = 10000,
    npart_sim: int = 30,
    nsubint: int = 100,
    dt: float = 1,
    q: float = 50,
    c_max: int = 1000,
    niter: int = 20,
    init_size: int = 10000,
    amortized: bool = False,
    always_train: bool = True,
):
    param_dim = len(prior_bounds)
    _, d, obs_len = obs.shape
    net = PENCNN(input_shape=(obs_len, d), output_shape=param_dim, pen_nr=1)
    mse = torch.nn.MSELoss()
    training_time = np.zeros(niter)
    abcround_time = np.zeros(niter)
    acc_rates = np.zeros(niter)
    use_gpu = True if torch.cuda.is_available() else False
    best_model_paths = []
    obs_flat = obs[0].flatten()

    def _summarize(x: np.ndarray):
        """Inner function to summarize a given path.

        Args:
            x (np.ndarray): Sample path.

        Returns:
            torch.tensor: S(x).
        """
        with torch.no_grad():
            return net(torch.tensor(x, dtype=torch.float32))[0]

    def _sample_pps(paths, params):
        nsamples = len(paths)
        proposal = partial(prior_proposal, prior_bounds)
        for i in tqdm(range(nsamples)):
            while True:
                # Prior-predictive sample.
                particle = proposal()
                path = forward(particle)
                # Check for numerical errors.
                if np.any(np.isnan(path)) or np.any(path >= 10e7):
                    # Resample if error occured.
                    continue
                break
            paths[i] = torch.tensor(path).float()
            params[i] = torch.tensor(np.log(particle)).float()

    def _train_network(t, train_paths, train_params, val_paths, val_params, net):

        # Set the stopping flag to False to begin with.
        stop_learning = False

        # Setup trainer and fit the neural network.
        datamodule = PENDataModule(train_paths, train_params, val_paths, val_params)
        if t == -1:
            filename = "best-model-pretrained"
        else:
            filename = "best-model-round-" + str(t)

        # Stop if validation loss doesn't change after 200 epochs.
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=200,
            verbose=False,
            mode="min",
        )

        # Add ModelCheckpoint to automatically save the best model.
        model_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath="./models/",
            filename=filename,
            save_top_k=1,
            mode="min",
        )

        # Setup trainer.
        trainer = pl.Trainer(
            max_epochs=1000,
            accelerator="gpu" if use_gpu else None,
            devices=1 if use_gpu else None,
            callbacks=[early_stopping, model_checkpoint],
        )

        # Train and measure time.
        start = time.time()
        trainer.fit(net, datamodule=datamodule)
        end = time.time()

        # Store best model.
        best_model_paths.append(model_checkpoint.best_model_path)

        # Extract best model.
        updated_net = PENCNN.load_from_checkpoint(
            checkpoint_path=model_checkpoint.best_model_path,
            input_shape=(obs_len, d),
            output_shape=param_dim,
            pen_nr=1,
        )
        updated_net.eval()

        # Convert to CPU for inference.
        if use_gpu:
            updated_net = updated_net.to("cpu")

        if t == -1:
            training_time[0] += end - start
        else:
            training_time[t + 1] += end - start

            # Check how the new model performs as compared
            # to the old one on this new validation set.
            prev_net = PENCNN.load_from_checkpoint(
                checkpoint_path=best_model_paths[-2],
                input_shape=(obs_len, d),
                output_shape=param_dim,
                pen_nr=1,
            )
            prev_net.eval()
            if use_gpu:
                prev_net = prev_net.to("cpu")

            with torch.no_grad():
                out = prev_net(val_paths)
                prev_loss = float(mse(out, val_params))

                out = updated_net(val_paths)
                new_loss = float(mse(out, val_params))

                print("Validation loss of the previous model", prev_loss)
                print("Validation loss of the current model", new_loss)

            # Determine best model.
            rel_improvement = (prev_loss - new_loss) / prev_loss
            print("Relative improvement", rel_improvement)

            if rel_improvement <= 0.01 and t > 2 and always_train is False:
                print("Relative improvement was less than 1% !")
                # If rel improvement is negative, that means that
                # the previous error was smaller, therefore
                # we need to load the previous model
                model_idx = -2 if rel_improvement < 0 else -1
                if model_idx == -2:
                    print(
                        "We need to load the previous model because it performed better!"
                    )
                else:
                    print("Loading the newest model!")
                updated_net = PENCNN.load_from_checkpoint(
                    checkpoint_path=best_model_paths[model_idx],
                    input_shape=(obs_len, d),
                    output_shape=param_dim,
                    pen_nr=1,
                )
                updated_net.eval()
                if use_gpu:
                    updated_net = updated_net.to("cpu")
                print("Saving the model and continuing without retraining!")
                stop_learning = True
            else:
                print("Continuing with relearning!")

        return updated_net, stop_learning

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
        bs_x = np.zeros((npart, npart_sim, d, obs_len))
        for j in nb.prange(npart_sim):
            for i in range(npart):
                bs_x[i, j] = backward(
                    dt=dt,
                    x=lsis_x[i],
                    w=lsis_w[i],
                    theta=thetas[i],
                )[0]
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
            batch_size (int): Size of batch for processing.

        Returns:
            np.ndarray: Log ratios for every summary statistic.
        """
        # Backward paths.
        bs_x = torch.tensor(
            _parallel_bs(
                lsis_x=lsis_x,
                lsis_w=lsis_w,
                thetas=thetas,
            ),
            dtype=torch.float32,
        ).reshape(-1, d, obs_len)

        # Forward paths.
        lsis_x = torch.tensor(lsis_x, dtype=torch.float32).reshape(-1, d, obs_len)

        # Processing.
        batch_size = 1200
        n_batches = (lsis_x.shape[0] + batch_size - 1) // batch_size

        # SL ratios and condition numbers.
        sl_ratios = np.zeros(npart)
        cond_nums = np.zeros(npart)

        k = 0
        total = time.time()
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, lsis_x.shape[0])
            actual_batch_size = end_idx - start_idx

            with torch.no_grad():
                fs = net(lsis_x[start_idx:end_idx])
                bs = net(bs_x[start_idx:end_idx])

            for j in range(0, actual_batch_size, npart_sim):

                # Compute mean and covariance
                mu_f, cov_f = fs[j : j + npart_sim].mean(axis=0), np.cov(
                    fs[j : j + npart_sim].T
                )
                mu_b, cov_b = bs[j : j + npart_sim].mean(axis=0), np.cov(
                    bs[j : j + npart_sim].T
                )

                # Initialize condition number and sl_ratios for the current subset
                current_cond_num = np.linalg.cond(cov_b)
                current_sl_ratio = -1000  # Default in case of high condition number

                if current_cond_num < c_max:
                    try:
                        f_sl = mvn(mu_f.numpy(), cov_f).logpdf(summaries[k])
                        b_sl = mvn(mu_b.numpy(), cov_b).logpdf(summaries[k])
                        current_sl_ratio = f_sl - b_sl
                    except:
                        current_sl_ratio = -1000

                # Update the arrays with results for the current group of 30
                cond_nums[k] = current_cond_num
                sl_ratios[k] = current_sl_ratio
                k += 1

        print("Compute total", time.time() - total)
        return sl_ratios, cond_nums

    # Particle system, covariance matrix, thresholds.
    abc_particles = np.zeros((niter, npart, param_dim))
    abc_weights = np.zeros((niter, npart))
    abc_cov = np.zeros((param_dim, param_dim))
    abc_thresholds = np.zeros(niter)

    # Summaries across rounds.
    obssummaries = torch.zeros(niter, param_dim)
    accepted_summaries = np.zeros((npart, param_dim))

    # Training data.
    init_tsize = int(init_size * 0.8)
    init_vsize = init_size - init_tsize
    tsize = int(npart * 0.8)
    vsize = npart - tsize

    # Generate initial training and validation data.
    print("Generating training and validation data...")
    nn_paths = torch.zeros(init_size + niter * npart, d, obs_len)
    nn_params = torch.zeros(init_size + niter * npart, param_dim)
    _sample_pps(nn_paths[:init_size], nn_params[:init_size])

    # Train network.
    net, stop_learning = _train_network(
        t=-1,
        train_paths=nn_paths[:init_tsize],
        train_params=nn_params[:init_tsize],
        val_paths=nn_paths[init_tsize:init_size],
        val_params=nn_params[init_tsize:init_size],
        net=net,
    )

    # Initialize ABC-SMC.
    abc_thresholds[0] = np.inf
    abc_weights[0] = np.log(1 / npart)
    obssummaries[0] = _summarize(obs)

    # Lookahead SIS paths and weights.
    lsis_x = np.zeros((npart, npart_sim, d, obs_len))
    lsis_w = np.zeros((npart, npart_sim, obs_len))

    # Run ABC-SMC.
    for t in range(niter):
        print("Runing round", t + 1)
        # Instantiate the parameter proposal density.
        # This way one can simly write proposal() to
        # propose a parameter.
        if t == 0:
            proposal = partial(prior_proposal, prior_bounds)
        else:
            proposal = partial(
                gaussian_proposal,
                abc_particles[t - 1],
                abc_weights[t - 1],
                np.linalg.cholesky(abc_cov),
                prior_bounds,
            )

        # Iterate over particles.
        total_sim = 0
        round_start = time.time()
        distances = np.zeros(npart)
        for i in tqdm(range(npart)):
            # Accept/reject.
            while True:
                # Sample.
                particle = proposal()
                x, w = lookahead(
                    xo=obs,
                    A=nsubint,
                    P=npart_sim,
                    dt=dt,
                    theta=particle,
                )
                if np.any(np.isnan(x)) or np.any(x >= 10e7):
                    continue
                total_sim += 1

                # Sample trajectory and summarize.
                path = backward(dt=dt, x=x, w=w, theta=particle)
                simsummary = _summarize(path)

                # Accept if within distance.
                distances[i] = distance(obssummaries[t], simsummary)
                if distances[i] <= abc_thresholds[t]:
                    # Store the particle.
                    abc_particles[t, i] = particle

                    # Store the forward particle system to compute the
                    # ratio of synthetic likelihoods.
                    lsis_x[i] = x
                    lsis_w[i] = w

                    # Store summary.
                    accepted_summaries[i] = simsummary.numpy()

                    # Store data for retraining.
                    nn_params[init_size + t * npart + i] = torch.tensor(
                        np.log(particle)
                    ).float()

                    # Compute euclidean distance between
                    # observation and the forward samples.
                    x_flat = x.reshape(npart_sim, -1)
                    dists = np.sqrt(((x_flat - obs_flat) ** 2).sum(axis=1))

                    nn_paths[init_size + t * npart + i] = torch.tensor(
                        x[np.argmin(dists)], dtype=torch.float32
                    )
                    break

        # Compute the synthetic likelihood ratios.
        synlik_ratios, condition_numbers = _compute_sl_ratios(
            lsis_x=lsis_x,
            lsis_w=lsis_w,
            summaries=accepted_summaries,
            thetas=abc_particles[t],
        )
        synlik_ratios[np.where(synlik_ratios >= 0)[0]] = -1000
        for i in range(npart):
            if condition_numbers[i] >= c_max:
                synlik_ratios[i] = -1000

        # Once all particles have been sampled, compute the weights.
        if t > 0:
            compute_param_ratio(
                t=t,
                prior_bounds=prior_bounds,
                particles=abc_particles,
                weights=abc_weights,
                cov_det=np.linalg.det(abc_cov),
                cov_inv=np.linalg.inv(abc_cov),
            )
        abc_weights[t] += synlik_ratios
        abc_weights[t] = np.exp(abc_weights[t] - logsumexp(abc_weights[t]))
        abc_cov = 2.0 * np.cov(abc_particles[t].T, aweights=abc_weights[t])

        # Total round time.
        abcround_time[t] = time.time() - round_start

        # Compute acceptance rate and stop if it's below 1.5%.
        acceptance_rate = (npart / total_sim) * 100
        acc_rates[t] = acceptance_rate

        # Stop if it's the last round.
        if t == niter - 1:
            break

        # Determine new threshold.
        new_threshold = np.percentile(distances, q)
        if new_threshold <= abc_thresholds[t]:
            abc_thresholds[t + 1] = new_threshold
        else:
            abc_thresholds[t + 1] = abc_thresholds[t] * 0.95

        print(
            "The threshold for this round was",
            round(abc_thresholds[t], 2),
            "\nAcceptance rate is",
            round(acceptance_rate, 2),
        )

        if t > 2 and acceptance_rate < 1.5:
            print("Ending ABC-SMC..")
            print("Round is ", t)
            break

        if t < niter - 1 and stop_learning is False and amortized is False:

            # Prepare new dataset.
            train_paths = torch.zeros(init_tsize + tsize * (t + 1), d, obs_len)
            train_params = torch.zeros(init_tsize + tsize * (t + 1), param_dim)
            val_paths = torch.zeros(init_vsize + vsize * (t + 1), d, obs_len)
            val_params = torch.zeros(init_vsize + vsize * (t + 1), param_dim)

            # Pretrained data first.
            train_paths[:init_tsize] = nn_paths[:init_tsize]
            train_params[:init_tsize] = nn_params[:init_tsize]
            val_paths[:init_vsize] = nn_paths[init_tsize:init_size]
            val_params[:init_vsize] = nn_params[init_tsize:init_size]

            # Sampled paths.
            for k in range(t + 1):
                # Training chunks.
                tstart, tend = init_size + k * npart, init_size + k * npart + tsize
                train_paths[init_tsize + k * tsize : init_tsize + (k + 1) * tsize] = (
                    nn_paths[tstart:tend]
                )
                train_params[init_tsize + k * tsize : init_tsize + (k + 1) * tsize] = (
                    nn_params[tstart:tend]
                )

                # Validation chunks.
                vstart, vend = (
                    init_size + k * npart + tsize,
                    init_size + (k + 1) * npart,
                )
                val_paths[init_vsize + k * vsize : init_vsize + (k + 1) * vsize] = (
                    nn_paths[vstart:vend]
                )
                val_params[init_vsize + k * vsize : init_vsize + (k + 1) * vsize] = (
                    nn_params[vstart:vend]
                )

            net, stop_learning = _train_network(
                t=t,
                train_paths=train_paths,
                train_params=train_params,
                val_paths=val_paths,
                val_params=val_params,
                net=net,
            )

        # Update summary.
        obssummaries[t + 1] = _summarize(obs)
        print(
            "Difference in summary statistics:",
            mse(obssummaries[t], obssummaries[t + 1]),
        )
        print("Next summary", torch.exp(obssummaries[t + 1]))

    return (
        abc_particles[: t + 1],
        abc_weights[: t + 1],
        abcround_time[: t + 1],
        training_time[: t + 2],
        acc_rates[: t + 1],
        obssummaries[: t + 1],
        abc_thresholds[: t + 1],
    )


def abcsmc_forward(
    obs: np.ndarray,
    prior_bounds: np.ndarray,
    simulator: Callable,
    npart: int = 10000,
    q: float = 50,
    niter: int = 20,
    init_size: int = 10000,
    amortized: bool = False,
    always_train: bool = True,
):

    param_dim = len(prior_bounds)
    _, d, obs_len = obs.shape
    net = PENCNN(input_shape=(obs_len, d), output_shape=param_dim, pen_nr=1)
    mse = torch.nn.MSELoss()
    training_time = np.zeros(niter)
    abcround_time = np.zeros(niter)
    acc_rates = np.zeros(niter)
    use_gpu = True if torch.cuda.is_available() else False
    best_model_paths = []

    def _summarize(x: np.ndarray):
        """Inner function to summarize a given path.

        Args:
            x (np.ndarray): Sample path.

        Returns:
            torch.tensor: S(x).
        """
        with torch.no_grad():
            return net(torch.tensor(x).float())[0]

    def _sample_pps(paths, params):
        nsamples = len(paths)
        proposal = partial(prior_proposal, prior_bounds)
        for i in tqdm(range(nsamples)):
            while True:
                # Prior-predictive sample.
                particle = proposal()
                path = simulator(particle)
                # Check for numerical errors.
                if np.any(np.isnan(path)) or np.any(path >= 10e7):
                    # Resample if error occured.
                    continue
                break
            paths[i] = torch.tensor(path).float()
            params[i] = torch.tensor(np.log(particle)).float()

    def _train_network(t, train_paths, train_params, val_paths, val_params, net):

        # Set the stopping flag to False to begin with.
        stop_learning = False

        # Setup trainer and fit the neural network.
        datamodule = PENDataModule(train_paths, train_params, val_paths, val_params)
        if t == -1:
            filename = "best-model-pretrained"
        else:
            filename = "best-model-round-" + str(t)

        # Stop if validation loss doesn't change after 200 epochs.
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=200,
            verbose=False,
            mode="min",
        )

        # Add ModelCheckpoint to automatically save the best model.
        model_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath="./models/",
            filename=filename,
            save_top_k=1,
            mode="min",
        )

        # Setup trainer.
        trainer = pl.Trainer(
            max_epochs=1000,
            accelerator="gpu" if use_gpu else None,
            devices=1 if use_gpu else None,
            callbacks=[early_stopping, model_checkpoint],
        )

        # Train and measure time.
        start = time.time()
        trainer.fit(net, datamodule=datamodule)
        end = time.time()

        # Store best model.
        best_model_paths.append(model_checkpoint.best_model_path)

        # Extract best model.
        updated_net = PENCNN.load_from_checkpoint(
            checkpoint_path=model_checkpoint.best_model_path,
            input_shape=(obs_len, d),
            output_shape=param_dim,
            pen_nr=1,
        )
        updated_net.eval()

        # Convert to CPU for inference.
        if use_gpu:
            updated_net = updated_net.to("cpu")

        if t == -1:
            training_time[0] += end - start
        else:
            training_time[t + 1] += end - start

            # Check how the new model performs as compared
            # to the old one on this new validation set.
            prev_net = PENCNN.load_from_checkpoint(
                checkpoint_path=best_model_paths[-2],
                input_shape=(obs_len, d),
                output_shape=param_dim,
                pen_nr=1,
            )
            prev_net.eval()
            if use_gpu:
                prev_net = prev_net.to("cpu")

            with torch.no_grad():
                out = prev_net(val_paths)
                prev_loss = float(mse(out, val_params))

                out = updated_net(val_paths)
                new_loss = float(mse(out, val_params))

                print("Validation loss of the previous model", prev_loss)
                print("Validation loss of the current model", new_loss)

            # Determine best model.
            rel_improvement = (prev_loss - new_loss) / prev_loss
            print("Relative improvement", rel_improvement)

            if rel_improvement <= 0.01 and t > 2 and always_train is False:
                print("Relative improvement was less than 1% !")
                # If rel improvement is negative, that means that
                # the previous error was smaller, therefore
                # we need to load the previous model
                model_idx = -2 if rel_improvement < 0 else -1
                if model_idx == -2:
                    print(
                        "We need to load the previous model because it performed better!"
                    )
                else:
                    print("Loading the newest model!")
                updated_net = PENCNN.load_from_checkpoint(
                    checkpoint_path=best_model_paths[model_idx],
                    input_shape=(obs_len, d),
                    output_shape=param_dim,
                    pen_nr=1,
                )
                updated_net.eval()
                if use_gpu:
                    updated_net = updated_net.to("cpu")
                print("Saving the model and continuing without retraining!")
                stop_learning = True
            else:
                print("Continuing with relearning!")

        return updated_net, stop_learning

    # Particle system, covariance matrix, thresholds.
    abc_particles = np.zeros((niter, npart, param_dim))
    abc_weights = np.zeros((niter, npart))
    abc_cov = np.zeros((param_dim, param_dim))
    abc_thresholds = np.zeros(niter)

    # Summaries across rounds.
    obssummaries = torch.zeros(niter, param_dim)

    # Training data.
    init_tsize = int(init_size * 0.8)
    init_vsize = init_size - init_tsize
    tsize = int(npart * 0.8)
    vsize = npart - tsize

    # Generate initial training and validation data.
    print("Generating training and validation data...")
    nn_paths = torch.zeros(init_size + niter * npart, d, obs_len)
    nn_params = torch.zeros(init_size + niter * npart, param_dim)
    _sample_pps(nn_paths[:init_size], nn_params[:init_size])

    # Train network.
    net, stop_learning = _train_network(
        t=-1,
        train_paths=nn_paths[:init_tsize],
        train_params=nn_params[:init_tsize],
        val_paths=nn_paths[init_tsize:init_size],
        val_params=nn_params[init_tsize:init_size],
        net=net,
    )

    # Initialize ABC-SMC.
    abc_thresholds[0] = np.inf
    abc_weights[0] = np.log(1 / npart)
    obssummaries[0] = _summarize(obs)

    # Run ABC-SMC.
    for t in range(niter):
        print("Runing round", t + 1)
        # Instantiate the parameter proposal density.
        if t == 0:
            proposal = partial(prior_proposal, prior_bounds)
        else:
            proposal = partial(
                gaussian_proposal,
                abc_particles[t - 1],
                abc_weights[t - 1],
                np.linalg.cholesky(abc_cov),
                prior_bounds,
            )

        # Iterate over particles.
        total_sim = 0
        round_start = time.time()
        distances = np.zeros(npart)
        for i in tqdm(range(npart)):
            # Accept/reject.
            while True:
                # Sample.
                particle = proposal()
                path = simulator(particle)
                if np.any(np.isnan(path)) or np.any(path >= 10e7):
                    continue
                total_sim += 1

                # Summarize.
                simsummary = _summarize(path)

                # Accept / reject step.
                distances[i] = distance(obssummaries[t], simsummary)
                if distances[i] <= abc_thresholds[t]:
                    # Store the particle.
                    abc_particles[t, i] = particle

                    # Store data for retraining.
                    nn_params[init_size + t * npart + i] = torch.tensor(
                        np.log(particle)
                    ).float()

                    nn_paths[init_size + t * npart + i] = torch.tensor(path).float()
                    break

        # Once all particles have been sampled, compute the weights.
        if t > 0:
            compute_param_ratio(
                t=t,
                prior_bounds=prior_bounds,
                particles=abc_particles,
                weights=abc_weights,
                cov_det=np.linalg.det(abc_cov),
                cov_inv=np.linalg.inv(abc_cov),
            )

        # Normalize weights and compute covariance matrix for the next round.
        abc_weights[t] = np.exp(abc_weights[t] - logsumexp(abc_weights[t]))
        abc_cov = 2.0 * np.cov(abc_particles[t].T, aweights=abc_weights[t])

        # Total round time.
        abcround_time[t] = time.time() - round_start

        # Compute acceptance rate and stop if it's below 1.5%.
        acceptance_rate = (npart / total_sim) * 100
        acc_rates[t] = acceptance_rate

        # Stop if it's the last round.
        if t == niter - 1:
            break

        # Determine new threshold.
        new_threshold = np.percentile(distances, q)
        if new_threshold <= abc_thresholds[t]:
            abc_thresholds[t + 1] = new_threshold
        else:
            abc_thresholds[t + 1] = abc_thresholds[t] * 0.95

        print(
            "The threshold for this round was",
            round(abc_thresholds[t], 2),
            "\nAcceptance rate is",
            round(acceptance_rate, 2),
        )
        if t > 2 and acceptance_rate < 1.5:
            print("Ending ABC-SMC..")
            print("Round is ", t)
            break

        if t < niter - 1 and stop_learning is False and amortized is False:

            # Prepare new dataset.
            train_paths = torch.zeros(init_tsize + tsize * (t + 1), d, obs_len)
            train_params = torch.zeros(init_tsize + tsize * (t + 1), param_dim)
            val_paths = torch.zeros(init_vsize + vsize * (t + 1), d, obs_len)
            val_params = torch.zeros(init_vsize + vsize * (t + 1), param_dim)

            # Pretrained data first.
            train_paths[:init_tsize] = nn_paths[:init_tsize]
            train_params[:init_tsize] = nn_params[:init_tsize]
            val_paths[:init_vsize] = nn_paths[init_tsize:init_size]
            val_params[:init_vsize] = nn_params[init_tsize:init_size]

            # Sampled paths.
            for k in range(t + 1):
                # Training chunks.
                tstart, tend = init_size + k * npart, init_size + k * npart + tsize
                train_paths[init_tsize + k * tsize : init_tsize + (k + 1) * tsize] = (
                    nn_paths[tstart:tend]
                )
                train_params[init_tsize + k * tsize : init_tsize + (k + 1) * tsize] = (
                    nn_params[tstart:tend]
                )

                # Validation chunks.
                vstart, vend = (
                    init_size + k * npart + tsize,
                    init_size + (k + 1) * npart,
                )
                val_paths[init_vsize + k * vsize : init_vsize + (k + 1) * vsize] = (
                    nn_paths[vstart:vend]
                )
                val_params[init_vsize + k * vsize : init_vsize + (k + 1) * vsize] = (
                    nn_params[vstart:vend]
                )

            net, stop_learning = _train_network(
                t=t,
                train_paths=train_paths,
                train_params=train_params,
                val_paths=val_paths,
                val_params=val_params,
                net=net,
            )

        # Update summary.
        obssummaries[t + 1] = _summarize(obs)
        print(
            "Difference in summary statistics:",
            mse(obssummaries[t], obssummaries[t + 1]),
        )
        print("Next summary", torch.exp(obssummaries[t + 1]))

    print("Saving until", t + 1)
    return (
        abc_particles[: t + 1],
        abc_weights[: t + 1],
        abcround_time[: t + 1],
        training_time[: t + 2],
        acc_rates[: t + 1],
        obssummaries[: t + 1],
        abc_thresholds[: t + 1],
    )


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
