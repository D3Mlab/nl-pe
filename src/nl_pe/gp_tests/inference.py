import os
import pandas as pd
import sys
import json
import csv
import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from nl_pe.utils.setup_logging import setup_logging
from pathlib import Path
import yaml
import time
from contextlib import nullcontext


class GPInference:
    def __init__(self, config):
        self.config = config        
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")
        self.exp_dir = Path(self.config['exp_dir']) 

    def run_inference(self):
        """
        Runs a single GP regression experiment:
        - sample training data from a ground-truth function on [0,1]^d
        - fit an Exact GP
        - evaluate posterior mean + variance on test points
        - record timing + marginal log likelihood
        """

        # ------------------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------------------
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ------------------------------------------------------------------
        # Read experiment configuration
        # ------------------------------------------------------------------
        n_obs = self.config.get("n_obs")        # number of observed points
        n_unobs = self.config.get("n_unobs")    # number of test points
        d = self.config.get("d")                # input dimensionality
        gt_func = self.config.get("gt_func")    # e.g. "sin"
        device_cfg = self.config.get("device")  # "cpu" or "cuda"
        fast_pred = self.config.get("fast_pred", False)
        self.logger.info(f"Fast prediction mode: {fast_pred}")

        # Decide device: only use CUDA if explicitly requested + available
        device = torch.device(
            "cuda" if device_cfg == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Using device: {device}")

        # ------------------------------------------------------------------
        # Ground-truth function
        # ------------------------------------------------------------------
        if gt_func == "sin":
            f_gt = make_sin_ground_truth(d, seed=seed)
        else:
            raise ValueError(f"Unknown ground-truth function: {gt_func}")

        # ------------------------------------------------------------------
        # Generate training data in [0,1]^d
        # ------------------------------------------------------------------
        # Sample inputs uniformly from the hypercube
        X_train_np = np.random.rand(n_obs, d)
        y_train_np = f_gt(X_train_np)

        # Convert to torch tensors
        train_x = torch.from_numpy(X_train_np).float().to(device)
        train_y = torch.from_numpy(y_train_np).float().to(device)

        # ------------------------------------------------------------------
        # Define Exact GP model
        # ------------------------------------------------------------------
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()

                # RBF kernel with outputscale
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # ------------------------------------------------------------------
        # Model + likelihood initialization (timed)
        # ------------------------------------------------------------------
        init_time_start = time.time()

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x, train_y, likelihood).to(device)

        init_time = time.time() - init_time_start


        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        final_mll = mll(model(train_x), train_y).item()

        # ------------------------------------------------------------------
        # Generate test points in [0,1]^d
        # ------------------------------------------------------------------
        X_test_np = np.random.rand(n_unobs, d)
        test_x = torch.from_numpy(X_test_np).float().to(device)

        # ------------------------------------------------------------------
        # Evaluation (posterior mean and *function* variance)
        # ------------------------------------------------------------------
        eval_time_start = time.time()

        model.eval()
        likelihood.eval()

        #todo -- adjust to use fast , optionally as per config setting

        fast_ctx = gpytorch.settings.fast_pred_var() if fast_pred else nullcontext()
        with torch.no_grad(), fast_ctx:
            # Latent function posterior (no observation noise)
            posterior = model(test_x)
            mean = posterior.mean
            std = posterior.variance.sqrt()

        eval_time = time.time() - eval_time_start

        #todo - save results to a mean_std.csv file in exp_dir with columns: mean, std

        # ------------------------------------------------------------------
        # Save results
        # ------------------------------------------------------------------
        results = {
            "n_obs": n_obs,
            "n_unobs": n_unobs,
            "d": d,
            "gt_func": gt_func,
            "device": str(device),
            "mll": final_mll,
            "init_time": init_time,
            "eval_time": eval_time,
        }

        results_path = self.exp_dir / "detailed_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results written to {results_path}")

        # Move tensors to CPU and convert to numpy
        # mean_np = mean.detach().cpu().numpy()
        # std_np = std.detach().cpu().numpy()

        # df = pd.DataFrame({
        #     "mean": mean_np,
        #     "std": std_np,
        # })

        # csv_path = self.exp_dir / "results.csv"
        # df.to_csv(csv_path, index=False)

        # self.logger.info(f"Saved predictive mean/std to {csv_path}")



def make_sin_ground_truth(d, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(d)
    b = rng.uniform(0.5, 1.5, size=d)
    phi = rng.uniform(0, 2*np.pi, size=d)
    w = rng.randn(d)

    def f(x):
        """
        x: shape (..., d) or (d,)
        returns: shape (...) evaluations of f
        """
        x = np.asarray(x)
        # Broadcast to (..., d)
        per_dim = (a * np.sin(2 * np.pi * b * x + phi)).sum(axis=-1)
        mix = 0.5 * np.sin(x @ w)   # x @ w handles (..., d) @ (d,)
        return per_dim + mix

    return f