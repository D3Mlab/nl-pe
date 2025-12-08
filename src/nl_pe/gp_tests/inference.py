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

class GPInference:
    def __init__(self, config_path):
        # config_path: path to config.yaml
        self.config_path = Path(config_path).resolve()
        self.exp_dir = self.config_path.parent

        # load config.yaml
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.logger = setup_logging(
            self.__class__.__name__,
            config=self.config,
            output_file=self.exp_dir / "experiment.log",
        )

        self.logger.debug(
            f"Initializing {self.__class__.__name__} "
            f"with config at {self.config_path}"
        )

    def run_inference_test(self):

        #use 42 to seed all random number generators for reproducibility

        #read from config
        n_obs = self.config.get('n_obs')
        n_unobs = self.config.get('n_unobs')
        d = self.config.get('d')
        gt_func = self.config.get('gt_func')
        devince = self.config.get('device')

        #if gt_func is 'sin' then use the make_sin_ground_truth function 

        #todo: device , use cuda if gpu and available, ow cpu

        #to do -- test gpu exact vs approximate inference 
        
        #gt generation function example usage
        d = 5
        f = make_sin_ground_truth(d, seed=42)

        X = np.random.rand(100, d)
        y = f(X)  # shape (100,)

                #otherwise modify the following tutorial code to fit into our framework

        #modify this to randomly space n_obs points accross the [0,1]xd hypercube
        train_x = torch.linspace(0, 1, 100)
        # get true function values, modify to use OUR function, not this 1d sin function 
        train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            #

        init_time_start = #now

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        init_time = #now - init_time_start

        #modify to only use cuda if available and requested!
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

        #modify this to use n_unobs randomly spaces test points in [0,1]xd
        test_x = torch.linspace(0, 1, 51).cuda()

        #start eval timer
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()


        #modify this to get the std of the function at test points (so not the likelihood and confidence region!)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()

        #end eval timer 

        #record results to detailed_results.json in self.exp_dir

        # in the results includ mll for marginal log loss under "mll" key in json
        #also store
        #"init_time"
        #"eval_time"


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