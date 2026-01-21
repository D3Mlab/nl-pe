import numpy as np
import pandas as pd
import os
from nl_pe.utils.setup_logging import setup_logging
from pathlib import Path
import json
from scipy.stats import gamma

class HyperpriorFitter():
    def __init__(self, config):
        self.config = config
        self.exp_dir = self.config['exp_dir']
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))          
     
    def fit_all(self):
        path = Path(self.exp_dir).parents[1] / 'trained_params.csv'
        self.df = pd.read_csv(path)

        #read whether ard
        self.ard = self.config.get('ard')

        #read config to get methods to handle sig_noise, obs_noise, lengthscale, etc
        self.f_sig_noise = getattr(self, self.config["f_sig_noise"]) if "f_sig_noise" in self.config else None
        self.f_obs_noise = getattr(self, self.config["f_obs_noise"]) if "f_obs_noise" in self.config else None
        self.f_lengthscale = getattr(self, self.config["f_lengthscale"]) if "f_lengthscale" in self.config else None

        self.res = {}

        # sig_noise
        if "sig_noise" in self.df.columns:
            sig_noise_vals = self.df["sig_noise"].to_numpy()
            self.f_sig_noise(sig_noise_vals, "sig_noise")
        else:
            self.logger.info("column sig_noise missing from trained_params.csv")

        # obs_noise
        if "obs_noise" in self.df.columns:
            obs_noise_vals = self.df["obs_noise"].to_numpy()
            self.f_obs_noise(obs_noise_vals, "obs_noise")
        else:
            self.logger.info("column obs_noise missing from trained_params.csv")

        # lengthscale 
        if not self.ard:
            if "lengthscale" in self.df.columns:
                lengthscale_1d_vals = self.df["lengthscale"].to_numpy()
                self.f_lengthscale(lengthscale_1d_vals, "lengthscale", self.res)
            else:
                self.logger.info("column lengthscale missing from trained_params.csv")
        else: #ard case
            ls_cols = [c for c in self.df.columns if c.startswith("lengthscale_")]
            if ls_cols:
                ls_cols = sorted(ls_cols, key=lambda c: int(c.split("_")[1]))
                D = len(ls_cols)
                expected = [f"lengthscale_{i}" for i in range(D)]
                if ls_cols == expected:
                    lengthscale_2d_vals = self.df[ls_cols].to_numpy()
                    self.f_lengthscale(lengthscale_2d_vals, "lengthscale")
                else:
                    self.logger.info("ARD lengthscale columns are missing or non-contiguous")
            else:
                self.logger.info("no ARD lengthscale_* columns found in trained_params.csv")

        # dump res as a json (nicely formatted for readability) to self.exp_dir / hyperpriors.json
        out_path = Path(self.exp_dir) / "hyperpriors.json"
        with open(out_path, "w") as f:
            json.dump(self.res, f, indent=2, sort_keys=True)
        self.logger.info(f"wrote hyperpriors to {out_path}")

    def fit_lognormal_1d(self, x, name):
        x = np.asarray(x)
        if np.any(x <= 0):
            self.logger.info(f"non-positive values found for {name}, cannot fit lognormal")
            return
        logx = np.log(x)
        mu = logx.mean()
        sigma = logx.std(ddof=1)
        self.res[name] = {"dist": "lognormal", "mu": mu, "sigma": sigma}
        self.logger.info(f"fit lognormal for {name}: mu={mu:.4f}, sigma={sigma:.4f}")

    def fit_indep_gamma_2d(self, x, name):
        x = np.asarray(x)
        Q, D = x.shape
        self.res[name] = {}
        for d in range(D):
            xd = x[:, d]
            if any(v <= 0 for v in xd):
                self.logger.info(f"non-positive values found for {name}_{d}, cannot fit gamma")
                continue
            try:
                alpha, loc, scale = gamma.fit(xd, floc=0)
            except (ValueError, RuntimeWarning):
                #if variance between vals is near 0, crash.
                # Add small relative noise (1e-0.5 of the value)
                jitter = xd * np.random.uniform(1e-3, 1e-1, size=xd.shape)
                alpha, loc, scale = gamma.fit(xd + jitter, floc=0)
                self.logger.info(f"MLE succeeded after adding jitter for {name}_{d}")
            self.res[name][f"{name}_{d}"] = {"dist": "indep_gamma", "alpha": alpha, "scale": scale}
            self.logger.debug(f"fit indep gamma (MLE) for {name}_{d}: alpha={alpha:.4f}, scale={scale:.4f}")

    def fit_gamma_1d(self, x, name):
        x = np.asarray(x)
        if np.any(x <= 0):
            self.logger.info(f"non-positive values found for {name}, cannot fit gamma")
            return
        alpha, loc, scale = gamma.fit(x, floc=0)
        self.res[name] = {"dist": "gamma", "alpha": alpha, "scale": scale}
        self.logger.debug(f"fit gamma for {name}: alpha={alpha:.4f}, scale={scale:.4f}")
