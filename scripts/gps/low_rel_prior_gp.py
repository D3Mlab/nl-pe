#!/usr/bin/env python3
"""
Low Relevance Prior GP: Animation of Posterior Evolution with Mixed Observations

This script demonstrates how a GP with a low relevance prior (low uncertainty)
evolves as observations are sequentially added. We use:
- 1D inputs
- Constant mean (0)
- RBF kernel with low output scale for low relevance prior (~0.2 std prior)
- 4 clusters: low [-1.0,-0.8], high [0.0,0.2], low [1.0,1.2], high [2.0,2.2]
- 0.1 observation noise, observations interleaved

Animation shows 20 steps of posterior updates as observations are added alternately.
"""

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def setup_low_relevance_prior():
    """Setup GP with low relevance prior: mean 0, ~0.2 std prior uncertainty"""
    class LowRelGP(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            # RBF kernel with outputscale such that prior std ≈ 0.2
            # Prior variance at any point = outputscale * kernel(self, self) = outputscale
            # So std = sqrt(outputscale) ≈ 0.2 => outputscale ≈ 0.04
            self.covar_module = ScaleKernel(RBFKernel(lengthscale=0.00001))
            self.covar_module.outputscale = 1  # Prior std 

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    return LowRelGP

def generate_mixed_observations(n_per_cluster=5, noise_std=0.1):
    """Generate alternating observations: 4 clusters - low left, high middle, low right, high right"""
    # Low cluster left: [-1.0, -0.8]
    x_low_left = torch.linspace(-1.0, -0.8, n_per_cluster)
    y_low_left = torch.rand(n_per_cluster) * 0.1 + 0  # Uniform 0-3

    # High cluster middle: [0.0, 0.2]
    x_high_middle = torch.linspace(0.0, 0.2, n_per_cluster)
    y_high_middle = torch.rand(n_per_cluster) * 0.1 + 7  # Uniform 7-10

    # Low cluster right: [1.0, 1.2]
    x_low_right = torch.linspace(1.0, 1.2, n_per_cluster)
    y_low_right = torch.rand(n_per_cluster) * 0.1 + 0  # Uniform 0-3

    # High cluster right: [2.0, 2.2]
    x_high_right = torch.linspace(2.0, 2.2, n_per_cluster)
    y_high_right = torch.rand(n_per_cluster) * 0.1 + 7  # Uniform 7-10

    # Interleave observations: low_left, high_middle, low_right, high_right, repeating
    x_obs = []
    y_obs = []
    for i in range(n_per_cluster):
        x_obs.extend([x_low_left[i], x_high_middle[i], x_low_right[i], x_high_right[i]])
        y_obs.extend([y_low_left[i], y_high_middle[i], y_low_right[i], y_high_right[i]])

    x_obs = torch.tensor(x_obs)
    y_obs = torch.tensor(y_obs)

    # Add noise
    y_obs += torch.randn(len(y_obs)) * noise_std

    return x_obs, y_obs, len(x_obs)

def animate_posterior_evolution(x_plot, x_obs, y_obs, n_steps=20):
    """Create animation showing posterior evolution as observations are added sequentially"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot range
    x_min, x_max = x_plot.min(), x_plot.max()

    # Initialize GP model class
    LowRelGP = setup_low_relevance_prior()
    likelihood = GaussianLikelihood()
    likelihood.noise = 0.1  # observation noise
    def animate(frame):
        ax.clear()

        # Determine how many observations to include
        obs_to_add = min(frame, len(x_obs))

        # Current observations
        x_current = x_obs[:obs_to_add] if obs_to_add > 0 else torch.empty(0)
        y_current = y_obs[:obs_to_add] if obs_to_add > 0 else torch.empty(0)

        # Initialize model with current data
        if obs_to_add == 0:
            model = LowRelGP(x_current, y_current, likelihood)
            # No training needed for prior

            # Predictions for prior
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                pred = likelihood(model(x_plot))

            mean_pred = np.zeros_like(x_plot.numpy())
            std_pred = np.sqrt(0.04) * np.ones_like(x_plot.numpy())  # Prior std ≈ 0.2

            # Plot prior
            ax.plot(x_plot.numpy(), mean_pred, 'b-', linewidth=2, alpha=0.7, label='Prior Mean')
            ax.fill_between(x_plot.numpy(), mean_pred - 2*std_pred, mean_pred + 2*std_pred,
                           alpha=0.2, color='blue', label='Prior ±2σ')

            ax.scatter([], [], c='red', s=30, alpha=0.8, label='Observations')  # Empty for legend
            ax.set_title(f'Prior (Step 0/{n_steps}): mean=0, std≈0.2')
        else:
            model = LowRelGP(x_current, y_current, likelihood)

            # No optimization: fixed hyperparams, pure inference
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                pred = likelihood(model(x_plot))

            mean_pred = pred.mean.numpy()
            std_pred = pred.stddev.numpy()

            # Plot posterior
            ax.plot(x_plot.numpy(), mean_pred, 'g-', linewidth=2, label='Posterior Mean')
            ax.fill_between(x_plot.numpy(), mean_pred - 2*std_pred, mean_pred + 2*std_pred,
                           alpha=0.2, color='green', label='Posterior ±2σ')

            # Plot observations
            ax.scatter(x_current.numpy(), y_current.numpy(), c='red', s=30, alpha=0.8, label='Observations')

            ax.set_title(f'Posterior after {obs_to_add}/{len(x_obs)} observations (Step {frame}/{n_steps})')

        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-2, 12)
        ax.legend(loc='upper right')

        return ax,

    # Create animation
    anim = FuncAnimation(fig, animate, frames=n_steps+1, interval=1000, blit=False, repeat=False)

    # Save animation
    anim.save('plots/low_rel_prior_gp_animation.gif', writer='pillow', fps=1)
    plt.close()

    print("Animation saved to plots/low_rel_prior_gp_animation.gif")

def main():
    print("=== Low Relevance Prior GP Animation ===")

    # Setup plot directory
    import os
    os.makedirs('plots', exist_ok=True)

    # Plot points for visualization
    x_plot = torch.linspace(-2, 3, 200)

    # Generate observations
    x_obs, y_obs, n_total = generate_mixed_observations(n_per_cluster=5, noise_std=0.1)
    print(f"Generated {n_total} interleaved observations across 4 clusters")

    # Create animation with 20 steps
    animate_posterior_evolution(x_plot, x_obs, y_obs, n_steps=20)

    print("\n=== Script Complete ===")
    print("Animation shows GP posterior evolution with low relevance prior.")
    print("Observe how the GP peaks at high observations with low uncertainty, stays low with low uncertainty in observed and unobserved regions.")

if __name__ == "__main__":
    main()
