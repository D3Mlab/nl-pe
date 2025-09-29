#!/usr/bin/env python3
"""
Introduction to GPyTorch: A script to explore main features of Gaussian Processes with lots of visualizations.

This script demonstrates:
- Basic 1D GP regression
- Kernel comparisons
- Hyperparameter optimization
- Multi-dimensional inputs
- Variational GPs for large datasets
- Different mean functions

Requires: torch, gpytorch, matplotlib, pandas, numpy
"""

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def generate_data(n_train=50, n_test=100, noise_std=0.2):
    """Generate synthetic 1D data: sin(x) + 0.1*x^2 + noise"""
    x_train = torch.linspace(0, 5, n_train)
    x_test = torch.linspace(-1, 6, n_test)

    def true_f(x):
        return np.sin(x) + 0.1 * x**2

    y_train = true_f(x_train) + torch.randn(n_train) * noise_std
    y_test = true_f(x_test)

    return x_train, y_train, x_test, y_test, true_f

def create_basic_model(x_train, y_train):
    """Create a basic ExactGP model with RBF kernel"""
    class ExactGPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)
    return model, likelihood

def train_model(model, likelihood, x_train, y_train, num_epochs=50):
    """Train the GP model using Adam optimizer"""
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

def plot_results(x_train, y_train, x_test, y_test, model, likelihood, true_f, title="GP Regression"):
    """Plot training data, true function, and predictions with uncertainty"""
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_test))

    plt.figure(figsize=(12, 8))

    # Plot training data
    plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.6, c='red', s=20, label='Training Data')

    # Plot true function
    plt.plot(x_test.numpy(), y_test.numpy(), 'b--', linewidth=2, label='True Function')

    # Plot GP predictions
    pred_mean = observed_pred.mean.numpy()
    pred_std = observed_pred.stddev.numpy()

    plt.plot(x_test.numpy(), pred_mean, 'g-', linewidth=2, label='GP Mean')
    plt.fill_between(x_test.numpy(), pred_mean - 2*pred_std, pred_mean + 2*pred_std,
                    alpha=0.3, color='green', label='GP ±2σ')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{title.lower().replace(" ", "_")}_plot.png')
    plt.show()

def display_hyperparameters(model, likelihood):
    """Display model hyperparameters in a table"""
    params = {
        'Parameter': [],
        'Value': []
    }

    if hasattr(model.mean_module, 'constant'):
        params['Parameter'].append('Mean')
        params['Value'].append(f"{model.mean_module.constant.item():.4f}")

    params['Parameter'].append('Output Scale')
    params['Value'].append(f"{model.covar_module.outputscale.item():.4f}")

    # Handle lengthscale - could be scalar or tensor for ARD
    lengthscale = model.covar_module.base_kernel.lengthscale
    if lengthscale.numel() == 1:
        params['Parameter'].append('RBF Lengthscale')
        params['Value'].append(f"{lengthscale.item():.4f}")
    else:
        params['Parameter'].append('RBF Lengthscales (ARD)')
        params['Value'].append(f"{lengthscale.squeeze().tolist()}")

    params['Parameter'].append('Likelihood Noise')
    params['Value'].append(f"{likelihood.noise.item():.4f}")

    df = pd.DataFrame(params)
    print("\n=== Model Hyperparameters ===")
    print(df.to_string(index=False))

def compare_kernels(x_train, y_train, x_test, num_epochs=30):
    """Compare different kernels: RBF, Matern 1/2, Matern 3/2, Periodic"""
    kernels = {
        'RBF': ScaleKernel(RBFKernel()),
        'Matern 1/2': ScaleKernel(MaternKernel(nu=0.5)),
        'Matern 3/2': ScaleKernel(MaternKernel(nu=1.5)),
        'Periodic': ScaleKernel(PeriodicKernel())
    }

    results = {}
    plt.figure(figsize=(16, 12))

    for i, (kernel_name, kernel) in enumerate(kernels.items()):
        # Train model with this kernel
        class ExactGPModel(ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.covar_module = kernel

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        likelihood = GaussianLikelihood()
        model = ExactGPModel(x_train, y_train, likelihood)
        train_model(model, likelihood, x_train, y_train, num_epochs)

        # Predictions
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            observed_pred = likelihood(model(x_test))

        results[kernel_name] = {'model': model, 'likelihood': likelihood, 'pred': observed_pred}

        # Plot
        plt.subplot(2, 2, i+1)
        plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.6, c='red', s=20, label='Training Data')
        pred_mean = observed_pred.mean.numpy()
        pred_std = observed_pred.stddev.numpy()
        plt.plot(x_test.numpy(), pred_mean, 'g-', linewidth=2, label='GP Mean')
        plt.fill_between(x_test.numpy(), pred_mean - 2*pred_std, pred_mean + 2*pred_std,
                        alpha=0.3, color='green', label='GP ±2σ')
        plt.title(f'Kernel: {kernel_name}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/kernel_comparison.png')
    plt.show()

    return results

def multi_dimensional_example(n_train=100, n_test_1d=100):
    """2D input example with linear mean function"""
    # Generate 2D training data
    x_train_2d = torch.rand(n_train, 2) * 6  # 2D inputs in [0,6]x[0,6]
    # True function: sin(x1) + cos(x2) + 0.1*(x1*x2)
    def true_f_2d(x):
        return torch.sin(x[:,0]) + torch.cos(x[:,1]) + 0.1 * x[:,0] * x[:,1]

    y_train_2d = true_f_2d(x_train_2d) + torch.randn(n_train) * 0.1

    # Create 1D test grid for visualization
    x1_test = torch.linspace(0, 6, n_test_1d)
    x2_test = torch.linspace(0, 6, n_test_1d)
    X1, X2 = torch.meshgrid(x1_test, x2_test, indexing='ij')
    x_test_2d = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

    # Model with linear mean and RBF kernel
    class ExactGPModel2D(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = LinearMean(input_size=2)
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=2))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood_2d = GaussianLikelihood()
    model_2d = ExactGPModel2D(x_train_2d, y_train_2d, likelihood_2d)

    # Train
    losses = train_model(model_2d, likelihood_2d, x_train_2d, y_train_2d, num_epochs=50)

    # Predictions
    model_2d.eval()
    likelihood_2d.eval()
    with torch.no_grad():
        observed_pred = likelihood_2d(model_2d(x_test_2d))

    # Plot
    pred_mean = observed_pred.mean.reshape(n_test_1d, n_test_1d).numpy()
    pred_std = observed_pred.stddev.reshape(n_test_1d, n_test_1d).numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Training data scatter
    ax1.scatter(x_train_2d[:,0].numpy(), x_train_2d[:,1].numpy(), c=y_train_2d.numpy(),
               cmap='viridis', alpha=0.8, s=30)
    ax1.set_title('Training Data (2D)')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')

    # Predicted mean surface
    im1 = ax2.contourf(X1.numpy(), X2.numpy(), pred_mean, cmap='viridis', levels=20)
    ax2.set_title('GP Mean Prediction')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    plt.colorbar(im1, ax=ax2)

    # Prediction uncertainty
    im2 = ax3.contourf(X1.numpy(), X2.numpy(), pred_std, cmap='plasma', levels=20)
    ax3.set_title('GP Uncertainty (±σ)')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    plt.colorbar(im2, ax=ax3)

    plt.tight_layout()
    plt.savefig('plots/multi_dimensional_gp.png')
    plt.show()

    print("\n=== 2D GP Hyperparameters ===")
    display_hyperparameters(model_2d, likelihood_2d)

def variational_gp_example(n_train=1000, n_inducing=50):
    """Variational GP for larger datasets"""
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

    # Generate larger training data
    x_train = torch.linspace(0, 5, n_train).unsqueeze(-1)
    y_train = torch.sin(x_train.squeeze()) + 0.1 * x_train.squeeze()**2 + torch.randn(n_train) * 0.2

    x_test = torch.linspace(-1, 6, 200).unsqueeze(-1)

    class VariationalGPModel(ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
            super().__init__(variational_strategy)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Initialize inducing points randomly from training data
    inducing_points = x_train[:n_inducing]
    model_vgp = VariationalGPModel(inducing_points)
    likelihood_vgp = GaussianLikelihood()

    model_vgp.train()
    likelihood_vgp.train()

    optimizer = torch.optim.Adam([
        {'params': model_vgp.parameters()},
        {'params': likelihood_vgp.parameters()},
    ], lr=0.1)

    mll = gpytorch.mlls.VariationalELBO(likelihood_vgp, model_vgp, num_data=y_train.size(0))

    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        output = model_vgp(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f'VGP Epoch {epoch}: Loss = {loss.item():.4f}')

    # Predictions
    model_vgp.eval()
    likelihood_vgp.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood_vgp(model_vgp(x_test))

    plt.figure(figsize=(12, 8))

    # Plot training data (sample)
    idx_sample = torch.randperm(n_train)[:200]  # Show random subset
    plt.scatter(x_train[idx_sample].squeeze().numpy(), y_train[idx_sample].numpy(),
               alpha=0.6, c='red', s=10, label='Sample of Training Data')

    # Plot inducing points
    inducing_locs = model_vgp.variational_strategy.inducing_points.detach()
    pred_inducing = likelihood_vgp(model_vgp(inducing_locs))
    plt.scatter(inducing_locs[:,0].numpy(), pred_inducing.mean.detach().numpy(),
               c='orange', s=50, marker='s', alpha=0.9, label='Inducing Points')

    # Plot predictions
    pred_mean = observed_pred.mean.numpy()
    pred_std = observed_pred.stddev.numpy()
    plt.plot(x_test.squeeze().numpy(), pred_mean, 'g-', linewidth=2, label='VGP Mean')
    plt.fill_between(x_test.squeeze().numpy(), pred_mean - 2*pred_std, pred_mean + 2*pred_std,
                    alpha=0.3, color='green', label='VGP ±2σ')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Variational GP ({n_train} training points, {n_inducing} inducing points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/variational_gp.png')
    plt.show()

    # Display hyperparameters
    display_hyperparameters(model_vgp, likelihood_vgp)

def main():
    print("=== GPyTorch Introduction Script ===")
    print("This script demonstrates key features of Gaussian Processes with GPyTorch\n")

    # Set matplotlib backend for saving plots
    import os
    os.makedirs('plots', exist_ok=True)

    # Section 1: Basic 1D GP Regression
    print("Section 1: Basic 1D GP Regression with RBF Kernel")
    x_train, y_train, x_test, y_test, true_f = generate_data(n_train=50, n_test=100, noise_std=0.2)

    model, likelihood = create_basic_model(x_train, y_train)
    losses = train_model(model, likelihood, x_train, y_train, num_epochs=50)

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/training_loss.png')
    plt.show()

    plot_results(x_train, y_train, x_test, y_test, model, likelihood, true_f, "Basic GP Regression")
    display_hyperparameters(model, likelihood)

    # Section 2: Kernel Comparison
    print("\nSection 2: Kernel Comparison")
    kernel_results = compare_kernels(x_train, y_train, x_test, num_epochs=30)

    # Display hyperparameters for each kernel
    kernel_params = []
    for name, result in kernel_results.items():
        params = {
            'Kernel': name,
            'Output Scale': f"{result['model'].covar_module.outputscale.item():.4f}",
            'Noise': f"{result['likelihood'].noise.item():.6f}"
        }
        # Add kernel-specific parameters
        if hasattr(result['model'].covar_module.base_kernel, 'lengthscale'):
            params['Lengthscale'] = f"{result['model'].covar_module.base_kernel.lengthscale.item():.4f}"
        elif hasattr(result['model'].covar_module.base_kernel, 'period_length'):
            params['Period'] = f"{result['model'].covar_module.base_kernel.period_length.item():.4f}"
        kernel_params.append(params)

    df_kernels = pd.DataFrame(kernel_params)
    print("\n=== Kernel Comparison Hyperparameters ===")
    print(df_kernels.to_string(index=False))

    # Section 3: Multi-dimensional Inputs
    print("\nSection 3: Multi-dimensional GP")
    multi_dimensional_example(n_train=200)

    # Section 4: Variational GP for Large Datasets
    print("\nSection 4: Variational GP for Large Datasets")
    variational_gp_example(n_train=1000, n_inducing=50)

    print("\n=== Script Complete ===")
    print("All plots have been saved to the 'plots/' directory.")
    print("Feel free to modify parameters and explore different features!")

if __name__ == "__main__":
    main()
