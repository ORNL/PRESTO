"""
Core Machine Learning and Gaussian Process functionality for PRESTO.

This module contains the core differentially private machine learning functions
and Gaussian Process models that form the foundation of PRESTO's ML capabilities.
"""

import torch
import numpy as np
import gpytorch
from opacus import PrivacyEngine
from bayes_opt import BayesianOptimization
from torch.utils.data import DataLoader

from .utils import _to_tensor


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Gaussian Process model using exact inference.

    This class implements a standard GP model with RBF kernel for regression tasks
    in the context of differential privacy.
    """

    def __init__(self, train_x, train_y, likelihood):
        """
        Initialize the GP model.

        Args:
            train_x (torch.Tensor): Training input data
            train_y (torch.Tensor): Training target data
            likelihood (gpytorch.likelihoods): GP likelihood function
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """Forward pass through the GP model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def gpr_gpytorch(train_x, train_y, test_x, training_iter):
    """
    Gaussian Process Regression using GPyTorch.

    Args:
        train_x (array-like): Training input data
        train_y (array-like): Training target data
        test_x (array-like): Test input data
        training_iter (int): Number of optimization iterations

    Returns:
        tuple: (predictions, lower_bounds, upper_bounds) with confidence intervals
    """
    # Convert to tensors
    train_x = _to_tensor(train_x)
    train_y = _to_tensor(train_y)
    test_x = _to_tensor(test_x)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Make predictions
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    return (
        observed_pred.mean.numpy(),
        observed_pred.confidence_region()[0].numpy(),
        observed_pred.confidence_region()[1].numpy(),
    )


def dp_function(noise_multiplier, max_grad_norm, model_class, train_dataset, X_train):
    """
    Train a differentially private neural network model.

    Args:
        noise_multiplier (float): Noise multiplier for differential privacy
        max_grad_norm (float): Maximum gradient norm for clipping
        model_class: Neural network model class
        train_dataset: Training dataset
        X_train: Training input features

    Returns:
        tuple: (trained_model, optimizer, criterion)
    """
    try:
        # Initialize model components
        model = model_class()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss()

        # Setup privacy engine
        privacy_engine = PrivacyEngine()
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=DataLoader(train_dataset, batch_size=32, shuffle=True),
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        return model, optimizer, criterion

    except Exception as e:
        print(f"Error in dp_function: {e}")
        return None, None, None


def dp_function_train_and_pred(
    model, optimizer, criterion, train_dataloader, X_test, y_test
):
    """
    Train a DP model and make predictions.

    Args:
        model: Differentially private model
        optimizer: Model optimizer
        criterion: Loss function
        train_dataloader: Training data loader
        X_test: Test input features
        y_test: Test target values

    Returns:
        float: Mean squared error on test set
    """
    try:
        # Training phase
        model.train()
        for epoch in range(10):  # Fixed number of epochs
            for batch_idx, (data, target) in enumerate(train_dataloader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            X_test_tensor = _to_tensor(X_test)
            y_test_tensor = _to_tensor(y_test)

            predictions = model(X_test_tensor)
            mse = torch.nn.functional.mse_loss(predictions, y_test_tensor)

        return mse.item()

    except Exception as e:
        print(f"Error in dp_function_train_and_pred: {e}")
        return float("inf")


def dp_target(
    noise_multiplier, max_grad_norm, model_class, X_test, train_dataset, y_test
):
    """
    Target function for Bayesian optimization of DP hyperparameters.

    Args:
        noise_multiplier (float): Noise multiplier for DP
        max_grad_norm (float): Maximum gradient norm
        model_class: Neural network model class
        X_test: Test input features
        train_dataset: Training dataset
        y_test: Test target values

    Returns:
        float: Negative MSE (for maximization in Bayesian optimization)
    """
    # Get DP model components
    model, optimizer, criterion = dp_function(
        noise_multiplier, max_grad_norm, model_class, train_dataset, None
    )

    if model is None:
        return -float("inf")

    # Create dataloader for training
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train and evaluate
    mse = dp_function_train_and_pred(
        model, optimizer, criterion, train_dataloader, X_test, y_test
    )

    # Return negative MSE for maximization
    return -mse


def dp_pareto_front(x1, x2, model_class, X_test, train_dataset, y_test):
    """
    Generate Pareto front for privacy-utility trade-off analysis.

    Args:
        x1 (array-like): Range of noise multiplier values
        x2 (array-like): Range of max gradient norm values
        model_class: Neural network model class
        X_test: Test input features
        train_dataset: Training dataset
        y_test: Test target values

    Returns:
        tuple: (privacy_values, utility_values) for Pareto front
    """
    privacy_values = []
    utility_values = []

    try:
        for noise_mult in x1:
            for grad_norm in x2:
                # Calculate utility (negative MSE)
                utility = dp_target(
                    noise_mult,
                    grad_norm,
                    model_class,
                    X_test,
                    train_dataset,
                    y_test,
                )

                # Privacy level is inversely related to noise multiplier
                privacy = 1.0 / (
                    noise_mult + 1e-6
                )  # Add small epsilon to avoid division by zero

                privacy_values.append(privacy)
                utility_values.append(-utility)  # Convert back to positive MSE

    except Exception as e:
        print(f"Error in dp_pareto_front: {e}")

    return privacy_values, utility_values


def dp_hyper(model_class, train_dataset, test_dataset, seed=42):
    """
    Hyperparameter optimization for differentially private models using Bayesian optimization.

    Args:
        model_class: Neural network model class
        train_dataset: Training dataset
        test_dataset: Test dataset
        seed (int): Random seed for reproducibility

    Returns:
        dict: Optimal hyperparameters and performance metrics
    """
    try:
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Extract test data
        X_test = []
        y_test = []
        for data, target in test_dataset:
            X_test.append(data.numpy())
            y_test.append(target.numpy())

        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)

        # Define objective function for Bayesian optimization
        def objective(noise_multiplier, max_grad_norm):
            return dp_target(
                noise_multiplier,
                max_grad_norm,
                model_class,
                X_test,
                train_dataset,
                y_test,
            )

        # Define parameter bounds
        pbounds = {"noise_multiplier": (0.1, 2.0), "max_grad_norm": (0.1, 2.0)}

        # Perform Bayesian optimization
        optimizer = BayesianOptimization(
            f=objective, pbounds=pbounds, random_state=seed, verbose=0
        )

        optimizer.maximize(init_points=5, n_iter=10)

        # Get best parameters
        best_params = optimizer.max["params"]
        best_value = optimizer.max["target"]

        return {
            "best_noise_multiplier": best_params["noise_multiplier"],
            "best_max_grad_norm": best_params["max_grad_norm"],
            "best_utility": -best_value,  # Convert back to MSE
            "optimization_history": optimizer.res,
        }

    except Exception as e:
        print(f"Error in dp_hyper: {e}")
        return {
            "best_noise_multiplier": 1.0,
            "best_max_grad_norm": 1.0,
            "best_utility": float("inf"),
            "optimization_history": [],
        }
