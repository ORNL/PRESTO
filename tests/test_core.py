"""
Tests for core ML/GP functionality.
"""

import pytest
import torch
import numpy as np
from ornl_presto import gpr_gpytorch, dp_function, dp_target, dp_pareto_front, dp_hyper


class SimpleModel(torch.nn.Module):
    """Simple neural network for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def test_gpr_gpytorch_basic():
    """Test basic Gaussian Process regression functionality."""
    # Generate simple test data
    torch.manual_seed(42)
    train_x = torch.randn(20, 2)
    train_y = train_x[:, 0] + train_x[:, 1] + 0.1 * torch.randn(20)
    test_x = torch.randn(10, 2)

    # Run GP regression
    predictions, lower, upper = gpr_gpytorch(train_x, train_y, test_x, training_iter=50)

    # Basic checks
    assert len(predictions) == 10
    assert len(lower) == 10
    assert len(upper) == 10
    assert np.all(lower <= predictions)
    assert np.all(predictions <= upper)


def test_dp_function_initialization():
    """Test differential privacy function initialization."""
    # Create dummy training data
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

    # Test DP function initialization
    model, optimizer, criterion = dp_function(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        model_class=SimpleModel,
        train_dataset=train_dataset,
        X_train=X_train,
    )

    # Check that components are properly initialized
    assert model is not None
    assert optimizer is not None
    assert criterion is not None
    # Model is wrapped in GradSampleModule for differential privacy
    assert hasattr(model, "module") or isinstance(model._module, SimpleModel)


def test_dp_target_evaluation():
    """Test DP target function evaluation."""
    # Create dummy data
    X_test = torch.randn(50, 10)
    y_test = torch.randn(50, 1)
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

    # Test DP target evaluation
    score = dp_target(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        model_class=SimpleModel,
        X_test=X_test,
        train_dataset=train_dataset,
        y_test=y_test,
    )

    # Should return a finite negative number (negative MSE)
    assert isinstance(score, float)
    assert score <= 0  # Negative MSE
    assert np.isfinite(score)


def test_dp_pareto_front():
    """Test Pareto front generation."""
    # Create dummy data
    X_test = torch.randn(30, 10)
    y_test = torch.randn(30, 1)
    X_train = torch.randn(50, 10)
    y_train = torch.randn(50, 1)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

    # Test with small ranges for quick execution
    x1 = [0.5, 1.0]  # noise multiplier range
    x2 = [0.5, 1.0]  # max grad norm range

    privacy_vals, utility_vals = dp_pareto_front(
        x1, x2, SimpleModel, X_test, train_dataset, y_test
    )

    # Check outputs
    assert len(privacy_vals) == len(x1) * len(x2)
    assert len(utility_vals) == len(x1) * len(x2)
    assert all(isinstance(p, float) for p in privacy_vals)
    assert all(isinstance(u, float) for u in utility_vals)


def test_dp_hyper_basic():
    """Test basic hyperparameter optimization functionality."""
    # Create small datasets for quick testing
    X_train = torch.randn(30, 10)
    y_train = torch.randn(30, 1)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

    X_test = torch.randn(20, 10)
    y_test = torch.randn(20, 1)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # Run hyperparameter optimization with minimal iterations
    result = dp_hyper(SimpleModel, train_dataset, test_dataset, seed=42)

    # Check result structure
    assert isinstance(result, dict)
    assert "best_noise_multiplier" in result
    assert "best_max_grad_norm" in result
    assert "best_utility" in result
    assert "optimization_history" in result

    # Check reasonable ranges
    assert 0.1 <= result["best_noise_multiplier"] <= 2.0
    assert 0.1 <= result["best_max_grad_norm"] <= 2.0
    assert isinstance(result["best_utility"], float)
