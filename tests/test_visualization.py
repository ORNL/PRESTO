"""
Tests for visualization functionality.
"""

import pytest
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from ornl_presto import (
    visualize_data,
    visualize_similarity,
    visualize_top3,
    visualize_confidence,
    visualize_confidence_top3,
    visualize_overlay_original_and_private,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    return torch.randn(100)


@pytest.fixture
def sample_results():
    """Generate sample algorithm results for testing."""
    return [
        {
            "algorithm": "gaussian",
            "epsilon": 1.0,
            "mean": 0.5,
            "mean_rmse": 0.5,
            "ci_width": 0.2,
            "reliability": 0.8,
            "score": -0.5,
        },
        {
            "algorithm": "laplace",
            "epsilon": 1.2,
            "mean": 0.6,
            "mean_rmse": 0.6,
            "ci_width": 0.25,
            "reliability": 0.7,
            "score": -0.6,
        },
    ]


def test_visualize_data_basic(sample_data):
    """Test basic data visualization."""
    # Should not raise an error
    try:
        visualize_data(sample_data.tolist(), title="Test Data")
        plt.close()  # Clean up
        success = True
    except Exception as e:
        success = False
        print(f"Error: {e}")

    assert success


def test_visualize_similarity(sample_data):
    """Test similarity visualization."""
    try:
        visualize_similarity(sample_data.tolist(), "gaussian", epsilon=1.0)
        plt.close()
        success = True
    except Exception as e:
        success = False
        print(f"Error: {e}")

    assert success


def test_visualize_top3(sample_results):
    """Test top 3 algorithms visualization."""
    try:
        visualize_top3(sample_results)
        plt.close()
        success = True
    except Exception as e:
        success = False
        print(f"Error: {e}")

    assert success


def test_visualize_confidence(sample_data):
    """Test confidence visualization."""
    try:
        visualize_confidence(sample_data.tolist(), "gaussian", epsilon=1.0, n_evals=3)
        plt.close()
        success = True
    except Exception as e:
        success = False
        print(f"Error: {e}")

    assert success


def test_visualize_confidence_top3(sample_data, sample_results):
    """Test top 3 confidence visualization."""
    try:
        visualize_confidence_top3(sample_data.tolist(), sample_results, n_evals=3)
        plt.close()
        success = True
    except Exception as e:
        success = False
        print(f"Error: {e}")

    assert success


def test_visualize_overlay_original_and_private(sample_data, sample_results):
    """Test overlay visualization."""
    try:
        visualize_overlay_original_and_private(sample_data.tolist(), sample_results)
        plt.close()
        success = True
    except Exception as e:
        success = False
        print(f"Error: {e}")

    assert success
