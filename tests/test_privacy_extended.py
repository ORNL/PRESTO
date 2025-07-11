"""
Extended tests for privacy mechanisms.
"""

import pytest
import torch
import numpy as np
from ornl_presto import (
    applySVTAboveThreshold_full,
    percentilePrivacy,
    count_mean_sketch,
    hadamard_mechanism,
    hadamard_response,
    rappor,
    get_noise_generators,
)


@pytest.fixture
def sample_tensor():
    """Generate sample tensor data."""
    torch.manual_seed(42)
    return torch.randn(50)


@pytest.fixture
def sample_list():
    """Generate sample list data."""
    np.random.seed(42)
    return np.random.randn(20).tolist()


def test_svt_above_threshold(sample_list):
    """Test Sparse Vector Technique."""
    result = applySVTAboveThreshold_full(sample_list, epsilon=1.0)

    assert len(result) == len(sample_list)
    assert isinstance(result, list)


def test_percentile_privacy(sample_list):
    """Test percentile privacy mechanism."""
    result = percentilePrivacy(sample_list, percentile=75)

    assert len(result) == len(sample_list)
    assert isinstance(result, list)

    # Check that some values are zeroed
    zeros_count = sum(1 for x in result if x == 0)
    assert zeros_count > 0


def test_percentile_privacy_edge_cases():
    """Test percentile privacy edge cases."""
    data = [1, 2, 3, 4, 5]

    # Test 0th percentile (all values should remain)
    result = percentilePrivacy(data, percentile=0)
    assert all(r != 0 for r in result)

    # Test 100th percentile (all but max should be zeroed)
    result = percentilePrivacy(data, percentile=100)
    non_zero_count = sum(1 for x in result if x != 0)
    assert non_zero_count >= 1  # At least the maximum value

    # Test invalid percentile
    with pytest.raises(ValueError):
        percentilePrivacy(data, percentile=150)


def test_count_mean_sketch(sample_tensor):
    """Test count mean sketch mechanism."""
    result = count_mean_sketch(sample_tensor, epsilon=1.0, bins=10)

    assert isinstance(result, torch.Tensor)
    assert result.shape == sample_tensor.shape
    assert torch.all(torch.isfinite(result))


def test_hadamard_mechanism(sample_tensor):
    """Test Hadamard mechanism."""
    result = hadamard_mechanism(sample_tensor[:32], epsilon=1.0)  # Use power of 2 size

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 32
    assert torch.all(torch.isfinite(result))


def test_hadamard_response():
    """Test Hadamard response mechanism."""
    # Use integer data for local DP
    data = torch.randint(0, 5, (20,))
    result = hadamard_response(data, epsilon=1.0)

    assert isinstance(result, torch.Tensor)
    assert result.shape == data.shape
    assert torch.all(result >= 0)


def test_rappor_mechanism():
    """Test RAPPOR mechanism."""
    # Use integer data
    data = torch.randint(0, 10, (15,))
    result = rappor(data, epsilon=1.0, m=16, k=2)

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 15
    assert torch.all(torch.isfinite(result))


def test_get_noise_generators():
    """Test noise generators dictionary."""
    generators = get_noise_generators()

    assert isinstance(generators, dict)
    assert len(generators) > 0

    # Check that all expected mechanisms are present
    expected_keys = [
        "DP_Gaussian",
        "DP_Exponential",
        "DP_Laplace",
        "gaussian",
        "exponential",
        "laplace",
        "svt",
        "percentile",
        "count_mean_sketch",
        "hadamard",
        "hadamard_response",
        "rappor",
    ]

    for key in expected_keys:
        assert key in generators
        assert callable(generators[key])


def test_privacy_mechanisms_with_different_input_types():
    """Test that mechanisms handle different input types correctly."""
    # Test data in different formats
    list_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    numpy_data = np.array(list_data)
    tensor_data = torch.tensor(list_data)

    from ornl_presto import applyDPGaussian

    # All should work and return same type as input
    result_list = applyDPGaussian(list_data, epsilon=1.0)
    result_numpy = applyDPGaussian(numpy_data, epsilon=1.0)
    result_tensor = applyDPGaussian(tensor_data, epsilon=1.0)

    assert isinstance(result_list, list)
    assert isinstance(result_numpy, np.ndarray)
    assert isinstance(result_tensor, torch.Tensor)

    # All should have same length
    assert len(result_list) == len(list_data)
    assert len(result_numpy) == len(numpy_data)
    assert len(result_tensor) == len(tensor_data)
