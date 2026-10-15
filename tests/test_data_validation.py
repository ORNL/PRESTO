"""
Tests for PRESTO data validation and preprocessing.
"""

import pytest
import numpy as np
import torch
from ornl_presto.data_validation import (
    DataValidator,
    DataPreprocessor,
    validate_and_preprocess,
    recommend_preprocessing_strategy,
)


class TestDataValidator:
    """Test data validation functionality."""

    def test_valid_data(self):
        """Test validation of good quality data."""
        data = torch.randn(100)
        validator = DataValidator()
        results = validator.validate_data(data)

        assert results["valid"] is True
        assert len(results["errors"]) == 0
        assert "statistics" in results
        assert results["statistics"]["size"] == 100

    def test_data_too_small(self):
        """Test validation of too small datasets."""
        data = torch.randn(5)  # Below default minimum of 10
        validator = DataValidator(min_size=10)
        results = validator.validate_data(data)

        assert results["valid"] is False
        assert len(results["errors"]) > 0
        assert any("too small" in error.lower() for error in results["errors"])

    def test_data_with_nan(self):
        """Test validation of data with NaN values."""
        data = torch.tensor([1.0, 2.0, float("nan"), 4.0])
        validator = DataValidator()
        results = validator.validate_data(data)

        assert results["valid"] is False
        assert any("NaN" in error for error in results["errors"])

    def test_data_with_inf(self):
        """Test validation of data with infinite values."""
        data = torch.tensor([1.0, 2.0, float("inf"), 4.0])
        validator = DataValidator()
        results = validator.validate_data(data)

        assert results["valid"] is False
        assert any("infinite" in error for error in results["errors"])

    def test_statistics_calculation(self):
        """Test statistical calculations."""
        data = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        )  # 10 elements
        validator = DataValidator()
        results = validator.validate_data(data)

        stats = results["statistics"]
        assert abs(stats["mean"] - 5.5) < 1e-6  # Mean of 1-10 is 5.5
        assert abs(stats["median"] - 5.5) < 1e-6
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["size"] == 10

    def test_outlier_detection(self):
        """Test outlier detection functionality."""
        # Create data with clear outliers
        normal_data = np.random.normal(0, 1, 90)
        outliers = np.array([10, -10])  # Clear outliers
        data = torch.tensor(np.concatenate([normal_data, outliers]))

        validator = DataValidator()
        results = validator.validate_data(data)

        assert "outlier_percentage_iqr" in results["statistics"]
        assert results["statistics"]["outlier_percentage_iqr"] > 0

    def test_constant_data(self):
        """Test validation of constant data."""
        data = torch.tensor(
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        )  # 10 elements
        validator = DataValidator()
        results = validator.validate_data(data)

        assert results["valid"] is False
        assert any("constant" in error.lower() for error in results["errors"])


class TestDataPreprocessor:
    """Test data preprocessing functionality."""

    def test_basic_preprocessing(self):
        """Test basic preprocessing pipeline."""
        np.random.seed(42)
        data = torch.tensor(np.random.normal(50, 10, 100))

        preprocessor = DataPreprocessor()
        processed_data, info = preprocessor.preprocess_data(data)

        assert isinstance(processed_data, torch.Tensor)
        assert len(processed_data) <= len(data)  # May be smaller due to outlier removal
        assert "original_shape" in info
        assert "final_stats" in info
        assert "steps_applied" in info

    def test_standardization(self):
        """Test data standardization."""
        data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

        preprocessor = DataPreprocessor()
        processed_data, info = preprocessor.preprocess_data(
            data, standardize=True, handle_outliers=False
        )

        # Check that data is approximately standardized
        assert abs(processed_data.mean().item()) < 1e-6  # Should be close to 0
        assert (
            abs(processed_data.std().item() - 1.0) < 0.2
        )  # Should be close to 1, allow some tolerance

    def test_outlier_removal_iqr(self):
        """Test IQR-based outlier removal."""
        # Create data with clear outliers
        normal_data = torch.randn(90)
        outliers = torch.tensor([10.0, -10.0])
        data = torch.cat([normal_data, outliers])

        preprocessor = DataPreprocessor()
        processed_data, info = preprocessor.preprocess_data(
            data, handle_outliers=True, outlier_method="iqr"
        )

        assert len(processed_data) < len(data)  # Some outliers should be removed
        assert info["outliers_removed"] > 0
        assert "outlier_removal_iqr" in info["steps_applied"]

    def test_outlier_removal_zscore(self):
        """Test Z-score based outlier removal."""
        normal_data = torch.randn(90)
        outliers = torch.tensor([5.0, -5.0])  # 5 sigma outliers
        data = torch.cat([normal_data, outliers])

        preprocessor = DataPreprocessor()
        processed_data, info = preprocessor.preprocess_data(
            data, handle_outliers=True, outlier_method="zscore", outlier_threshold=3.0
        )

        assert len(processed_data) < len(data)
        assert info["outliers_removed"] > 0

    def test_no_preprocessing(self):
        """Test with no preprocessing steps."""
        data = torch.randn(50)

        preprocessor = DataPreprocessor()
        processed_data, info = preprocessor.preprocess_data(
            data, standardize=False, handle_outliers=False
        )

        assert torch.allclose(data, processed_data, atol=1e-6)
        assert len(info["steps_applied"]) == 0


class TestPreprocessingRecommendations:
    """Test preprocessing recommendation system."""

    def test_recommend_for_normal_data(self):
        """Test recommendations for normal data."""
        data = torch.randn(100)
        recommendations = recommend_preprocessing_strategy(data)

        assert isinstance(recommendations, dict)
        assert "standardize" in recommendations
        assert "handle_outliers" in recommendations
        assert "rationale" in recommendations

    def test_recommend_for_skewed_data(self):
        """Test recommendations for skewed data."""
        # Create highly skewed data
        data = torch.tensor(np.random.exponential(2.0, 100))
        recommendations = recommend_preprocessing_strategy(data)

        # Should recommend transformation for skewed data
        assert recommendations.get("transformation") in ["log", "sqrt"]
        assert any(
            "skewness" in rationale.lower()
            for rationale in recommendations["rationale"]
        )

    def test_recommend_for_outlier_data(self):
        """Test recommendations for data with outliers."""
        normal_data = np.random.normal(0, 1, 90)
        outliers = np.array([10, -10, 15, -15])
        data = torch.tensor(np.concatenate([normal_data, outliers]))

        recommendations = recommend_preprocessing_strategy(data)

        assert recommendations["handle_outliers"] is True
        assert any(
            "outlier" in rationale.lower() for rationale in recommendations["rationale"]
        )


class TestIntegratedValidationPreprocessing:
    """Test integrated validation and preprocessing."""

    def test_validate_and_preprocess_success(self):
        """Test successful validation and preprocessing."""
        data = torch.randn(100)

        processed_data, info = validate_and_preprocess(data, auto_preprocess=True)

        assert isinstance(processed_data, torch.Tensor)
        assert "validation" in info
        assert "preprocessing" in info
        assert info["validation"]["valid"] is True

    def test_validate_and_preprocess_invalid_data(self):
        """Test with invalid data."""
        data = torch.tensor([float("nan"), 1.0, 2.0])

        with pytest.raises(ValueError):
            validate_and_preprocess(data)

    def test_validate_and_preprocess_custom_params(self):
        """Test with custom preprocessing parameters."""
        data = torch.randn(100)

        processed_data, info = validate_and_preprocess(
            data, auto_preprocess=False, standardize=True, handle_outliers=False
        )

        assert isinstance(processed_data, torch.Tensor)
        assert "standardization" in info["preprocessing"]["steps_applied"]
        assert "outlier_removal" not in str(info["preprocessing"]["steps_applied"])


def test_different_input_types():
    """Test with different input data types."""
    # Test with list (make it larger than minimum size)
    data_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    validator = DataValidator()
    results = validator.validate_data(data_list)
    assert results["valid"] is True

    # Test with numpy array
    data_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    results = validator.validate_data(data_np)
    assert results["valid"] is True

    # Test with torch tensor
    data_torch = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    results = validator.validate_data(data_torch)
    assert results["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__])
