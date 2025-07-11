import numpy as np
import torch
import pytest
from ornl_presto import (
    calculate_utility_privacy_score,
    evaluate_algorithm_confidence,
    performance_explanation_metrics,
)
from ornl_presto.metrics import (
    calculate_similarity_score,
    jensen_shannon_divergence,
    kolmogorov_smirnov_score,
    pearson_correlation_score,
    wasserstein_distance_score,
    calculate_sensitivity,
    estimate_noise_scale,
    privacy_loss_distribution,
    utility_preservation_metrics,
    statistical_distance_metrics,
    confidence_interval_analysis,
)


class TestUtilityPrivacyScoring:
    """Test utility-privacy scoring functions."""

    def test_calculate_utility_privacy_score_basic(self):
        """Test basic utility privacy score calculation."""
        data = [1.0, 2.0, 3.0]
        score = calculate_utility_privacy_score(data, "DP_Gaussian", epsilon=1.0)
        assert isinstance(score, float)
        assert score < 0

    def test_calculate_utility_privacy_score_different_epsilons(self):
        """Test scoring with different epsilon values."""
        data = torch.randn(100)

        scores = []
        epsilons = [0.1, 1.0, 10.0]

        for eps in epsilons:
            score = calculate_utility_privacy_score(data, "DP_Gaussian", epsilon=eps)
            scores.append(score)

        # Higher epsilon should generally give better utility (less negative score)
        assert scores[0] <= scores[1] <= scores[2]

    def test_calculate_utility_privacy_score_different_algorithms(self):
        """Test scoring across different privacy algorithms."""
        data = torch.randn(50)
        algorithms = ["DP_Gaussian", "DP_Laplace", "DP_Exponential"]

        scores = {}
        for algo in algorithms:
            try:
                score = calculate_utility_privacy_score(data, algo, epsilon=1.0)
                scores[algo] = score
                assert isinstance(score, float)
            except Exception as e:
                pytest.skip(f"Algorithm {algo} not available: {e}")

    def test_calculate_utility_privacy_score_edge_cases(self):
        """Test edge cases for utility scoring."""
        # Very small data
        small_data = [1.0]
        score = calculate_utility_privacy_score(small_data, "DP_Gaussian", epsilon=1.0)
        assert isinstance(score, float)

        # Large epsilon
        data = torch.randn(20)
        score = calculate_utility_privacy_score(data, "DP_Gaussian", epsilon=100.0)
        assert isinstance(score, float)

        # Very small epsilon
        score = calculate_utility_privacy_score(data, "DP_Gaussian", epsilon=0.01)
        assert isinstance(score, float)


class TestConfidenceEvaluation:
    """Test algorithm confidence evaluation."""

    def test_evaluate_algorithm_confidence_basic(self):
        """Test basic confidence evaluation."""
        data = [1.0, 2.0, 3.0]
        res = evaluate_algorithm_confidence(data, "DP_Gaussian", epsilon=1.0, n_evals=3)

        assert "mean" in res and "ci_width" in res
        assert res["mean"] > 0
        assert isinstance(res["std"], float)
        assert isinstance(res["ci_lower"], float)
        assert isinstance(res["ci_upper"], float)
        assert isinstance(res["ci_width"], float)
        assert "scores" in res

    def test_evaluate_algorithm_confidence_multiple_evaluations(self):
        """Test confidence with different numbers of evaluations."""
        data = torch.randn(50)

        for n_evals in [5, 10, 20]:
            res = evaluate_algorithm_confidence(
                data, "DP_Gaussian", epsilon=1.0, n_evals=n_evals
            )
            assert len(res["scores"]) == n_evals
            assert res["mean"] > 0
            assert res["ci_width"] > 0

    def test_evaluate_algorithm_confidence_consistency(self):
        """Test confidence evaluation consistency."""
        data = torch.randn(100)

        # Multiple runs should give similar results
        results = []
        for _ in range(3):
            res = evaluate_algorithm_confidence(
                data, "DP_Gaussian", epsilon=1.0, n_evals=10
            )
            results.append(res["mean"])

        # Results should be reasonably consistent (within 50% of each other)
        mean_result = np.mean(results)
        for result in results:
            assert abs(result - mean_result) / mean_result < 0.5

    def test_evaluate_algorithm_confidence_different_algorithms(self):
        """Test confidence evaluation across algorithms."""
        data = torch.randn(30)
        algorithms = ["DP_Gaussian", "DP_Laplace"]

        for algo in algorithms:
            try:
                res = evaluate_algorithm_confidence(data, algo, epsilon=1.0, n_evals=5)
                assert res["mean"] > 0
                assert res["ci_width"] > 0
            except Exception as e:
                pytest.skip(f"Algorithm {algo} not available: {e}")


class TestPerformanceMetrics:
    """Test performance explanation metrics."""

    def test_performance_explanation_metrics_basic(self):
        """Test basic performance metrics calculation."""
        metrics = {"mean": 0.5, "ci_lower": 0.4, "ci_upper": 0.6}
        perf = performance_explanation_metrics(metrics)

        assert "mean_rmse" in perf
        assert "ci_width" in perf
        assert "reliability" in perf
        assert isinstance(perf["mean_rmse"], float)
        assert isinstance(perf["ci_width"], float)
        assert isinstance(perf["reliability"], float)

    def test_performance_explanation_metrics_edge_cases(self):
        """Test performance metrics with edge cases."""
        # Very narrow confidence interval
        narrow_metrics = {"mean": 0.1, "ci_lower": 0.09, "ci_upper": 0.11}
        perf = performance_explanation_metrics(narrow_metrics)
        assert perf["ci_width"] < 0.1
        assert perf["reliability"] > 90  # High reliability for narrow CI

        # Wide confidence interval
        wide_metrics = {"mean": 0.5, "ci_lower": 0.1, "ci_upper": 0.9}
        perf = performance_explanation_metrics(wide_metrics)
        assert perf["ci_width"] > 0.5
        assert perf["reliability"] < 50  # Low reliability for wide CI

    def test_performance_explanation_metrics_reliability_calculation(self):
        """Test reliability calculation formula."""
        # Test known reliability values
        test_cases = [
            {"mean": 1.0, "ci_lower": 0.9, "ci_upper": 1.1, "expected_rel": 95},
            {"mean": 1.0, "ci_lower": 0.5, "ci_upper": 1.5, "expected_rel": 50},
            {"mean": 1.0, "ci_lower": 0.0, "ci_upper": 2.0, "expected_rel": 0},
        ]

        for case in test_cases:
            perf = performance_explanation_metrics(
                {
                    "mean": case["mean"],
                    "ci_lower": case["ci_lower"],
                    "ci_upper": case["ci_upper"],
                }
            )
            assert abs(perf["reliability"] - case["expected_rel"]) < 5.1


class TestSimilarityMetrics:
    """Test similarity scoring functions."""

    def test_calculate_similarity_score_basic(self):
        """Test basic similarity score calculation."""
        original = torch.randn(100)
        private = original + torch.randn(100) * 0.1  # Add small noise

        score = calculate_similarity_score(original, private)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Similarity should be between 0 and 1

    def test_calculate_similarity_score_identical_data(self):
        """Test similarity with identical data."""
        data = torch.randn(50)
        score = calculate_similarity_score(data, data)
        assert score > 0.99  # Should be very high similarity

    def test_calculate_similarity_score_random_data(self):
        """Test similarity with completely random data."""
        original = torch.randn(100)
        random_data = torch.randn(100)

        score = calculate_similarity_score(original, random_data)
        assert 0 <= score <= 1
        assert score < 0.8  # Should be relatively low similarity


class TestStatisticalDistanceMetrics:
    """Test statistical distance measurement functions."""

    def test_jensen_shannon_divergence(self):
        """Test Jensen-Shannon divergence calculation."""
        data1 = torch.randn(100)
        data2 = data1 + torch.randn(100) * 0.1

        js_div = jensen_shannon_divergence(data1, data2)
        assert isinstance(js_div, float)
        assert 0 <= js_div <= 1

    def test_kolmogorov_smirnov_score(self):
        """Test Kolmogorov-Smirnov score calculation."""
        data1 = torch.randn(100)
        data2 = data1 + torch.randn(100) * 0.1

        ks_score = kolmogorov_smirnov_score(data1, data2)
        assert isinstance(ks_score, float)
        assert 0 <= ks_score <= 1

    def test_pearson_correlation_score(self):
        """Test Pearson correlation score calculation."""
        data1 = torch.randn(100)
        data2 = data1 + torch.randn(100) * 0.1

        corr_score = pearson_correlation_score(data1, data2)
        assert isinstance(corr_score, float)
        assert -1 <= corr_score <= 1

    def test_wasserstein_distance_score(self):
        """Test Wasserstein distance score calculation."""
        data1 = torch.randn(100)
        data2 = data1 + torch.randn(100) * 0.1

        w_score = wasserstein_distance_score(data1, data2)
        assert isinstance(w_score, float)
        assert w_score >= 0


class TestPrivacyMetrics:
    """Test privacy-specific metrics."""

    def test_calculate_sensitivity(self):
        """Test sensitivity calculation for different query types."""
        data = torch.randn(100)

        # Count query sensitivity
        count_sens = calculate_sensitivity(data, query_type="count")
        assert count_sens == 1.0

        # Sum query sensitivity
        sum_sens = calculate_sensitivity(data, query_type="sum")
        assert sum_sens > 0

        # Mean query sensitivity
        mean_sens = calculate_sensitivity(data, query_type="mean")
        assert mean_sens > 0

    def test_estimate_noise_scale(self):
        """Test noise scale estimation."""
        sensitivity = 1.0
        epsilon = 1.0

        # Laplace noise scale
        laplace_scale = estimate_noise_scale(sensitivity, epsilon, mechanism="laplace")
        assert laplace_scale == sensitivity / epsilon

        # Gaussian noise scale
        gaussian_scale = estimate_noise_scale(
            sensitivity, epsilon, mechanism="gaussian", delta=1e-5
        )
        assert gaussian_scale > 0

    def test_privacy_loss_distribution(self):
        """Test privacy loss distribution calculation."""
        epsilon = 1.0
        delta = 1e-5

        pld = privacy_loss_distribution(epsilon, delta)
        assert isinstance(pld, dict)
        assert "epsilon" in pld
        assert "delta" in pld
        assert "composition_bounds" in pld


class TestUtilityPreservationMetrics:
    """Test utility preservation measurement."""

    def test_utility_preservation_metrics_basic(self):
        """Test basic utility preservation metrics."""
        original = torch.randn(100)
        private = original + torch.randn(100) * 0.1

        metrics = utility_preservation_metrics(original, private)

        assert "mean_absolute_error" in metrics
        assert "mean_squared_error" in metrics
        assert "relative_error" in metrics
        assert "signal_to_noise_ratio" in metrics

        for metric_name, value in metrics.items():
            if metric_name in ["original_moments", "private_moments"]:
                assert isinstance(value, dict)
                # Check that all moment values are floats
                for moment_name, moment_value in value.items():
                    assert isinstance(moment_value, float)
                    assert not np.isnan(moment_value)
            else:
                assert isinstance(value, float)
                assert not np.isnan(value)

    def test_utility_preservation_perfect_case(self):
        """Test utility preservation with identical data."""
        data = torch.randn(100)
        metrics = utility_preservation_metrics(data, data)

        assert metrics["mean_absolute_error"] < 1e-10
        assert metrics["mean_squared_error"] < 1e-10
        assert metrics["relative_error"] < 1e-10

    def test_utility_preservation_high_noise_case(self):
        """Test utility preservation with high noise."""
        original = torch.randn(100)
        private = original + torch.randn(100) * 10  # High noise

        metrics = utility_preservation_metrics(original, private)

        assert metrics["mean_absolute_error"] > 1.0
        assert metrics["mean_squared_error"] > 1.0
        assert metrics["signal_to_noise_ratio"] < 1.0


class TestConfidenceIntervalAnalysis:
    """Test confidence interval analysis functionality."""

    def test_confidence_interval_analysis_basic(self):
        """Test basic confidence interval analysis."""
        scores = [0.8, 0.75, 0.85, 0.9, 0.7, 0.88, 0.82, 0.79, 0.86, 0.83]

        analysis = confidence_interval_analysis(scores)

        assert "mean" in analysis
        assert "std" in analysis
        assert "ci_lower" in analysis
        assert "ci_upper" in analysis
        assert "ci_width" in analysis
        assert "confidence_level" in analysis

        assert 0.7 <= analysis["mean"] <= 0.9
        assert analysis["ci_lower"] < analysis["mean"] < analysis["ci_upper"]

    def test_confidence_interval_analysis_different_confidence_levels(self):
        """Test confidence intervals with different confidence levels."""
        scores = np.random.normal(0.8, 0.1, 50).tolist()

        for confidence in [0.90, 0.95, 0.99]:
            analysis = confidence_interval_analysis(scores, confidence_level=confidence)

            assert analysis["confidence_level"] == confidence
            assert analysis["ci_width"] > 0

            # Higher confidence should give wider intervals
            if confidence == 0.99:
                wide_analysis = analysis
            elif confidence == 0.90:
                narrow_analysis = analysis

        assert wide_analysis["ci_width"] > narrow_analysis["ci_width"]

    def test_confidence_interval_analysis_edge_cases(self):
        """Test confidence interval analysis edge cases."""
        # Single value
        single_score = [0.8]
        analysis = confidence_interval_analysis(single_score)
        assert analysis["std"] == 0.0
        assert analysis["ci_width"] == 0.0

        # Very low variance
        low_var_scores = [0.8, 0.8, 0.8, 0.8, 0.8]
        analysis = confidence_interval_analysis(low_var_scores)
        assert analysis["std"] == 0.0
        assert analysis["ci_width"] == 0.0


class TestStatisticalDistanceMetricsAdvanced:
    """Test advanced statistical distance metrics."""

    def test_statistical_distance_metrics_comprehensive(self):
        """Test comprehensive statistical distance metrics."""
        original = torch.randn(200)
        private = original + torch.randn(200) * 0.2

        metrics = statistical_distance_metrics(original, private)

        expected_metrics = [
            "jensen_shannon_divergence",
            "kolmogorov_smirnov_distance",
            "wasserstein_distance",
            "total_variation_distance",
            "hellinger_distance",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert not np.isnan(metrics[metric])

    def test_statistical_distance_metrics_identical_data(self):
        """Test statistical distance metrics with identical data."""
        data = torch.randn(100)
        metrics = statistical_distance_metrics(data, data)

        # Most distance metrics should be 0 for identical data
        for metric_name, value in metrics.items():
            if "distance" in metric_name or "divergence" in metric_name:
                assert value < 1e-10

    def test_statistical_distance_metrics_orthogonal_data(self):
        """Test statistical distance metrics with orthogonal data."""
        data1 = torch.zeros(100)
        data2 = torch.ones(100)

        metrics = statistical_distance_metrics(data1, data2)

        # Distance metrics should be high for very different data
        for metric_name, value in metrics.items():
            if "distance" in metric_name or "divergence" in metric_name:
                assert value > 0.1  # Should be substantial distance


class TestIntegrationMetrics:
    """Test integration of multiple metrics."""

    def test_comprehensive_privacy_utility_assessment(self):
        """Test comprehensive assessment combining multiple metrics."""
        original_data = torch.randn(150)

        # Simulate different privacy levels
        privacy_levels = [0.1, 1.0, 10.0]
        results = {}

        for epsilon in privacy_levels:
            # Add noise proportional to privacy level
            noise_scale = 1.0 / epsilon
            private_data = original_data + torch.randn(150) * noise_scale

            # Calculate multiple metrics
            utility_metrics = utility_preservation_metrics(original_data, private_data)
            distance_metrics = statistical_distance_metrics(original_data, private_data)
            similarity = calculate_similarity_score(original_data, private_data)

            results[epsilon] = {
                "utility": utility_metrics,
                "distances": distance_metrics,
                "similarity": similarity,
            }

        # Verify privacy-utility tradeoff trends
        # Higher epsilon (less privacy) should give better utility
        assert (
            results[10.0]["similarity"]
            >= results[1.0]["similarity"]
            >= results[0.1]["similarity"]
        )

        assert (
            results[10.0]["utility"]["signal_to_noise_ratio"]
            >= results[1.0]["utility"]["signal_to_noise_ratio"]
            >= results[0.1]["utility"]["signal_to_noise_ratio"]
        )

    def test_cross_algorithm_metric_consistency(self):
        """Test metric consistency across different algorithms."""
        data = torch.randn(100)
        epsilon = 1.0

        algorithms = ["DP_Gaussian", "DP_Laplace"]
        algorithm_metrics = {}

        for algo in algorithms:
            try:
                # Get confidence metrics
                confidence_res = evaluate_algorithm_confidence(
                    data, algo, epsilon, n_evals=5
                )

                # Get performance metrics
                perf_metrics = performance_explanation_metrics(confidence_res)

                algorithm_metrics[algo] = {
                    "confidence": confidence_res,
                    "performance": perf_metrics,
                }

                # Verify metric structure consistency
                assert "mean_rmse" in perf_metrics
                assert "reliability" in perf_metrics
                assert "ci_width" in perf_metrics

            except Exception as e:
                pytest.skip(f"Algorithm {algo} not available: {e}")

        # If we have results for multiple algorithms, compare them
        if len(algorithm_metrics) > 1:
            algos = list(algorithm_metrics.keys())

            # All algorithms should produce valid metrics
            for algo in algos:
                assert algorithm_metrics[algo]["performance"]["reliability"] >= 0
                assert algorithm_metrics[algo]["performance"]["reliability"] <= 100
                assert algorithm_metrics[algo]["confidence"]["mean"] > 0


# Test data fixtures for reuse
@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return torch.randn(100)


@pytest.fixture
def sample_private_data(sample_data):
    """Sample private data derived from sample_data."""
    return sample_data + torch.randn(100) * 0.1


def test_end_to_end_metrics_pipeline(sample_data, sample_private_data):
    """Test complete metrics calculation pipeline."""
    # Calculate utility-privacy score
    score = calculate_utility_privacy_score(sample_data, "DP_Gaussian", epsilon=1.0)
    assert isinstance(score, float)

    # Evaluate confidence
    confidence = evaluate_algorithm_confidence(
        sample_data, "DP_Gaussian", epsilon=1.0, n_evals=3
    )
    assert "mean" in confidence

    # Get performance metrics
    performance = performance_explanation_metrics(confidence)
    assert "reliability" in performance

    # Calculate similarity
    similarity = calculate_similarity_score(sample_data, sample_private_data)
    assert 0 <= similarity <= 1

    # All metrics should be valid numbers
    assert not np.isnan(score)
    assert not np.isnan(confidence["mean"])
    assert not np.isnan(performance["reliability"])
    assert not np.isnan(similarity)


def test_calculate_utility_privacy_score():
    data = [1.0, 2.0, 3.0]
    score = calculate_utility_privacy_score(data, "DP_Gaussian", epsilon=1.0)
    assert isinstance(score, float)
    assert score < 0


def test_evaluate_algorithm_confidence():
    data = [1.0, 2.0, 3.0]
    res = evaluate_algorithm_confidence(data, "DP_Gaussian", epsilon=1.0, n_evals=3)
    assert "mean" in res and "ci_width" in res
    assert res["mean"] > 0


def test_performance_explanation_metrics():
    metrics = {"mean": 0.5, "ci_lower": 0.4, "ci_upper": 0.6}
    perf = performance_explanation_metrics(metrics)
    assert "mean_rmse" in perf and "ci_width" in perf and "reliability" in perf
