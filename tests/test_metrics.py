import numpy as np
from ornl_presto import calculate_utility_privacy_score, evaluate_algorithm_confidence, performance_explanation_metrics

def test_calculate_utility_privacy_score():
    data = [1.0, 2.0, 3.0]
    score = calculate_utility_privacy_score(data, 'DP_Gaussian', epsilon=1.0)
    assert isinstance(score, float)
    assert score < 0

def test_evaluate_algorithm_confidence():
    data = [1.0, 2.0, 3.0]
    res = evaluate_algorithm_confidence(data, 'DP_Gaussian', epsilon=1.0, n_evals=3)
    assert 'mean' in res and 'ci_width' in res
    assert res['mean'] > 0

def test_performance_explanation_metrics():
    metrics = {'mean': 0.5, 'ci_lower': 0.4, 'ci_upper': 0.6}
    perf = performance_explanation_metrics(metrics)
    assert 'mean_rmse' in perf and 'ci_width' in perf and 'reliability' in perf
