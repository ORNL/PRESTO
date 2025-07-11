# Privacy mechanisms
from .privacy_mechanisms import (
    applyDPGaussian,
    applyDPExponential,
    applyDPLaplace,
    above_threshold_SVT,
    applySVTAboveThreshold_full,
    percentilePrivacy,
    count_mean_sketch,
    hadamard_mechanism,
    hadamard_response,
    rappor,
    get_noise_generators,
)

# Metrics and evaluation
from .metrics import (
    calculate_utility_privacy_score,
    evaluate_algorithm_confidence,
    performance_explanation_metrics,
    recommend_top3,
    recommend_best_algorithms,
)

# Visualization
from .visualization import (
    visualize_data,
    visualize_similarity,
    visualize_top3,
    visualize_confidence,
    visualize_confidence_top3,
    visualize_overlay_original_and_private,
)

# Core ML/GP functionality (still in core.py)
from .core import (
    dp_function,
    dp_function_train_and_pred,
    dp_target,
    dp_pareto_front,
    gpr_gpytorch,
    dp_hyper,
)

# Configuration management
from .config import (
    PRESTOConfig,
    PrivacyConfig,
    OptimizationConfig,
    DataConfig,
    VisualizationConfig,
    ConfigManager,
    get_domain_recommendations,
)

# Data validation and preprocessing
from .data_validation import (
    DataValidator,
    DataPreprocessor,
    validate_and_preprocess,
    recommend_preprocessing_strategy,
)

__version__ = "1.0.0"
__author__ = "ORNL PRESTO Team"
__all__ = [
    # Privacy mechanisms
    "applyDPGaussian",
    "applyDPExponential",
    "applyDPLaplace",
    "above_threshold_SVT",
    "applySVTAboveThreshold_full",
    "percentilePrivacy",
    "count_mean_sketch",
    "hadamard_mechanism",
    "hadamard_response",
    "rappor",
    "get_noise_generators",
    # Metrics
    "calculate_utility_privacy_score",
    "evaluate_algorithm_confidence",
    "performance_explanation_metrics",
    "recommend_top3",
    "recommend_best_algorithms",
    # Visualization
    "visualize_data",
    "visualize_similarity",
    "visualize_top3",
    "visualize_confidence",
    "visualize_confidence_top3",
    "visualize_overlay_original_and_private",
    # Core functionality
    "dp_function",
    "dp_function_train_and_pred",
    "dp_target",
    "dp_pareto_front",
    "gpr_gpytorch",
    "dp_hyper",
    # Configuration management
    "PRESTOConfig",
    "PrivacyConfig",
    "OptimizationConfig",
    "DataConfig",
    "VisualizationConfig",
    "ConfigManager",
    "get_domain_recommendations",
    # Data validation and preprocessing
    "DataValidator",
    "DataPreprocessor",
    "validate_and_preprocess",
    "recommend_preprocessing_strategy",
]
