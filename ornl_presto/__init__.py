"""
PRESTO: Privacy REcommendation and SecuriTy Optimization

A Python package for automated differential privacy mechanism selection and optimization.
PRESTO analyzes datasets and recommends optimal privacy-preserving algorithms with
quantified utility-privacy trade-offs, confidence intervals, and reliability metrics.

Main Features:
- Automated privacy mechanism selection using Bayesian optimization
- Support for multiple differential privacy algorithms (Laplace, Gaussian,
  Exponential, etc.)
- Comprehensive utility and privacy metrics with statistical significance testing
- Data-driven recommendations with confidence intervals and reliability analysis
- Extensible architecture for custom privacy mechanisms
- Integration with popular machine learning libraries (PyTorch, NumPy, Pandas)
- Rich visualization tools for privacy-utility analysis

Quick Start:
    Install PRESTO and get privacy recommendations in minutes:

    >>> pip install ornl-presto
    >>> import numpy as np
    >>> from ornl_presto import recommend_top3, visualize_similarity
    >>>
    >>> # Load your dataset
    >>> data = np.random.randn(1000)  # Replace with your actual data
    >>>
    >>> # Get top 3 privacy mechanism recommendations
    >>> recommendations = recommend_top3(data)
    >>> for i, rec in enumerate(recommendations, 1):
    ...     print(f"{i}. {rec['algorithm']} (ε={rec['epsilon']:.4f}, "
    ...           f"utility={rec['score']:.4f})")
    >>>
    >>> # Visualize the best recommendation
    >>> best = recommendations[0]
    >>> visualize_similarity(data, best['algorithm'], best['epsilon'])

Advanced Usage:
    For domain-specific optimization and custom configurations:

    >>> from ornl_presto import (evaluate_algorithm_confidence, PRESTOConfig,
    ...                         ConfigManager, visualize_confidence)
    >>>
    >>> # Configure for medical domain with strict privacy requirements
    >>> config = ConfigManager.get_domain_config('medical')
    >>> config.privacy.min_epsilon = 0.01  # Very strict privacy
    >>> config.privacy.max_epsilon = 1.0
    >>>
    >>> # Evaluate mechanism reliability with confidence intervals
    >>> confidence = evaluate_algorithm_confidence(
    ...     data, 'laplace', epsilon=0.5, n_evals=20
    ... )
    >>> print(f"Mean utility: {confidence['mean']:.4f} ± {confidence['std']:.4f}")
    >>> print(f"95% CI: [{confidence['ci_lower']:.4f}, {confidence['ci_upper']:.4f}]")
    >>>
    >>> # Visualize confidence intervals
    >>> visualize_confidence(confidence)

Custom Privacy Mechanisms:
    Extend PRESTO with domain-specific privacy algorithms:

    >>> from ornl_presto.privacy_mechanisms import get_noise_generators
    >>> import torch
    >>>
    >>> def custom_mechanism(data, epsilon, custom_param=1.0):
    ...     \"\"\"Custom privacy mechanism implementation.\"\"\"
    ...     noise_scale = custom_param / epsilon
    ...     noise = torch.laplace(torch.zeros_like(data), noise_scale)
    ...     return data + noise
    >>>
    >>> # Register custom mechanism
    >>> generators = get_noise_generators()
    >>> generators['custom'] = custom_mechanism
    >>>
    >>> # Use in recommendations
    >>> score = calculate_utility_privacy_score(data, 'custom', 1.0, custom_param=2.0)

Research and Development:
    For advanced privacy research with multi-objective optimization:

    >>> from ornl_presto.core import dp_pareto_front, gpr_gpytorch
    >>>
    >>> # Generate Pareto front for privacy-utility trade-offs
    >>> pareto_results = dp_pareto_front(
    ...     data, algorithm='gaussian',
    ...     epsilon_range=(0.1, 10.0), n_points=20
    ... )
    >>>
    >>> # Use Gaussian Process Regression for hyperparameter optimization
    >>> model = gpr_gpytorch(data)
    >>> optimized_params = model.optimize_acquisition()

Supported Domains:
    PRESTO provides optimized configurations for various application domains:
    - Medical/Healthcare (HIPAA compliance)
    - Financial (regulatory requirements)
    - Energy/Smart grids (infrastructure privacy)
    - General research (flexible parameters)

For comprehensive documentation and tutorials, visit:
https://ornl-presto.readthedocs.io/
"""

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

__version__ = "0.1.28"
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
