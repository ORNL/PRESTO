# PRESTO Advanced Features Guide

## Table of Contents
- [Configuration Management](#configuration-management)
- [Data Validation & Preprocessing](#data-validation--preprocessing)
- [Advanced Examples](#advanced-examples)
- [Performance Benchmarking](#performance-benchmarking)
- [Production Deployment](#production-deployment)

## Configuration Management

PRESTO now includes a comprehensive configuration management system for different use cases and deployment scenarios.

### Quick Start with Configurations

```python
from ornl_presto.config import ConfigManager

# Get predefined configuration for healthcare
config = ConfigManager.get_config("healthcare")

# List all available configurations
available_configs = ConfigManager.list_configs()
print(available_configs)
# Output: ['healthcare', 'finance', 'research', 'iot_sensors', 'survey_data', 'production_fast', 'development']
```

### Available Predefined Configurations

| Configuration | Use Case | Privacy Level | Recommended ε | Key Features |
|---------------|----------|---------------|---------------|--------------|
| `healthcare` | Medical data, HIPAA compliance | Very High | 0.01 - 0.5 | Strict privacy, high utility preservation |
| `finance` | Financial data, regulatory compliance | High | 0.1 - 2.0 | Multi-algorithm evaluation, audit trails |
| `research` | Academic research with IRB approval | Medium-High | 0.5 - 5.0 | Reproducibility, broad exploration |
| `iot_sensors` | IoT sensor data and telemetry | Medium | 1.0 - 10.0 | Large volumes, real-time processing |
| `survey_data` | Survey responses, questionnaires | Medium-High | 0.1 - 2.0 | Local DP, categorical handling |
| `production_fast` | Production with speed requirements | Medium | 1.0 - 5.0 | Fast optimization, parallel processing |
| `development` | Development and testing | Low | 0.1 - 10.0 | Quick iterations, verbose output |

### Custom Configuration

```python
from ornl_presto.config import PRESTOConfig, PrivacyConfig, OptimizationConfig

# Create custom configuration
config = PRESTOConfig()

# Customize privacy requirements
config.privacy.epsilon_min = 0.05
config.privacy.epsilon_max = 2.0
config.privacy.required_algorithms = ["DP_Gaussian", "DP_Laplace"]
config.privacy.utility_threshold = 0.85

# Customize optimization
config.optimization.n_evals = 15
config.optimization.init_points = 5
config.optimization.n_iter = 25

# Save for reuse
ConfigManager.save_config(config, "my_custom_config.json")

# Load later
config = ConfigManager.load_config("my_custom_config.json")
```

### Domain-Specific Recommendations

```python
from ornl_presto.config import get_domain_recommendations

recommendations = get_domain_recommendations()
healthcare_rec = recommendations["healthcare"]

print(f"Description: {healthcare_rec['description']}")
print(f"Privacy Level: {healthcare_rec['privacy_level']}")
print(f"Recommended ε: {healthcare_rec['recommended_epsilon']}")
print("Key Considerations:")
for consideration in healthcare_rec['key_considerations']:
    print(f"  - {consideration}")
```

## Data Validation & Preprocessing

PRESTO includes advanced data validation and preprocessing capabilities to ensure optimal privacy algorithm performance.

### Automatic Data Validation

```python
from ornl_presto.data_validation import DataValidator
import torch

# Your data
data = torch.randn(1000)

# Validate data quality
validator = DataValidator()
results = validator.validate_data(data)

print(f"Data valid: {results['valid']}")
print(f"Warnings: {len(results['warnings'])}")
print(f"Recommendations: {len(results['recommendations'])}")

# View specific issues
for warning in results['warnings']:
    print(f"{warning}")

for rec in results['recommendations']:
    print(f"{rec}")
```

### Smart Preprocessing

```python
from ornl_presto.data_validation import validate_and_preprocess

# Automatic validation and preprocessing
processed_data, info = validate_and_preprocess(
    data,
    auto_preprocess=True  # Automatically applies recommended preprocessing
)

print(f"Original size: {info['preprocessing']['original_stats']['size']}")
print(f"Final size: {info['preprocessing']['final_stats']['size']}")
print(f"Outliers removed: {info['preprocessing']['outliers_removed']}")
print(f"Steps applied: {info['preprocessing']['steps_applied']}")
```

### Manual Preprocessing Control

```python
from ornl_presto.data_validation import DataPreprocessor

preprocessor = DataPreprocessor()

# Custom preprocessing
processed_data, info = preprocessor.preprocess_data(
    data,
    standardize=True,
    handle_outliers=True,
    outlier_method="iqr",  # Options: "iqr", "zscore", "isolation_forest"
    transformation="log"   # Options: "log", "sqrt", "box_cox"
)
```

### Preprocessing Recommendations

```python
from ornl_presto.data_validation import recommend_preprocessing_strategy

# Get smart recommendations
recommendations = recommend_preprocessing_strategy(data)

print(f"Standardize: {recommendations['standardize']}")
print(f"Handle outliers: {recommendations['handle_outliers']}")
print(f"Outlier method: {recommendations['outlier_method']}")
print(f"Transformation: {recommendations['transformation']}")
print("Rationale:")
for reason in recommendations['rationale']:
    print(f"  - {reason}")
```

## Advanced Examples

### Healthcare Data Analysis with Configuration

```python
from ornl_presto import recommend_top3, visualize_top3
from ornl_presto.config import ConfigManager
from ornl_presto.data_validation import validate_and_preprocess

# Load healthcare configuration
config = ConfigManager.get_config("healthcare")

# Simulate patient heart rate data
import numpy as np
np.random.seed(42)
heart_rates = np.random.normal(72, 12, 500)  # 500 patients
heart_rates = np.clip(heart_rates, 50, 120)   # Realistic range

# Validate and preprocess
processed_data, info = validate_and_preprocess(heart_rates)

# Get recommendations using healthcare config
top3 = recommend_top3(
    processed_data,
    n_evals=config.optimization.n_evals,
    init_points=config.optimization.init_points,
    n_iter=config.optimization.n_iter
)

# Filter by required algorithms (from config)
filtered_top3 = [
    rec for rec in top3
    if rec['algorithm'] in config.privacy.required_algorithms
]

print("Healthcare-Compliant Privacy Recommendations:")
for i, rec in enumerate(filtered_top3, 1):
    # Check if meets privacy requirements
    compliant = (config.privacy.epsilon_min <= rec['epsilon'] <= config.privacy.epsilon_max)
    utility_ok = rec.get('utility_preservation', 0) >= config.privacy.utility_threshold

    status = "[COMPLIANT]" if compliant and utility_ok else "[NON-COMPLIANT]"
    print(f"{i}. {rec['algorithm']} (ε={rec['epsilon']:.3f}) {status}")

# Visualize results
visualize_top3(filtered_top3)
```

### Production Pipeline with Fast Configuration

```python
from ornl_presto.config import ConfigManager
from ornl_presto import get_noise_generators, calculate_utility_privacy_score
import time

# Production-optimized configuration
config = ConfigManager.get_config("production_fast")

# Simulate production data processing
def production_privacy_pipeline(data, target_epsilon=1.0):
    start_time = time.time()

    # Quick validation (reduced checks for speed)
    processed_data, _ = validate_and_preprocess(data, auto_preprocess=True)

    # Fast algorithm evaluation using config settings
    noise_generators = get_noise_generators()
    results = {}

    for algo_name in config.privacy.required_algorithms:
        if algo_name in noise_generators:
            try:
                # Apply privacy mechanism
                private_data = noise_generators[algo_name](processed_data, target_epsilon)

                # Quick utility evaluation
                utility_score = calculate_utility_privacy_score(
                    processed_data, algo_name, target_epsilon
                )

                results[algo_name] = {
                    'utility_score': utility_score,
                    'algorithm': algo_name,
                    'epsilon': target_epsilon
                }
            except Exception as e:
                print(f"Error with {algo_name}: {e}")

    # Select best algorithm
    best_algo = max(results.values(), key=lambda x: x['utility_score'])

    processing_time = time.time() - start_time

    return best_algo, processing_time

# Test production pipeline
import torch
production_data = torch.randn(10000)  # Large production dataset

best_algorithm, time_taken = production_privacy_pipeline(production_data)
print(f"Best algorithm: {best_algorithm['algorithm']}")
print(f"Utility score: {best_algorithm['utility_score']:.4f}")
print(f"Processing time: {time_taken:.2f} seconds")
```

## Performance Benchmarking

### Comprehensive Benchmarking

```python
from examples.benchmark_performance import PRESTOBenchmark, run_comprehensive_benchmark

# Run full benchmark suite
performance_df, scalability_df, quality_results = run_comprehensive_benchmark()

# View top performers
print("Top 3 Fastest Algorithms:")
speed_ranking = performance_df.groupby('algorithm')['execution_time'].mean().sort_values()
for i, (algo, time) in enumerate(speed_ranking.head(3).items(), 1):
    print(f"{i}. {algo}: {time:.4f}s average")

print("\nTop 3 Utility Preserving Algorithms:")
utility_ranking = performance_df.groupby('algorithm')['utility_score'].mean().sort_values(ascending=False)
for i, (algo, score) in enumerate(utility_ranking.head(3).items(), 1):
    print(f"{i}. {algo}: {score:.4f} utility score")
```

### Custom Benchmarking

```python
from examples.benchmark_performance import PRESTOBenchmark

benchmark = PRESTOBenchmark()

# Test specific algorithms and conditions
custom_results = benchmark.benchmark_algorithm_performance(
    algorithms=["DP_Gaussian", "DP_Laplace"],
    data_sizes=[1000, 5000, 10000],
    data_types=["normal", "exponential"],
    epsilon_values=[0.5, 1.0, 2.0]
)

# Generate performance report
report = benchmark.generate_performance_report(custom_results)
print(report)

# Create visualizations
benchmark.plot_performance_summary(custom_results, "my_benchmark_results.png")
```

## Production Deployment

### Environment Setup

```python
from ornl_presto.config import ConfigManager

# Production environment configuration
config = ConfigManager.get_config("production_fast")

# Enable GPU acceleration if available
import torch
if torch.cuda.is_available():
    config.gpu_acceleration = True
    print("GPU acceleration enabled")

# Set parallel processing
config.parallel_workers = 4  # Adjust based on your system

# Save production config
ConfigManager.save_config(config, "/etc/presto/production.json")
```

### Production-Ready Privacy Service

```python
class PrivacyService:
    """Production-ready privacy preservation service."""

    def __init__(self, config_path="/etc/presto/production.json"):
        self.config = ConfigManager.load_config(config_path)
        self.noise_generators = get_noise_generators()
        self.cache = {}

    def apply_privacy(self, data, epsilon=None, algorithm=None):
        """Apply privacy mechanism with production optimizations."""

        # Use default epsilon from config if not specified
        if epsilon is None:
            epsilon = (self.config.privacy.epsilon_min + self.config.privacy.epsilon_max) / 2

        # Validate epsilon bounds
        if not (self.config.privacy.epsilon_min <= epsilon <= self.config.privacy.epsilon_max):
            raise ValueError(f"Epsilon {epsilon} outside allowed range")

        # Auto-select algorithm if not specified
        if algorithm is None:
            # Use cached recommendation or compute new one
            cache_key = f"best_algo_{len(data)}_{epsilon}"
            if cache_key in self.cache:
                algorithm = self.cache[cache_key]
            else:
                # Quick algorithm evaluation
                best_score = float('-inf')
                best_algo = self.config.privacy.required_algorithms[0]

                for algo in self.config.privacy.required_algorithms:
                    if algo in self.noise_generators:
                        score = calculate_utility_privacy_score(data, algo, epsilon)
                        if score > best_score:
                            best_score = score
                            best_algo = algo

                algorithm = best_algo
                self.cache[cache_key] = algorithm

        # Apply privacy mechanism
        if algorithm not in self.noise_generators:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        private_data = self.noise_generators[algorithm](data, epsilon)

        return {
            'private_data': private_data,
            'algorithm': algorithm,
            'epsilon': epsilon,
            'utility_score': calculate_utility_privacy_score(data, algorithm, epsilon)
        }

# Usage in production
service = PrivacyService()

# Process incoming data
import torch
incoming_data = torch.randn(1000)  # Your production data

result = service.apply_privacy(
    incoming_data,
    epsilon=1.0,
    algorithm="DP_Gaussian"
)

print(f"Applied {result['algorithm']} with ε={result['epsilon']}")
print(f"Utility preservation: {result['utility_score']:.3f}")
```

### Monitoring and Logging

```python
import logging
from datetime import datetime

# Setup production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/presto/privacy_service.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('presto_privacy_service')

class MonitoredPrivacyService(PrivacyService):
    """Privacy service with monitoring and logging."""

    def apply_privacy(self, data, epsilon=None, algorithm=None):
        start_time = time.time()
        data_size = len(data)

        try:
            result = super().apply_privacy(data, epsilon, algorithm)

            processing_time = time.time() - start_time

            # Log successful operation
            logger.info(
                f"Privacy applied successfully: "
                f"algorithm={result['algorithm']}, "
                f"epsilon={result['epsilon']}, "
                f"data_size={data_size}, "
                f"utility={result['utility_score']:.3f}, "
                f"processing_time={processing_time:.3f}s"
            )

            return result

        except Exception as e:
            # Log error
            logger.error(
                f"Privacy application failed: "
                f"data_size={data_size}, "
                f"epsilon={epsilon}, "
                f"algorithm={algorithm}, "
                f"error={str(e)}"
            )
            raise

# Production service with monitoring
monitored_service = MonitoredPrivacyService()
```

### Health Checks and Validation

```python
def validate_privacy_service():
    """Validate that the privacy service is working correctly."""

    test_cases = [
        torch.randn(100),      # Normal case
        torch.randn(10),       # Small data
        torch.randn(10000),    # Large data
    ]

    service = MonitoredPrivacyService()

    for i, test_data in enumerate(test_cases):
        try:
            result = service.apply_privacy(test_data, epsilon=1.0)

            # Validate result structure
            assert 'private_data' in result
            assert 'algorithm' in result
            assert 'epsilon' in result
            assert 'utility_score' in result

            # Validate privacy was applied (data should be different)
            assert not torch.allclose(test_data, result['private_data'], atol=1e-6)

            print(f"[PASS] Test case {i+1} passed")

        except Exception as e:
            print(f"[FAIL] Test case {i+1} failed: {e}")
            return False

    print("[SUCCESS] All health checks passed")
    return True

# Run health check
if __name__ == "__main__":
    validate_privacy_service()
```

This advanced features guide demonstrates PRESTO's evolution into a production-ready, enterprise-grade privacy-preserving ML platform with comprehensive configuration management, data validation, and deployment capabilities.
