# Getting Started with PRESTO

Welcome to PRESTO! This guide will help you get up and running with automated differential privacy in just a few minutes.

## Installation

### Prerequisites

- Python 3.9 or higher
- PyTorch 1.9 or higher
- NumPy, SciPy, Matplotlib

### Install from PyPI

```bash
pip install ornl-presto
```

### Install from Source

```bash
git clone https://github.com/ORNL/PRESTO.git
cd PRESTO
pip install -e .
```

### Verify Installation

```python
import ornl_presto
print(ornl_presto.__version__)
```

## Quick Start Guide

### Basic Usage

```python
import torch
from ornl_presto import recommend_top3, visualize_top3

# Your sensitive data
data = torch.randn(1000)

# Get automated privacy recommendations
recommendations = recommend_top3(data, n_evals=10)

# View results
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['algorithm']} (ε={rec['epsilon']:.3f}, score={rec['score']:.4f})")

# Visualize recommendations
visualize_top3(recommendations)
```

### Apply Privacy Protection

```python
from ornl_presto import get_noise_generators

# Get the best recommendation
best_algo = recommendations[0]

# Apply privacy mechanism
noise_fn = get_noise_generators()[best_algo['algorithm']]
private_data = noise_fn(data, best_algo['epsilon'])

print(f"Original mean: {data.mean():.4f}")
print(f"Private mean: {private_data.mean():.4f}")
print(f"Privacy level: ε = {best_algo['epsilon']:.3f}")
```

### Advanced Configuration

```python
from ornl_presto.config import ConfigManager
from ornl_presto.data_validation import validate_and_preprocess

# Use domain-specific configuration
config = ConfigManager.get_config("healthcare")

# Validate and preprocess data
processed_data, info = validate_and_preprocess(data)

# Get recommendations with domain constraints
recommendations = recommend_top3(
    processed_data,
    n_evals=config.optimization.n_evals,
    init_points=config.optimization.init_points,
    n_iter=config.optimization.n_iter
)

# Filter by required algorithms
filtered_recs = [
    rec for rec in recommendations
    if rec['algorithm'] in config.privacy.required_algorithms
]
```

## Core Concepts

### Differential Privacy

Differential privacy provides mathematical guarantees about individual privacy in datasets. A mechanism M satisfies ε-differential privacy if:

$$\\Pr[M(D) \\in S] \\leq e^{\\varepsilon} \\cdot \\Pr[M(D') \\in S]$$

for all neighboring datasets D and D' and all possible outputs S.

### Privacy Parameter (ε)

- **Lower ε** = Higher privacy, Lower utility
- **Higher ε** = Lower privacy, Higher utility
- **Typical values**: 0.1 (very private) to 10.0 (low privacy)

### PRESTO's Automated Selection

PRESTO automates three critical decisions:

1. **Algorithm Selection**: Which privacy mechanism to use
2. **Parameter Tuning**: Optimal ε value for your data
3. **Utility Assessment**: Expected data quality after privacy

## Available Privacy Mechanisms

| Algorithm | Best For | Characteristics |
|-----------|----------|-----------------|
| `DP_Gaussian` | Continuous data, ML applications | Smooth noise, good for optimization |
| `DP_Laplace` | General purpose, standard choice | Geometric noise, robust |
| `DP_Exponential` | Ranking, selection queries | Utility-aware selection |
| `count_mean_sketch` | High-dimensional, streaming | Efficient for large datasets |
| `randomized_response` | Binary/categorical data | Survey data, yes/no questions |
| `DP_Geometric` | Integer data, counting | Discrete noise for counts |

## Configuration Profiles

PRESTO includes pre-configured profiles for different domains:

### Healthcare
```python
config = ConfigManager.get_config("healthcare")
# ε ∈ [0.01, 1.0], utility_threshold=0.9, required_algorithms=["DP_Gaussian"]
```

### Finance
```python
config = ConfigManager.get_config("finance")
# ε ∈ [0.1, 2.0], utility_threshold=0.8, robust algorithms
```

### Research
```python
config = ConfigManager.get_config("research")
# ε ∈ [0.5, 10.0], utility_threshold=0.7, flexible privacy
```

### IoT/Sensors
```python
config = ConfigManager.get_config("iot_sensors")
# ε ∈ [1.0, 20.0], streaming-optimized, efficient algorithms
```

## Data Validation and Preprocessing

PRESTO includes comprehensive data quality checking:

```python
from ornl_presto.data_validation import DataValidator, validate_and_preprocess

# Validate data quality
validator = DataValidator()
results = validator.validate_data(data)

if not results['valid']:
    print("Data issues found:")
    for error in results['errors']:
        print(f"  - {error}")

# Automatic preprocessing
processed_data, info = validate_and_preprocess(data, auto_preprocess=True)
print(f"Preprocessing applied: {info['preprocessing']['steps_applied']}")
```

## Understanding Results

### Recommendation Structure
```python
recommendation = {
    'algorithm': 'DP_Gaussian',      # Selected mechanism
    'epsilon': 1.234,               # Optimal ε value
    'score': -0.0567,              # Utility score (higher is better)
    'reliability': 87.5,           # Confidence score (0-100)
    'utility_preservation': 0.92   # Expected utility retention
}
```

### Evaluation Metrics

- **Score**: Negative RMSE between original and private data
- **Reliability**: Statistical confidence in the recommendation
- **Utility Preservation**: Fraction of original data utility retained
- **Privacy Level**: ε parameter indicating privacy strength

## Common Use Cases

### 1. Data Release
```python
# Prepare data for public release
data = load_sensitive_dataset()
recommendations = recommend_top3(data, n_evals=20)
best = recommendations[0]

# Apply privacy and release
private_data = apply_privacy(data, best['algorithm'], best['epsilon'])
release_dataset(private_data)
```

### 2. Privacy-Preserving Analytics
```python
# Maintain statistical properties
original_mean = data.mean()
original_std = data.std()

private_data = apply_optimal_privacy(data)

print(f"Mean preservation: {abs(private_data.mean() - original_mean):.4f}")
print(f"Std preservation: {abs(private_data.std() - original_std):.4f}")
```

### 3. Machine Learning with Privacy
```python
from ornl_presto import dp_function_train_and_pred

# Train ML model with differential privacy
accuracy = dp_function_train_and_pred(
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    train_dataset=train_data,
    model_class=MyModel,
    X_test=X_test,
    y_test=y_test
)

print(f"Private model accuracy: {accuracy:.1%}")
```

## Best Practices

### 1. Start with Data Validation
Always validate your data before applying privacy:
```python
processed_data, validation_info = validate_and_preprocess(raw_data)
```

### 2. Use Domain Configurations
Leverage pre-built domain expertise:
```python
config = ConfigManager.get_config("your_domain")
```

### 3. Evaluate Multiple Recommendations
Don't just use the top recommendation:
```python
top3 = recommend_top3(data)
for rec in top3:
    evaluate_recommendation(rec, your_requirements)
```

### 4. Monitor Utility Preservation
Set minimum utility thresholds:
```python
min_utility = 0.8
suitable_recs = [r for r in recommendations if r['utility_preservation'] >= min_utility]
```

### 5. Consider Privacy Budget
Plan your privacy budget across multiple analyses:
```python
total_epsilon_budget = 1.0
analysis_count = 5
per_analysis_epsilon = total_epsilon_budget / analysis_count
```

## Troubleshooting

### Common Issues

**Low utility scores**: Try higher ε values or different algorithms
```python
recommendations = recommend_top3(data, epsilon_range=(1.0, 10.0))
```

**Algorithm failures**: Check data validation results
```python
validation = validator.validate_data(data)
print(validation['warnings'])
```

**Slow performance**: Reduce optimization iterations
```python
recommendations = recommend_top3(data, n_evals=5, n_iter=3)
```

### Getting Help

- **Documentation**: [https://presto-privacy.readthedocs.io](https://presto-privacy.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/ORNL/PRESTO/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ORNL/PRESTO/discussions)
- **Email**: presto-support@ornl.gov

## Next Steps

- Explore the [Examples Gallery](examples.md) for domain-specific use cases
- Read the [Advanced Features Guide](advanced_features.md) for production deployment
- Check out the [API Reference](api_reference.md) for detailed function documentation
- Join our [Community Discussions](https://github.com/ORNL/PRESTO/discussions) to share experiences

---

**Ready to protect your data?** Continue with our [Tutorial Series](tutorials.md) or dive into [Real-World Examples](examples.md).
