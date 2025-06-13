![# PRESTO](images/PRESTO-logo-tagline-no-bg.png)

# PRESTO
PRESTO: Privacy REcommendation and SecuriTy Optimization is a Python package that provides recomendation for the best privacy preservation algorithm based on user preferences. Traditional privacy preservation libraries provide an implementation of a set of algorithms but the user need to experiment and detemine which of them is the best for the given dataset.

## Summary
This package includes functions for:
- Defines reliability, confidence, and similarity score.
- Modular solution so new privacy preservation algorithms or privacy preservation library can be easily integrated.
- Determines the best algorithm, privacy loss, confidence interval, reliability using Bayesian Optimization.
- Recommends the best privacy preservation algorithms for a given dataset and user requirements.
- Calculate the privacy-utility, similarity and reliability score.
- Finds the best privacy preservation and machine learning settings for a given algorithm, dataset, and user requirements.
- Visualize the top 3 algorithms and their confidence intervals.
- Visualize the original and private datasets.
- Visualize the similarity between the datasets and reliability score.
- Integration with existing privacy preservation libraries (e.g., Opacus) for finding the optimal parameters.

## Installation
You can install the package from source:

```bash
git clone https://github.com/ORNL/PRESTO.git
cd presto
pip install -e .
```

## Quick Start
Here's a simple example of how to use `presto` for time-series data.
```
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import PRESTO functions from your latest module
from ornl_presto import (
    get_noise_generators,
    recommend_top3,
    visualize_data,
    visualize_similarity
)

# 1) Generate a synthetic energy consumption time series
#    Simulate one week of hourly data (168 points)
np.random.seed(42)
hours = np.arange(0, 168)
# Base consumption: sinusoidal daily pattern + trend + noise
daily_pattern = 2.0 * np.sin(2 * np.pi * hours / 24)
trend = 0.01 * hours
noise = np.random.normal(0, 0.3, size=hours.shape)
consumption = 5.0 + daily_pattern + trend + noise

# Convert to PyTorch tensor
data = torch.tensor(consumption, dtype=torch.float32)

# 2) Visualize original time series distribution
visualize_data(data, title="Original Energy Consumption Distribution")

# 3) Recommend top-3 privacy algorithms
top3 = recommend_top3(data, n_evals=5, init_points=3, n_iter=10)

print("Top-3 recommended privacy algorithms for energy data:")
for rank, rec in enumerate(top3, start=1):
    print(f"{rank}. {rec['algorithm']} | ε={rec['epsilon']:.2f} | score={rec['score']:.4f} "
          f"| mean_rmse={rec['mean']:.4f} | ci_width={rec['ci_width']:.4f} | rel={rec['reliability']:.2f}")

# 4) For each top algorithm, visualize privatized data and similarity metrics
for rec in top3:
    algo = rec['algorithm']
    eps  = rec['epsilon']
    noise_fn = get_noise_generators()[algo]

    # 1) Generate private data and visualize its distribution
    private = noise_fn(data, eps)
    if not torch.is_tensor(private):
        private = torch.as_tensor(private, dtype=data.dtype)
    visualize_data(private, title=f"Private Data ({algo}, ε={eps:.2f})")

    # 2) Invoke visualize_similarity with (domain, key, epsilon)
    metrics = visualize_similarity(
        domain  = data.numpy(),  # pass the raw series
        key     = algo,
        epsilon = eps
    )
    print(f"{algo} similarity metrics: {metrics}")
```

## Detailed Examples
For more examples see the Tutorial folder, there are examples using real-world datasets for electric grid and medical domains.

## Experimental Results
Top-3 recommended privacy algorithms for energy data:
1. DP_Exponential | ε=4.87 | score=-0.2798 | mean_rmse=0.2812 | ci_width=0.0283 | rel=125.66
2. DP_Laplace | ε=4.55 | score=-0.2916 | mean_rmse=0.3155 | ci_width=0.0455 | rel=69.66
3. DP_Gaussian | ε=4.88 | score=-0.6738 | mean_rmse=0.6944 | ci_width=0.0524 | rel=27.48

Visualization of original, private, and similarity metrics ![Visualization metrics](images/all_top_dp.png)

## API Reference

### Core Functions
- `get_noise_generators()`: Returns a dictionary of privacy algorithms.
- `recommend_top3_mean(data, n_evals=5, init_points=3, n_iter=10)`: Calculate the Top-3 Recommendation via Bayesian Optimization. Sort by mean then by narrower CI.
- `recommend_top3_reliability(domain, n_evals=3, init_points=2, n_iter=5)`: Calculate the Top-3 Recommendation via Bayesian Optimization. Sort by reliability.
- `recommend_best_algorithms(data, epsilon, get_noise_generators, calculate_utility_privacy_score, evaluate_algorithm_confidence, performance_explanation_metrics):`: Calculate the best algorithm(s) for privacy, reliability, and similarity.
- `evaluate_algorithm_confidence(domain, key, epsilon, n_evals=10, **params)`: Calculate the confidence score.
- `calculate_utility_privacy_score(domain, key, epsilon, **params)`: Calculate Utility-Privacy Scoring.
- `performance_explanation_metrics(metrics)`: Calculates the performance explanation metrics: RMSE, Confidence Interval and REliability.
- `visualize_similarity(domain, key, epsilon, **params)`: Visualize similarity using KS Statistic, Jensen–Shannon Divergence, and Pearson Correlation.
- `visualize_top3(recommendations)`: Visualize Top 3 Privacy Mechanism Recommendations.
- `visualize_confidence(domain, key, epsilon, n_evals=10, **params)`: Visualize confidence for top algorithm.
- `visualize_confidence_top3(domain, recommendations, n_evals=10)`: Visualize the Confidence Intervals for Top-3 Mechanisms.
- `visualize_overlay_original_and_private(domain, top3)`: Visualize overlay Original vs Top-3 Privatized Distributions.
  
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research under Contract No. DE-AC05-00OR22725. This manuscript has been co-authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

## References
Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science, 9(3–4), 211-407.

