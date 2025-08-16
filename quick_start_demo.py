import torch
import numpy as np
import matplotlib.pyplot as plt

# Import PRESTO functions from your latest module
from ornl_presto import (
    get_noise_generators,
    recommend_top3,
    visualize_data,
    visualize_similarity,
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

print("Energy consumption data generated successfully!")
print(f"Data shape: {data.shape}")
print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
print(f"Data mean: {data.mean():.2f}")

# 2) Visualize original time series distribution
print("\n2) Visualizing original energy consumption distribution...")
visualize_data(data, title="Original Energy Consumption Distribution")

# 3) Recommend top-3 privacy algorithms
print("\n3) Running Bayesian optimization to recommend top-3 privacy algorithms...")
print("This may take a moment...")
top3 = recommend_top3(data, n_evals=5, init_points=3, n_iter=10)

print("\nTop-3 recommended privacy algorithms for energy data:")
for rank, rec in enumerate(top3, start=1):
    print(
        f"{rank}. {rec['algorithm']} | ε={rec['epsilon']:.2f} | score={rec['score']:.4f} "
        f"| mean_rmse={rec['mean_rmse']:.4f} | ci_width={rec['ci_width']:.4f} | rel={rec['reliability']:.2f}"
    )

# 4) For each top algorithm, visualize privatized data and similarity metrics
print("\n4) Analyzing each recommended algorithm...")
for i, rec in enumerate(top3, start=1):
    algo = rec["algorithm"]
    eps = rec["epsilon"]
    noise_fn = get_noise_generators()[algo]

    print(f"\nAnalyzing algorithm {i}: {algo} (ε={eps:.2f})")

    # 1) Generate private data and visualize its distribution
    private = noise_fn(data, eps)
    if not torch.is_tensor(private):
        private = torch.as_tensor(private, dtype=data.dtype)
    visualize_data(private, title=f"Private Data ({algo}, ε={eps:.2f})")

    # 2) Invoke visualize_similarity with (domain, key, epsilon)
    metrics = visualize_similarity(
        domain=data.numpy(), key=algo, epsilon=eps  # pass the raw series
    )
    print(f"{algo} similarity metrics: {metrics}")

print("\nQuick Start demo completed successfully!")
print("Check the generated plots to see the privacy analysis results.")
