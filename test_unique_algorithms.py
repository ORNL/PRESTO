import torch
import numpy as np
import matplotlib.pyplot as plt
from ornl_presto import (
    get_noise_generators,
    recommend_top3
)

# Generate the same synthetic data
np.random.seed(42)
hours = np.arange(0, 168)
daily_pattern = 2.0 * np.sin(2 * np.pi * hours / 24)
trend = 0.01 * hours
noise = np.random.normal(0, 0.3, size=hours.shape)
consumption = 5.0 + daily_pattern + trend + noise
data = torch.tensor(consumption, dtype=torch.float32)

print("Getting top-3 unique algorithms...")
top3 = recommend_top3(data, n_evals=5, init_points=3, n_iter=10)

# Filter out duplicates (exponential vs DP_Exponential)
unique_algorithms = []
seen_algorithms = set()

for rec in top3:
    algo = rec['algorithm']
    # Map both exponential variants to the same base name
    base_algo = 'exponential' if algo in ['exponential', 'DP_Exponential'] else algo
    
    if base_algo not in seen_algorithms:
        # Use the shorter name for exponential
        if algo in ['exponential', 'DP_Exponential']:
            rec['algorithm'] = 'exponential'
        unique_algorithms.append(rec)
        seen_algorithms.add(base_algo)

# If we have less than 3 unique algorithms, get more
if len(unique_algorithms) < 3:
    print("Need more unique algorithms, running extended search...")
    extended_top = recommend_top3(data, n_evals=3, init_points=2, n_iter=15)
    
    for rec in extended_top:
        algo = rec['algorithm']
        base_algo = 'exponential' if algo in ['exponential', 'DP_Exponential'] else algo
        
        if base_algo not in seen_algorithms and len(unique_algorithms) < 3:
            if algo in ['exponential', 'DP_Exponential']:
                rec['algorithm'] = 'exponential'
            unique_algorithms.append(rec)
            seen_algorithms.add(base_algo)

print("\nTop-3 unique recommended privacy algorithms for energy data:")
for rank, rec in enumerate(unique_algorithms[:3], start=1):
    print(f"{rank}. {rec['algorithm']} | ε={rec['epsilon']:.2f} | score={rec['score']:.4f} "
          f"| mean_rmse={rec['mean_rmse']:.4f} | ci_width={rec['ci_width']:.4f} | rel={rec['reliability']:.2f}")

# Check available algorithms
print("\nAll available algorithms:")
generators = get_noise_generators()
for name in generators.keys():
    print(f"- {name}")
