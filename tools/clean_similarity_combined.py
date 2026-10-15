#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from ornl_presto import get_noise_generators
from ornl_presto.metrics import similarity_metrics

# Generate the same energy consumption data as in Quick Start
np.random.seed(42)
hours = np.arange(0, 168)
daily_pattern = 2.0 * np.sin(2 * np.pi * hours / 24)
trend = 0.01 * hours
noise = np.random.normal(0, 0.3, size=hours.shape)
consumption = 5.0 + daily_pattern + trend + noise
data = torch.tensor(consumption, dtype=torch.float32)

# Use the exact epsilon values from the README results
algorithms_config = [
    {'name': 'DP_Exponential', 'key': 'exponential', 'epsilon': 5.00},
    {'name': 'DP_Gaussian', 'key': 'gaussian', 'epsilon': 3.85},
    {'name': 'DP_Laplace', 'key': 'laplace', 'epsilon': 4.72}
]

# Get noise generators
noise_generators = get_noise_generators()

# Create figure with 3 rows and 3 columns - exactly like the attached image
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for i, config in enumerate(algorithms_config):
    algo_name = config['name']
    algo_key = config['key']
    epsilon = config['epsilon']
    
    # Generate private data
    private_data = noise_generators[algo_key](data, epsilon)
    if not torch.is_tensor(private_data):
        private_data = torch.as_tensor(private_data, dtype=data.dtype)
    
    # Calculate similarity metrics
    metrics = similarity_metrics(data.numpy(), private_data.numpy())
    
    # Add row title above each row
    fig.text(0.5, 0.88 - i*0.29, f'Similarity Analysis: {algo_name} (ε={epsilon:.4f})', 
             ha='center', fontsize=14, fontweight='bold', color='black')
    
    # Column 1: Original Data Distribution
    axes[i, 0].hist(data.numpy(), bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[i, 0].set_title('Original Data Distribution', fontsize=12, color='black')
    axes[i, 0].set_xlabel('Value', fontsize=10, color='black')
    axes[i, 0].set_ylabel('Density', fontsize=10, color='black')
    axes[i, 0].tick_params(colors='black')
    
    # Column 2: Private Data Distribution
    private_title = f'Private Data ({algo_name}, ε={epsilon:.2f})'
    axes[i, 1].hist(private_data.numpy(), bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[i, 1].set_title(private_title, fontsize=12, color='black')
    axes[i, 1].set_xlabel('Value', fontsize=10, color='black')
    axes[i, 1].set_ylabel('Density', fontsize=10, color='black')
    axes[i, 1].tick_params(colors='black')
    
    # Column 3: Similarity Metrics as bar chart (matching the attached image style)
    metric_names = ['KS', 'JSD', 'Pearson']
    metric_values = [metrics['KS'], metrics['JSD'], metrics['Pearson']]
    
    bars = axes[i, 2].bar(metric_names, metric_values, color=['lightblue', 'lightgreen', 'steelblue'])
    axes[i, 2].set_title('Similarity Metrics', fontsize=12, color='black')
    axes[i, 2].set_ylabel('Score', fontsize=10, color='black')
    axes[i, 2].set_ylim(0, 1.0)
    axes[i, 2].tick_params(colors='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[i, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9, color='black')

# Adjust layout
plt.tight_layout(rect=[0, 0.02, 1, 0.92])

# Save the figure
plt.savefig('images/similarity_analysis_combined.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("✅ Similarity analysis figure saved as 'images/similarity_analysis_combined.png'")
print("This matches the format of your attached image with:")
print("- 3 rows (one per algorithm)")
print("- 3 columns (Original Data, Private Data, Similarity Metrics)")
print("- Exact epsilon values from README results")

plt.show()
