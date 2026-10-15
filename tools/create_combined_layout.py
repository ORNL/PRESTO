#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ornl_presto import get_noise_generators, visualize_similarity

# Set style for clean plots
plt.style.use('default')
sns.set_palette("husl")

# Generate the same energy consumption data as in Quick Start
np.random.seed(42)
hours = np.arange(0, 168)
daily_pattern = 2.0 * np.sin(2 * np.pi * hours / 24)
trend = 0.01 * hours
noise = np.random.normal(0, 0.3, size=hours.shape)
consumption = 5.0 + daily_pattern + trend + noise
original_data = torch.tensor(consumption, dtype=torch.float32)

# Get noise generators
noise_generators = get_noise_generators()

# Define the three algorithms with their optimal epsilon values from README results
algorithms_config = [
    {'name': 'DP_Exponential', 'key': 'exponential', 'epsilon': 5.00},
    {'name': 'DP_Gaussian', 'key': 'gaussian', 'epsilon': 3.85},
    {'name': 'DP_Laplace', 'key': 'laplace', 'epsilon': 4.72}
]

# Create the combined figure with 3 rows (one per algorithm) and 3 columns
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('PRESTO: Privacy-Preserving Data Analysis Comparison', fontsize=16, fontweight='bold', y=0.95)

for i, config in enumerate(algorithms_config):
    algo_name = config['name']
    algo_key = config['key']
    epsilon = config['epsilon']
    
    # Generate private data using the noise generator
    private_data = noise_generators[algo_key](original_data, epsilon)
    if not torch.is_tensor(private_data):
        private_data = torch.as_tensor(private_data, dtype=original_data.dtype)
    
    # Get similarity metrics
    plt.ioff()  # Turn off interactive plotting to suppress visualize_similarity plot
    metrics = visualize_similarity(original_data.numpy(), algo_key, epsilon)
    plt.ion()   # Turn interactive plotting back on
    
    # Row title with epsilon value
    fig.text(0.5, 0.88 - i*0.29, f'Similarity Analysis: {algo_name} (ε={epsilon:.4f})', 
             ha='center', fontsize=14, fontweight='bold', color='black')
    
    # Column 1: Original Data Distribution
    axes[i, 0].hist(original_data.numpy(), bins=30, alpha=0.7, color='steelblue', edgecolor='black')
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
    
    # Column 3: Similarity Metrics Bar Chart
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
plt.tight_layout(rect=[0, 0.02, 1, 0.93])

# Save the figure
plt.savefig('images/combined_similarity_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("Combined similarity analysis figure saved as 'images/combined_similarity_analysis.png'")
