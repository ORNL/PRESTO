import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ornl_presto import (
    get_noise_generators,
    visualize_similarity
)

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Regenerate the same data as in the quick start
np.random.seed(42)
hours = np.arange(0, 168)
daily_pattern = 2.0 * np.sin(2 * np.pi * hours / 24)
trend = 0.01 * hours
noise = np.random.normal(0, 0.3, size=hours.shape)
consumption = 5.0 + daily_pattern + trend + noise
data = torch.tensor(consumption, dtype=torch.float32)

# Manual top3 results as shown in README
top3_results = [
    {'algorithm': 'exponential', 'epsilon': 5.00, 'score': -0.2689, 'mean_rmse': 0.2705, 'ci_width': 0.0201, 'reliability': 96.48},
    {'algorithm': 'laplace', 'epsilon': 4.72, 'score': -0.2855, 'mean_rmse': 0.2899, 'ci_width': 0.0232, 'reliability': 96.20},
    {'algorithm': 'gaussian', 'epsilon': 3.85, 'score': -0.3156, 'mean_rmse': 0.3201, 'ci_width': 0.0298, 'reliability': 89.34}
]

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# Define colors for each algorithm
colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green
algorithm_colors = {rec['algorithm']: colors[i] for i, rec in enumerate(top3_results)}

# 1. Original time series (top left)
ax1 = plt.subplot(4, 4, (1, 2))
plt.plot(hours, data.numpy(), linewidth=2.5, color='#2C3E50', alpha=0.9)
plt.title('Original Energy Consumption Time Series', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Hours', fontsize=12)
plt.ylabel('Energy (kWh)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 168)

# 2. Original data distribution (top right)
ax2 = plt.subplot(4, 4, (3, 4))
plt.hist(data.numpy(), bins=25, alpha=0.8, color='#2C3E50', edgecolor='white', linewidth=1.5)
plt.title('Original Data Distribution', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Energy (kWh)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)

# 3. Performance comparison - RMSE (middle left)
ax3 = plt.subplot(4, 4, (5, 6))
algorithms = [rec['algorithm'] for rec in top3_results]
rmse_values = [rec['mean_rmse'] for rec in top3_results]
colors_list = [algorithm_colors[algo] for algo in algorithms]

bars = plt.bar(algorithms, rmse_values, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
plt.title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('RMSE', fontsize=12)
plt.xticks(rotation=45, fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 4. Reliability scores (middle right)
ax4 = plt.subplot(4, 4, (7, 8))
reliability_values = [rec['reliability'] for rec in top3_results]
bars = plt.bar(algorithms, reliability_values, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
plt.title('Reliability Scores (Higher is Better)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Reliability', fontsize=12)
plt.xticks(rotation=45, fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, reliability_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 5-7. Private data distributions for top 3 algorithms
noise_generators = get_noise_generators()
for i, rec in enumerate(top3_results):
    ax = plt.subplot(4, 4, 9 + i)
    algo = rec['algorithm']
    eps = rec['epsilon']
    
    # Generate private data
    private = noise_generators[algo](data, eps)
    if not torch.is_tensor(private):
        private = torch.as_tensor(private, dtype=data.dtype)
    
    plt.hist(private.numpy(), bins=25, alpha=0.8, color=algorithm_colors[algo], 
             edgecolor='white', linewidth=1.5)
    plt.title(f'{algo.capitalize()}\n(ε={eps:.2f})', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Energy (kWh)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)

# 8. Epsilon values comparison
ax8 = plt.subplot(4, 4, 12)
epsilon_values = [rec['epsilon'] for rec in top3_results]
bars = plt.bar(algorithms, epsilon_values, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
plt.title('Privacy Budget (ε)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Epsilon (ε)', fontsize=12)
plt.xticks(rotation=45, fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, epsilon_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 9. Similarity metrics heatmap
ax9 = plt.subplot(4, 4, (13, 16))
similarity_data = []
metrics_names = ['KS Statistic', 'Jensen-Shannon Div', 'Pearson Correlation']

print("Calculating similarity metrics...")
for rec in top3_results:
    algo = rec['algorithm']
    eps = rec['epsilon']
    
    # Get similarity metrics (suppressing the plot output)
    plt.ioff()  # Turn off interactive plotting temporarily
    metrics = visualize_similarity(data.numpy(), algo, eps)
    plt.close('all')  # Close any plots created by visualize_similarity
    plt.ion()  # Turn interactive plotting back on
    
    similarity_data.append([metrics['KS'], metrics['JSD'], metrics['Pearson']])

# Create heatmap
similarity_array = np.array(similarity_data)
im = plt.imshow(similarity_array, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
cbar = plt.colorbar(im, ax=ax9, shrink=0.8)
cbar.set_label('Metric Value', fontsize=10)

# Add labels
plt.xticks(range(len(metrics_names)), metrics_names, rotation=45, ha='right', fontsize=10)
plt.yticks(range(len(algorithms)), [algo.capitalize() for algo in algorithms], fontsize=11)
plt.title('Similarity Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)

# Add text annotations
for i in range(len(algorithms)):
    for j in range(len(metrics_names)):
        color = 'white' if similarity_array[i, j] > 0.5 else 'black'
        text = plt.text(j, i, f'{similarity_array[i, j]:.3f}',
                       ha="center", va="center", color=color, fontweight='bold', fontsize=9)

# Add a comprehensive title and subtitle
fig.suptitle('PRESTO: Privacy-Preserving Energy Data Analysis', 
             fontsize=22, fontweight='bold', y=0.98)
plt.figtext(0.5, 0.94, 'Top-3 Unique Privacy Algorithms Evaluation and Comparison', 
            ha='center', fontsize=15, style='italic')

# Add algorithm ranking text box with updated results
ranking_text = "Algorithm Rankings:\n"
for rank, rec in enumerate(top3_results, start=1):
    ranking_text += f"{rank}. {rec['algorithm'].capitalize()} (ε={rec['epsilon']:.2f}, RMSE={rec['mean_rmse']:.4f})\n"

plt.figtext(0.02, 0.02, ranking_text, fontsize=11, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

# Add performance summary
summary_text = "Performance Summary:\n"
summary_text += f"Best Accuracy: {top3_results[0]['algorithm'].capitalize()} ({top3_results[0]['mean_rmse']:.4f} RMSE)\n"
summary_text += f"Highest Reliability: {max(top3_results, key=lambda x: x['reliability'])['algorithm'].capitalize()} ({max(rec['reliability'] for rec in top3_results):.1f})\n"
summary_text += f"Privacy Range: ε = {min(rec['epsilon'] for rec in top3_results):.2f} - {max(rec['epsilon'] for rec in top3_results):.2f}"

plt.figtext(0.98, 0.02, summary_text, fontsize=11, ha='right',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.08, 1, 0.92])

# Save the comprehensive figure
output_path = '/Users/ok0/Library/CloudStorage/OneDrive-OakRidgeNationalLaboratory/Work in progress/Projects/ORNL/ASCR_FL_Privacy/code/PRESTO/images/all_top_dp.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

print(f"\nNew comprehensive visualization saved to: {output_path}")
print("The updated figure includes:")
print("- Original time series and distribution")
print("- Performance metrics for exponential, laplace, and gaussian algorithms")
print("- Private data distributions for all 3 unique algorithms")
print("- Updated similarity metrics heatmap")
print("- Algorithm rankings and performance summary")
