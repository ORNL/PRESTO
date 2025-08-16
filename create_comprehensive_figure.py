import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ornl_presto import (
    get_noise_generators,
    recommend_top3,
    visualize_data,
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

# Get recommendations
print("Getting top-3 recommendations...")
top3 = recommend_top3(data, n_evals=5, init_points=3, n_iter=10)

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# Define colors for each algorithm
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
algorithm_colors = {rec['algorithm']: colors[i] for i, rec in enumerate(top3)}

# 1. Original time series (top left)
ax1 = plt.subplot(4, 4, (1, 2))
plt.plot(hours, data.numpy(), linewidth=2, color='#2C3E50', alpha=0.8)
plt.title('Original Energy Consumption Time Series', fontsize=14, fontweight='bold')
plt.xlabel('Hours')
plt.ylabel('Energy (kWh)')
plt.grid(True, alpha=0.3)

# 2. Original data distribution (top right)
ax2 = plt.subplot(4, 4, (3, 4))
plt.hist(data.numpy(), bins=25, alpha=0.7, color='#2C3E50', edgecolor='white', linewidth=1)
plt.title('Original Data Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Energy (kWh)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 3. Performance comparison bar chart (middle left)
ax3 = plt.subplot(4, 4, (5, 6))
algorithms = [rec['algorithm'] for rec in top3]
rmse_values = [rec['mean_rmse'] for rec in top3]
colors_list = [algorithm_colors[algo] for algo in algorithms]

bars = plt.bar(algorithms, rmse_values, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
plt.title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# 4. Reliability scores (middle right)
ax4 = plt.subplot(4, 4, (7, 8))
reliability_values = [rec['reliability'] for rec in top3]
bars = plt.bar(algorithms, reliability_values, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
plt.title('Reliability Scores (Higher is Better)', fontsize=14, fontweight='bold')
plt.ylabel('Reliability')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, reliability_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

# 5-7. Private data distributions for top 3 algorithms
noise_generators = get_noise_generators()
for i, rec in enumerate(top3):
    ax = plt.subplot(4, 4, 9 + i)
    algo = rec['algorithm']
    eps = rec['epsilon']
    
    # Generate private data
    private = noise_generators[algo](data, eps)
    if not torch.is_tensor(private):
        private = torch.as_tensor(private, dtype=data.dtype)
    
    plt.hist(private.numpy(), bins=25, alpha=0.7, color=algorithm_colors[algo], 
             edgecolor='white', linewidth=1)
    plt.title(f'{algo}\n(ε={eps:.2f})', fontsize=12, fontweight='bold')
    plt.xlabel('Energy (kWh)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

# 8. Epsilon values comparison
ax8 = plt.subplot(4, 4, 12)
epsilon_values = [rec['epsilon'] for rec in top3]
bars = plt.bar(algorithms, epsilon_values, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
plt.title('Privacy Budget (ε)', fontsize=14, fontweight='bold')
plt.ylabel('Epsilon (ε)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, epsilon_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

# 9. Similarity metrics heatmap
ax9 = plt.subplot(4, 4, (13, 16))
similarity_data = []
metrics_names = ['KS Statistic', 'Jensen-Shannon Div', 'Pearson Correlation']

for rec in top3:
    algo = rec['algorithm']
    eps = rec['epsilon']
    
    # Get similarity metrics
    metrics = visualize_similarity(data.numpy(), algo, eps)
    similarity_data.append([metrics['KS'], metrics['JSD'], metrics['Pearson']])

# Create heatmap
similarity_array = np.array(similarity_data)
im = plt.imshow(similarity_array, cmap='RdYlBu_r', aspect='auto')
plt.colorbar(im, ax=ax9, shrink=0.8)

# Add labels
plt.xticks(range(len(metrics_names)), metrics_names, rotation=45, ha='right')
plt.yticks(range(len(algorithms)), algorithms)
plt.title('Similarity Metrics Heatmap', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(algorithms)):
    for j in range(len(metrics_names)):
        text = plt.text(j, i, f'{similarity_array[i, j]:.3f}',
                       ha="center", va="center", color="black", fontweight='bold')

# Add a comprehensive title and subtitle
fig.suptitle('PRESTO: Privacy-Preserving Energy Data Analysis', 
             fontsize=20, fontweight='bold', y=0.98)
plt.figtext(0.5, 0.94, 'Comprehensive Privacy Algorithm Evaluation and Comparison', 
            ha='center', fontsize=14, style='italic')

# Add algorithm ranking text box
ranking_text = "Algorithm Rankings:\n"
for rank, rec in enumerate(top3, start=1):
    ranking_text += f"{rank}. {rec['algorithm']} (ε={rec['epsilon']:.2f}, RMSE={rec['mean_rmse']:.4f})\n"

plt.figtext(0.02, 0.02, ranking_text, fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.05, 1, 0.92])

# Save the comprehensive figure
output_path = '/Users/ok0/Library/CloudStorage/OneDrive-OakRidgeNationalLaboratory/Work in progress/Projects/ORNL/ASCR_FL_Privacy/code/PRESTO/images/all_top_dp.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

print(f"\nComprehensive visualization saved to: {output_path}")
print("The figure includes:")
print("- Original time series and distribution")
print("- Performance metrics comparison")
print("- Private data distributions for top 3 algorithms")
print("- Similarity metrics heatmap")
print("- Algorithm rankings and epsilon values")
