"""
PRESTO Basic Example: Energy Data Privacy Analysis

This example demonstrates the core PRESTO workflow:
1. Load and visualize data
2. Get top-3 algorithm recommendations
3. Compare algorithm performance
4. Visualize results with confidence intervals
"""

import torch
import numpy as np
from ornl_presto import (
    recommend_top3,
    visualize_data,
    visualize_top3,
    visualize_confidence_top3,
    visualize_overlay_original_and_private,
    get_noise_generators,
)


def main():
    # Generate synthetic energy consumption data
    np.random.seed(42)
    torch.manual_seed(42)

    # Simulate hourly energy consumption (kWh) over a month
    base_consumption = 2.5  # Base load in kWh
    daily_pattern = np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.8  # Daily cycle
    weekly_pattern = (
        np.sin(np.linspace(0, 2 * np.pi * 4, 24 * 30)) * 0.3
    )  # Monthly variations
    noise = np.random.normal(0, 0.2, 24 * 30)  # Random variations

    energy_data = []
    for day in range(30):
        for hour in range(24):
            idx = day * 24 + hour
            consumption = (
                base_consumption
                + daily_pattern[hour]
                + weekly_pattern[idx]
                + noise[idx]
            )
            energy_data.append(max(0.1, consumption))  # Ensure positive values

    data = torch.tensor(energy_data, dtype=torch.float32)

    print("🏠 PRESTO Energy Privacy Analysis")
    print("=" * 50)
    print(f"Dataset: {len(data)} hourly energy readings")
    print(f"Mean consumption: {data.mean():.2f} kWh")
    print(f"Std deviation: {data.std():.2f} kWh")
    print()

    # Step 1: Visualize original data
    print("📊 Step 1: Visualizing original energy data...")
    visualize_data(data, title="Original Energy Consumption Distribution")

    # Step 2: Get top-3 algorithm recommendations
    print("🔍 Step 2: Finding optimal privacy algorithms...")
    print("Running Bayesian optimization (this may take a moment)...")

    top3 = recommend_top3(data, n_evals=3, init_points=2, n_iter=5)

    print("\n🏆 Top-3 Recommended Privacy Algorithms:")
    print("-" * 60)
    for rank, rec in enumerate(top3, start=1):
        print(f"{rank}. {rec['algorithm']}")
        print(f"   ε = {rec['epsilon']:.3f}")
        print(f"   Score = {rec['score']:.4f}")
        print(f"   Mean RMSE = {rec['mean']:.4f}")
        print(f"   CI Width = {rec['ci_width']:.4f}")
        print(f"   Reliability = {rec['reliability']:.1f}")
        print()

    # Step 3: Visualize top-3 recommendations
    print("📈 Step 3: Visualizing algorithm comparison...")
    visualize_top3(top3)

    # Step 4: Show confidence intervals
    print("📏 Step 4: Analyzing confidence intervals...")
    visualize_confidence_top3(data, top3, n_evals=5)

    # Step 5: Compare original vs privatized data
    print("🔒 Step 5: Comparing original vs privatized distributions...")
    visualize_overlay_original_and_private(data, top3)

    # Step 6: Algorithm analysis
    print("🧠 Step 6: Algorithm Analysis")
    print("-" * 40)

    for i, rec in enumerate(top3):
        algo_name = rec["algorithm"]
        epsilon = rec["epsilon"]

        print(f"\n{algo_name} (ε={epsilon:.3f}):")

        # Generate private data
        noise_fn = get_noise_generators()[algo_name]
        private_data = noise_fn(data, epsilon)

        # Calculate privacy-utility metrics
        original_mean = data.mean().item()
        private_mean = torch.mean(private_data).item()
        mean_error = abs(original_mean - private_mean)

        original_std = data.std().item()
        private_std = torch.std(private_data).item()
        std_error = abs(original_std - private_std)

        print(
            f"  Mean preservation: {mean_error:.4f} error ({mean_error/original_mean*100:.1f}%)"
        )
        print(
            f"  Std preservation: {std_error:.4f} error ({std_error/original_std*100:.1f}%)"
        )
        print(f"  Utility score: {rec['score']:.4f}")
        print(f"  Confidence: ±{rec['ci_width']:.4f}")

    print("\n[SUCCESS] Analysis Complete!")
    print(
        "💡 Recommendation: Use",
        top3[0]["algorithm"],
        f"with ε={top3[0]['epsilon']:.3f} for optimal privacy-utility balance",
    )


if __name__ == "__main__":
    main()
