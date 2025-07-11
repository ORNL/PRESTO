"""
PRESTO Advanced Example: Healthcare Data Privacy with ML Integration

This example demonstrates advanced PRESTO features:
1. Healthcare data simulation and preprocessing
2. Differential privacy with machine learning
3. Custom algorithm comparison
4. Pareto frontier analysis
5. Production deployment considerations
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ornl_presto import (
    recommend_top3,
    dp_function_train_and_pred,
    dp_pareto_front,
    get_noise_generators,
    visualize_data,
    visualize_similarity,
    recommend_best_algorithms,
    calculate_utility_privacy_score,
    evaluate_algorithm_confidence,
    performance_explanation_metrics,
)


class HealthcareMLModel(nn.Module):
    """Simple neural network for healthcare prediction tasks."""

    def __init__(self, input_size=10, hidden_size=64, output_size=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.network(x)


def generate_healthcare_data(n_patients=1000):
    """Generate synthetic healthcare data with realistic patterns."""
    np.random.seed(42)

    # Patient demographics and vitals
    age = np.random.normal(45, 15, n_patients)
    age = np.clip(age, 18, 90)

    # Correlated health metrics
    bmi = np.random.normal(25 + (age - 45) * 0.1, 4, n_patients)
    bmi = np.clip(bmi, 15, 45)

    blood_pressure_sys = np.random.normal(
        120 + (age - 45) * 0.5 + (bmi - 25) * 0.8, 15, n_patients
    )
    blood_pressure_dia = blood_pressure_sys * 0.6 + np.random.normal(0, 5, n_patients)

    heart_rate = np.random.normal(72 - (age - 45) * 0.1, 12, n_patients)
    heart_rate = np.clip(heart_rate, 50, 120)

    cholesterol = np.random.normal(
        180 + (age - 45) * 0.8 + (bmi - 25) * 1.2, 30, n_patients
    )
    glucose = np.random.normal(95 + (bmi - 25) * 1.5 + (age - 45) * 0.3, 15, n_patients)

    # Lab values
    hemoglobin = np.random.normal(14 - (age - 45) * 0.02, 1.5, n_patients)
    creatinine = np.random.normal(1.0 + (age - 45) * 0.005, 0.2, n_patients)

    # Lifestyle factors
    exercise_hours = np.random.exponential(3, n_patients)
    exercise_hours = np.clip(exercise_hours, 0, 20)

    smoking_score = np.random.binomial(1, 0.3, n_patients) * np.random.uniform(
        1, 5, n_patients
    )

    # Combine into feature matrix
    features = np.column_stack(
        [
            age,
            bmi,
            blood_pressure_sys,
            blood_pressure_dia,
            heart_rate,
            cholesterol,
            glucose,
            hemoglobin,
            creatinine,
            exercise_hours,
            smoking_score,
        ]
    )

    # Create risk labels (simplified binary classification)
    risk_score = (
        (age > 55) * 0.3
        + (bmi > 30) * 0.2
        + (blood_pressure_sys > 140) * 0.2
        + (cholesterol > 200) * 0.15
        + (glucose > 100) * 0.1
        + (smoking_score > 2) * 0.15
        + (exercise_hours < 2) * 0.1
    )

    labels = (risk_score > 0.5).astype(int)

    return features, labels


def main():
    print("🏥 PRESTO Advanced Healthcare Privacy Analysis")
    print("=" * 60)

    # Generate synthetic healthcare dataset
    print("📊 Generating synthetic healthcare data...")
    features, labels = generate_healthcare_data(n_patients=500)

    # Focus on heart rate data for privacy analysis
    heart_rate_data = torch.tensor(
        features[:, 4], dtype=torch.float32
    )  # Heart rate column

    print(f"Dataset: {len(heart_rate_data)} patient heart rate measurements")
    print(f"Mean heart rate: {heart_rate_data.mean():.1f} bpm")
    print(f"Std deviation: {heart_rate_data.std():.1f} bpm")
    print(f"Range: {heart_rate_data.min():.1f} - {heart_rate_data.max():.1f} bpm")
    print()

    # Step 1: Comprehensive privacy analysis
    print("🔍 Step 1: Comprehensive Privacy Algorithm Analysis")
    print("-" * 50)

    # Test multiple epsilon values
    epsilon_values = [0.5, 1.0, 2.0, 5.0]
    algorithm_performance = {}

    for eps in epsilon_values:
        print(f"\nAnalyzing ε = {eps}...")

        best_algos = recommend_best_algorithms(
            heart_rate_data,
            eps,
            get_noise_generators,
            calculate_utility_privacy_score,
            evaluate_algorithm_confidence,
            performance_explanation_metrics,
        )

        algorithm_performance[eps] = best_algos

        print(
            f"  Best Similarity: {best_algos['max_similarity']['algorithm']} "
            f"(score: {best_algos['max_similarity']['score']:.4f})"
        )
        print(
            f"  Best Reliability: {best_algos['max_reliability']['algorithm']} "
            f"(score: {best_algos['max_reliability']['score']:.4f})"
        )
        print(
            f"  Best Privacy: {best_algos['max_privacy']['algorithm']} "
            f"(score: {best_algos['max_privacy']['score']:.4f})"
        )

    # Step 2: Detailed algorithm comparison
    print("\n🧪 Step 2: Detailed Algorithm Analysis")
    print("-" * 40)

    target_epsilon = 1.0
    noise_generators = get_noise_generators()

    print(f"\nComparing all algorithms at ε = {target_epsilon}:")
    print()

    for algo_name, algo_func in noise_generators.items():
        try:
            # Test algorithm
            private_data = algo_func(heart_rate_data, target_epsilon)

            # Calculate metrics
            score = calculate_utility_privacy_score(
                heart_rate_data, algo_name, target_epsilon
            )
            confidence = evaluate_algorithm_confidence(
                heart_rate_data, algo_name, target_epsilon, n_evals=3
            )
            performance = performance_explanation_metrics(confidence)

            print(
                f"{algo_name:20s}: Score={score:7.4f}, "
                f"RMSE={performance['mean_rmse']:6.4f}, "
                f"Reliability={performance['reliability']:7.1f}"
            )

        except Exception as e:
            print(f"{algo_name:20s}: Error - {str(e)[:40]}...")

    # Step 3: Healthcare-specific privacy requirements
    print("\n🏥 Step 3: Healthcare Privacy Requirements Analysis")
    print("-" * 50)

    # HIPAA-like privacy levels
    hipaa_epsilon = 0.1  # Very strict
    moderate_epsilon = 1.0  # Moderate
    relaxed_epsilon = 5.0  # Relaxed

    privacy_levels = {
        "HIPAA-Strict": hipaa_epsilon,
        "Moderate": moderate_epsilon,
        "Research-Relaxed": relaxed_epsilon,
    }

    print("Recommended algorithms for different privacy requirements:")
    print()

    for level_name, eps in privacy_levels.items():
        top3 = recommend_top3(heart_rate_data, n_evals=2, init_points=2, n_iter=3)
        best_algo = top3[0]

        # Calculate data utility preservation
        noise_fn = get_noise_generators()[best_algo["algorithm"]]
        private_data = noise_fn(heart_rate_data, eps)

        utility_preservation = (
            1
            - abs(heart_rate_data.mean() - torch.mean(private_data))
            / heart_rate_data.mean()
        )

        print(
            f"{level_name:15s} (ε={eps:4.1f}): {best_algo['algorithm']:15s} "
            f"- Utility: {utility_preservation:.1%}, "
            f"Reliability: {best_algo['reliability']:5.1f}"
        )

    # Step 4: Privacy-ML integration demo
    print("\n🤖 Step 4: Privacy-Preserving Machine Learning Demo")
    print("-" * 50)

    print("Demonstrating DP training with healthcare model...")

    # Prepare data for ML
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Create train/test datasets
    n_train = int(0.8 * len(X))
    train_X, test_X = X[:n_train], X[n_train:]
    train_y, test_y = y[:n_train], y[n_train:]

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)

    # Test different privacy levels
    privacy_configs = [
        {"noise_multiplier": 0.0, "name": "No Privacy"},
        {"noise_multiplier": 0.5, "name": "Light Privacy"},
        {"noise_multiplier": 1.0, "name": "Moderate Privacy"},
        {"noise_multiplier": 2.0, "name": "Strong Privacy"},
    ]

    print()
    for config in privacy_configs:
        try:
            # Train with privacy
            accuracy = dp_function_train_and_pred(
                noise_multiplier=config["noise_multiplier"],
                max_grad_norm=1.0,
                model_class=HealthcareMLModel,
                X_test=test_X,
                train_dataset=train_dataset,
                y_test=test_y,
            )

            print(f"{config['name']:15s}: Accuracy = {accuracy:.1%}")

        except Exception as e:
            print(f"{config['name']:15s}: Error - {str(e)[:30]}...")

    # Step 5: Production recommendations
    print("\n🚀 Step 5: Production Deployment Recommendations")
    print("-" * 50)

    print("\nFor healthcare data deployment:")
    print("• Recommended algorithm: DP_Gaussian or DP_Laplace")
    print("• Strict privacy: ε ≤ 0.5 (HIPAA compliance)")
    print("• Research use: ε = 1.0-2.0 (balance utility/privacy)")
    print("• Always validate with domain experts")
    print("• Monitor utility preservation > 80%")
    print("• Use confidence intervals for uncertainty quantification")

    print("\n📊 Generating final comparison visualization...")

    # Final visualization
    best_algorithm = recommend_top3(
        heart_rate_data, n_evals=2, init_points=2, n_iter=3
    )[0]
    print(
        f"\nFinal Recommendation: {best_algorithm['algorithm']} with ε={best_algorithm['epsilon']:.3f}"
    )

    # Visualize similarity metrics
    visualize_similarity(
        heart_rate_data, best_algorithm["algorithm"], best_algorithm["epsilon"]
    )

    print("\n✅ Advanced Analysis Complete!")
    print("📝 See generated plots for detailed privacy-utility analysis")


if __name__ == "__main__":
    main()
