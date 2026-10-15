"""
PRESTO Medical Example: Clinical Trial Data Privacy

This example demonstrates PRESTO's application to clinical trial data:
1. Patient enrollment data protection
2. Adverse event reporting with privacy
3. Treatment efficacy analysis
4. Biomarker discovery with differential privacy
5. Regulatory compliance (FDA, EMA) considerations
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ornl_presto import (
    recommend_top3,
    visualize_top3,
    get_noise_generators,
    calculate_utility_privacy_score,
)
from ornl_presto.config import ConfigManager
from ornl_presto.data_validation import validate_and_preprocess


def generate_clinical_trial_data(n_patients=1000):
    """Generate synthetic clinical trial data with realistic patterns."""
    np.random.seed(42)

    # Patient demographics
    age = np.random.normal(55, 15, n_patients)
    age = np.clip(age, 18, 85)

    # Binary sex (0=Female, 1=Male)
    sex = np.random.binomial(1, 0.52, n_patients)

    # Treatment groups (0=Placebo, 1=Treatment A, 2=Treatment B)
    treatment_group = np.random.choice([0, 1, 2], n_patients, p=[0.33, 0.34, 0.33])

    # Baseline biomarkers
    baseline_ldl = np.random.normal(140, 25, n_patients)  # LDL cholesterol
    baseline_hdl = np.random.normal(50, 15, n_patients)  # HDL cholesterol
    baseline_bp = np.random.normal(135, 20, n_patients)  # Blood pressure

    # Primary endpoint: LDL reduction after treatment
    # Treatment effect varies by group
    treatment_effects = {0: 0, 1: -25, 2: -35}  # Placebo, Treatment A, Treatment B
    noise_std = 15

    ldl_change = np.array(
        [
            treatment_effects[group] + np.random.normal(0, noise_std)
            for group in treatment_group
        ]
    )

    # Add age and sex effects
    ldl_change += (age - 55) * 0.2  # Older patients respond better
    ldl_change += sex * (-5)  # Males respond slightly better

    # Secondary endpoints
    bp_change = ldl_change * 0.3 + np.random.normal(0, 8, n_patients)

    # Adverse events (binary outcomes)
    # Higher risk with treatment, age-dependent
    ae_base_risk = 0.1 + (age - 18) / 670  # Age-dependent baseline risk
    ae_treatment_multiplier = [1.0, 1.3, 1.5]  # Risk multiplier by treatment

    adverse_events = np.array(
        [
            np.random.binomial(1, ae_base_risk * ae_treatment_multiplier[group])
            for group in treatment_group
        ]
    )

    # Time to event (e.g., cardiovascular events)
    # Exponential survival with treatment effects
    hazard_base = 0.02
    hazard_multipliers = [1.0, 0.7, 0.5]  # Hazard reduction by treatment

    time_to_event = np.array(
        [
            np.random.exponential(1 / (hazard_base * hazard_multipliers[group]))
            for group in treatment_group
        ]
    )

    # Censoring at 24 months
    censored = time_to_event > 24
    time_to_event = np.minimum(time_to_event, 24)

    # Laboratory values with measurement error
    week_12_ldl = baseline_ldl + ldl_change + np.random.normal(0, 5, n_patients)
    week_12_hdl = baseline_hdl + np.random.normal(
        2, 8, n_patients
    )  # Slight HDL increase

    clinical_data = {
        "patient_id": np.arange(1, n_patients + 1),
        "age": age,
        "sex": sex,
        "treatment_group": treatment_group,
        "baseline_ldl": baseline_ldl,
        "baseline_hdl": baseline_hdl,
        "baseline_bp": baseline_bp,
        "ldl_change": ldl_change,
        "bp_change": bp_change,
        "week_12_ldl": week_12_ldl,
        "week_12_hdl": week_12_hdl,
        "adverse_events": adverse_events,
        "time_to_event": time_to_event,
        "censored": censored.astype(int),
    }

    return pd.DataFrame(clinical_data)


def analyze_treatment_efficacy(data, private_data=None):
    """Analyze treatment efficacy with and without privacy."""

    results = {}

    # Use original or private data
    analysis_data = private_data if private_data is not None else data

    # Primary efficacy analysis: LDL reduction by treatment group
    for group in [0, 1, 2]:
        group_data = analysis_data[analysis_data["treatment_group"] == group]

        mean_reduction = group_data["ldl_change"].mean()
        std_reduction = group_data["ldl_change"].std()
        n_patients = len(group_data)

        # 95% confidence interval
        se = std_reduction / np.sqrt(n_patients)
        ci_lower = mean_reduction - 1.96 * se
        ci_upper = mean_reduction + 1.96 * se

        group_name = ["Placebo", "Treatment A", "Treatment B"][group]

        results[group_name] = {
            "mean_ldl_change": mean_reduction,
            "std_ldl_change": std_reduction,
            "n_patients": n_patients,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    # Calculate treatment effects vs placebo
    placebo_mean = results["Placebo"]["mean_ldl_change"]

    for treatment in ["Treatment A", "Treatment B"]:
        treatment_effect = results[treatment]["mean_ldl_change"] - placebo_mean
        results[treatment]["treatment_effect_vs_placebo"] = treatment_effect

    return results


def calculate_clinical_utility_metrics(original_results, private_results):
    """Calculate clinical trial specific utility metrics."""

    metrics = {}

    # Treatment effect preservation
    for treatment in ["Treatment A", "Treatment B"]:
        orig_effect = original_results[treatment]["treatment_effect_vs_placebo"]
        priv_effect = private_results[treatment]["treatment_effect_vs_placebo"]

        effect_preservation = 1 - abs(orig_effect - priv_effect) / abs(orig_effect)
        metrics[
            f'{treatment.lower().replace(" ", "_")}_effect_preservation'
        ] = effect_preservation

    # Confidence interval overlap
    for group in ["Placebo", "Treatment A", "Treatment B"]:
        orig_ci = (
            original_results[group]["ci_lower"],
            original_results[group]["ci_upper"],
        )
        priv_ci = (
            private_results[group]["ci_lower"],
            private_results[group]["ci_upper"],
        )

        # Calculate CI overlap
        overlap_lower = max(orig_ci[0], priv_ci[0])
        overlap_upper = min(orig_ci[1], priv_ci[1])
        overlap_width = max(0, overlap_upper - overlap_lower)

        orig_width = orig_ci[1] - orig_ci[0]
        priv_width = priv_ci[1] - priv_ci[0]
        avg_width = (orig_width + priv_width) / 2

        ci_overlap = overlap_width / avg_width if avg_width > 0 else 0
        metrics[f'{group.lower().replace(" ", "_")}_ci_overlap'] = ci_overlap

    # Overall clinical utility
    all_preservations = [
        v for k, v in metrics.items() if "preservation" in k or "overlap" in k
    ]
    metrics["overall_clinical_utility"] = np.mean(all_preservations)

    return metrics


def analyze_regulatory_compliance():
    """Analyze different regulatory privacy requirements."""

    regulatory_frameworks = {
        "FDA_Phase_III": {
            "epsilon": 0.5,
            "description": "FDA Phase III trial data sharing",
            "requirements": [
                "Individual patient data protection",
                "Efficacy signal preservation",
                "Safety profile maintenance",
                "Statistical power conservation",
            ],
        },
        "EMA_Submission": {
            "epsilon": 0.3,
            "description": "European Medicines Agency submission",
            "requirements": [
                "GDPR compliance",
                "Clinical study report accuracy",
                "Benefit-risk assessment integrity",
                "Post-marketing surveillance compatibility",
            ],
        },
        "Real_World_Evidence": {
            "epsilon": 1.0,
            "description": "Real-world evidence generation",
            "requirements": [
                "Electronic health record privacy",
                "Comparative effectiveness research",
                "Health technology assessment",
                "Pharmacovigilance compliance",
            ],
        },
    }

    return regulatory_frameworks


def main():
    print("PRESTO Clinical Trial Privacy Analysis")
    print("=" * 50)

    # Generate synthetic clinical trial dataset
    print("Generating synthetic clinical trial data...")
    clinical_data = generate_clinical_trial_data(n_patients=600)

    print(f"Clinical trial dataset: {len(clinical_data)} patients")
    print(
        f"Treatment groups: {clinical_data['treatment_group'].value_counts().to_dict()}"
    )
    print(
        f"Age range: {clinical_data['age'].min():.1f} - {clinical_data['age'].max():.1f}"
    )
    print(f"Adverse event rate: {clinical_data['adverse_events'].mean():.1%}")
    print()

    # Step 1: Baseline efficacy analysis
    print("Step 1: Baseline Treatment Efficacy Analysis")
    print("-" * 40)

    original_results = analyze_treatment_efficacy(clinical_data)

    for group, results in original_results.items():
        print(f"{group}:")
        print(
            f"  LDL change: {results['mean_ldl_change']:.1f} ± {results['std_ldl_change']:.1f} mg/dL"
        )
        print(f"  95% CI: [{results['ci_lower']:.1f}, {results['ci_upper']:.1f}]")
        print(f"  N = {results['n_patients']}")

        if "treatment_effect_vs_placebo" in results:
            print(
                f"  Treatment effect vs placebo: {results['treatment_effect_vs_placebo']:.1f} mg/dL"
            )
        print()

    # Step 2: Privacy analysis for primary endpoint
    print("Step 2: Privacy Analysis for Primary Endpoint")
    print("-" * 40)

    # Focus on LDL change as primary endpoint
    ldl_change_tensor = torch.tensor(
        clinical_data["ldl_change"].values, dtype=torch.float32
    )

    # Validate and preprocess
    processed_ldl, validation_info = validate_and_preprocess(ldl_change_tensor)

    print("Data validation results:")
    print(
        f"  Data quality: {'PASS' if validation_info['validation']['valid'] else 'FAIL'}"
    )
    print(f"  Outliers removed: {validation_info['preprocessing']['outliers_removed']}")
    print(f"  Preprocessing steps: {validation_info['preprocessing']['steps_applied']}")
    print()

    # Get privacy recommendations
    print("Privacy algorithm recommendations:")
    top3 = recommend_top3(processed_ldl, n_evals=3, init_points=2, n_iter=3)

    for i, algo in enumerate(top3, 1):
        print(
            f"  {i}. {algo['algorithm']} (ε = {algo['epsilon']:.3f}, score = {algo['score']:.4f})"
        )
    print()

    # Step 3: Regulatory compliance analysis
    print("Step 3: Regulatory Compliance Analysis")
    print("-" * 40)

    regulatory_frameworks = analyze_regulatory_compliance()

    compliance_results = {}

    for framework_name, config in regulatory_frameworks.items():
        print(f"\n{framework_name} (ε ≤ {config['epsilon']}):")
        print(f"  {config['description']}")

        # Find suitable algorithms
        suitable_algos = [algo for algo in top3 if algo["epsilon"] <= config["epsilon"]]

        if suitable_algos:
            best_algo = suitable_algos[0]
            compliance_results[framework_name] = best_algo

            print(f"  ✅ Recommended: {best_algo['algorithm']}")
            print(f"  Privacy level: ε = {best_algo['epsilon']:.3f}")
            print(f"  Expected utility: {best_algo['score']:.4f}")
        else:
            print(
                f"  WARNING: No suitable algorithms found for ε ≤ {config['epsilon']}"
            )

        print(f"  Requirements:")
        for req in config["requirements"]:
            print(f"    • {req}")

    # Step 4: Clinical utility preservation analysis
    print("\nStep 4: Clinical Utility Preservation Analysis")
    print("-" * 40)

    if compliance_results:
        # Use FDA framework for detailed analysis
        test_config = compliance_results.get(
            "FDA_Phase_III", list(compliance_results.values())[0]
        )

        # Apply privacy to LDL change data
        noise_fn = get_noise_generators()[test_config["algorithm"]]
        private_ldl = noise_fn(ldl_change_tensor, test_config["epsilon"])

        # Create private dataset
        private_clinical_data = clinical_data.copy()
        private_clinical_data["ldl_change"] = private_ldl.numpy()

        # Recalculate week 12 LDL with private change
        private_clinical_data["week_12_ldl"] = (
            private_clinical_data["baseline_ldl"] + private_clinical_data["ldl_change"]
        )

        # Analyze private data efficacy
        private_results = analyze_treatment_efficacy(private_clinical_data)

        # Calculate clinical utility metrics
        utility_metrics = calculate_clinical_utility_metrics(
            original_results, private_results
        )

        print("Clinical utility preservation:")
        print(
            f"  Treatment A effect preservation: {utility_metrics['treatment_a_effect_preservation']:.3f}"
        )
        print(
            f"  Treatment B effect preservation: {utility_metrics['treatment_b_effect_preservation']:.3f}"
        )
        print(f"  Placebo CI overlap: {utility_metrics['placebo_ci_overlap']:.3f}")
        print(
            f"  Treatment A CI overlap: {utility_metrics['treatment_a_ci_overlap']:.3f}"
        )
        print(
            f"  Treatment B CI overlap: {utility_metrics['treatment_b_ci_overlap']:.3f}"
        )
        print(
            f"  Overall clinical utility: {utility_metrics['overall_clinical_utility']:.3f}"
        )
        print()

        # Compare treatment effects
        print("Treatment effect comparison (Original vs Private):")
        for treatment in ["Treatment A", "Treatment B"]:
            orig_effect = original_results[treatment]["treatment_effect_vs_placebo"]
            priv_effect = private_results[treatment]["treatment_effect_vs_placebo"]

            print(f"  {treatment}:")
            print(f"    Original: {orig_effect:.1f} mg/dL reduction vs placebo")
            print(f"    Private:  {priv_effect:.1f} mg/dL reduction vs placebo")
            print(f"    Difference: {abs(orig_effect - priv_effect):.1f} mg/dL")

    # Step 5: Safety analysis considerations
    print("\n[SECURITY] Step 5: Safety Analysis Considerations")
    print("-" * 40)

    ae_rate_by_group = clinical_data.groupby("treatment_group")["adverse_events"].mean()

    print("Adverse event rates by treatment group:")
    for group, rate in ae_rate_by_group.items():
        group_name = ["Placebo", "Treatment A", "Treatment B"][group]
        print(f"  {group_name}: {rate:.1%}")

    print("\nSafety considerations for privacy:")
    print("  • Adverse event rates must be preserved for regulatory safety")
    print("  • Serious adverse events require individual case safety reports")
    print("  • Privacy may be relaxed for safety-critical outcomes")
    print("  • Consider separate privacy budgets for efficacy vs safety")

    # Step 6: Recommendations
    print("\nStep 6: Clinical Trial Privacy Recommendations")
    print("-" * 40)

    print("For clinical trial data sharing:")
    print("• Regulatory submissions: Use ε ≤ 0.5 with robust utility preservation")
    print("• Phase III efficacy: Maintain treatment effect within 10% of original")
    print("• Safety analysis: Consider separate privacy budgets or no privacy")
    print("• Multi-site trials: Aggregate privacy budgets across sites")
    print("• Real-world evidence: Use ε = 0.5-1.0 for broader utility")
    print("• Always validate statistical power preservation")
    print("• Implement interim analysis privacy budget allocation")

    print("\n[SUCCESS] Clinical trial privacy analysis complete!")
    print("Generated comprehensive clinical privacy recommendations")


if __name__ == "__main__":
    main()
