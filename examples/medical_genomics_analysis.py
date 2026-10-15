"""
PRESTO Medical Example: Genomics Privacy Analysis

This example demonstrates PRESTO's application to genomic data privacy:
1. SNP (Single Nucleotide Polymorphism) data protection
2. GWAS (Genome-Wide Association Study) privacy considerations
3. Population genetics with differential privacy
4. Biobank data sharing protocols
5. Compliance with genomic privacy regulations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ornl_presto import (
    recommend_top3,
    visualize_top3,
    dp_pareto_front,
    get_noise_generators,
    calculate_utility_privacy_score,
    evaluate_algorithm_confidence,
)
from ornl_presto.config import ConfigManager
from ornl_presto.data_validation import validate_and_preprocess


def generate_snp_data(n_individuals=1000, n_snps=50000):
    """Generate synthetic SNP data mimicking real genomic patterns."""
    np.random.seed(42)

    # SNP data: 0, 1, 2 (homozygous ref, heterozygous, homozygous alt)
    # Minor Allele Frequencies (MAF) following realistic distribution
    maf = np.random.beta(0.5, 2, n_snps)  # Skewed toward rare variants
    maf = np.clip(maf, 0.01, 0.5)  # Realistic MAF range

    # Generate genotypes based on Hardy-Weinberg equilibrium
    snp_data = np.zeros((n_individuals, n_snps))

    for i, freq in enumerate(maf):
        # Allele frequencies: p (major), q (minor)
        q = freq
        p = 1 - q

        # Hardy-Weinberg proportions
        prob_00 = p**2  # Homozygous major
        prob_01 = 2 * p * q  # Heterozygous
        prob_11 = q**2  # Homozygous minor

        # Generate genotypes
        genotypes = np.random.choice(
            [0, 1, 2], size=n_individuals, p=[prob_00, prob_01, prob_11]
        )
        snp_data[:, i] = genotypes

    # Create population structure (simulate ancestry)
    population_labels = np.random.choice(["EUR", "AFR", "EAS", "AMR"], n_individuals)

    # Add population-specific allele frequency shifts
    for pop_idx, pop in enumerate(["EUR", "AFR", "EAS", "AMR"]):
        pop_mask = population_labels == pop
        if np.any(pop_mask):
            # Slight frequency shifts to simulate population structure
            freq_shift = np.random.normal(0, 0.05, n_snps)
            freq_shift = np.clip(freq_shift, -0.1, 0.1)

            for snp_idx in range(
                min(1000, n_snps)
            ):  # Subset for computational efficiency
                if np.random.random() < 0.1:  # 10% of SNPs show population differences
                    shift = freq_shift[snp_idx]
                    snp_data[pop_mask, snp_idx] = np.clip(
                        snp_data[pop_mask, snp_idx] + shift, 0, 2
                    )

    return snp_data, maf, population_labels


def calculate_genomic_metrics(original_snps, private_snps, maf):
    """Calculate genomics-specific utility metrics."""

    # 1. Allele frequency preservation
    orig_af = np.mean(original_snps, axis=0) / 2  # Convert to allele frequency
    priv_af = np.mean(private_snps, axis=0) / 2
    af_rmse = np.sqrt(np.mean((orig_af - priv_af) ** 2))

    # 2. Linkage disequilibrium preservation (sample correlation)
    orig_ld = np.corrcoef(original_snps[:, :100].T)  # Sample of SNPs for efficiency
    priv_ld = np.corrcoef(private_snps[:, :100].T)
    ld_preservation = np.corrcoef(orig_ld.flatten(), priv_ld.flatten())[0, 1]

    # 3. Hardy-Weinberg equilibrium preservation
    def calc_hw_deviation(snps, af):
        """Calculate deviation from Hardy-Weinberg equilibrium."""
        observed_het = np.mean(snps == 1, axis=0)  # Heterozygote frequency
        expected_het = 2 * af * (1 - af)  # Expected under HWE
        return np.mean(np.abs(observed_het - expected_het))

    orig_hw_dev = calc_hw_deviation(original_snps, orig_af)
    priv_hw_dev = calc_hw_deviation(private_snps, priv_af)
    hw_preservation = 1 - abs(orig_hw_dev - priv_hw_dev) / orig_hw_dev

    return {
        "allele_frequency_rmse": af_rmse,
        "linkage_disequilibrium_preservation": ld_preservation,
        "hardy_weinberg_preservation": hw_preservation,
        "overall_genomic_utility": np.mean(
            [1 - af_rmse, ld_preservation, hw_preservation]
        ),
    }


def analyze_genomic_privacy_requirements():
    """Analyze different genomic privacy requirement levels."""

    privacy_levels = {
        "Biobank_Sharing": {
            "epsilon": 0.1,
            "description": "Strict privacy for public biobank sharing",
            "regulations": ["GDPR", "HIPAA", "NIH Genomic Data Sharing Policy"],
        },
        "Research_Collaboration": {
            "epsilon": 1.0,
            "description": "Moderate privacy for research collaborations",
            "regulations": ["IRB Approved", "Data Use Agreements"],
        },
        "Population_Studies": {
            "epsilon": 5.0,
            "description": "Relaxed privacy for population-level studies",
            "regulations": ["Aggregate Analysis Only"],
        },
    }

    return privacy_levels


def main():
    print("PRESTO Genomics Privacy Analysis")
    print("=" * 50)

    # Generate synthetic genomic dataset
    print("Generating synthetic SNP data...")
    snp_data, maf, populations = generate_snp_data(n_individuals=500, n_snps=1000)

    print(f"Dataset: {snp_data.shape[0]} individuals, {snp_data.shape[1]} SNPs")
    print(f"Populations: {np.unique(populations, return_counts=True)}")
    print(f"Mean MAF: {np.mean(maf):.4f}")
    print(f"MAF range: {np.min(maf):.4f} - {np.max(maf):.4f}")
    print()

    # Focus on allele frequencies for privacy analysis
    allele_frequencies = np.mean(snp_data, axis=0) / 2  # Convert to allele frequencies
    af_tensor = torch.tensor(allele_frequencies, dtype=torch.float32)

    # Step 1: Data validation and preprocessing
    print("Step 1: Genomic Data Validation")
    print("-" * 40)

    processed_af, validation_info = validate_and_preprocess(af_tensor)

    print("Validation Results:")
    print(f"• Original range: {af_tensor.min():.4f} - {af_tensor.max():.4f}")
    print(
        f"• Data quality: {'PASS' if validation_info['validation']['valid'] else 'FAIL'}"
    )
    print(f"• Outliers removed: {validation_info['preprocessing']['outliers_removed']}")
    print()

    # Step 2: Privacy requirement analysis
    print("Step 2: Genomic Privacy Requirements")
    print("-" * 40)

    privacy_levels = analyze_genomic_privacy_requirements()

    genomic_results = {}

    for level_name, config in privacy_levels.items():
        print(f"\n{level_name} (ε = {config['epsilon']}):")
        print(f"  Description: {config['description']}")
        print(f"  Regulations: {', '.join(config['regulations'])}")

        # Get privacy recommendations
        top3 = recommend_top3(processed_af, n_evals=3, init_points=2, n_iter=3)

        # Filter for epsilon constraint
        suitable_algos = [
            algo
            for algo in top3
            if algo["epsilon"] <= config["epsilon"] * 1.5  # Allow some flexibility
        ]

        if suitable_algos:
            best_algo = suitable_algos[0]
            genomic_results[level_name] = best_algo

            print(
                f"  Recommended: {best_algo['algorithm']} (ε = {best_algo['epsilon']:.3f})"
            )
            print(f"  Expected utility: {best_algo.get('score', 0):.4f}")
        else:
            print(
                f"  WARNING: No suitable algorithms found for ε ≤ {config['epsilon']}"
            )

    # Step 3: Genomic-specific utility analysis
    print("\nStep 3: Genomic Utility Preservation Analysis")
    print("-" * 40)

    if genomic_results:
        # Use the research collaboration setting for detailed analysis
        best_config = genomic_results.get(
            "Research_Collaboration", list(genomic_results.values())[0]
        )

        # Apply privacy mechanism to full SNP data
        noise_fn = get_noise_generators()[best_config["algorithm"]]

        # Apply to subset for computational efficiency
        subset_snps = snp_data[:, :100]  # First 100 SNPs
        subset_tensor = torch.tensor(subset_snps, dtype=torch.float32)

        private_snps_tensor = noise_fn(subset_tensor, best_config["epsilon"])
        private_snps = private_snps_tensor.numpy()

        # Ensure SNP values remain in valid range [0, 2]
        private_snps = np.clip(np.round(private_snps), 0, 2)

        # Calculate genomic metrics
        genomic_metrics = calculate_genomic_metrics(
            subset_snps, private_snps, maf[:100]
        )

        print("Genomic Utility Metrics:")
        print(
            f"• Allele frequency RMSE: {genomic_metrics['allele_frequency_rmse']:.6f}"
        )
        print(
            f"• Linkage disequilibrium preservation: {genomic_metrics['linkage_disequilibrium_preservation']:.4f}"
        )
        print(
            f"• Hardy-Weinberg preservation: {genomic_metrics['hardy_weinberg_preservation']:.4f}"
        )
        print(
            f"• Overall genomic utility: {genomic_metrics['overall_genomic_utility']:.4f}"
        )

    # Step 4: Population stratification analysis
    print("\n🌍 Step 4: Population Privacy Analysis")
    print("-" * 40)

    population_af = {}
    for pop in np.unique(populations):
        pop_mask = populations == pop
        pop_snps = snp_data[pop_mask]
        pop_af = np.mean(pop_snps, axis=0) / 2
        population_af[pop] = pop_af

        print(
            f"{pop}: {np.sum(pop_mask)} individuals, mean MAF = {np.mean(pop_af):.4f}"
        )

    # Check population differentiation after privacy
    if len(population_af) > 1:
        pop_names = list(population_af.keys())
        orig_fst = calculate_fst(
            population_af[pop_names[0]], population_af[pop_names[1]]
        )
        print(f"\nOriginal population differentiation (Fst): {orig_fst:.6f}")

    # Step 5: Compliance recommendations
    print("\nStep 5: Genomic Privacy Compliance Recommendations")
    print("-" * 40)

    print("For genomic data sharing:")
    print("• Public biobanks: Use ε ≤ 0.1 with DP_Gaussian mechanism")
    print("• Research collaborations: Use ε = 0.5-1.0 with robust algorithms")
    print("• Population studies: Use ε ≤ 5.0 with utility-optimized mechanisms")
    print("• Always validate Hardy-Weinberg equilibrium preservation")
    print("• Monitor allele frequency accuracy (RMSE < 0.01)")
    print("• Consider linkage disequilibrium patterns in utility assessment")
    print("• Implement per-SNP sensitivity analysis for rare variants")

    print("\n[SUCCESS] Genomic privacy analysis complete!")
    print("Generated comprehensive genomic privacy recommendations")


def calculate_fst(pop1_af, pop2_af):
    """Calculate Fst (population differentiation) between two populations."""
    # Wright's Fst calculation
    p_total = (pop1_af + pop2_af) / 2

    # Heterozygosity within populations
    h_pop1 = 2 * pop1_af * (1 - pop1_af)
    h_pop2 = 2 * pop2_af * (1 - pop2_af)
    h_within = (h_pop1 + h_pop2) / 2

    # Total heterozygosity
    h_total = 2 * p_total * (1 - p_total)

    # Fst calculation
    fst = (h_total - h_within) / h_total
    return np.nanmean(fst)  # Handle division by zero


if __name__ == "__main__":
    main()
