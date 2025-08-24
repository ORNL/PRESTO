"""
Visualization functions for PRESTO - Privacy REcommendation and SecuriTy Optimization.

This module provides comprehensive visualization tools for analyzing differential
privacy mechanisms, comparing algorithm performance, and understanding privacy-utility
tradeoffs.
The visualizations help users make informed decisions about privacy mechanism
selection and parameter tuning.

Key Features:
- Data distribution analysis and comparison
- Privacy mechanism performance visualization
- Confidence interval and reliability plots
- Privacy-utility tradeoff exploration
- Algorithm similarity analysis

Example:
    >>> import numpy as np
    >>> from ornl_presto.visualization import visualize_data, visualize_similarity
    >>>
    >>> # Visualize original data distribution
    >>> data = np.random.randn(1000)
    >>> visualize_data(data, "Original Data Distribution")
    >>>
    >>> # Compare algorithm performance
    >>> algorithms = ['laplace', 'gaussian']
    >>> epsilons = [0.1, 0.5, 1.0, 2.0]
    >>> visualize_similarity(data, algorithms, epsilons)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import (
    evaluate_algorithm_confidence,
    similarity_metrics,
)
from .privacy_mechanisms import get_noise_generators


# Visualize the distribution of the input data
def visualize_data(domain, title="Data Distribution"):
    """
    Create a comprehensive visualization of data distribution with histogram and KDE.

    This function provides an overview of the input data's statistical properties
    through a combined histogram and kernel density estimation (KDE) plot. This
    visualization is essential for understanding data characteristics before
    applying differential privacy mechanisms.

    Args:
        domain: Input data to visualize. Can be:
            - list: Python list of numerical values
            - np.ndarray: NumPy array of any shape (will be flattened)
            - torch.Tensor: PyTorch tensor (will be converted to NumPy)
        title (str, optional): Custom title for the plot. 
            Default is "Data Distribution".

    Returns:
        None: Displays the plot directly using matplotlib.

    Example:
        >>> import numpy as np
        >>> from ornl_presto.visualization import visualize_data
        >>>
        >>> # Visualize normal distribution
        >>> normal_data = np.random.randn(1000)
        >>> visualize_data(normal_data, "Normal Distribution Sample")
        >>>
        >>> # Visualize skewed data
        >>> skewed_data = np.random.exponential(2, 1000)
        >>> visualize_data(skewed_data, "Exponential Distribution Sample")

    Note:
        The plot includes both histogram bars and a smooth KDE curve to show
        both the empirical distribution and the estimated probability density.
        Grid lines are added for easier reading of values.
    """
    arr = np.array(domain)
    plt.figure(figsize=(12, 6))
    sns.histplot(arr, bins=30, kde=True, alpha=0.6)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()


# Visualize similarity metrics between original and privatized data
def visualize_similarity(domain, key, epsilon, **params):
    """
    Plot side-by-side histograms and similarity metrics for original and privatized data.
    Args:
        domain: Input data.
        key: Name of the privacy mechanism.
        epsilon: Privacy parameter.
        **params: Additional parameters for the mechanism.
    Returns:
        dict: Similarity metrics (KS, JSD, Pearson).
    """
    NOISE_GENERATORS = get_noise_generators()
    priv = NOISE_GENERATORS[key](domain, epsilon, **params)
    o = np.array(domain)
    p = np.array(priv)
    metrics = similarity_metrics(o, p)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(o, bins=30, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Original Data Distribution")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.3)
    sns.histplot(p, bins=30, kde=True, ax=axes[1], color="orange")
    axes[1].set_title(f"Private Data ({key}, ε={epsilon:.2f})")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.3)
    sns.barplot(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        ax=axes[2],
        hue=list(metrics.keys()),
        palette="Blues",
        legend=False,
    )
    axes[2].set_title("Similarity Metrics")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.3)
    plt.suptitle(f"{key} (ε={epsilon:.4f})", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return metrics


# Visualize the top-3 recommended privacy mechanisms
def visualize_top3(recommendations):
    """
    Plot a bar chart of the top-3 recommended privacy mechanisms.
    Args:
        recommendations: List of recommendation dicts.
    """
    labels = [
        f"{r['algorithm']}\nε={r['epsilon']:.2f}\nmean={r['mean']:.2f}\nwidth={r['ci_width']:.2f}"
        for r in recommendations
    ]
    scores = [r["score"] for r in recommendations]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, scores, capsize=5)
    plt.title("Top 3 Privacy Mechanism Recommendations")
    plt.ylabel("Mean Utility-Privacy Score")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


# Visualize confidence interval for a privacy mechanism
def visualize_confidence(domain, key, epsilon, n_evals=10, **params):
    """
    Plot the mean and confidence interval for a privacy mechanism.
    Args:
        domain: Input data.
        key: Name of the privacy mechanism.
        epsilon: Privacy parameter.
        n_evals: Number of evaluations.
        **params: Additional parameters for the mechanism.
    Returns:
        dict: Confidence interval results.
    """
    res = evaluate_algorithm_confidence(domain, key, epsilon, n_evals, **params)
    mean, lower, upper = res["mean"], res["ci_lower"], res["ci_upper"]
    plt.figure(figsize=(6, 4))
    plt.bar([key], [mean], yerr=[[mean - lower], [upper - mean]], capsize=5)
    plt.title(f"Confidence: {key} (ε={epsilon:.2f})")
    plt.ylabel("Mean Utility-Privacy Score")
    plt.grid(alpha=0.3)
    plt.show()
    return res


# Visualize confidence intervals for the top-3 mechanisms
def visualize_confidence_top3(domain, recommendations, n_evals=10):
    """
    Plot 95% confidence intervals for each of the top-3 recommended mechanisms.
    Args:
        domain: Input data.
        recommendations: List of recommendation dicts.
        n_evals: Number of evaluations.
    """
    labels = []
    means = []
    error_lower = []
    error_upper = []
    for rec in recommendations:
        alg = rec["algorithm"]
        eps = rec["epsilon"]
        conf = evaluate_algorithm_confidence(domain, alg, eps, n_evals)
        labels.append(f"{alg} ε={eps:.2f}")
        means.append(conf["mean"])
        error_lower.append(conf["mean"] - conf["ci_lower"])
        error_upper.append(conf["ci_upper"] - conf["mean"])
    plt.figure(figsize=(8, 6))
    plt.bar(labels, means, yerr=[error_lower, error_upper], capsize=5)
    plt.title("95% Confidence Intervals for Top-3 Mechanisms")
    plt.ylabel("Mean Utility-Privacy Score")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


# Visualize overlay of original and top-3 privatized distributions
def visualize_overlay_original_and_private(domain, top3):
    """
    Plot KDE overlays of the original data and the top-3 privatized distributions.
    Args:
        domain: Input data.
        top3: List of top-3 recommendation dicts.
    """
    arr_orig = np.array(domain)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(arr_orig, label="Original", fill=False)
    for rec in top3:
        key, eps = rec["algorithm"], rec["epsilon"]
        NOISE_GENERATORS = get_noise_generators()
        priv = NOISE_GENERATORS[key](domain, eps)
        arr_priv = np.array(priv)
        sns.kdeplot(arr_priv, label=f"{key} ε={eps:.4f}", fill=False)
    plt.title("Overlay: Original vs Top-3 Privatized Distributions")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
