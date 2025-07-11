"""
Metrics and scoring functions for PRESTO.
"""

import numpy as np
from scipy.stats import ks_2samp, pearsonr
from scipy.spatial.distance import jensenshannon
from .privacy_mechanisms import get_noise_generators
from .utils import flatten_and_shape
from typing import List, Dict, Any, Tuple
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization


def calculate_utility_privacy_score(domain, key, epsilon, **params):
    """
    Calculate the negative RMSE between the original and privatized data for a given privacy mechanism.
    Args:
        domain: Original data (list, np.ndarray, or torch.Tensor).
        key: Name of the privacy mechanism.
        epsilon: Privacy parameter.
        **params: Additional parameters for the mechanism.
    Returns:
        float: Negative RMSE (higher is better utility).
    """
    data_list, _ = flatten_and_shape(domain)
    privatized = get_noise_generators()[key](domain, **{**params, "epsilon": epsilon})
    priv_list, _ = flatten_and_shape(privatized)
    rmse = np.sqrt(np.mean((np.array(data_list) - np.array(priv_list)) ** 2))
    return -rmse


def evaluate_algorithm_confidence(domain, key, epsilon, n_evals=10, **params):
    """
    Evaluate the confidence interval for a privacy mechanism by running multiple utility evaluations.
    Args:
        domain: Original data.
        key: Name of the privacy mechanism.
        epsilon: Privacy parameter.
        n_evals: Number of evaluations to run.
        **params: Additional parameters for the mechanism.
    Returns:
        dict: Mean, std, CI bounds, CI width, and individual scores.
    """
    scores = [
        abs(calculate_utility_privacy_score(domain, key, epsilon, **params))
        for _ in range(n_evals)
    ]
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    ci = 1.96 * std / np.sqrt(n_evals)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_lower": round(mean - ci, 4),
        "ci_upper": round(mean + ci, 4),
        "ci_width": round(2 * ci, 4),
        "scores": [round(s, 4) for s in scores],
    }


def performance_explanation_metrics(metrics):
    """
    Calculate performance metrics: mean RMSE, CI width, and reliability.
    Args:
        metrics: Dictionary with 'mean', 'ci_lower', 'ci_upper'.
    Returns:
        dict: mean_rmse, ci_width, reliability.
    """
    rmse = metrics["mean"]
    width = metrics["ci_upper"] - metrics["ci_lower"]
    if width > 0 and rmse > 0:
        reliability = round(1 / (rmse * width), 4)
    else:
        reliability = np.inf
    return {"mean_rmse": rmse, "ci_width": round(width, 4), "reliability": reliability}


def similarity_metrics(original, privatized):
    """
    Compute similarity metrics (KS, JSD, Pearson) between original and privatized data.
    Args:
        original: Original data (array-like).
        privatized: Privatized data (array-like).
    Returns:
        dict: KS statistic, JSD, and Pearson correlation.
    """
    o = np.array(original)
    p = np.array(privatized)
    ks = round(ks_2samp(o, p)[0], 4)
    hist_o, bins = np.histogram(o, bins=30, density=True)
    hist_p, _ = np.histogram(p, bins=bins, density=True)
    jsd = round(jensenshannon(hist_o, hist_p, base=2) ** 2, 4)
    corr = round(pearsonr(o, p)[0], 4)
    return {"KS": ks, "JSD": jsd, "Pearson": corr}


def recommend_top3(
    domain, n_evals: int = 5, init_points: int = 2, n_iter: int = 5
) -> List[Dict[str, Any]]:
    """
    Recommend the top 3 differential privacy algorithms based on performance metrics.

    Args:
        domain: Input data for evaluation
        n_evals: Number of evaluations per algorithm
        init_points: Initial points for Bayesian optimization
        n_iter: Number of optimization iterations

    Returns:
        List of top 3 algorithms with their performance metrics
    """
    results = []
    NOISE_GENERATORS = get_noise_generators()

    for key in NOISE_GENERATORS:
        # Objective: maximize negative RMSE (i.e., minimize RMSE)
        def target(epsilon):
            scores = [
                calculate_utility_privacy_score(domain, key, epsilon)
                for _ in range(n_evals)
            ]
            return float(np.mean(scores))  # Mean negative RMSE

        # Bayesian Optimization to find best ε in [0.1, 5.0]
        optimizer = BayesianOptimization(
            f=target, pbounds={"epsilon": (0.1, 5.0)}, verbose=0, random_state=1
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        best = optimizer.max

        # Extract best ε and evaluate confidence at that point
        eps_opt = best["params"]["epsilon"]
        conf = evaluate_algorithm_confidence(domain, key, eps_opt)
        perf = performance_explanation_metrics(conf)

        # Record performance metrics
        results.append(
            {
                "algorithm": key,
                "epsilon": eps_opt,
                "mean_rmse": perf["mean_rmse"],  # Accuracy
                "ci_width": perf["ci_width"],  # Stability
                "reliability": perf["reliability"],  # Confidence metric
                "score": best["target"],  # Optimization score (neg RMSE)
            }
        )

    # Rank by: lower RMSE → lower ε → narrower CI
    ranked = sorted(
        results, key=lambda x: (x["mean_rmse"], x["epsilon"], x["ci_width"])
    )

    return ranked[:3]  # Return top 3 mechanisms


def recommend_best_algorithms(
    data: torch.Tensor,
    epsilon: float,
    get_noise_generators_func=None,
    calculate_utility_privacy_score_func=None,
    evaluate_algorithm_confidence_func=None,
    performance_explanation_metrics_func=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns the algorithms with:
      1) Maximum similarity (Pearson) between original & privatized data
      2) Maximum reliability (mean RMSE / CI width) at given ε
      3) Maximum privacy strength (mean absolute noise)
    Also plots, side-by-side, the original vs privatized distributions for each of these three.

    Args:
        data: Input data tensor
        epsilon: Privacy parameter
        get_noise_generators_func: Function to get noise generators (uses default if None)
        calculate_utility_privacy_score_func: Function to calculate utility (uses default if None)
        evaluate_algorithm_confidence_func: Function to evaluate confidence (uses default if None)
        performance_explanation_metrics_func: Function to get performance metrics (uses default if None)

    Returns:
        Dictionary with best algorithms for different criteria
    """
    # Use default functions if not provided
    if get_noise_generators_func is None:
        get_noise_generators_func = get_noise_generators
    if calculate_utility_privacy_score_func is None:
        calculate_utility_privacy_score_func = calculate_utility_privacy_score
    if evaluate_algorithm_confidence_func is None:
        evaluate_algorithm_confidence_func = evaluate_algorithm_confidence
    if performance_explanation_metrics_func is None:
        performance_explanation_metrics_func = performance_explanation_metrics

    # Ensure data is a CPU tensor
    if not torch.is_tensor(data):
        data = torch.as_tensor(data, dtype=torch.float32)
    data = data.to("cpu")
    orig_np = data.numpy()

    noise_gens = get_noise_generators_func()
    best_sim = ("", -1.0)
    best_rel = ("", -1.0)
    best_priv = ("", -1.0)

    # Identify top algorithms
    for algo, fn in noise_gens.items():
        # Generate private data
        private = fn(data, epsilon)
        if not torch.is_tensor(private):
            private = torch.as_tensor(private, dtype=data.dtype)
        priv_np = private.cpu().numpy()

        # 1) Similarity (Pearson)
        sim, _ = pearsonr(orig_np, priv_np)
        if sim > best_sim[1]:
            best_sim = (algo, round(sim, 4))

        # 2) Reliability (evaluate at this ε)
        conf = evaluate_algorithm_confidence_func(data, algo, epsilon)
        perf = performance_explanation_metrics_func(conf)
        rel = perf["reliability"]
        if rel > best_rel[1]:
            best_rel = (algo, round(rel, 4))

        # 3) Privacy strength (mean absolute noise)
        priv_strength = float(torch.mean((data - private).abs()).item())
        if priv_strength > best_priv[1]:
            best_priv = (algo, round(priv_strength, 4))

    # Gather the three best
    winners = {
        "max_similarity": {"algorithm": best_sim[0], "score": best_sim[1]},
        "max_reliability": {"algorithm": best_rel[0], "score": best_rel[1]},
        "max_privacy": {"algorithm": best_priv[0], "score": best_priv[1]},
    }

    # Plot original vs. private distributions side-by-side
    plt.figure(figsize=(18, 5))
    for idx, (key, info) in enumerate(winners.items(), start=1):
        algo = info["algorithm"]
        fn = noise_gens[algo]
        private = fn(data, epsilon)
        if not torch.is_tensor(private):
            private = torch.as_tensor(private, dtype=data.dtype)
        priv_np = private.cpu().numpy()

        ax = plt.subplot(1, 3, idx)
        sns.histplot(
            orig_np, bins=30, kde=True, color="skyblue", label="Original", ax=ax
        )
        sns.histplot(
            priv_np, bins=30, kde=True, color="orange", label=f"Private ({algo})", ax=ax
        )
        ax.set_title(
            f"{key.replace('_', ' ').title()}\n{algo} (ε={epsilon:.2f})", fontsize=12
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"Original vs. Private Distributions (ε={epsilon:.2f})",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return winners
