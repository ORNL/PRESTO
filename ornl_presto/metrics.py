"""
Metrics and scoring functions for PRESTO.
"""
import numpy as np
from scipy.stats import ks_2samp, pearsonr
from scipy.spatial.distance import jensenshannon
from .privacy_mechanisms import get_noise_generators
from .utils import flatten_and_shape

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
    privatized = get_noise_generators()[key](domain, **{**params, 'epsilon': epsilon})
    priv_list, _ = flatten_and_shape(privatized)
    rmse = np.sqrt(np.mean((np.array(data_list) - np.array(priv_list))**2))
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
    scores = [abs(calculate_utility_privacy_score(domain, key, epsilon, **params)) for _ in range(n_evals)]
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    ci = 1.96 * std / np.sqrt(n_evals)
    return {
        'mean': round(mean, 4),
        'std': round(std, 4),
        'ci_lower': round(mean - ci, 4),
        'ci_upper': round(mean + ci, 4),
        'ci_width': round(2 * ci, 4),
        'scores': [round(s, 4) for s in scores]
    }

def performance_explanation_metrics(metrics):
    """
    Calculate performance metrics: mean RMSE, CI width, and reliability.
    Args:
        metrics: Dictionary with 'mean', 'ci_lower', 'ci_upper'.
    Returns:
        dict: mean_rmse, ci_width, reliability.
    """
    rmse = metrics['mean']
    width = metrics['ci_upper'] - metrics['ci_lower']
    if width > 0 and rmse > 0:
        reliability = round(1 / (rmse * width), 4)
    else:
        reliability = np.inf
    return {
        'mean_rmse': rmse,
        'ci_width': round(width, 4),
        'reliability': reliability
    }

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
    return {'KS': ks, 'JSD': jsd, 'Pearson': corr}
