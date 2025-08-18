"""
Differential privacy mechanisms for PRESTO.
"""

from typing import Union, List, Tuple, Any
import numpy as np
import torch
import random
from hashlib import sha256
from .utils import (
    flatten_and_shape,
    restore_type_and_shape,
    ensure_tensor,
    fast_walsh_hadamard_transform,
)

ArrayLike = Union[List, np.ndarray, torch.Tensor]


# Gaussian Mechanism for Differential Privacy
def applyDPGaussian(
    domain: ArrayLike,
    sensitivity: float = 1,
    delta: float = 1e-5,
    epsilon: float = 1,
    gamma: float = 1,
) -> ArrayLike:
    """
    Apply the Gaussian mechanism to privatize data.
    Args:
        domain: Input data (list, np.ndarray, or torch.Tensor).
        sensitivity: Sensitivity of the query.
        delta: Target delta for (epsilon, delta)-DP.
        epsilon: Privacy parameter.
        gamma: Scaling parameter.
    Returns:
        Privatized data matching the input type and shape.
    """
    data, shape = flatten_and_shape(domain)
    sigma = np.sqrt(sensitivity * np.log(1.25 / delta)) * gamma / epsilon
    privatized = np.array(data) + np.random.normal(0, sigma, size=len(data))
    return restore_type_and_shape(domain, privatized.tolist(), shape)


# Exponential Mechanism for Differential Privacy
def applyDPExponential(
    domain: ArrayLike, sensitivity: float = 1, epsilon: float = 1, gamma: float = 1.0
) -> ArrayLike:
    """
    Apply the Exponential mechanism to privatize data.
    Args:
        domain: Input data.
        sensitivity: Sensitivity of the query.
        epsilon: Privacy parameter.
        gamma: Scaling parameter.
    Returns:
        Privatized data matching the input type and shape.
    """
    data, shape = flatten_and_shape(domain)
    scale = sensitivity * gamma / epsilon
    noise = np.random.exponential(scale, size=len(data))
    signs = np.random.choice([-1, 1], size=len(data))
    priv = np.array(data) + noise * signs
    return restore_type_and_shape(domain, priv.tolist(), shape)


# Laplace Mechanism for Differential Privacy
def applyDPLaplace(
    domain: ArrayLike, sensitivity: float = 1, epsilon: float = 1, gamma: float = 1
) -> ArrayLike:
    """
    Apply the Laplace mechanism to privatize data.
    Args:
        domain: Input data.
        sensitivity: Sensitivity of the query.
        epsilon: Privacy parameter.
        gamma: Scaling parameter.
    Returns:
        Privatized data matching the input type and shape.
    """
    data, shape = flatten_and_shape(domain)
    noise = np.random.laplace(0, sensitivity * gamma / epsilon, size=len(data))
    privatized = np.array(data) + noise
    return restore_type_and_shape(domain, privatized.tolist(), shape)


# Helper for Sparse Vector Technique (SVT)
def above_threshold_SVT(
    val: float, domain_list: List[float], T: float, epsilon: float
) -> bool:
    """
    Helper function for SVT: returns value if above noisy threshold, else random value.
    """
    T_hat = T + np.random.laplace(0, 2 / epsilon)
    nu_i = np.random.laplace(0, 4 / epsilon)
    if val + nu_i >= T_hat:
        return val
    return random.choice(domain_list)


# Sparse Vector Technique (SVT) for Differential Privacy
def applySVTAboveThreshold_full(domain: ArrayLike, epsilon: float) -> ArrayLike:
    """
    Apply the full SVT to privatize data.
    Args:
        domain: Input data.
        epsilon: Privacy parameter.
    Returns:
        Privatized data matching the input type and shape.
    """
    data_list, shape = flatten_and_shape(domain)
    T = np.mean(data_list)
    privatized = [above_threshold_SVT(val, data_list, T, epsilon) for val in data_list]
    return restore_type_and_shape(domain, privatized, shape)


# Percentile Privacy Mechanism
def percentilePrivacy(domain: ArrayLike, percentile: float = 50) -> ArrayLike:
    """
    Apply percentile privacy: values below the percentile are zeroed.
    Args:
        domain: Input data.
        percentile: Percentile threshold (0-100).
    Returns:
        Privatized data matching the input type and shape.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")
    data, shape = flatten_and_shape(domain)
    arr = np.array(data)
    threshold = np.percentile(arr, percentile)
    result = np.where(arr >= threshold, arr, 0)
    return restore_type_and_shape(domain, result.tolist(), shape)


# Count Mean Sketch with Laplace Noise
def count_mean_sketch(data: ArrayLike, epsilon: float, bins: int = 10) -> torch.Tensor:
    """
    Estimate the mean using a histogram with Laplace noise.
    Args:
        data: Input data (tensor or array).
        epsilon: Privacy parameter.
        bins: Number of histogram bins.
    Returns:
        Tensor of mean estimates.
    """
    data = ensure_tensor(data)
    min_val = float(data.min().item())
    max_val = float(data.max().item())
    counts = torch.histc(data, bins=bins, min=min_val, max=max_val)
    scale = 1.0 / epsilon
    noise = torch.distributions.Laplace(0, scale).sample(counts.size()).to(data.device)
    noisy = counts.float() + noise
    edges = torch.linspace(min_val, max_val, steps=bins + 1, device=data.device)
    centers = (edges[:-1] + edges[1:]) / 2
    mean_est = (noisy * centers).sum() / noisy.sum()
    return torch.full_like(data, mean_est)


# Hadamard Mechanism for Differential Privacy
def hadamard_mechanism(data: ArrayLike, epsilon: float) -> torch.Tensor:
    """
    Apply the Hadamard mechanism: add Laplace noise in the Hadamard domain.
    Args:
        data: Input data (tensor or array).
        epsilon: Privacy parameter.
    Returns:
        Privatized tensor.
    """
    data = ensure_tensor(data)
    n = data.numel()
    m = 1 << ((n - 1).bit_length())
    x = torch.zeros(m, dtype=data.dtype, device=data.device)
    x[:n] = data
    y = fast_walsh_hadamard_transform(x) / np.sqrt(m)
    scale = np.sqrt(m) / epsilon
    noise = torch.distributions.Laplace(0, scale).sample((m,)).to(data.device)
    y_noisy = y + noise
    x_noisy = fast_walsh_hadamard_transform(y_noisy) / np.sqrt(m)
    return x_noisy[:n]


# Hadamard Response (Local DP)
def hadamard_response(data: ArrayLike, epsilon: float) -> torch.Tensor:
    """
    Local-DP via randomized response in the Hadamard domain.
    Args:
        data: Input data (tensor or array).
        epsilon: Privacy parameter.
    Returns:
        Privatized tensor.
    """
    data = ensure_tensor(data)
    d = int(data.max().item()) + 1
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    flip = torch.bernoulli(torch.full(data.size(), p, device=data.device))
    rand = torch.randint(0, d, data.size(), device=data.device)
    return torch.where(flip.bool(), data.long(), rand)


# RAPPOR Mechanism (Bloom filter + randomized response)
def rappor(data: ArrayLike, epsilon: float, m: int = 16, k: int = 2) -> torch.Tensor:
    """
    Basic RAPPOR: Bloom filter encoding + randomized response.
    Args:
        data: Input data (tensor or array).
        epsilon: Privacy parameter.
        m: Bloom filter size.
        k: Number of hash functions.
    Returns:
        Privatized tensor.
    """
    data = ensure_tensor(data)
    n = data.numel()
    bloom = torch.zeros((n, m), dtype=torch.bool, device=data.device)
    for i in range(n):
        val = int(data[i].item())
        for j in range(k):
            h = int(sha256(f"{val}_{j}".encode()).hexdigest(), 16)
            bloom[i, h % m] = True
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    q = 1.0 / (np.exp(epsilon) + 1)
    rnd = torch.rand((n, m), device=data.device)
    priv = torch.where(bloom, rnd < p, rnd < q)
    out = priv.int().sum(dim=1).float() / p
    return out


def get_noise_generators() -> dict:
    """
    Return a dictionary of available privacy mechanisms.
    Returns:
        dict: Mapping of mechanism names to their functions.
    """
    return {
        "gaussian": applyDPGaussian,
        "exponential": applyDPExponential,
        "laplace": applyDPLaplace,
        "svt": applySVTAboveThreshold_full,
        "percentile": percentilePrivacy,
        "count_mean_sketch": count_mean_sketch,
        "hadamard": hadamard_mechanism,
        "hadamard_response": hadamard_response,
        "rappor": rappor,
    }
