"""
This is the code for PRESTO: Privacy Recommendation and Security Optimization tool.

Bayesian optimization is used to automatically select the best privacy mechanism and its optimal epsilon.
Top-3 Recommendations, Confidence Analysis, Performance Explanation, GPU Support, and Visualization.

Metrics Explanations:
- KS Statistic: measures the maximum difference between two cumulative distribution functions; desired value near 0.
- Jensen–Shannon Divergence (JSD): quantifies how two distributions diverge in bits; desired value near 0 bits.
- Pearson Correlation: indicates linear alignment between original and private data; desired value close to +1.
- Mean Utility-Privacy Score: negative root mean square error of privatization; desired to be as high as possible (closer to 0).
- Confidence Interval (CI) Width: range of the 95% confidence interval for the utility score; desired width to be small (e.g., <0.05).
- Reliability: ratio of mean utility score to CI width, reflecting consistency; desired to be high (e.g., >10).

Desired Metrics:
- Similarity Metrics:
    KS ≈ 0 (identical distributions)
    JSD ≈ 0 bits (minimal divergence)
    Pearson ≈ +1 (strong linear fidelity)
- Confidence Metrics:
    mean as high as possible (closer to 0, since we use negative RMSE)
    CI width small (e.g., <0.05 for stable performance)
- Performance Explanation:
    A small confidence interval (CI) width indicates consistent performance across evaluations.
    Reliability is computed as 1 / (RMSE × CI width); higher values (e.g., >10) suggest a reliable and stable privacy-utility trade-off.
"""

import math
import torch
import random
import numpy as np
import seaborn as sns
import pandas as pd, re
from hashlib import sha256
import matplotlib.pyplot as plt

import gpytorch
from bayes_opt import BayesianOptimization

from scipy import stats as st
from scipy.stats import ks_2samp, pearsonr
from scipy.spatial.distance import jensenshannon

try:
    import torch.nn as nn
    from torch.nn import RMSNorm
except AttributeError:
    RMSNorm = None  # or define a fallback

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
torch.set_default_dtype(torch.float64)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from opacus.privacy_engine import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

print(f"Using device: {device}")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
def gpr_gpytorch(train_x, train_y, test_x, training_iter):
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)
    test_x = torch.tensor(test_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    f_preds = model(test_x)
    y_preds = likelihood(model(test_x))

    f_mean = f_preds.mean
    f_covar = f_preds.covariance_matrix
    return f_mean, f_covar


def numpy_to_list(nd_array):
    """
        Converts a PyTorch tensor to a flattened Python list using NumPy.
    
        Args:
            nd_array (torch.Tensor): A PyTorch tensor (any shape, assumed to be on CPU or GPU).
    
        Returns:
            list: A flattened list of tensor values.
    """
    return nd_array.cpu().numpy().flatten().tolist()


def _to_tensor(data):
    """
        Ensures input is a PyTorch tensor. If it's not, converts it using float32 precision.
    
        Args:
            data (list, np.ndarray, or torch.Tensor): The input data to convert.
    
        Returns:
            torch.Tensor: The input as a torch tensor with dtype=float32.
    """
    if not torch.is_tensor(data):
        return torch.as_tensor(data, dtype=torch.float32)
    return data


def type_checking_and_return_lists(domain):
    """
        Converts a list, numpy array, or tensor to a flattened list and captures its original shape.
    
        Args:
            domain (list, np.ndarray, or torch.Tensor): Input data to be flattened.
    
        Returns:
            tuple:
                - list: Flattened data as a list.
                - tuple: Original shape of the input data.
    """
    arr = np.array(domain)
    return arr.flatten().tolist(), arr.shape


def type_checking_return_actual_dtype(domain, data_list, shape):
    """
        Reconstructs the original structure and type of a dataset from a flat list.
    
        Args:
            domain (list, np.ndarray, or torch.Tensor): Original data used to infer target type and shape.
            data_list (list): Flattened list of privatized or transformed values.
            shape (tuple): Original shape to reshape the flat list into.
    
        Returns:
            torch.Tensor or list: Data reshaped and converted back to match the original input type.
    """
    arr = np.array(data_list).reshape(shape)
    if isinstance(domain, torch.Tensor):
        return torch.from_numpy(arr).to(dtype=domain.dtype, device=domain.device)
    return arr.tolist()


def applyDPGaussian(domain, sensitivity=1, delta=1e-5, epsilon=1, gamma=1):
    """
        Parameters:
            domain ([type]): Description.
            sensitivity=1 ([type]): Description.
            delta=1e-5 ([type]): Description.
            epsilon=1 ([type]): Description.
            gamma=1 ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    data, shape = type_checking_and_return_lists(domain)
    sigma = np.sqrt(sensitivity * np.log(1.25 / delta)) * gamma / epsilon
    privatized = np.array(data) + np.random.normal(0, sigma, size=len(data))
    return type_checking_return_actual_dtype(domain, privatized.tolist(), shape)


def applyDPExponential(domain, sensitivity=1, epsilon=1, gamma=1.0):
    """
        Parameters:
            domain ([type]): Description.
            sensitivity=1 ([type]): Description.
            epsilon=1 ([type]): Description.
            gamma=1.0 ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    data, shape = type_checking_and_return_lists(domain)
    scale = sensitivity * gamma / epsilon
    noise = np.random.exponential(scale, size=len(data))
    signs = np.random.choice([-1, 1], size=len(data))
    priv = np.array(data) + noise * signs
    return type_checking_return_actual_dtype(domain, priv.tolist(), shape)


def applyDPLaplace(domain, sensitivity=1, epsilon=1, gamma=1):
    """
        Parameters:
            domain ([type]): Description.
            sensitivity=1 ([type]): Description.
            epsilon=1 ([type]): Description.
            gamma=1 ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    data, shape = type_checking_and_return_lists(domain)
    noise = np.random.laplace(0, sensitivity * gamma / epsilon, size=len(data))
    privatized = np.array(data) + noise
    return type_checking_return_actual_dtype(domain, privatized.tolist(), shape)


def above_threshold_SVT(val, domain_list, T, epsilon):
    """
        Parameters:
            val ([type]): Description.
            domain_list ([type]): Description.
            T ([type]): Description.
            epsilon ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    T_hat = T + np.random.laplace(0, 2/epsilon)
    nu_i = np.random.laplace(0, 4/epsilon)
    if val + nu_i >= T_hat:
        return val
    return random.choice(domain_list)


def applySVTAboveThreshold_full(domain, epsilon):
    """
        Parameters:
            domain ([type]): Description.
            epsilon ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """
    data_list, shape = type_checking_and_return_lists(domain)
    T = np.mean(data_list)
    privatized = [above_threshold_SVT(val, data_list, T, epsilon) for val in data_list]
    return type_checking_return_actual_dtype(domain, privatized, shape)


def percentilePrivacy(domain, percentile=50):
    """
        Parameters:
            domain ([type]): Description.
            percentile=50 ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")
    data, shape = type_checking_and_return_lists(domain)
    arr = np.array(data)
    threshold = np.percentile(arr, percentile)
    result = np.where(arr >= threshold, arr, 0)
    return type_checking_return_actual_dtype(domain, result.tolist(), shape)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Perform Fast Walsh–Hadamard Transform (length must be power of two)."""
    h = 1
    y = x.clone()
    n = y.numel()
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = y[j]
                v = y[j + h]
                y[j]     = u + v
                y[j + h] = u - v
        h *= 2
    return y


def count_mean_sketch(data, epsilon: float, bins: int = 10) -> torch.Tensor:
    """Histogram with Laplace noise, then mean estimation."""
    data = _to_tensor(data)
    min_val = float(data.min().item())
    max_val = float(data.max().item())
    counts = torch.histc(data, bins=bins, min=min_val, max=max_val)
    scale = 1.0 / epsilon
    noise = torch.distributions.Laplace(0, scale).sample(counts.size()).to(data.device)
    noisy = counts.float() + noise
    edges = torch.linspace(min_val, max_val, steps=bins+1, device=data.device)
    centers = (edges[:-1] + edges[1:]) / 2
    mean_est = (noisy * centers).sum() / noisy.sum()
    return torch.full_like(data, mean_est)


def hadamard_mechanism(data, epsilon: float) -> torch.Tensor:
    """Add Laplace noise in the Hadamard transform domain."""
    data = _to_tensor(data)
    n = data.numel()
    m = 1 << ((n - 1).bit_length())  # next power of two
    x = torch.zeros(m, dtype=data.dtype, device=data.device)
    x[:n] = data
    y = fwht(x) / math.sqrt(m)
    scale = math.sqrt(m) / epsilon
    noise = torch.distributions.Laplace(0, scale).sample((m,)).to(data.device)
    y_noisy = y + noise
    x_noisy = fwht(y_noisy) / math.sqrt(m)
    return x_noisy[:n]


def hadamard_response(data, epsilon: float) -> torch.Tensor:
    """Local-DP via simplified randomized response over integer categories."""
    data = _to_tensor(data)
    d = int(data.max().item()) + 1
    p = math.exp(epsilon) / (math.exp(epsilon) + 1)
    flip = torch.bernoulli(torch.full(data.size(), p, device=data.device))
    rand = torch.randint(0, d, data.size(), device=data.device)
    return torch.where(flip.bool(), data.long(), rand)


def rappor(data, epsilon: float, m: int = 16, k: int = 2) -> torch.Tensor:
    """Basic RAPPOR: Bloom filter + randomized response."""
    data = _to_tensor(data)
    n = data.numel()
    bloom = torch.zeros((n, m), dtype=torch.bool, device=data.device)
    for i in range(n):
        val = int(data[i].item())
        for j in range(k):
            h = int(sha256(f"{val}_{j}".encode()).hexdigest(), 16)
            bloom[i, h % m] = True
    p = math.exp(epsilon) / (math.exp(epsilon) + 1)
    q = 1.0 / (math.exp(epsilon) + 1)
    rnd = torch.rand((n, m), device=data.device)
    priv = torch.where(bloom, rnd < p, rnd < q)
    out = priv.int().sum(dim=1).float() / p
    return out


def get_noise_generators():
    """
        Parameters:
    
        Returns:
            [type]: Description of the return value.
    """

    return {
        "DP_Gaussian": lambda d, epsilon, **kw: applyDPGaussian(d, delta=kw.get('delta',1e-5), epsilon=epsilon, gamma=kw.get('gamma',1)),
        "DP_Exponential": lambda d, epsilon, **kw: applyDPExponential(d, sensitivity=kw.get('sensitivity',1), epsilon=epsilon, gamma=kw.get('gamma',1.0)),
        "DP_Laplace": lambda d, epsilon, **kw: applyDPLaplace(d, sensitivity=kw.get('sensitivity',1), epsilon=epsilon, gamma=kw.get('gamma',1)),
        "SVT_AboveThreshold": lambda d, epsilon, **kw: applySVTAboveThreshold_full(d, epsilon),
        "PercentilePrivacy": lambda d, epsilon, **kw: percentilePrivacy(d, percentile=kw.get('percentile',50)),
        'Count_Mean_Sketch': count_mean_sketch,
        'Hadamard_Mechanism': hadamard_mechanism,
        # 'Hadamard_Response': hadamard_response,  # (commented out; can be enabled if needed)
        'RAPPOR': rappor,
    }


def calculate_utility_privacy_score(domain, key, epsilon, **params):
    """
        Parameters:
            domain ([type]): Description.
            key ([type]): Description.
            epsilon ([type]): Description.
            **params ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    # Convert original domain to flat list
    data_list, _ = type_checking_and_return_lists(domain)

    # Apply the selected privacy mechanism from get_noise_generators()
    privatized = get_noise_generators()[key](domain, **{**params, 'epsilon': epsilon})

    # Convert privatized output to flat list
    priv_list, _ = type_checking_and_return_lists(privatized)

    # Compute RMSE between original and privatized data
    rmse = np.sqrt(np.mean((np.array(data_list) - np.array(priv_list))**2))

    # Return negative RMSE (higher = better utility)
    return -rmse


def evaluate_algorithm_confidence(domain, key, epsilon, n_evals=10, **params):
    """
        Parameters:
            domain ([type]): Description.
            key ([type]): Description.
            epsilon ([type]): Description.
            n_evals=10 ([type]): Description.
            **params ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    # Run the utility score multiple times (abs(-RMSE) → RMSE)
    scores = [abs(calculate_utility_privacy_score(domain, key, epsilon, **params)) for _ in range(n_evals)]

    # Compute statistics over the scores
    mean = np.mean(scores)                      # Mean RMSE across trials
    std = np.std(scores, ddof=1)                # Sample standard deviation
    ci = 1.96 * std / np.sqrt(n_evals)          # 95% Confidence Interval (normal approx.)

    # Return rounded summary metrics and individual scores
    return {
        'mean': round(mean, 4),                 # Average RMSE
        'std': round(std, 4),                   # Standard deviation of RMSE
        'ci_lower': round(mean - ci, 4),        # Lower bound of 95% CI
        'ci_upper': round(mean + ci, 4),        # Upper bound of 95% CI
        'ci_width': round(2 * ci, 4),           # Total CI width
        'scores': [round(s, 4) for s in scores] # Individual RMSE scores (rounded)
    }


def performance_explanation_metrics(metrics):
    """
        Parameters:
            metrics ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    rmse = metrics['mean']
    width = metrics['ci_upper'] - metrics['ci_lower']

    # Reliability is undefined if RMSE or CI width is zero; assign ∞ in such cases
    if width > 0 and rmse > 0:
        reliability = round(1 / (rmse * width), 4)
    else:
        reliability = np.inf  # Ideal or degenerate case

    return {
        'mean_rmse': rmse,               # Central accuracy metric
        'ci_width': round(width, 4),     # Stability metric
        'reliability': reliability       # Combined performance-confidence score
    }


def recommend_top3(domain, n_evals=5, init_points=2, n_iter=5):
    """
        Parameters:
            domain ([type]): Description.
            n_evals=5 ([type]): Description.
            init_points=2 ([type]): Description.
            n_iter=5 ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
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
            f=target,
            pbounds={'epsilon': (0.1, 5.0)},
            verbose=0,
            random_state=1
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        best = optimizer.max

        # Extract best ε and evaluate confidence at that point
        eps_opt = best['params']['epsilon']
        conf    = evaluate_algorithm_confidence(domain, key, eps_opt)
        perf    = performance_explanation_metrics(conf)

        # Record performance metrics
        results.append({
            'algorithm':   key,
            'epsilon':     eps_opt,
            'mean_rmse':   perf['mean_rmse'],   # Accuracy
            'ci_width':    perf['ci_width'],    # Stability
            'reliability': perf['reliability'], # Confidence metric
            'score':       best['target']       # Optimization score (neg RMSE)
        })

    # Rank by: lower RMSE → lower ε → narrower CI
    ranked = sorted(
        results,
        key=lambda x: (x['mean_rmse'], x['epsilon'], x['ci_width'])
    )

    return ranked[:3]  # Return top 3 mechanisms


def visualize_data(domain, title="Data Distribution"):
    """
        Parameters:
            domain ([type]): Description.
            title="Data Distribution" ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    arr = np.array(domain)
    plt.figure(figsize=(12,6))
    sns.histplot(arr, bins=30, kde=True, alpha=0.6)
    plt.title(title); plt.xlabel("Value"); plt.ylabel("Frequency"); plt.grid(alpha=0.3); plt.show()


def visualize_similarity(domain, key, epsilon, **params):
    """
        Parameters:
            domain ([type]): Description.
            key ([type]): Description.
            epsilon ([type]): Description.
            **params ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    """
    Computes similarity metrics between original and privatized data and
    displays three side-by-side plots:
      1) Histogram of original data
      2) Histogram of private data
      3) Bar chart of KS, JSD, Pearson

    Parameters:
    - domain   : list or array of original values
    - key      : name of the DP mechanism in NOISE_GENERATORS
    - epsilon  : privacy parameter
    - **params : any additional arguments for the noise function
    """
    # Generate private data
    NOISE_GENERATORS = get_noise_generators()
    priv = NOISE_GENERATORS[key](domain, epsilon, **params)

    # Convert to numpy arrays
    o = np.array(domain)
    p = np.array(priv)

    # Compute similarity metrics
    ks   = round(ks_2samp(o, p)[0], 4)
    hist_o, bins = np.histogram(o, bins=30, density=True)
    hist_p, _    = np.histogram(p, bins=bins, density=True)
    jsd  = round(jensenshannon(hist_o, hist_p, base=2) ** 2, 4)
    corr = round(pearsonr(o, p)[0], 4)
    metrics = {'KS': ks, 'JSD': jsd, 'Pearson': corr}

    # Create a 1x3 subplot layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Original data histogram
    sns.histplot(o, bins=30, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Original Data Distribution")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.3)

    # Panel 2: Private data histogram
    sns.histplot(p, bins=bins, kde=True, ax=axes[1], color='orange')
    axes[1].set_title(f"Private Data ({key}, ε={epsilon:.2f})")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.3)

    # Panel 3: Similarity metrics bar chart
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=axes[2], palette="Blues")
    axes[2].set_title("Similarity Metrics")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0, 1)  # since KS, JSD, and Pearson are in [0,1]
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle(f"Similarity Analysis: {key} (ε={epsilon:.4f})", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return metrics

# Visualize top-3 recommendations
def visualize_top3(recommendations):
    """
        Parameters:
            recommendations ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    labels = [f"{r['algorithm']}\nε={r['epsilon']:.2f}\nmean={r['mean']:.2f}\nwidth={r['ci_width']:.2f}" for r in recommendations]
    scores = [r['score'] for r in recommendations]
    plt.figure(figsize=(8,6))
    plt.bar(labels, scores, capsize=5)
    plt.title("Top 3 Privacy Mechanism Recommendations")
    plt.ylabel("Mean Utility-Privacy Score")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# Visualize confidence for top algorithm. Desired: narrow error bars indicating small CI.
def visualize_confidence(domain, key, epsilon, n_evals=10, **params):
    """
        Parameters:
            domain ([type]): Description.
            key ([type]): Description.
            epsilon ([type]): Description.
            n_evals=10 ([type]): Description.
            **params ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    res = evaluate_algorithm_confidence(domain, key, epsilon, n_evals, **params)
    mean, lower, upper = res['mean'], res['ci_lower'], res['ci_upper']
    plt.figure(figsize=(6,4))
    plt.bar([key], [mean], yerr=[[mean-lower],[upper-mean]], capsize=5)
    plt.title(f"Confidence: {key} (ε={epsilon:.2f})")
    plt.ylabel("Mean Utility-Privacy Score"); plt.grid(alpha=0.3); plt.show()
    return res

# Confidence visualization for Top-3 Mechanisms.
def visualize_confidence_top3(domain, recommendations, n_evals=10):
    """
        Parameters:
            domain ([type]): Description.
            recommendations ([type]): Description.
            n_evals=10 ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    """
    Visualizes 95% confidence intervals for each algorithm in recommendations.
    """
    labels = []
    means = []
    error_lower = []
    error_upper = []
    for rec in recommendations:
        alg = rec['algorithm']
        eps = rec['epsilon']
        conf = evaluate_algorithm_confidence(domain, alg, eps, n_evals)
        labels.append(f"{alg} ε={eps:.2f}")
        means.append(conf['mean'])
        error_lower.append(conf['mean'] - conf['ci_lower'])
        error_upper.append(conf['ci_upper'] - conf['mean'])
    plt.figure(figsize=(8,6))
    plt.bar(labels, means, yerr=[error_lower, error_upper], capsize=5)
    plt.title("95% Confidence Intervals for Top-3 Mechanisms")
    plt.ylabel("Mean Utility-Privacy Score")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# Combined overlay plot
def visualize_overlay_original_and_private(domain, top3):
    """
        Parameters:
            domain ([type]): Description.
            top3 ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    arr_orig = np.array(domain)
    plt.figure(figsize=(10,6))
    sns.kdeplot(arr_orig, label='Original', fill=False)
    for rec in top3:
        key, eps = rec['algorithm'], rec['epsilon']
        NOISE_GENERATORS = get_noise_generators()
        priv = NOISE_GENERATORS[key](domain, eps)
        arr_priv = np.array(priv)
        sns.kdeplot(arr_priv, label=f"{key} ε={eps:.4f}", fill=False)
    plt.title("Overlay: Original vs Top-3 Privatized Distributions")
    plt.xlabel("Value"); plt.ylabel("Density"); plt.legend(); plt.grid(alpha=0.3); plt.show()

# Recomendation the best algorithms for privacy, reliability and similary for given epsilon.
def recommend_best_algorithms(data: torch.Tensor, epsilon: float, get_noise_generators, calculate_utility_privacy_score, evaluate_algorithm_confidence,performance_explanation_metrics):
    """
    Returns the algorithms with:
      1) Maximum similarity (Pearson) between original & privatized data
      2) Maximum reliability (mean RMSE / CI width) at given ε
      3) Maximum privacy strength (mean absolute noise)
    Also plots, side-by-side, the original vs privatized distributions for each of these three.
    """
    # Ensure data is a CPU tensor
    if not torch.is_tensor(data):
        data = torch.as_tensor(data, dtype=torch.float32)
    data = data.to('cpu')
    orig_np = data.numpy()

    noise_gens = get_noise_generators()
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
        conf = evaluate_algorithm_confidence(data, algo, epsilon)
        perf = performance_explanation_metrics(conf)
        rel = perf["reliability"]
        if rel > best_rel[1]:
            best_rel = (algo, round(rel, 4))

        # 3) Privacy strength (mean absolute noise)
        priv_strength = float(torch.mean((data - private).abs()).item())
        if priv_strength > best_priv[1]:
            best_priv = (algo, round(priv_strength, 4))

    # Gather the three best
    winners = {
        "max_similarity":  {"algorithm": best_sim[0], "score": best_sim[1]},
        "max_reliability": {"algorithm": best_rel[0], "score": best_rel[1]},
        "max_privacy":     {"algorithm": best_priv[0], "score": best_priv[1]},
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
        sns.histplot(orig_np, bins=30, kde=True, color='skyblue', label="Original", ax=ax)
        sns.histplot(priv_np, bins=30, kde=True, color='orange', label=f"Private ({algo})", ax=ax)
        ax.set_title(f"{key.replace('_', ' ').title()}\n{algo} (ε={epsilon:.2f})", fontsize=12)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Original vs. Private Distributions (ε={epsilon:.2f})", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return winners


def dp_function(noise_multiplier, max_grad_norm, model_class, train_dataset, X_train):
    """
        Parameters:
            noise_multiplier ([type]): Description.
            max_grad_norm ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    privacy_engine = PrivacyEngine()
    # Instantiate the model, loss function, and optimizer
    model_pre_dp = model_class(input_size=X_train.shape[1])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model_pre_dp.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # apply the privacy "engine"
    model, optimizer, data_loader = privacy_engine.make_private(
      module=model_pre_dp, 
      optimizer=optimizer, 
      data_loader=train_dataloader,
      noise_multiplier=noise_multiplier, 
      max_grad_norm=max_grad_norm,
    )
    return model, optimizer, criterion, data_loader, privacy_engine


def dp_function_train_and_pred(model, optimizer, criterion, train_dataloader, X_test, y_test):
    """
        Parameters:
            model ([type]): Description.
            optimizer ([type]): Description.
            criterion ([type]): Description.
            train_dataloader ([type]): Description.
            X_test ([type]): Description.
            y_test ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    # This code runs the DP-impacted model to compute the Accuracy

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        for inputs, labels in train_dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.double())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    with torch.no_grad():
        predicted = model(X_test)
        predicted_classes = (predicted > 0.5).float()
        accuracy = (predicted_classes.squeeze() == y_test).sum().item() / y_test.size(0)
        print(f'Accuracy: {accuracy:.4f}')

    return predicted_classes, accuracy


def dp_target(noise_multiplier, max_grad_norm, model_class, X_test, train_dataset, y_test):
    """
        Parameters:
            noise_multiplier ([type]): Description.
            max_grad_norm ([type]): Description.
            model: e.g., BinaryClassifier, LogisticRegression.
    
        Returns:
            [type]: Description of the return value.
    """

    # apply DP to ML
    model, optimizer, criterion, train_dataloader, privacy_engine = dp_function(noise_multiplier, max_grad_norm, model_class, train_dataset, X_test)
    # compute accuracy
    predicted_classes, accuracy = dp_function_train_and_pred(model, optimizer, criterion, train_dataloader, X_test, y_test)
    return accuracy, privacy_engine, predicted_classes


def dp_pareto_front(x1, x2, model_class, X_test, train_dataset, y_test):
    """
        Parameters:
            model ([type]): Description.
            optimizer ([type]): Description.
            criterion ([type]): Description.
            train_dataloader ([type]): Description.
            X_test ([type]): Description.
            y_test ([type]): Description.
    
        Returns:
            [type]: Description of the return value.
    """

    # delta is .1 below.
    delta = .1
    accuracy_ = []
    epsilon_ = []
    xg1, xg2 = np.meshgrid(x1, x2)
    Xg = np.hstack((xg1.flatten()[:,None], xg2.flatten()[:,None]))
    measured_points = []

    X_params = np.asarray([[100., 1.],[10,1.],[1.1, 1.],[.1, .1]])
    Y_params = np.asarray([[-.01, .51],[-.09, .84],[.62,.865],[650,.885]])

    alphas = np.asarray([1,1])

    for i in range(100):
        print('run',i)

        # which data points are unmeasured?
        unmeasured_points = np.setdiff1d(np.arange(0,Xg.shape[0]),measured_points)

        # train our GPs using all our prior data.
        training_iter = 50
        mean1, Cov1 = gpr_gpytorch(X_params, Y_params[:,0], Xg[unmeasured_points,:], training_iter)
        mean2, Cov2 = gpr_gpytorch(X_params, Y_params[:,1], Xg[unmeasured_points,:], training_iter)

        Z1  = np.random.multivariate_normal(mean1.detach().numpy().flatten(), Cov1.detach().numpy(), 1).T
        Z2  = np.random.multivariate_normal(mean2.detach().numpy().flatten(), Cov2.detach().numpy(), 1).T

        # Find the full pareto front.
        lam = np.random.dirichlet(alphas,1)

        # solve for the point that maximizes our objective function
        # !!! Want min epsilon, so negative for Z1
        pt = np.argmax(-lam[0,0]*Z1 + lam[0,1]*Z2)
        pt = unmeasured_points[pt]
        measured_points = np.append(measured_points, pt)

        accuracy, privacy_engine, predicted_classes = dp_target(Xg[pt,0], Xg[pt,1], model_class, X_test, train_dataset, y_test)
        epsilon = privacy_engine.get_epsilon(delta)
        
        epsilon_.append(epsilon)
        accuracy_.append(accuracy)
        X_params = np.vstack((X_params, np.asarray([Xg[pt,0], Xg[pt,1]])[None,:]))
        Y_params = np.vstack((Y_params, np.asarray([epsilon,accuracy])))
        print('epsilon', epsilon)
    return measured_points, epsilon_, accuracy_
  
  
def dp_hyper(model_class, train_dataset, test_dataset, seed=42):
    # 1) Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Seed everything for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    def train_with_privacy(noise_multiplier, batch_size, learning_rate, clipping_norm):
        batch_size = int(round(batch_size))

        # instantiate and move model to device
        model     = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # sampler + data loader with seeded generator
        sample_rate = batch_size / len(train_dataset)
        sampler = UniformWithReplacementSampler(
            num_samples=len(train_dataset),
            sample_rate=sample_rate,
        )

        gen = torch.Generator()
        gen.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            generator=gen,
        )

        # hook up DP
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=clipping_norm,
        )

        # training loop (single epoch; repeat if you want multiple)
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

        epsilon_spent = privacy_engine.accountant.get_epsilon(delta=1e-5)
        return model, epsilon_spent

    def evaluate_model(model):
        model.eval()
        correct, total = 0, 0

        gen = torch.Generator()
        gen.manual_seed(seed)

        eval_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            generator=gen,
        )

        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                pred = outputs.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return 100.0 * correct / total

    def bo_objective(noise_multiplier, batch_size, learning_rate, clipping_norm):
        model, eps = train_with_privacy(noise_multiplier, batch_size, learning_rate, clipping_norm)
        print(f"Epsilon spent: {eps:.3f}")
        if eps > 5.0:
            return 0.0

        acc = evaluate_model(model)
        print(f"Accuracy: {acc:.2f}%")
        if acc < 85.0:
            return 0.0

        return acc

    # bounds
    pbounds = {
        'noise_multiplier': (0.5, 3.0),
        'batch_size':       (16, 128),
        'learning_rate':    (1e-5, 1e-2),
        'clipping_norm':    (0.1, 2.0),
    }

    optimizer = BayesianOptimization(f=bo_objective, pbounds=pbounds, random_state=seed, verbose=2)
    optimizer.maximize(init_points=5, n_iter=15)

    best_params   = optimizer.max['params']
    best_accuracy = optimizer.max['target']
    
    return best_params, best_accuracy
