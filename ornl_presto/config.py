# Configuration Management for PRESTO

"""
Configuration management system for PRESTO privacy recommendations.
Provides standardized configurations for different use cases and deployment scenarios.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import json


@dataclass
class PrivacyConfig:
    """Configuration for privacy requirements."""

    epsilon_min: float = 0.1
    epsilon_max: float = 10.0
    delta: float = 1e-5
    required_algorithms: List[str] = field(
        default_factory=lambda: ["gaussian", "laplace"]
    )
    excluded_algorithms: List[str] = field(default_factory=list)
    utility_threshold: float = 0.8  # Minimum acceptable utility preservation


@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization."""

    n_evals: int = 10
    init_points: int = 3
    n_iter: int = 15
    acquisition_function: str = "ucb"
    kappa: float = 2.576  # For UCB acquisition
    xi: float = 0.01  # For EI acquisition


@dataclass
class DataConfig:
    """Configuration for data handling."""

    standardize: bool = True
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 3.0
    min_data_size: int = 50
    max_data_size: int = 100000


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = "seaborn"
    color_palette: str = "husl"
    save_plots: bool = True
    plot_directory: str = "presto_plots"


@dataclass
class PRESTOConfig:
    """Main PRESTO configuration container."""

    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    random_seed: int = 42
    verbose: bool = True
    parallel_workers: int = 1
    gpu_acceleration: bool = False


class ConfigManager:
    """Manages PRESTO configurations for different use cases."""

    PREDEFINED_CONFIGS = {
        "healthcare": {
            "privacy": {
                "epsilon_min": 0.01,
                "epsilon_max": 1.0,
                "delta": 1e-6,
                "required_algorithms": ["gaussian", "laplace"],
                "utility_threshold": 0.9,
            },
            "optimization": {"n_evals": 15, "init_points": 5, "n_iter": 25},
            "data": {
                "handle_outliers": True,
                "outlier_method": "iqr",
                "standardize": True,
            },
        },
        "finance": {
            "privacy": {
                "epsilon_min": 0.1,
                "epsilon_max": 5.0,
                "delta": 1e-5,
                "required_algorithms": ["gaussian", "laplace", "exponential"],
                "utility_threshold": 0.85,
            },
            "optimization": {"n_evals": 12, "init_points": 4, "n_iter": 20},
        },
        "research": {
            "privacy": {
                "epsilon_min": 0.5,
                "epsilon_max": 10.0,
                "delta": 1e-4,
                "utility_threshold": 0.7,
            },
            "optimization": {"n_evals": 20, "init_points": 6, "n_iter": 30},
        },
        "iot_sensors": {
            "privacy": {
                "epsilon_min": 1.0,
                "epsilon_max": 20.0,
                "required_algorithms": ["laplace", "count_mean_sketch"],
                "utility_threshold": 0.75,
            },
            "data": {
                "handle_outliers": True,
                "outlier_method": "zscore",
                "max_data_size": 50000,
            },
        },
        "survey_data": {
            "privacy": {
                "epsilon_min": 0.1,
                "epsilon_max": 2.0,
                "required_algorithms": ["laplace", "rappor", "hadamard_response"],
                "utility_threshold": 0.8,
            }
        },
        "production_fast": {
            "privacy": {
                "epsilon_min": 1.0,
                "epsilon_max": 5.0,
                "required_algorithms": ["laplace", "gaussian"],
                "utility_threshold": 0.8,
            },
            "optimization": {"n_evals": 5, "init_points": 2, "n_iter": 8},
            "parallel_workers": 4,
        },
        "development": {
            "privacy": {
                "epsilon_min": 0.1,
                "epsilon_max": 10.0,
                "utility_threshold": 0.6,
            },
            "optimization": {"n_evals": 3, "init_points": 2, "n_iter": 5},
            "verbose": True,
        },
    }

    @classmethod
    def get_config(cls, config_name: str) -> PRESTOConfig:
        """Get a predefined configuration by name."""
        if config_name not in cls.PREDEFINED_CONFIGS:
            available = list(cls.PREDEFINED_CONFIGS.keys())
            raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

        # Start with default config
        config = PRESTOConfig()

        # Apply predefined settings
        predefined = cls.PREDEFINED_CONFIGS[config_name]

        # Update privacy settings
        if "privacy" in predefined and isinstance(predefined["privacy"], dict):
            for key, value in predefined["privacy"].items():
                setattr(config.privacy, key, value)

        # Update optimization settings
        if "optimization" in predefined and isinstance(
            predefined["optimization"], dict
        ):
            for key, value in predefined["optimization"].items():
                setattr(config.optimization, key, value)

        # Update data settings
        if "data" in predefined and isinstance(predefined["data"], dict):
            for key, value in predefined["data"].items():
                setattr(config.data, key, value)

        # Update visualization settings
        if "visualization" in predefined and isinstance(
            predefined["visualization"], dict
        ):
            for key, value in predefined["visualization"].items():
                setattr(config.visualization, key, value)

        # Update top-level settings
        for key in ["random_seed", "verbose", "parallel_workers", "gpu_acceleration"]:
            if isinstance(predefined, dict) and key in predefined:
                setattr(config, key, predefined[key])

        return config

    @classmethod
    def list_configs(cls) -> List[str]:
        """List all available predefined configurations."""
        return list(cls.PREDEFINED_CONFIGS.keys())

    @classmethod
    def save_config(cls, config: PRESTOConfig, filepath: str):
        """Save configuration to a JSON file."""
        config_dict = {
            "privacy": {
                "epsilon_min": config.privacy.epsilon_min,
                "epsilon_max": config.privacy.epsilon_max,
                "delta": config.privacy.delta,
                "required_algorithms": config.privacy.required_algorithms,
                "excluded_algorithms": config.privacy.excluded_algorithms,
                "utility_threshold": config.privacy.utility_threshold,
            },
            "optimization": {
                "n_evals": config.optimization.n_evals,
                "init_points": config.optimization.init_points,
                "n_iter": config.optimization.n_iter,
                "acquisition_function": config.optimization.acquisition_function,
                "kappa": config.optimization.kappa,
                "xi": config.optimization.xi,
            },
            "data": {
                "standardize": config.data.standardize,
                "handle_outliers": config.data.handle_outliers,
                "outlier_method": config.data.outlier_method,
                "outlier_threshold": config.data.outlier_threshold,
                "min_data_size": config.data.min_data_size,
                "max_data_size": config.data.max_data_size,
            },
            "visualization": {
                "figure_size": config.visualization.figure_size,
                "dpi": config.visualization.dpi,
                "style": config.visualization.style,
                "color_palette": config.visualization.color_palette,
                "save_plots": config.visualization.save_plots,
                "plot_directory": config.visualization.plot_directory,
            },
            "random_seed": config.random_seed,
            "verbose": config.verbose,
            "parallel_workers": config.parallel_workers,
            "gpu_acceleration": config.gpu_acceleration,
        }

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, filepath: str) -> PRESTOConfig:
        """Load configuration from a JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)

        config = PRESTOConfig()

        # Load privacy config
        if "privacy" in config_dict:
            for key, value in config_dict["privacy"].items():
                setattr(config.privacy, key, value)

        # Load optimization config
        if "optimization" in config_dict:
            for key, value in config_dict["optimization"].items():
                setattr(config.optimization, key, value)

        # Load data config
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                setattr(config.data, key, value)

        # Load visualization config
        if "visualization" in config_dict:
            for key, value in config_dict["visualization"].items():
                if key == "figure_size":
                    value = tuple(value)
                setattr(config.visualization, key, value)

        # Load top-level settings
        for key in ["random_seed", "verbose", "parallel_workers", "gpu_acceleration"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config


def get_domain_recommendations() -> Dict[str, Dict[str, Any]]:
    """Get domain-specific recommendations for PRESTO configuration."""
    return {
        "healthcare": {
            "description": "HIPAA-compliant privacy for medical data",
            "privacy_level": "Very High",
            "recommended_epsilon": "0.01 - 0.5",
            "key_considerations": [
                "Patient privacy is paramount",
                "Regulatory compliance required",
                "High utility preservation needed",
                "Prefer Gaussian/Laplace mechanisms",
            ],
        },
        "finance": {
            "description": "Financial data with regulatory requirements",
            "privacy_level": "High",
            "recommended_epsilon": "0.1 - 2.0",
            "key_considerations": [
                "Regulatory compliance (GDPR, CCPA)",
                "Balance utility and privacy",
                "Multiple algorithm evaluation",
                "Audit trail important",
            ],
        },
        "research": {
            "description": "Academic research with IRB approval",
            "privacy_level": "Medium-High",
            "recommended_epsilon": "0.5 - 5.0",
            "key_considerations": [
                "IRB/ethics approval typically required",
                "Publication of methods necessary",
                "Reproducibility important",
                "Broader algorithm exploration",
            ],
        },
        "iot_sensors": {
            "description": "IoT sensor data and telemetry",
            "privacy_level": "Medium",
            "recommended_epsilon": "1.0 - 10.0",
            "key_considerations": [
                "Large data volumes",
                "Real-time processing needs",
                "Computational efficiency",
                "Geographic privacy concerns",
            ],
        },
        "survey_data": {
            "description": "Survey responses and questionnaires",
            "privacy_level": "Medium-High",
            "recommended_epsilon": "0.1 - 2.0",
            "key_considerations": [
                "Respondent anonymity",
                "Categorical data handling",
                "Local differential privacy",
                "Response bias minimization",
            ],
        },
    }


def print_config_guide():
    """Print a guide for choosing PRESTO configurations."""
    print("PRESTO Configuration Guide")
    print("=" * 50)
    print()

    print("Available Configurations:")
    for config_name in ConfigManager.list_configs():
        print(f"  • {config_name}")
    print()

    print("Domain-Specific Recommendations:")
    recommendations = get_domain_recommendations()

    for domain, info in recommendations.items():
        print(f"\n{domain.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Privacy Level: {info['privacy_level']}")
        print(f"  Recommended ε: {info['recommended_epsilon']}")
        print("  Key Considerations:")
        for consideration in info["key_considerations"]:
            print(f"    - {consideration}")

    print("\nUsage Examples:")
    print("  config = ConfigManager.get_config('healthcare')")
    print("  config = ConfigManager.get_config('research')")
    print("  ConfigManager.save_config(config, 'my_config.json')")
    print("  config = ConfigManager.load_config('my_config.json')")


if __name__ == "__main__":
    print_config_guide()
