"""
Tests for PRESTO configuration management.
"""

import pytest
import tempfile
import os
from ornl_presto.config import (
    PRESTOConfig,
    PrivacyConfig,
    ConfigManager,
    get_domain_recommendations,
)


class TestPRESTOConfig:
    """Test PRESTO configuration classes."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = PRESTOConfig()

        assert config.privacy.epsilon_min == 0.1
        assert config.privacy.epsilon_max == 10.0
        assert config.optimization.n_evals == 10
        assert config.random_seed == 42
        assert config.verbose is True

    def test_privacy_config_customization(self):
        """Test privacy configuration customization."""
        privacy_config = PrivacyConfig(
            epsilon_min=0.01,
            epsilon_max=1.0,
            required_algorithms=["gaussian"],
            utility_threshold=0.9,
        )

        assert privacy_config.epsilon_min == 0.01
        assert privacy_config.epsilon_max == 1.0
        assert privacy_config.required_algorithms == ["gaussian"]
        assert privacy_config.utility_threshold == 0.9


class TestConfigManager:
    """Test configuration manager functionality."""

    def test_list_configs(self):
        """Test listing available configurations."""
        configs = ConfigManager.list_configs()

        assert isinstance(configs, list)
        assert len(configs) > 0
        assert "healthcare" in configs
        assert "finance" in configs
        assert "research" in configs

    def test_get_healthcare_config(self):
        """Test getting healthcare configuration."""
        config = ConfigManager.get_config("healthcare")

        assert config.privacy.epsilon_min == 0.01
        assert config.privacy.epsilon_max == 1.0
        assert config.privacy.utility_threshold == 0.9
        assert "gaussian" in config.privacy.required_algorithms

    def test_get_research_config(self):
        """Test getting research configuration."""
        config = ConfigManager.get_config("research")

        assert config.privacy.epsilon_min == 0.5
        assert config.privacy.epsilon_max == 10.0
        assert config.privacy.utility_threshold == 0.7

    def test_invalid_config_name(self):
        """Test error handling for invalid config names."""
        with pytest.raises(ValueError):
            ConfigManager.get_config("nonexistent_config")

    def test_save_and_load_config(self):
        """Test saving and loading configuration to/from file."""
        # Create a test config
        config = ConfigManager.get_config("healthcare")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            ConfigManager.save_config(config, temp_path)

            # Load the config back
            loaded_config = ConfigManager.load_config(temp_path)

            # Verify key properties match
            assert loaded_config.privacy.epsilon_min == config.privacy.epsilon_min
            assert loaded_config.privacy.epsilon_max == config.privacy.epsilon_max
            assert loaded_config.optimization.n_evals == config.optimization.n_evals
            assert loaded_config.random_seed == config.random_seed

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDomainRecommendations:
    """Test domain-specific recommendations."""

    def test_get_domain_recommendations(self):
        """Test getting domain recommendations."""
        recommendations = get_domain_recommendations()

        assert isinstance(recommendations, dict)
        assert "healthcare" in recommendations
        assert "finance" in recommendations

        # Check healthcare recommendations
        healthcare = recommendations["healthcare"]
        assert "description" in healthcare
        assert "privacy_level" in healthcare
        assert "recommended_epsilon" in healthcare
        assert "key_considerations" in healthcare
        assert isinstance(healthcare["key_considerations"], list)

    def test_domain_recommendation_structure(self):
        """Test structure of domain recommendations."""
        recommendations = get_domain_recommendations()

        for domain, info in recommendations.items():
            assert "description" in info
            assert "privacy_level" in info
            assert "recommended_epsilon" in info
            assert "key_considerations" in info
            assert isinstance(info["key_considerations"], list)
            assert len(info["key_considerations"]) > 0


def test_config_integration():
    """Test configuration integration with different scenarios."""

    # Test production-fast config
    fast_config = ConfigManager.get_config("production_fast")
    assert fast_config.optimization.n_evals == 5
    assert fast_config.optimization.n_iter == 8
    assert fast_config.parallel_workers == 4

    # Test development config
    dev_config = ConfigManager.get_config("development")
    assert dev_config.optimization.n_evals == 3
    assert dev_config.verbose is True

    # Test IoT config
    iot_config = ConfigManager.get_config("iot_sensors")
    assert iot_config.privacy.epsilon_min == 1.0
    assert "count_mean_sketch" in iot_config.privacy.required_algorithms


if __name__ == "__main__":
    pytest.main([__file__])
