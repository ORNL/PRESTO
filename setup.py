from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="ornl_presto",
    version="0.1.28",
    packages=["ornl_presto"],  
    install_requires=[
        "torch",
        "numpy",
        "seaborn",
        "pandas",
        "scipy",
        "matplotlib",
        "bayesian-optimization",
        "gpytorch",
        "scikit-learn",
        "opacus"
    ],
    author="Olivera Kotevska",
    author_email="kotevskao@ornl.gov",
    description="A Python package for privacy preservation algorithm recommendation",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    keywords="differential-privacy privacy-preserving machine-learning security optimization",
    url="https://github.com/OKotevska/PRESTO/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "Development Status :: 4 - Beta"
    ],
    include_package_data=True,
    python_requires=">=3.7",
)
