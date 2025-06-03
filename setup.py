from setuptools import setup, find_packages

setup(
    name="PRESTO",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
          "math",
          "torch",
          "random",
          "numpy",
          "seaborn",
          "pandas",
          "hashlib",
          "scipy",
          "matplotlib",
          "bayes_opt",
          "GPy",
          "gpytorch",
          "sklearn",
          "opacus"
    ],
    author="Olivera Kotevska",
    author_email="kotevskao@ornl.gov",
    description="A Python package for privacy preservation algorithm recomendation",
    long_description=open("README.md").read(),
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
    python_requires=">=3.7",
    include_package_data=True,
)
