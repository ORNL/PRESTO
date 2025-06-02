from setuptools import setup, find_packages

setup(
    name="presto",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "diffprivlib",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "lifelines",
        "statsmodels"
    ],
    author="Olivera Kotevska",
    author_email="kotevskao@ornl.gov",
    description="A Python package for recomendation tool",
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
