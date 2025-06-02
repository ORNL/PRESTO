---
title: 'presto: A Python package for recomending privacy preservation algorithm based on user preferences.'
tags:
  - Python
  - differential privacy
  - security
  - optimization
  - machine learning
  - privacy-preserving
authors:
  - name: Olivera Kotevska
    orcid: 0000-0003-1677-2243
    affiliation: 1
  - name: A. Gilad Kusne
    orcid: 0000-0001-8904-2087
    affiliation: 2
  - name: Prasanna Balaprakash
    orcid: 0000-0002-0292-5715
    affiliation: 1
  - name: Robert Patton
    orcid: 0000-0002-8101-0571
    affiliation: 1
affiliations:
  - name: Oak Ridge National Laboratory, United States
    index: 1
  - name: National Institute of Standards and Technology, United States
    index: 1
date: 26 May 2025
bibliography: paper.bib
---

# Summary
PRESTO (Privacy REcommendation and SecuriTy Optimization) is a Python-based toolkit that automates the selection of differential-privacy mechanisms to balance data utility and privacy loss. By integrating descriptive and inferential statistics, Bayesian optimization, and data-similarity metrics, PRESTO analyzes arbitrary datasets—numerical, categorical, or structured—and recommends the optimal privacy algorithm and ε-parameter setting. Its modular design supports CPU/GPU execution, streaming and batch data, and extensibility for new algorithms and utility metrics. PRESTO’s automated multi-objective optimization delivers application-specific, data-driven recommendations with quantified uncertainty, empowering both experts and non-experts to integrate privacy-preserving methods seamlessly into their workflows.

# Statement of Need
As data collection proliferates across healthcare, finance, IoT, and beyond, safeguarding individual privacy without handicapping downstream analytics has become critical. Existing differential-privacy tools often require deep theoretical knowledge, manual tuning of privacy parameters, and trial-and-error to discover the right trade-off between noise injection and data utility. This steep adoption barrier impedes widespread deployment of privacy-preserving analytics in industrial and research settings. There is a pressing need for an intuitive, automated solution that can—given any dataset—identify the most suitable privacy mechanism and its optimal ε, quantify the remaining utility, and provide confidence intervals on its recommendations. PRESTO fills this gap, reducing the technical burden and accelerating safe, compliant data analysis.

# State of the Field
A variety of packages—such as IBM’s Diffprivlib, Google’s PyDP (and TensorFlow Privacy), Facebook’s Opacus, and LDP-Pure—offer implementations of noise-based DP mechanisms (Laplace, Gaussian, Exponential), local-DP protocols (Randomized Response, RAPPOR), and gradient perturbation for machine learning. However, they typically expose raw APIs, leaving users responsible for selecting and tuning algorithms, and provide limited guidance on choosing ε. Recent research has explored automatic hyperparameter tuning via cross-validation or surrogate modeling, but these approaches rarely integrate multi-objective optimization or deliver quantitative uncertainty measures.

PRESTO advances the state of the art by unifying statistical dataset analysis, Bayesian optimization, and data-similarity metrics into a single recommendation engine. It implements a broad suite of privacy mechanisms—including both batch and streaming algorithms—and automates their selection based on data characteristics and user-specified privacy–utility trade-offs, while providing 95% confidence intervals on its recommendations. Crucially, PRESTO is built on a modular architecture, enabling users to plug in new privacy algorithms or custom utility metrics at any time without modifying core logic. This extensibility ensures that PRESTO can evolve alongside emerging research and domain-specific needs, making it uniquely adaptable compared to existing static libraries.

# Methodology
1. **Dataset Profiling**  
   - Compute descriptive (mean, variance, skewness, kurtosis) and, for categorical data, domain‐size and frequency distributions.  
2. **Mechanism Library**  
   - Maintain a dictionary of privacy functions (`get_noise_generators()`), each mapping `(data, ε) → privatized_data`.  
3. **Bayesian Optimization of ε**  
   - For each mechanism, define  
     ```math
       f(ε) = -\mathrm{RMSE}\bigl(\text{data},\,\text{mechanism}_ε(\text{data})\bigr)
     ```  
     and maximize over ε∈[εₘᵢₙ,εₘₐₓ] using Gaussian‐process Bayesian optimization.  
4. **Confidence & Reliability**  
   - Compute 95% CI on RMSE at ε*; define  
     ```math
       \text{Reliability} = \frac{1}{\text{Mean RMSE} \times \text{CI Width}}.
     ```  
5. **Similarity Assessment**  
   - Measure distributional similarity via Kolmogorov–Smirnov, Jensen–Shannon, Pearson correlation.  
6. **Multi‐Objective Ranking**  
   - Recommend top mechanisms on **max similarity**, **max reliability**, and **max privacy** axes.

# Experiments
We conducted experiments to evaluate the effectiveness of our approach.

### Energy Time‐Series with Bayesian Optimization (Dataset: Hourly Consumption (Min))
1. Privacy loss (epsilon) vs utility (RMSE) for selected/prefered privacy algorithms ![Feature MAE comparison for MIC-DP vs. state-of-art approaches](energy.png)

### Medical Time‐Series with Bayesian Optimization (Dataset: Heart Rate (Min))
1. Privacy loss (epsilon) vs utility (RMSE) for selected/prefered privacy algorithms ![Feature MAE comparison for MIC-DP vs. state-of-art approaches](medical.png)

### Finance Time‐Series with Bayesian Optimization (Dataset: Payment Transactions (Min))
1. Privacy loss (epsilon) vs utility (RMSE) for selected/prefered privacy algorithms ![Feature MAE comparison for MIC-DP vs. state-of-art approaches](finance.png)

### Energy Consumption with Fixed epsilon = 1
Best by Similarity:  {'algorithm': 'PercentilePrivacy', 'score': np.float32(0.9841)}
Best by Reliability: {'algorithm': 'PercentilePrivacy', 'score': inf}
Best by Privacy:     {'algorithm': 'Hadamard_Mechanism', 'score': 71.6581}

### ML Classification with Private Gradients
Baseline Accuracy (no privacy): 93.00%
DP Accuracy with 'PercentilePrivacy': 94.00%

# Conclusion
PRESTO delivers a data-driven, automated, and extensible framework for differential-privacy mechanism selection and tuning. By profiling statistical properties, optimizing ε via Bayesian methods, and quantifying both utility and uncertainty, PRESTO guides users to the privacy solution best suited for their data. Its modular design allows seamless integration of new algorithms and metrics, positioning PRESTO as a flexible platform for both practitioners and researchers aiming to embed privacy guarantees in diverse analytical workflows.

# Acknowledgements
This manuscript has been authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).
