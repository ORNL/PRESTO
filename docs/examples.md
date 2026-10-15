# Examples Gallery

Explore real-world applications of PRESTO across different domains and use cases.

## Healthcare & Life Sciences

### Clinical Trial Data Privacy
Protect patient data while preserving clinical efficacy signals.

```python
from ornl_presto import recommend_top3
from ornl_presto.config import ConfigManager
import torch

# Load clinical trial data (simulated)
patient_outcomes = torch.tensor([
    # Treatment group outcomes
    85.2, 82.1, 88.5, 90.3, 87.8, 85.9, 89.1, 86.4,
    # Control group outcomes
    78.3, 76.8, 80.1, 77.9, 79.5, 81.2, 78.8, 80.0
])

# Use healthcare configuration
config = ConfigManager.get_config("healthcare")

# Get privacy recommendations
recommendations = recommend_top3(
    patient_outcomes,
    n_evals=config.optimization.n_evals
)

# Apply strict healthcare privacy requirements
healthcare_recs = [
    rec for rec in recommendations
    if rec['epsilon'] <= 1.0 and rec['utility_preservation'] >= 0.9
]

if healthcare_recs:
    best = healthcare_recs[0]
    print(f"Healthcare-compliant recommendation: {best['algorithm']}")
    print(f"Privacy level: ε = {best['epsilon']:.3f}")
    print(f"Utility preservation: {best['utility_preservation']:.1%}")
else:
    print("No recommendations meet strict healthcare requirements")
```

### Genomic Data Sharing
Protect individual genomic information while enabling population studies.

```python
# Simulate allele frequency data
import numpy as np
np.random.seed(42)

# Population allele frequencies (1000 SNPs)
allele_frequencies = np.random.beta(0.5, 2, 1000)  # Realistic MAF distribution
genomic_data = torch.tensor(allele_frequencies, dtype=torch.float32)

# Genomic privacy requirements
genomic_recommendations = recommend_top3(
    genomic_data,
    epsilon_range=(0.01, 0.5),  # Very strict for genomic data
    n_evals=15
)

# Evaluate Hardy-Weinberg equilibrium preservation
from ornl_presto import get_noise_generators

best_genomic = genomic_recommendations[0]
noise_fn = get_noise_generators()[best_genomic['algorithm']]
private_frequencies = noise_fn(genomic_data, best_genomic['epsilon'])

# Check statistical properties preservation
original_mean_af = genomic_data.mean()
private_mean_af = private_frequencies.mean()

print(f"Original mean allele frequency: {original_mean_af:.4f}")
print(f"Private mean allele frequency: {private_mean_af:.4f}")
print(f"Difference: {abs(original_mean_af - private_mean_af):.4f}")
```

### Electronic Health Records
Privacy-preserving analysis of medical records.

```python
# Simulate EHR vital signs data
vital_signs = {
    'heart_rate': torch.normal(72, 12, (500,)),
    'systolic_bp': torch.normal(120, 15, (500,)),
    'temperature': torch.normal(98.6, 1.0, (500,))
}

ehr_results = {}

for vital_type, data in vital_signs.items():
    # Clip to physiologically reasonable ranges
    if vital_type == 'heart_rate':
        data = torch.clamp(data, 50, 120)
    elif vital_type == 'systolic_bp':
        data = torch.clamp(data, 90, 180)
    elif vital_type == 'temperature':
        data = torch.clamp(data, 96, 102)

    recommendations = recommend_top3(data, n_evals=10)
    ehr_results[vital_type] = recommendations[0]

    print(f"{vital_type}: {recommendations[0]['algorithm']} "
          f"(ε={recommendations[0]['epsilon']:.3f})")
```

## Financial Services

### Transaction Monitoring
Detect fraud patterns while protecting customer privacy.

```python
# Simulate transaction amounts
transaction_amounts = torch.tensor([
    # Normal transactions
    *np.random.lognormal(3, 1.5, 900),  # $20-$8000 range
    # Suspicious large transactions
    *np.random.uniform(10000, 50000, 100)
])

# Financial regulatory requirements
finance_config = ConfigManager.get_config("finance")
fin_recommendations = recommend_top3(
    transaction_amounts,
    n_evals=finance_config.optimization.n_evals
)

# Ensure fraud detection capability is preserved
best_finance = fin_recommendations[0]
print(f"Financial recommendation: {best_finance['algorithm']}")
print(f"Regulatory privacy level: ε = {best_finance['epsilon']:.3f}")

# Apply privacy and check fraud detection preservation
private_transactions = get_noise_generators()[best_finance['algorithm']](
    transaction_amounts, best_finance['epsilon']
)

# Simple fraud threshold detection
fraud_threshold = 10000
original_fraud_rate = (transaction_amounts > fraud_threshold).float().mean()
private_fraud_rate = (private_transactions > fraud_threshold).float().mean()

print(f"Original fraud detection rate: {original_fraud_rate:.1%}")
print(f"Private fraud detection rate: {private_fraud_rate:.1%}")
print(f"Detection preservation: {1 - abs(original_fraud_rate - private_fraud_rate)/original_fraud_rate:.1%}")
```

### Credit Risk Assessment
Privacy-preserving credit scoring.

```python
# Simulate credit scores
credit_scores = torch.normal(680, 120, (1000,))
credit_scores = torch.clamp(credit_scores, 300, 850)  # FICO range

# Different privacy levels for different use cases
privacy_scenarios = {
    'internal_analytics': {'epsilon_max': 2.0, 'utility_min': 0.9},
    'third_party_sharing': {'epsilon_max': 0.5, 'utility_min': 0.8},
    'regulatory_reporting': {'epsilon_max': 0.1, 'utility_min': 0.7}
}

for scenario, requirements in privacy_scenarios.items():
    recs = recommend_top3(
        credit_scores,
        epsilon_range=(0.01, requirements['epsilon_max'])
    )

    suitable_recs = [
        r for r in recs
        if r['utility_preservation'] >= requirements['utility_min']
    ]

    if suitable_recs:
        best = suitable_recs[0]
        print(f"{scenario}: {best['algorithm']} "
              f"(ε={best['epsilon']:.3f}, utility={best['utility_preservation']:.1%})")
    else:
        print(f"{scenario}: No suitable recommendations found")
```

## Research & Academia

### Survey Data Protection
Protect survey respondent privacy while enabling statistical analysis.

```python
# Simulate Likert scale survey responses (1-7 scale)
survey_responses = torch.randint(1, 8, (500,), dtype=torch.float32)

# Add realistic response patterns
# Slight positive bias (people tend to respond 4-6)
bias_mask = torch.rand(500) < 0.7
survey_responses[bias_mask] = torch.randint(4, 7, (bias_mask.sum(),), dtype=torch.float32)

research_config = ConfigManager.get_config("research")
survey_recommendations = recommend_top3(
    survey_responses,
    n_evals=research_config.optimization.n_evals
)

# Check statistical significance preservation
from scipy import stats

best_survey = survey_recommendations[0]
private_responses = get_noise_generators()[best_survey['algorithm']](
    survey_responses, best_survey['epsilon']
)

# T-test against neutral (4.0)
original_tstat, original_pval = stats.ttest_1samp(survey_responses.numpy(), 4.0)
private_tstat, private_pval = stats.ttest_1samp(private_responses.numpy(), 4.0)

print(f"Original: t={original_tstat:.3f}, p={original_pval:.3f}")
print(f"Private: t={private_tstat:.3f}, p={private_pval:.3f}")
print(f"Statistical significance preserved: {(original_pval < 0.05) == (private_pval < 0.05)}")
```

### Educational Analytics
Protect student privacy in learning analytics.

```python
# Simulate student performance data
student_grades = torch.normal(75, 15, (200,))  # Grade distribution
student_grades = torch.clamp(student_grades, 0, 100)

# Educational research requirements
education_recommendations = recommend_top3(
    student_grades,
    epsilon_range=(0.5, 5.0),  # Moderate privacy for research
    n_evals=12
)

# Preserve grade distribution characteristics
best_edu = education_recommendations[0]
private_grades = get_noise_generators()[best_edu['algorithm']](
    student_grades, best_edu['epsilon']
)

print(f"Educational analytics: {best_edu['algorithm']}")
print(f"Original grade average: {student_grades.mean():.1f}")
print(f"Private grade average: {private_grades.mean():.1f}")
print(f"Distribution preservation: {1 - abs(student_grades.std() - private_grades.std())/student_grades.std():.1%}")
```

## IoT & Smart Systems

### Smart City Sensor Data
Protect individual location privacy while enabling urban analytics.

```python
# Simulate sensor readings (e.g., air quality, traffic flow)
sensor_data = {
    'air_quality_aqi': torch.randint(50, 150, (1000,), dtype=torch.float32),
    'traffic_count': torch.randint(10, 200, (1000,), dtype=torch.float32),
    'noise_level_db': torch.normal(60, 10, (1000,)),
}

iot_config = ConfigManager.get_config("iot_sensors")

for sensor_type, data in sensor_data.items():
    iot_recs = recommend_top3(
        data,
        epsilon_range=(1.0, iot_config.privacy.epsilon_max),
        n_evals=8
    )

    best_iot = iot_recs[0]
    print(f"{sensor_type}: {best_iot['algorithm']} "
          f"(ε={best_iot['epsilon']:.3f})")

    # Check temporal pattern preservation for time series
    if sensor_type == 'traffic_count':
        private_traffic = get_noise_generators()[best_iot['algorithm']](
            data, best_iot['epsilon']
        )

        # Simple trend analysis
        original_trend = torch.diff(data[:100])  # First 100 readings
        private_trend = torch.diff(private_traffic[:100])

        trend_correlation = torch.corrcoef(torch.stack([original_trend, private_trend]))[0,1]
        print(f"  Trend preservation: {trend_correlation:.3f}")
```

### Energy Grid Monitoring
Privacy-preserving smart meter data analysis.

```python
# Simulate hourly energy consumption (kWh)
daily_pattern = torch.sin(torch.linspace(0, 2*3.14159, 24)) * 2 + 5  # Base daily pattern
noise = torch.randn(24) * 0.5
energy_consumption = daily_pattern + noise
energy_consumption = torch.clamp(energy_consumption, 0, 20)  # Realistic range

# Energy sector privacy requirements
energy_recommendations = recommend_top3(
    energy_consumption,
    epsilon_range=(0.5, 3.0),  # Moderate privacy for grid operations
    n_evals=10
)

best_energy = energy_recommendations[0]
private_consumption = get_noise_generators()[best_energy['algorithm']](
    energy_consumption, best_energy['epsilon']
)

# Preserve peak/off-peak patterns
original_peak_hour = torch.argmax(energy_consumption)
private_peak_hour = torch.argmax(private_consumption)

print(f"Energy monitoring: {best_energy['algorithm']}")
print(f"Original peak hour: {original_peak_hour.item()}")
print(f"Private peak hour: {private_peak_hour.item()}")
print(f"Peak time preserved: {abs(original_peak_hour - private_peak_hour) <= 1}")
```

## Advanced Use Cases

### Multi-Domain Data Fusion
Combine data from multiple domains with appropriate privacy levels.

```python
# Multi-domain dataset
domains = {
    'healthcare': torch.randn(200),      # Strict privacy
    'finance': torch.randn(300),         # Moderate privacy
    'research': torch.randn(500),        # Flexible privacy
}

domain_configs = {
    'healthcare': {'epsilon_max': 0.5, 'utility_min': 0.9},
    'finance': {'epsilon_max': 1.0, 'utility_min': 0.8},
    'research': {'epsilon_max': 5.0, 'utility_min': 0.7}
}

fusion_results = {}

for domain, data in domains.items():
    config = domain_configs[domain]

    recs = recommend_top3(
        data,
        epsilon_range=(0.01, config['epsilon_max']),
        n_evals=10
    )

    suitable = [r for r in recs if r['utility_preservation'] >= config['utility_min']]

    if suitable:
        fusion_results[domain] = suitable[0]
        print(f"{domain}: {suitable[0]['algorithm']} "
              f"(ε={suitable[0]['epsilon']:.3f})")

# Combine private data from all domains
combined_private_data = []
for domain, result in fusion_results.items():
    private_data = get_noise_generators()[result['algorithm']](
        domains[domain], result['epsilon']
    )
    combined_private_data.append(private_data)

total_private_dataset = torch.cat(combined_private_data)
print(f"Combined dataset size: {len(total_private_dataset)}")
```

### Federated Learning with Privacy
Coordinate privacy across distributed learning.

```python
from ornl_presto import dp_function_train_and_pred

# Simulate federated learning scenario
federation_sites = ['hospital_a', 'hospital_b', 'hospital_c']
site_data_sizes = [200, 150, 300]

# Calculate privacy budget allocation
total_privacy_budget = 1.0
site_privacy_budgets = [
    total_privacy_budget * (size / sum(site_data_sizes))
    for size in site_data_sizes
]

print("Federated Privacy Budget Allocation:")
for site, budget in zip(federation_sites, site_privacy_budgets):
    print(f"  {site}: ε = {budget:.3f}")

# Simulate training at each site
federated_accuracies = []
for site, budget in zip(federation_sites, site_privacy_budgets):
    # Convert privacy budget to noise multiplier (approximate)
    noise_multiplier = 1.0 / budget

    print(f"Training at {site} with noise_multiplier = {noise_multiplier:.2f}")
    # Note: This would use actual federated learning in practice
```

### Real-Time Privacy Streaming
Handle continuous data streams with privacy.

```python
# Simulate streaming data (e.g., IoT sensor readings)
def generate_stream_batch(batch_size=100):
    return torch.randn(batch_size) * 10 + 50  # Temperature-like readings

# Real-time privacy parameters
stream_config = {
    'batch_size': 100,
    'privacy_budget_per_batch': 0.1,
    'total_budget': 10.0,
}

batches_processed = 0
remaining_budget = stream_config['total_budget']

print("Real-time streaming privacy processing:")

while remaining_budget > stream_config['privacy_budget_per_batch'] and batches_processed < 5:
    # Get new data batch
    batch_data = generate_stream_batch(stream_config['batch_size'])

    # Quick privacy recommendation (reduced optimization for speed)
    batch_recs = recommend_top3(
        batch_data,
        epsilon_range=(0.05, min(remaining_budget, 1.0)),
        n_evals=3,  # Fast recommendation
        n_iter=2
    )

    best_batch = batch_recs[0]

    # Apply privacy
    private_batch = get_noise_generators()[best_batch['algorithm']](
        batch_data, best_batch['epsilon']
    )

    # Update budget
    remaining_budget -= best_batch['epsilon']
    batches_processed += 1

    print(f"  Batch {batches_processed}: {best_batch['algorithm']} "
          f"(ε={best_batch['epsilon']:.3f}), Remaining budget: {remaining_budget:.3f}")

print(f"Processed {batches_processed} batches, Budget remaining: {remaining_budget:.3f}")
```

## Integration Examples

### Integration with Pandas
```python
import pandas as pd

# Load data with pandas
df = pd.DataFrame({
    'age': np.random.normal(40, 12, 1000),
    'income': np.random.lognormal(10, 0.5, 1000),
    'score': np.random.normal(650, 100, 1000)
})

# Apply privacy to each column
private_df = df.copy()

for column in df.columns:
    data_tensor = torch.tensor(df[column].values, dtype=torch.float32)

    recs = recommend_top3(data_tensor, n_evals=5)
    best = recs[0]

    private_data = get_noise_generators()[best['algorithm']](
        data_tensor, best['epsilon']
    )

    private_df[column] = private_data.numpy()
    print(f"{column}: {best['algorithm']} (ε={best['epsilon']:.3f})")

# Compare statistics
print("\\nOriginal vs Private Statistics:")
print(df.describe().round(2))
print("\\n" + "="*50 + "\\n")
print(private_df.describe().round(2))
```

### Integration with Scikit-learn
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate classification dataset
X = torch.randn(1000, 5)
y = (X.sum(dim=1) > 0).long()

# Apply privacy to features
X_private = X.clone()
for i in range(X.shape[1]):
    feature_data = X[:, i]
    recs = recommend_top3(feature_data, n_evals=5)
    best = recs[0]

    X_private[:, i] = get_noise_generators()[best['algorithm']](
        feature_data, best['epsilon']
    )

# Compare ML performance
X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(), y.numpy(), test_size=0.2, random_state=42
)
X_train_priv, X_test_priv, _, _ = train_test_split(
    X_private.numpy(), y.numpy(), test_size=0.2, random_state=42
)

# Train models
clf_original = RandomForestClassifier(random_state=42)
clf_private = RandomForestClassifier(random_state=42)

clf_original.fit(X_train, y_train)
clf_private.fit(X_train_priv, y_train)

# Evaluate
acc_original = accuracy_score(y_test, clf_original.predict(X_test))
acc_private = accuracy_score(y_test, clf_private.predict(X_test_priv))

print(f"Original model accuracy: {acc_original:.1%}")
print(f"Private model accuracy: {acc_private:.1%}")
print(f"Accuracy preservation: {acc_private/acc_original:.1%}")
```

## Next Steps

- Explore the [Advanced Features Guide](advanced_features.md) for production deployment
- Check out the [Tutorials](tutorials.md) for step-by-step guidance
- Review the [API Reference](api_reference.md) for detailed documentation
- Join our [Community](https://github.com/ORNL/PRESTO/discussions) to share your use cases

---

**Have a specific use case?** [Open a discussion](https://github.com/ORNL/PRESTO/discussions) and our community can help you implement privacy protection for your domain.
