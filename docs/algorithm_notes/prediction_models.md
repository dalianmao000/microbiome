# Prediction Models Documentation

## Overview

The prediction module provides a comprehensive suite of machine learning tools for microbiome-based disease classification, treatment efficacy prediction, and patient prognosis modeling.

## Architecture

```
Raw Abundance Matrix
        ↓
Preprocessing (CLR/Log/Percent transform + Variance filtering)
        ↓
Encoding (Optional: Autoencoder/VAE for dimensionality reduction)
        ↓
Classification (RF, SVM, XGBoost, MLP, GradientBoosting)
        OR
Survival Analysis (Cox Proportional Hazards)
        ↓
Evaluation (Cross-validation with multiple metrics)
        ↓
Interpretation (SHAP, Permutation Importance, Partial Dependence Plots)
```

## Key Classes

### MicrobiomePipeline

Complete prediction pipeline combining preprocessing, encoding, and classification.

```python
from biomekit.prediction import MicrobiomePipeline

pipeline = MicrobiomePipeline(
    preprocess='clr',     # Transformation: 'clr', 'log', 'percent'
    encode='autoencoder', # Encoding: 'autoencoder', 'vae', or None
    classifier='rf',      # Classifier: 'rf', 'svm', 'xgb', 'mlp', 'gb'
    latent_dim=16,        # Latent dimension for autoencoder
    n_estimators=100      # Number of trees (for tree-based models)
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
results = pipeline.evaluate(X_test, y_test, cv=5)
```

### Preprocessing

- `CLRTransformer`: Centered Log-Ratio transformation for compositional data
- `LogTransformer`: Log transformation with pseudocount
- `PercentTransformer`: Row-wise percentile/rank transformation
- `VarianceFilter`: Remove low-variance features
- `PreprocessingPipeline`: Configurable pipeline combining transforms and filtering

### Encoders

- `AutoencoderEncoder`: Neural network for dimensionality reduction
- `VAEEncoder`: Variational autoencoder for probabilistic representation

### Classifiers

- `MicrobiomeClassifier`: Unified interface for RF, SVM, XGBoost, MLP, GB

### Explainers

- `SHAPExplainer`: SHAP values for feature importance
- `PermutationImportance`: Permutation-based importance
- `PartialDependencePlot`: Feature vs prediction relationship

## Evaluation Metrics

The pipeline evaluates using multiple metrics:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- AUC-ROC (binary classification)
- AUC-PR (binary classification)

Cross-validation (default 5-fold stratified) provides mean and standard deviation for each metric.

## Example: Disease Classification

```python
from biomekit.prediction import MicrobiomePipeline
from biomekit.utils.simulate import simulate_abundance_data

# Generate data
X, metadata = simulate_abundance_data(n_samples=200, n_features=500)
y = (metadata['group'] == 'disease').astype(int).values

# Run pipeline
pipeline = MicrobiomePipeline(preprocess='clr', classifier='rf', n_estimators=100)
pipeline.fit(X, y)

# Evaluate
results = pipeline.evaluate(X, y, cv=5)
print(f"Accuracy: {results['accuracy'][0]:.3f} ± {results['accuracy'][1]:.3f}")
print(f"AUC-ROC: {results['auc_roc'][0]:.3f} ± {results['auc_roc'][1]:.3f}")

# Get important features
importance = pipeline.get_feature_importance()
```

## Example: Prognosis Modeling

```python
from biomekit.prediction import SurvivalRegressor

# Survival times and events
y_time = np.random.rand(100) * 100 + 10  # Survival time
y_event = np.random.randint(0, 2, 100)  # 1=death, 0=censored

regressor = SurvivalRegressor(model='cox')
regressor.fit(X, y_time, y_event)
risk_scores = regressor.predict_risk(X)
```

## Dependencies

- scikit-learn: Base ML utilities
- PyTorch: Neural network encoders
- xgboost: Gradient boosting
- shap: SHAP explanations
- lifelines: Survival analysis
- matplotlib/seaborn: Visualization