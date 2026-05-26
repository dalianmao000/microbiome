# Prediction Model Suite - Design Specification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive prediction module for microbiome-based disease classification, efficacy prediction, and prognosis modeling.

**Architecture:** Hybrid ensemble approach combining autoencoders (AE/VAE) for dimensionality reduction with traditional ML classifiers. Configurable preprocessing pipeline supports multiple transformation methods (CLR, log, percent, centered log-ratio). Full pipeline with standardized sklearn-compatible API.

**Tech Stack:** scikit-learn, PyTorch, xgboost, shap, lifelines

---

## Module Structure

```
biomekit/prediction/
├── __init__.py
├── preprocessing.py    # Configurable preprocessing (transform, filter, normalize)
├── encoders.py          # Autoencoder dim-reduction (AE, VAE)
├── classifiers.py        # Classification models (RF, SVM, XGBoost, MLP)
├── regressors.py        # Regression/survival models (CoxPH, Linear)
├── pipeline.py          # Complete pipeline (fit/predict/evaluate)
├── explainer.py         # Interpretation tools (SHAP, PDP, importance)
└── utils.py            # Marker extraction, visualization helpers
```

---

## Task 1: Preprocessing Module

**Files:**
- Create: `biomekit/prediction/preprocessing.py`
- Test: `tests/test_prediction/test_preprocessing.py`

- [ ] **Step 1: Write failing test**

```python
def test_clr_transform():
    import pandas as pd
    import numpy as np
    from biomekit.prediction.preprocessing import CLRTransformer

    # Create small test dataset
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
    transformer = CLRTransformer()
    result = transformer.fit_transform(X)
    assert result.shape == X.shape
    assert not np.any(np.isnan(result))
```

- [ ] **Step 2: Run test** → FAIL (module doesn't exist)

- [ ] **Step 3: Implement preprocessing.py**

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered Log-Ratio transformation for compositional data."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Add pseudocount to avoid log(0)
        X[X == 0] = np.min(X[X > 0]) / 2
        # CLR: log(x) - mean(log(x))
        log_x = np.log(X)
        return log_x - np.mean(log_x, axis=1, keepdims=True)

class VarianceFilter(BaseEstimator, TransformerMixin):
    """Filter features with low variance."""
    def __init__(self, threshold=0.01):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        self.var_ = np.var(X, axis=0)
        self.selected_ = self.var_ > self.threshold
        return self
    
    def transform(self, X):
        return X[:, self.selected_]

class PreprocessingPipeline:
    """Configurable preprocessing pipeline."""
    def __init__(self, transform='clr', filter_low_var=True, variance_threshold=0.01):
        self.transform = transform
        self.filter_low_var = filter_low_var
        self.variance_threshold = variance_threshold
        self.transformer = None
        self.var_filter = None
    
    def fit_transform(self, X, y=None):
        # Apply transformation
        if self.transform == 'clr':
            self.transformer = CLRTransformer()
        elif self.transform == 'log':
            self.transformer = LogTransformer()
        elif self.transform == 'percent':
            self.transformer = PercentTransformer()
        else:
            raise ValueError(f"Unknown transform: {self.transform}")
        
        X_t = self.transformer.fit_transform(X)
        
        # Filter low variance
        if self.filter_low_var:
            self.var_filter = VarianceFilter(threshold=self.variance_threshold)
            X_t = self.var_filter.fit_transform(X_t)
        
        return X_t
    
    def transform(self, X):
        X_t = self.transformer.transform(X)
        if self.filter_low_var:
            X_t = self.var_filter.transform(X_t)
        return X_t
```

- [ ] **Step 4: Run test** → PASS

- [ ] **Step 5: Commit**

---

## Task 2: Autoencoder Encoders

**Files:**
- Create: `biomekit/prediction/encoders.py`
- Test: `tests/test_prediction/test_encoders.py`

- [ ] **Step 1: Write failing test**

```python
def test_autoencoder_dim_reduction():
    import torch
    from biomekit.prediction.encoders import AutoencoderEncoder
    
    X = np.random.rand(50, 100)  # 50 samples, 100 features
    encoder = AutoencoderEncoder(latent_dim=16, hidden_dim=32)
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (50, 16)
```

- [ ] **Step 2: Run test** → FAIL

- [ ] **Step 3: Implement encoders.py**

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.encoder(x), self.decoder(x)

class AutoencoderEncoder(BaseEstimator, TransformerMixin):
    """Autoencoder for dimensionality reduction."""
    def __init__(self, latent_dim=16, hidden_dim=32, epochs=100, lr=0.001):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
    
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        input_dim = X.shape[1]
        self.model = Autoencoder(input_dim, self.hidden_dim, self.latent_dim)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X_tensor = torch.from_numpy(X)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            _, reconstructed = self.model(X_tensor)
            loss = nn.MSELoss()(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
        
        return self
    
    def transform(self, X):
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X)
        with torch.no_grad():
            latent, _ = self.model(X_tensor)
        return latent.numpy()
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class VAEEncoder(BaseEstimator, TransformerMixin):
    """Variational Autoencoder for dimensionality reduction."""
    def __init__(self, latent_dim=16, hidden_dim=32, epochs=100, lr=0.001):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
    
    def fit(self, X, y=None):
        # VAE implementation
        pass
    
    def transform(self, X):
        pass
```

- [ ] **Step 4: Run test** → PASS

- [ ] **Step 5: Commit**

---

## Task 3: Classification Models

**Files:**
- Create: `biomekit/prediction/classifiers.py`
- Test: `tests/test_prediction/test_classifiers.py`

- [ ] **Step 1: Write failing test**

```python
def test_random_forest_classifier():
    from biomekit.prediction.classifiers import MicrobiomeClassifier
    
    X = np.random.rand(100, 50)
    y = np.random.randint(0, 2, 100)
    
    clf = MicrobiomeClassifier(model='rf', n_estimators=100)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert len(pred) == 100
    assert hasattr(clf, 'feature_importances_')
```

- [ ] **Step 2: Run test** → FAIL

- [ ] **Step 3: Implement classifiers.py**

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import numpy as np

class MicrobiomeClassifier:
    """Unified classifier with multiple model options."""
    def __init__(self, model='rf', n_estimators=100, hidden_layer_sizes=(64, 32), 
                 learning_rate=0.01, max_depth=5, random_state=42):
        self.model_type = model
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
    
    def _create_model(self):
        if self.model_type == 'rf':
            return RandomForestClassifier(n_estimators=self.n_estimators, 
                                         random_state=self.random_state)
        elif self.model_type == 'svm':
            return SVC(kernel='rbf', probability=True, random_state=self.random_state)
        elif self.model_type == 'xgb':
            return xgb.XGBClassifier(n_estimators=self.n_estimators, 
                                     max_depth=self.max_depth,
                                     learning_rate=self.learning_rate,
                                     random_state=self.random_state)
        elif self.model_type == 'mlp':
            return MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                max_iter=500, random_state=self.random_state)
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(n_estimators=self.n_estimators,
                                              learning_rate=self.learning_rate,
                                              random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model: {self.model_type}")
    
    def fit(self, X, y):
        self.model = self._create_model()
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def feature_importances_(self):
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_params(self, deep=True):
        return {
            'model': self.model_type,
            'n_estimators': self.n_estimators,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
```

- [ ] **Step 4: Run test** → PASS

- [ ] **Step 5: Commit**

---

## Task 4: Regression/Survival Models

**Files:**
- Create: `biomekit/prediction/regressors.py`
- Test: `tests/test_prediction/test_regressors.py`

- [ ] **Step 1: Write failing test**

```python
def test_cox_regression():
    from biomekit.prediction.regressors import SurvivalRegressor
    
    X = np.random.rand(100, 50)
    y_time = np.random.rand(100) * 100
    y_event = np.random.randint(0, 2, 100)
    
    reg = SurvivalRegressor(model='cox')
    reg.fit(X, y_time, y_event)
    risk_scores = reg.predict_risk(X)
    assert len(risk_scores) == 100
```

- [ ] **Step 2: Run test** → FAIL

- [ ] **Step 3: Implement regressors.py**

```python
from sklearn.linear_model import LogisticRegression, LinearRegression
from lifelines import CoxPHRegressor
import numpy as np

class SurvivalRegressor:
    """Survival/regression models for prognosis prediction."""
    def __init__(self, model='cox', alpha=0.1):
        self.model_type = model
        self.alpha = alpha
        self.model = None
        self.fitted = False
    
    def _create_model(self):
        if self.model_type == 'cox':
            return CoxPHRegressor(alpha=self.alpha)
        elif self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'logistic':
            return LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model: {self.model_type}")
    
    def fit(self, X, y_time, y_event=None):
        self.model = self._create_model()
        
        if self.model_type == 'cox':
            from lifelines.utils import survival_events_from_df
            T, E = survival_events_from_df({'duration': y_time, 'event': y_event})
            self.model.fit(X, T, E)
        else:
            self.model.fit(X, y_time)
        
        self.fitted = True
        return self
    
    def predict_risk(self, X):
        if self.model_type == 'cox':
            return self.model.predict_partial_hazard(X)
        else:
            return self.model.predict(X)
    
    def predict(self, X):
        return self.predict_risk(X)
```

- [ ] **Step 4: Run test** → PASS

- [ ] **Step 5: Commit**

---

## Task 5: Complete Pipeline

**Files:**
- Create: `biomekit/prediction/pipeline.py`
- Test: `tests/test_prediction/test_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
def test_full_pipeline():
    from biomekit.prediction.pipeline import MicrobiomePipeline
    
    X = np.random.rand(100, 50)
    y = np.random.randint(0, 2, 100)
    
    pipeline = MicrobiomePipeline(
        preprocess='clr',
        encode='autoencoder',
        classifier='rf'
    )
    pipeline.fit(X, y)
    results = pipeline.evaluate(X, y, cv=5)
    assert 'accuracy' in results
    assert 'auc_roc' in results
    assert 'shap_values' in results
```

- [ ] **Step 2: Run test** → FAIL

- [ ] **Step 3: Implement pipeline.py**

```python
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, average_precision_score)
import pandas as pd

class MicrobiomePipeline:
    """Complete prediction pipeline with preprocessing, encoding, and classification."""
    def __init__(self, preprocess='clr', encode=None, classifier='rf',
                 latent_dim=16, variance_threshold=0.01, **classifier_kwargs):
        self.preprocess = preprocess
        self.encode = encode
        self.classifier = classifier
        self.latent_dim = latent_dim
        self.variance_threshold = variance_threshold
        self.classifier_kwargs = classifier_kwargs
        
        self.preprocessor = None
        self.encoder = None
        self.clf = None
        self.fitted = False
    
    def _init_components(self):
        from .preprocessing import PreprocessingPipeline, CLRTransformer
        from .encoders import AutoencoderEncoder
        from .classifiers import MicrobiomeClassifier
        
        # Preprocessing
        if self.preprocess == 'clr':
            self.preprocessor = CLRTransformer()
        else:
            self.preprocessor = PreprocessingPipeline(transform=self.preprocess)
        
        # Encoding (optional)
        if self.encode == 'autoencoder':
            self.encoder = AutoencoderEncoder(latent_dim=self.latent_dim)
        # else encoder = None (passthrough)
        
        # Classifier
        self.clf = MicrobiomeClassifier(model=self.classifier, **self.classifier_kwargs)
    
    def fit(self, X, y):
        self._init_components()
        
        # Preprocess
        X_t = self.preprocessor.fit_transform(X)
        
        # Encode
        if self.encoder is not None:
            X_t = self.encoder.fit_transform(X_t)
        
        # Classify
        self.clf.fit(X_t, y)
        self.fitted = True
        return self
    
    def predict(self, X):
        X_t = self.preprocessor.transform(X)
        if self.encoder is not None:
            X_t = self.encoder.transform(X_t)
        return self.clf.predict(X_t)
    
    def predict_proba(self, X):
        X_t = self.preprocessor.transform(X)
        if self.encoder is not None:
            X_t = self.encoder.transform(X_t)
        return self.clf.predict_proba(X_t)
    
    def evaluate(self, X, y, cv=5):
        """Full evaluation with multiple metrics and cross-validation."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'auc_roc': [], 'auc_pr': []
        }
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.fit(X_train, y_train)
            y_pred = self.predict(X_val)
            y_proba = self.predict_proba(X_val)[:, 1] if len(np.unique(y)) == 2 else None
            
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred, average='weighted'))
            metrics['recall'].append(recall_score(y_val, y_pred, average='weighted'))
            metrics['f1'].append(f1_score(y_val, y_pred, average='weighted'))
            
            if y_proba is not None:
                metrics['auc_roc'].append(roc_auc_score(y_val, y_proba))
                metrics['auc_pr'].append(average_precision_score(y_val, y_proba))
        
        # Return mean and std
        results = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
        return results
    
    def get_feature_importance(self):
        if self.clf and hasattr(self.clf, 'feature_importances_'):
            return self.clf.feature_importances_
        return None
    
    def get_params(self, deep=True):
        return {
            'preprocess': self.preprocess,
            'encode': self.encode,
            'classifier': self.classifier,
            'latent_dim': self.latent_dim,
            'variance_threshold': self.variance_threshold,
            **self.classifier_kwargs
        }
```

- [ ] **Step 4: Run test** → PASS

- [ ] **Step 5: Commit**

---

## Task 6: Explainability (SHAP, PDP, Importance)

**Files:**
- Create: `biomekit/prediction/explainer.py`
- Test: `tests/test_prediction/test_explainer.py`

- [ ] **Step 1: Write failing test**

```python
def test_shap_values():
    from biomekit.prediction.explainer import SHAPExplainer
    
    X = np.random.rand(50, 20)
    y = np.random.randint(0, 2, 50)
    
    # Train simple model first
    from .classifiers import MicrobiomeClassifier
    clf = MicrobiomeClassifier(model='rf')
    clf.fit(X, y)
    
    explainer = SHAPExplainer(clf)
    shap_values = explainer.shap_values(X)
    assert shap_values.shape == X.shape
```

- [ ] **Step 2: Run test** → FAIL

- [ ] **Step 3: Implement explainer.py**

```python
import numpy as np
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class SHAPExplainer:
    """SHAP-based model explainer."""
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
    
    def _init_explainer(self, X):
        if not SHAP_AVAILABLE:
            raise ImportError("shap library required for SHAP explanation")
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'predict_proba'):
            # Tree-based model
            self.explainer = shap.TreeExplainer(self.model.model)
        else:
            # KernelExplainer for others
            self.explainer = shap.KernelExplainer(
                self.model.model.predict_proba, X[:10]
            )
    
    def shap_values(self, X):
        self._init_explainer(X)
        return self.explainer.shap_values(X)

class PermutationImportance:
    """Permutation-based feature importance."""
    def __init__(self, model, n_repeats=10, random_state=42):
        self.model = model
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.importances_ = None
    
    def fit(self, X, y):
        from sklearn.inspection import permutation_importance
        result = permutation_importance(self.model, X, y, 
                                       n_repeats=self.n_repeats,
                                       random_state=self.random_state)
        self.importances_ = result.importances_mean
        return self
    
    def get_importance(self):
        return self.importances_

class PartialDependencePlot:
    """Partial dependence plot computation."""
    def __init__(self, model, feature_idx):
        self.model = model
        self.feature_idx = feature_idx
    
    def compute(self, X, feature_values):
        from sklearn.inspection import partial_dependence
        pd_results = partial_dependence(self.model, X, self.feature_idx,
                                        feature_values=feature_values)
        return pd_results['average']
```

- [ ] **Step 4: Run test** → PASS

- [ ] **Step 5: Commit**

---

## Task 7: Marker Extraction & Utils

**Files:**
- Create: `biomekit/prediction/utils.py`
- Test: `tests/test_prediction/test_utils.py`

- [ ] **Step 1: Write failing test**

```python
def test_extract_markers():
    from biomekit.prediction.utils import extract_top_markers
    
    importance = np.random.rand(50)
    feature_names = [f'feature_{i}' for i in range(50)]
    
    markers = extract_top_markers(importance, feature_names, top_n=10)
    assert len(markers) == 10
    assert all(isinstance(m, tuple) for m in markers)  # (name, score) pairs
```

- [ ] **Step 2: Run test** → FAIL

- [ ] **Step 3: Implement utils.py**

```python
import numpy as np
import pandas as pd

def extract_top_markers(importance, feature_names, top_n=20, direction='positive'):
    """Extract top microbial markers based on importance scores."""
    if len(importance) != len(feature_names):
        raise ValueError("importance and feature_names must have same length")
    
    # Sort by importance
    if direction == 'positive':
        sorted_idx = np.argsort(importance)[::-1]
    else:
        sorted_idx = np.argsort(importance)
    
    top_idx = sorted_idx[:top_n]
    markers = [(feature_names[i], importance[i]) for i in top_idx]
    return markers

def plot_feature_importance(importance, feature_names, top_n=20, save_path=None):
    """Plot feature importance as horizontal bar chart."""
    import matplotlib.pyplot as plt
    
    markers = extract_top_markers(importance, feature_names, top_n=top_n)
    names = [m[0] for m in markers]
    scores = [m[1] for m in markers]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(names)), scores)
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Microbial Markers')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_shap_summary(shap_values, feature_names=None, save_path=None):
    """Plot SHAP summary beeswarm plot."""
    import matplotlib.pyplot as plt
    try:
        import shap
        plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("shap library required for SHAP summary plot")

def learning_curve(estimator, X, y, cv=5, scoring='accuracy', save_path=None):
    """Plot learning curve for model evaluation."""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, 
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring)
    plt.title('Learning Curve')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
```

- [ ] **Step 4: Run test** → PASS

- [ ] **Step 5: Commit**

---

## Task 8: Module Init & Integration

**Files:**
- Modify: `biomekit/__init__.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Update __init__.py**

```python
# Add to __init__.py
try:
    from biomekit.prediction import MicrobiomePipeline, MicrobiomeClassifier, SHAPExplainer
except ImportError:
    pass
```

- [ ] **Step 2: Add integration test**

```python
def test_prediction_pipeline_integration():
    from biomekit.prediction import MicrobiomePipeline
    from biomekit.utils.simulate import simulate_abundance_data
    
    # Generate simulated data
    X, metadata = simulate_abundance_data(n_samples=100, n_features=200, n_diff_abundant=20)
    y = (metadata['group'] == 'disease').astype(int).values
    
    # Run full pipeline
    pipeline = MicrobiomePipeline(preprocess='clr', classifier='rf', encode=None)
    pipeline.fit(X, y)
    
    results = pipeline.evaluate(X, y, cv=3)
    assert results['accuracy'][0] > 0.5
    print(f"Accuracy: {results['accuracy'][0]:.3f} ± {results['accuracy'][1]:.3f}")
```

- [ ] **Step 3: Run all tests**

- [ ] **Step 4: Commit**

---

## Task 9: Update Documentation

**Files:**
- Create: `docs/algorithm_notes/prediction_models.md`
- Modify: `README.md` (add P1 section details)

- [ ] **Step 1: Write algorithm documentation**

- [ ] **Step 2: Update README**

- [ ] **Step 3: Commit**

---

## Dependencies Additions (to pyproject.toml)

```toml
[tool.poetry.dependencies]
# existing + add:
lifelines = "^0.27"
shap = "^0.42"
torch = "^2.0"
xgboost = "^2.0"
```

---

## Verification Checklist

- [ ] All 9 tasks completed
- [ ] 41 existing tests + new prediction tests pass
- [ ] SHAP values computed correctly
- [ ] Pipeline fit/predict/evaluate works end-to-end
- [ ] Documentation updated
- [ ] pyproject.toml dependencies added