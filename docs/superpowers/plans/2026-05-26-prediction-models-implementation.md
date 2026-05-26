# Prediction Model Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive prediction module for microbiome-based disease classification, efficacy prediction, and prognosis modeling.

**Architecture:** Hybrid ensemble approach combining autoencoders (AE/VAE) for dimensionality reduction with traditional ML classifiers. Configurable preprocessing pipeline supports multiple transformation methods (CLR, log, percent, centered log-ratio). Full pipeline with standardized sklearn-compatible API.

**Tech Stack:** scikit-learn, PyTorch, xgboost, shap, lifelines

---

## File Structure

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

tests/test_prediction/
├── __init__.py
├── test_preprocessing.py
├── test_encoders.py
├── test_classifiers.py
├── test_regressors.py
├── test_pipeline.py
├── test_explainer.py
└── test_utils.py
```

---

### Task 1: Preprocessing Module

**Files:**
- Create: `biomekit/prediction/preprocessing.py`
- Test: `tests/test_prediction/test_preprocessing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction/test_preprocessing.py
import pytest
import numpy as np
import pandas as pd
from biomekit.prediction.preprocessing import CLRTransformer, VarianceFilter, LogTransformer, PercentTransformer, PreprocessingPipeline

def test_clr_transform():
    """Test CLR transformation on small dataset."""
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
    transformer = CLRTransformer()
    result = transformer.fit_transform(X)
    assert result.shape == X.shape
    assert not np.any(np.isnan(result))

def test_log_transform():
    """Test log transformation."""
    X = np.array([[1, 2], [4, 8]])
    transformer = LogTransformer()
    result = transformer.fit_transform(X)
    assert result.shape == X.shape
    assert np.all(result >= 0)

def test_percent_transform():
    """Test percentile transformation."""
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    transformer = PercentTransformer()
    result = transformer.fit_transform(X)
    assert result.shape == X.shape
    assert np.all(result >= 0) and np.all(result <= 1)

def test_variance_filter():
    """Test variance-based feature filter."""
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    filter_obj = VarianceFilter(threshold=0.5)
    result = filter_obj.fit_transform(X)
    assert result.shape[1] < X.shape[1]  # Some features filtered

def test_preprocessing_pipeline_clr():
    """Test full preprocessing pipeline with CLR."""
    X = np.random.rand(50, 100) + 0.1  # Add small offset to avoid zeros
    pipeline = PreprocessingPipeline(transform='clr', filter_low_var=True, variance_threshold=0.01)
    result = pipeline.fit_transform(X)
    assert result.shape[0] == 50
    assert result.shape[1] <= 100
    assert not np.any(np.isnan(result))

def test_preprocessing_pipeline_log():
    """Test preprocessing pipeline with log transform."""
    X = np.random.rand(30, 50) + 0.1
    pipeline = PreprocessingPipeline(transform='log', filter_low_var=False)
    result = pipeline.fit_transform(X)
    assert result.shape == X.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction/test_preprocessing.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'biomekit.prediction'"

- [ ] **Step 3: Write implementation**

```python
# biomekit/prediction/preprocessing.py
"""Preprocessing module for microbiome prediction models."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered Log-Ratio transformation for compositional data.

    Transforms features using: log(x) - mean(log(x))
    Adds small pseudocount to handle zeros.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Add pseudocount to avoid log(0)
        min_positive = np.min(X[X > 0]) if np.any(X > 0) else 1e-10
        X[X == 0] = min_positive / 2
        # CLR: log(x) - mean(log(x))
        log_x = np.log(X)
        return log_x - np.mean(log_x, axis=1, keepdims=True)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log transformation with pseudocount."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        min_positive = np.min(X[X > 0]) if np.any(X > 0) else 1e-10
        X[X == 0] = min_positive / 2
        return np.log(X)


class PercentTransformer(BaseEstimator, TransformerMixin):
    """Percentile/rank transformation."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Row-wise percentile
        from scipy.stats import rankdata
        result = np.zeros_like(X)
        for i in range(X.shape[0]):
            result[i] = rankdata(X[i]) / len(X[i])
        return result


class VarianceFilter(BaseEstimator, TransformerMixin):
    """Filter features with variance below threshold."""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.var_ = np.var(X, axis=0)
        self.selected_ = self.var_ > self.threshold
        return self

    def transform(self, X):
        return X[:, self.selected_]

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, 'selected_'):
            return np.array([f"f{i}" for i in range(len(self.selected_))])[self.selected_]
        return input_features


class PreprocessingPipeline:
    """Configurable preprocessing pipeline.

    Parameters:
        transform: 'clr', 'log', or 'percent'
        filter_low_var: Whether to filter low variance features
        variance_threshold: Variance threshold for filtering
    """

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction/test_preprocessing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add biomekit/prediction/preprocessing.py tests/test_prediction/test_preprocessing.py
git commit -m "feat: add prediction preprocessing module (CLR, log, percent transforms)"
```

---

### Task 2: Autoencoder Encoders

**Files:**
- Create: `biomekit/prediction/encoders.py`
- Test: `tests/test_prediction/test_encoders.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction/test_encoders.py
import pytest
import numpy as np
import torch
from biomekit.prediction.encoders import AutoencoderEncoder, VAEEncoder

def test_autoencoder_dim_reduction():
    """Test autoencoder reduces dimensions correctly."""
    X = np.random.rand(50, 100).astype(np.float32) + 0.1
    encoder = AutoencoderEncoder(latent_dim=16, hidden_dim=32, epochs=10)
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (50, 16), f"Expected (50, 16), got {X_encoded.shape}"
    assert not np.any(np.isnan(X_encoded))

def test_autoencoder_passthrough():
    """Test autoencoder with None encoding (passthrough)."""
    X = np.random.rand(30, 50).astype(np.float32) + 0.1
    encoder = AutoencoderEncoder(latent_dim=50, hidden_dim=32, epochs=5)
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (30, 50)  # No reduction if latent >= input
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction/test_encoders.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write implementation**

```python
# biomekit/prediction/encoders.py
"""Autoencoder modules for dimensionality reduction."""
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Autoencoder(nn.Module):
    """Simple autoencoder for microbiome data."""

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
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class AutoencoderEncoder(BaseEstimator, TransformerMixin):
    """Autoencoder for dimensionality reduction.

    Parameters:
        latent_dim: Dimension of latent representation
        hidden_dim: Dimension of hidden layer
        epochs: Number of training epochs
        lr: Learning rate
    """

    def __init__(self, latent_dim=16, hidden_dim=32, epochs=100, lr=0.001):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.input_dim_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]

        # Don't encode if latent >= input
        if self.latent_dim >= self.input_dim_:
            return self

        self.model = Autoencoder(self.input_dim_, self.hidden_dim, self.latent_dim)
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
        # Passthrough if not fitted or latent >= input
        if self.model is None or self.latent_dim >= self.input_dim_:
            return np.asarray(X, dtype=np.float32)

        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X)
        with torch.no_grad():
            latent, _ = self.model(X_tensor)
        return latent.numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class VAEEncoder(BaseEstimator, TransformerMixin):
    """Variational Autoencoder for dimensionality reduction.

    Provides probabilistic latent representation.
    """

    def __init__(self, latent_dim=16, hidden_dim=32, epochs=100, lr=0.001):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.input_dim_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]

        if self.latent_dim >= self.input_dim_:
            return self

        # Simple VAE implementation
        class VAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc21 = nn.Linear(hidden_dim, latent_dim)
                self.fc22 = nn.Linear(hidden_dim, latent_dim)
                self.fc3 = nn.Linear(latent_dim, hidden_dim)
                self.fc4 = nn.Linear(hidden_dim, input_dim)

            def encode(self, x):
                h = torch.relu(self.fc1(x))
                return self.fc21(h), self.fc22(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                h = torch.relu(self.fc3(z))
                return self.fc4(h)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        self.model = VAE(self.input_dim_, self.hidden_dim, self.latent_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.from_numpy(X)
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon, mu, logvar = self.model(X_tensor)
            recon_loss = nn.MSELoss()(recon, X_tensor)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.001 * kl_loss
            loss.backward()
            optimizer.step()

        return self

    def transform(self, X):
        if self.model is None or self.latent_dim >= self.input_dim_:
            return np.asarray(X, dtype=np.float32)

        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X)
        with torch.no_grad():
            mu, _ = self.model.encode(X_tensor)
        return mu.numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction/test_encoders.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add biomekit/prediction/encoders.py tests/test_prediction/test_encoders.py
git commit -m "feat: add autoencoder and VAE encoders for dim reduction"
```

---

### Task 3: Classification Models

**Files:**
- Create: `biomekit/prediction/classifiers.py`
- Test: `tests/test_prediction/test_classifiers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction/test_classifiers.py
import pytest
import numpy as np
from biomekit.prediction.classifiers import MicrobiomeClassifier

def test_random_forest_classifier():
    """Test random forest classifier."""
    X = np.random.rand(100, 50)
    y = np.random.randint(0, 2, 100)

    clf = MicrobiomeClassifier(model='rf', n_estimators=50)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert len(pred) == 100
    assert hasattr(clf, 'feature_importances_')
    assert clf.feature_importances_ is not None

def test_xgboost_classifier():
    """Test XGBoost classifier."""
    X = np.random.rand(80, 40)
    y = np.random.randint(0, 2, 80)

    clf = MicrobiomeClassifier(model='xgb', n_estimators=50, max_depth=3)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert len(pred) == 80

def test_svm_classifier():
    """Test SVM classifier."""
    X = np.random.rand(60, 30)
    y = np.random.randint(0, 2, 60)

    clf = MicrobiomeClassifier(model='svm')
    clf.fit(X, y)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    assert proba.shape == (60, 2)

def test_mlp_classifier():
    """Test MLP classifier."""
    X = np.random.rand(50, 20)
    y = np.random.randint(0, 2, 50)

    clf = MicrobiomeClassifier(model='mlp', hidden_layer_sizes=(16, 8))
    clf.fit(X, y)
    pred = clf.predict(X)
    assert len(pred) == 50

def test_get_params():
    """Test get_params method."""
    clf = MicrobiomeClassifier(model='rf', n_estimators=100)
    params = clf.get_params()
    assert params['model'] == 'rf'
    assert params['n_estimators'] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction/test_classifiers.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write implementation**

```python
# biomekit/prediction/classifiers.py
"""Classification models for microbiome prediction."""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class MicrobiomeClassifier:
    """Unified classifier supporting multiple model types.

    Parameters:
        model: 'rf', 'svm', 'xgb', 'mlp', or 'gb'
        n_estimators: Number of trees (for RF, XGB, GB)
        hidden_layer_sizes: Tuple for MLP architecture
        learning_rate: Learning rate (for XGB, GB)
        max_depth: Maximum depth (for XGB, GB)
        random_state: Random seed
    """

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
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(kernel='rbf', probability=True, random_state=self.random_state)
        elif self.model_type == 'xgb':
            if not XGBOOST_AVAILABLE:
                raise ImportError("xgboost required for xgb model")
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=500,
                random_state=self.random_state
            )
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
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

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction/test_classifiers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add biomekit/prediction/classifiers.py tests/test_prediction/test_classifiers.py
git commit -m "feat: add MicrobiomeClassifier with RF, SVM, XGB, MLP, GB support"
```

---

### Task 4: Regression/Survival Models

**Files:**
- Create: `biomekit/prediction/regressors.py`
- Test: `tests/test_prediction/test_regressors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction/test_regressors.py
import pytest
import numpy as np
from biomekit.prediction.regressors import SurvivalRegressor

def test_cox_regression():
    """Test Cox proportional hazards model."""
    X = np.random.rand(100, 50)
    y_time = np.random.rand(100) * 100 + 10  # Survival times
    y_event = np.random.randint(0, 2, 100)  # Event indicators

    reg = SurvivalRegressor(model='cox')
    reg.fit(X, y_time, y_event)
    risk_scores = reg.predict_risk(X)
    assert len(risk_scores) == 100
    assert np.all(np.isfinite(risk_scores))

def test_linear_regression():
    """Test linear regression."""
    X = np.random.rand(80, 30)
    y = np.random.rand(80) * 100

    reg = SurvivalRegressor(model='linear')
    reg.fit(X, y)
    pred = reg.predict(X)
    assert len(pred) == 80

def test_logistic_regression():
    """Test logistic regression for binary outcomes."""
    X = np.random.rand(60, 25)
    y = np.random.randint(0, 2, 60)

    reg = SurvivalRegressor(model='logistic')
    reg.fit(X, y)
    proba = reg.predict_proba(X)
    assert proba.shape == (60, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction/test_regressors.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write implementation**

```python
# biomekit/prediction/regressors.py
"""Regression and survival analysis models."""
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

try:
    from lifelines import CoxPHRegressor
    from lifelines.utils import survival_events_from_df
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


class SurvivalRegressor:
    """Survival and regression models for prognosis prediction.

    Parameters:
        model: 'cox', 'linear', or 'logistic'
        alpha: Regularization parameter (for Cox)
    """

    def __init__(self, model='cox', alpha=0.1):
        self.model_type = model
        self.alpha = alpha
        self.model = None
        self.fitted = False

    def _create_model(self):
        if self.model_type == 'cox':
            if not LIFELINES_AVAILABLE:
                raise ImportError("lifelines required for Cox model. Install with: pip install lifelines")
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
            T, E = survival_events_from_df({'duration': y_time, 'event': y_event})
            self.model.fit(X, T, E)
        else:
            self.model.fit(X, y_time if y_event is None else y_event)

        self.fitted = True
        return self

    def predict_risk(self, X):
        if self.model_type == 'cox':
            return self.model.predict_partial_hazard(X)
        else:
            return self.model.predict(X)

    def predict(self, X):
        return self.predict_risk(X)

    def predict_proba(self, X):
        if self.model_type == 'logistic':
            return self.model.predict_proba(X)
        raise NotImplementedError("predict_proba only available for logistic regression")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction/test_regressors.py -v`
Expected: PASS (if lifelines installed) or SKIP (if not installed)

- [ ] **Step 5: Commit**

```bash
git add biomekit/prediction/regressors.py tests/test_prediction/test_regressors.py
git commit -m "feat: add SurvivalRegressor with Cox, linear, logistic models"
```

---

### Task 5: Complete Pipeline

**Files:**
- Create: `biomekit/prediction/pipeline.py`
- Test: `tests/test_prediction/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction/test_pipeline.py
import pytest
import numpy as np
from biomekit.prediction.pipeline import MicrobiomePipeline

def test_pipeline_fit_predict():
    """Test pipeline fit and predict."""
    X = np.random.rand(100, 50) + 0.1
    y = np.random.randint(0, 2, 100)

    pipeline = MicrobiomePipeline(
        preprocess='clr',
        encode=None,
        classifier='rf'
    )
    pipeline.fit(X, y)
    pred = pipeline.predict(X)
    assert len(pred) == 100

def test_pipeline_evaluate():
    """Test pipeline evaluation with cross-validation."""
    X = np.random.rand(100, 50) + 0.1
    y = np.random.randint(0, 2, 100)

    pipeline = MicrobiomePipeline(
        preprocess='clr',
        classifier='rf'
    )
    results = pipeline.evaluate(X, y, cv=3)
    assert 'accuracy' in results
    assert 'auc_roc' in results
    assert results['accuracy'][0] >= 0  # Mean should be valid

def test_pipeline_with_autoencoder():
    """Test pipeline with autoencoder encoding."""
    X = np.random.rand(80, 100).astype(np.float32) + 0.1
    y = np.random.randint(0, 2, 80)

    pipeline = MicrobiomePipeline(
        preprocess='clr',
        encode='autoencoder',
        classifier='rf',
        latent_dim=16
    )
    pipeline.fit(X, y)
    pred = pipeline.predict(X)
    assert len(pred) == 80

def test_get_feature_importance():
    """Test feature importance extraction."""
    X = np.random.rand(60, 40) + 0.1
    y = np.random.randint(0, 2, 60)

    pipeline = MicrobiomePipeline(classifier='rf')
    pipeline.fit(X, y)
    importance = pipeline.get_feature_importance()
    assert importance is not None
    assert len(importance) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction/test_pipeline.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write implementation**

```python
# biomekit/prediction/pipeline.py
"""Complete prediction pipeline for microbiome analysis."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

from .preprocessing import CLRTransformer, PreprocessingPipeline
from .encoders import AutoencoderEncoder
from .classifiers import MicrobiomeClassifier


class MicrobiomePipeline:
    """Complete prediction pipeline with preprocessing, encoding, and classification.

    Parameters:
        preprocess: Preprocessing transform ('clr', 'log', 'percent', or None)
        encode: Encoding method ('autoencoder', 'vae', or None)
        classifier: Classifier type ('rf', 'svm', 'xgb', 'mlp', 'gb')
        latent_dim: Latent dimension for autoencoder
        variance_threshold: Variance threshold for filtering
        **classifier_kwargs: Additional arguments for classifier
    """

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
        # Preprocessing
        if self.preprocess == 'clr':
            self.preprocessor = CLRTransformer()
        elif self.preprocess:
            self.preprocessor = PreprocessingPipeline(transform=self.preprocess)
        else:
            self.preprocessor = None

        # Encoding
        if self.encode == 'autoencoder':
            self.encoder = AutoencoderEncoder(latent_dim=self.latent_dim)
        elif self.encode == 'vae':
            from .encoders import VAEEncoder
            self.encoder = VAEEncoder(latent_dim=self.latent_dim)
        else:
            self.encoder = None

        # Classifier
        self.clf = MicrobiomeClassifier(model=self.classifier, **self.classifier_kwargs)

    def fit(self, X, y):
        self._init_components()

        # Preprocess
        if self.preprocessor is not None:
            X_t = self.preprocessor.fit_transform(X)
        else:
            X_t = np.asarray(X)

        # Encode
        if self.encoder is not None:
            X_t = self.encoder.fit_transform(X_t)

        # Classify
        self.clf.fit(X_t, y)
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        X_t = np.asarray(X)
        if self.preprocessor is not None:
            X_t = self.preprocessor.transform(X_t)
        if self.encoder is not None:
            X_t = self.encoder.transform(X_t)
        return self.clf.predict(X_t)

    def predict_proba(self, X):
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        X_t = np.asarray(X)
        if self.preprocessor is not None:
            X_t = self.preprocessor.transform(X_t)
        if self.encoder is not None:
            X_t = self.encoder.transform(X_t)
        return self.clf.predict_proba(X_t)

    def evaluate(self, X, y, cv=5):
        """Evaluate pipeline with cross-validation.

        Returns:
            Dictionary with metric means and standard deviations
        """
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

            # Binary vs multi-class handling
            n_classes = len(np.unique(y))
            if n_classes == 2:
                y_proba = self.predict_proba(X_val)[:, 1]
            else:
                y_proba = None

            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
            metrics['recall'].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
            metrics['f1'].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))

            if y_proba is not None:
                try:
                    metrics['auc_roc'].append(roc_auc_score(y_val, y_proba))
                    metrics['auc_pr'].append(average_precision_score(y_val, y_proba))
                except ValueError:
                    metrics['auc_roc'].append(np.nan)
                    metrics['auc_pr'].append(np.nan)

        # Return mean and std
        results = {k: (np.nanmean(v), np.nanstd(v)) for k, v in metrics.items()}
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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add biomekit/prediction/pipeline.py tests/test_prediction/test_pipeline.py
git commit -m "feat: add complete MicrobiomePipeline with fit/predict/evaluate"
```

---

### Task 6: Explainability (SHAP, PDP, Importance)

**Files:**
- Create: `biomekit/prediction/explainer.py`
- Test: `tests/test_prediction/test_explainer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction/test_explainer.py
import pytest
import numpy as np
from biomekit.prediction.explainer import SHAPExplainer, PermutationImportance, PartialDependencePlot
from biomekit.prediction.classifiers import MicrobiomeClassifier

def test_shap_explainer_tree():
    """Test SHAP explainer with tree model."""
    X = np.random.rand(50, 20)
    y = np.random.randint(0, 2, 50)

    clf = MicrobiomeClassifier(model='rf', n_estimators=50)
    clf.fit(X, y)

    explainer = SHAPExplainer(clf)
    shap_values = explainer.shap_values(X[:10])
    assert shap_values.shape == (10, 20)

def test_permutation_importance():
    """Test permutation importance."""
    X = np.random.rand(60, 15)
    y = np.random.randint(0, 2, 60)

    clf = MicrobiomeClassifier(model='rf', n_estimators=50)
    clf.fit(X, y)

    perm_imp = PermutationImportance(clf, n_repeats=5)
    perm_imp.fit(X, y)
    importance = perm_imp.get_importance()
    assert len(importance) == 15

def test_partial_dependence():
    """Test partial dependence plot."""
    X = np.random.rand(50, 10)
    y = np.random.randint(0, 2, 50)

    clf = MicrobiomeClassifier(model='rf', n_estimators=50)
    clf.fit(X, y)

    pdp = PartialDependencePlot(clf, feature_idx=0)
    feature_values = np.linspace(0, 1, 20)
    pd_result = pdp.compute(X, feature_values)
    assert len(pd_result) == len(feature_values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction/test_explainer.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write implementation**

```python
# biomekit/prediction/explainer.py
"""Model explainability tools for microbiome prediction."""
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """SHAP-based model explainer.

    Provides SHAP values for model interpretation.
    """

    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def _init_explainer(self, X):
        if not SHAP_AVAILABLE:
            raise ImportError("shap library required for SHAP explanation. Install with: pip install shap")

        model_obj = self.model.model if hasattr(self.model, 'model') else self.model

        if hasattr(model_obj, 'predict_proba'):
            # Tree-based or other models with predict_proba
            try:
                self.explainer = shap.TreeExplainer(model_obj)
            except Exception:
                # Fall back to kernel explainer
                self.explainer = shap.KernelExplainer(
                    model_obj.predict_proba, X[:20]
                )

    def shap_values(self, X):
        self._init_explainer(X)
        if self.explainer is None:
            raise RuntimeError("Could not initialize SHAP explainer")

        X = np.asarray(X, dtype=np.float32)
        shap_values = self.explainer.shap_values(X)

        # Handle list return (one per class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        return shap_values


class PermutationImportance:
    """Permutation-based feature importance.

    Measures importance by permuting feature values.
    """

    def __init__(self, model, n_repeats=10, random_state=42):
        self.model = model
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.importances_ = None

    def fit(self, X, y):
        from sklearn.inspection import permutation_importance

        model_obj = self.model.model if hasattr(self.model, 'model') else self.model
        result = permutation_importance(
            model_obj, X, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.importances_ = result.importances_mean
        return self

    def get_importance(self):
        if self.importances_ is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        return self.importances_


class PartialDependencePlot:
    """Partial dependence plot computation.

    Shows relationship between feature and prediction.
    """

    def __init__(self, model, feature_idx):
        self.model = model
        self.feature_idx = feature_idx

    def compute(self, X, feature_values=None):
        from sklearn.inspection import partial_dependence

        if feature_values is None:
            feature_values = np.linspace(0, 1, 20)

        model_obj = self.model.model if hasattr(self.model, 'model') else self.model

        pd_results = partial_dependence(
            model_obj, X, self.feature_idx,
            feature_values=[feature_values],
            kind='average'
        )
        return pd_results['average'][0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction/test_explainer.py -v`
Expected: PASS (if shap installed) or SKIP (if not installed)

- [ ] **Step 5: Commit**

```bash
git add biomekit/prediction/explainer.py tests/test_prediction/test_explainer.py
git commit -m "feat: add SHAP, permutation importance, and PDP explainers"
```

---

### Task 7: Marker Extraction & Utils

**Files:**
- Create: `biomekit/prediction/utils.py`
- Test: `tests/test_prediction/test_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction/test_utils.py
import pytest
import numpy as np
from biomekit.prediction.utils import extract_top_markers

def test_extract_top_markers():
    """Test marker extraction from importance scores."""
    importance = np.random.rand(50)
    feature_names = [f'feature_{i}' for i in range(50)]

    markers = extract_top_markers(importance, feature_names, top_n=10)
    assert len(markers) == 10
    assert all(isinstance(m, tuple) for m in markers)
    assert all(isinstance(m[0], str) for m in markers)
    assert all(isinstance(m[1], (int, float)) for m in markers)

def test_extract_markers_direction():
    """Test marker extraction with different directions."""
    importance = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    feature_names = ['a', 'b', 'c', 'd', 'e']

    markers_pos = extract_top_markers(importance, feature_names, top_n=3, direction='positive')
    assert markers_pos[0][0] == 'd'  # Highest

    markers_neg = extract_top_markers(importance, feature_names, top_n=3, direction='negative')
    assert markers_neg[0][0] == 'a'  # Lowest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction/test_utils.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write implementation**

```python
# biomekit/prediction/utils.py
"""Utility functions for microbiome prediction."""
import numpy as np
import matplotlib.pyplot as plt


def extract_top_markers(importance, feature_names, top_n=20, direction='positive'):
    """Extract top microbial markers based on importance scores.

    Parameters:
        importance: Array of importance scores
        feature_names: List of feature names
        top_n: Number of top markers to extract
        direction: 'positive' for highest, 'negative' for lowest

    Returns:
        List of (feature_name, score) tuples
    """
    if len(importance) != len(feature_names):
        raise ValueError("importance and feature_names must have same length")

    # Sort by importance
    if direction == 'positive':
        sorted_idx = np.argsort(importance)[::-1]
    else:
        sorted_idx = np.argsort(importance)

    top_idx = sorted_idx[:top_n]
    markers = [(feature_names[i], float(importance[i])) for i in top_idx]
    return markers


def plot_feature_importance(importance, feature_names=None, top_n=20,
                           direction='positive', save_path=None, figsize=(10, 8)):
    """Plot feature importance as horizontal bar chart.

    Parameters:
        importance: Array of importance scores
        feature_names: List of feature names (optional)
        top_n: Number of top features to plot
        direction: Sort direction
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(len(importance))]

    markers = extract_top_markers(importance, feature_names, top_n=top_n, direction=direction)
    names = [m[0] for m in markers]
    scores = [m[1] for m in markers]

    plt.figure(figsize=figsize)
    y_pos = np.arange(len(names))
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, names)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Microbial Markers')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_shap_summary(shap_values, feature_names=None, save_path=None, figsize=(10, 8)):
    """Plot SHAP summary beeswarm plot.

    Parameters:
        shap_values: SHAP values array (samples x features)
        feature_names: List of feature names (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    try:
        import shap
    except ImportError:
        print("shap library required for SHAP summary plot. Install with: pip install shap")
        return

    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def learning_curve(estimator, X, y, cv=5, scoring='accuracy', train_sizes=None,
                   save_path=None, figsize=(10, 6)):
    """Plot learning curve for model evaluation.

    Parameters:
        estimator: Fitted estimator or pipeline
        X: Feature matrix
        y: Target labels
        cv: Number of cross-validation folds
        scoring: Scoring metric
        train_sizes: Training set sizes (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        n_jobs=-1, train_sizes=train_sizes
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=figsize)
    plt.plot(train_sizes_abs, train_mean, 'o-', label='Training score', color='blue')
    plt.plot(train_sizes_abs, test_mean, 'o-', label='Cross-validation score', color='orange')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring)
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None, figsize=(8, 6)):
    """Plot confusion matrix.

    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction/test_utils.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add biomekit/prediction/utils.py tests/test_prediction/test_utils.py
git commit -m "feat: add prediction utils (marker extraction, plotting)"
```

---

### Task 8: Module Init & Integration

**Files:**
- Create: `biomekit/prediction/__init__.py`
- Modify: `biomekit/__init__.py`
- Create: `tests/test_prediction/__init__.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Create prediction __init__.py**

```python
# biomekit/prediction/__init__.py
"""Prediction module for microbiome-based disease classification and prognosis."""
from .preprocessing import CLRTransformer, LogTransformer, PercentTransformer, VarianceFilter, PreprocessingPipeline
from .encoders import AutoencoderEncoder, VAEEncoder
from .classifiers import MicrobiomeClassifier
from .regressors import SurvivalRegressor
from .pipeline import MicrobiomePipeline
from .explainer import SHAPExplainer, PermutationImportance, PartialDependencePlot
from .utils import extract_top_markers, plot_feature_importance, plot_shap_summary, learning_curve

__all__ = [
    # Preprocessing
    'CLRTransformer', 'LogTransformer', 'PercentTransformer', 'VarianceFilter', 'PreprocessingPipeline',
    # Encoders
    'AutoencoderEncoder', 'VAEEncoder',
    # Classifiers
    'MicrobiomeClassifier',
    # Regressors
    'SurvivalRegressor',
    # Pipeline
    'MicrobiomePipeline',
    # Explainers
    'SHAPExplainer', 'PermutationImportance', 'PartialDependencePlot',
    # Utils
    'extract_top_markers', 'plot_feature_importance', 'plot_shap_summary', 'learning_curve',
]
```

- [ ] **Step 2: Update main __init__.py**

```python
# Add to biomekit/__init__.py (after existing try blocks)
try:
    from biomekit.prediction import (
        MicrobiomePipeline, MicrobiomeClassifier, SurvivalRegressor,
        SHAPExplainer, PermutationImportance,
        extract_top_markers, plot_feature_importance, plot_shap_summary
    )
except ImportError:
    pass

# Add to __all__:
# Prediction (if available)
"MicrobiomePipeline", "MicrobiomeClassifier", "SurvivalRegressor",
"SHAPExplainer", "PermutationImportance",
"extract_top_markers", "plot_feature_importance", "plot_shap_summary",
```

- [ ] **Step 3: Create tests/test_prediction/__init__.py**

```python
# tests/test_prediction/__init__.py
"""Tests for prediction module."""
```

- [ ] **Step 4: Add integration test**

Add to `tests/test_integration.py`:

```python
def test_prediction_pipeline_integration():
    """Test full prediction pipeline with simulated data."""
    from biomekit.prediction import MicrobiomePipeline
    from biomekit.utils.simulate import simulate_abundance_data

    # Generate simulated data
    X, metadata = simulate_abundance_data(n_samples=100, n_features=200, n_diff_abundant=20)
    y = (metadata['group'] == 'disease').astype(int).values

    # Run full pipeline
    pipeline = MicrobiomePipeline(preprocess='clr', classifier='rf', encode=None)
    pipeline.fit(X, y)

    results = pipeline.evaluate(X, y, cv=3)
    assert results['accuracy'][0] >= 0  # Mean should be valid
    assert results['accuracy'][0] <= 1.0

    # Test get_feature_importance
    importance = pipeline.get_feature_importance()
    assert importance is not None

    print(f"Pipeline test passed. Accuracy: {results['accuracy'][0]:.3f} ± {results['accuracy'][1]:.3f}")
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All existing tests pass, new prediction tests pass

- [ ] **Step 6: Commit**

```bash
git add biomekit/prediction/__init__.py biomekit/__init__.py
git add tests/test_prediction/__init__.py tests/test_integration.py
git commit -m "feat: add prediction module init and integration tests"
```

---

### Task 9: Update Documentation

**Files:**
- Create: `docs/algorithm_notes/prediction_models.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write algorithm documentation**

```markdown
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
```

- [ ] **Step 2: Update README.md**

Add to roadmap table row for P1:
```
| P1 | Prediction Model Suite | ✅ Complete | Disease classification, efficacy prediction, prognosis models (ML/DL) |
```

- [ ] **Step 3: Update CHANGELOG.md**

```markdown
## [0.2.0] - 2026-05-26

### Added
- Prediction model module (biomekit.prediction)
  - Preprocessing: CLR, log, percent transforms, variance filtering
  - Autoencoders: AE and VAE for dimensionality reduction
  - Classifiers: RF, SVM, XGBoost, MLP, GradientBoosting
  - Survival: Cox proportional hazards, linear, logistic regression
  - Pipeline: Complete fit/predict/evaluate workflow
  - Explainers: SHAP, permutation importance, partial dependence
  - Utils: Marker extraction, visualization helpers
```

- [ ] **Step 4: Commit**

```bash
git add docs/algorithm_notes/prediction_models.md README.md CHANGELOG.md
git commit -m "docs: add prediction models documentation and update README"
```

---

## Dependencies to Add (pyproject.toml)

```toml
[tool.poetry.dependencies]
# Add these lines:
lifelines = "^0.27"
shap = "^0.42"
torch = "^2.0"
xgboost = "^2.0"
```

Run: `poetry add lifelines shap torch xgboost`

---

## Verification Checklist

- [ ] All 9 tasks completed
- [ ] 41 existing tests + new prediction tests pass
- [ ] SHAP values computed correctly
- [ ] Pipeline fit/predict/evaluate works end-to-end
- [ ] Documentation updated
- [ ] pyproject.toml dependencies added
- [ ] GitHub push with new tag v0.2.0