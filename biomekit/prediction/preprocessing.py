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