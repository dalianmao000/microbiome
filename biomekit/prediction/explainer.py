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
            try:
                self.explainer = shap.TreeExplainer(model_obj)
            except Exception:
                self.explainer = shap.KernelExplainer(
                    model_obj.predict_proba, X[:20]
                )

    def shap_values(self, X):
        self._init_explainer(X)
        if self.explainer is None:
            raise RuntimeError("Could not initialize SHAP explainer")

        X = np.asarray(X, dtype=np.float32)
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            # For binary classification, shap returns list of arrays for each class
            # Select class 1 (positive class) for binary classification
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # For classifiers, shape is (n_samples, n_features, n_classes)
        # Average across classes or take positive class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # Take positive class

        return shap_values


class PermutationImportance:
    """Permutation-based feature importance."""

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
    """Partial dependence plot computation."""

    def __init__(self, model, feature_idx):
        self.model = model
        self.feature_idx = feature_idx

    def compute(self, X, feature_values=None):
        from sklearn.inspection import partial_dependence

        model_obj = self.model.model if hasattr(self.model, 'model') else self.model

        # sklearn's partial_dependence uses percentiles and grid_resolution
        # instead of explicit feature_values in newer versions
        if feature_values is not None:
            # If explicit values are provided, compute manually
            results = []
            for val in feature_values:
                X_mod = X.copy()
                X_mod[:, self.feature_idx] = val
                pred = model_obj.predict_proba(X_mod)[:, 1]
                results.append(pred.mean())
            return np.array(results)

        # Use sklearn's built-in grid
        pd_results = partial_dependence(
            model_obj, X, self.feature_idx,
            method='auto',
            kind='average'
        )
        return pd_results['average'][0]