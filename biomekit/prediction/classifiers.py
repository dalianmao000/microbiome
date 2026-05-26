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