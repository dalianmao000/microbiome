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