# tests/test_prediction/test_regressors.py
import pytest
import numpy as np
from biomekit.prediction.regressors import SurvivalRegressor

def test_cox_regression():
    """Test Cox proportional hazards model."""
    pytest.importorskip("lifelines", reason="lifelines required for Cox model")

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