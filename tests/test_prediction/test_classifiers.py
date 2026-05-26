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
    pytest.importorskip("xgboost")
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