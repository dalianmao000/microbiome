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