import pytest
import pandas as pd
import numpy as np
from biomekit.automl.prepare import AutoMLPrepare

class TestAutoMLPrepare:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 50
        X = pd.DataFrame(
            np.random.rand(n_samples, 20),
            columns=[f'feature_{i}' for i in range(20)]
        )
        y = np.array([0, 1] * 25)
        return X, y

    def test_constructor(self, sample_data):
        X, y = sample_data
        prep = AutoMLPrepare(X, y, cv_folds=5)
        assert prep.cv_folds == 5
        assert prep.X is not None
        assert prep.y is not None

    def test_get_fold_indices(self, sample_data):
        X, y = sample_data
        prep = AutoMLPrepare(X, y, cv_folds=5)
        folds = prep.get_fold_indices()
        assert len(folds) == 5
        assert all(len(f) > 0 for f in folds)

    def test_evaluate_config(self, sample_data):
        X, y = sample_data
        prep = AutoMLPrepare(X, y, cv_folds=3)
        config = {
            'feature_strategy': 'rf_importance',
            'model_type': 'rf',
            'n_features': 10,
            'hyperparams': {'n_estimators': 100, 'max_depth': 5},
        }
        result = prep.evaluate_config(config)
        assert 'score' in result
        assert 'metrics' in result
        assert 'selected_features' in result
        assert 0.0 <= result['score'] <= 1.0

    def test_evaluate_config_with_stability(self, sample_data):
        X, y = sample_data
        prep = AutoMLPrepare(X, y, cv_folds=5)
        config = {
            'feature_strategy': 'rf_importance',
            'model_type': 'rf',
            'n_features': 10,
            'hyperparams': {'n_estimators': 100, 'max_depth': 5},
        }
        result = prep.evaluate_config(config, compute_stability=True)
        assert 'stability' in result['metrics']