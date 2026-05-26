import pytest
import pandas as pd
import numpy as np
from biomekit.automl.train import AutoMLTrain

class TestAutoMLTrain:
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
        trainer = AutoMLTrain(X, y)
        assert trainer.X is not None
        assert trainer.y is not None

    def test_generate_config(self, sample_data):
        trainer = AutoMLTrain(*sample_data)
        config = trainer.generate_config()
        assert 'feature_strategy' in config
        assert 'model_type' in config
        assert 'n_features' in config
        assert 'hyperparams' in config

    def test_generate_config_with_constraints(self, sample_data):
        trainer = AutoMLTrain(*sample_data)
        config = trainer.generate_config(model_type='xgb')
        assert config['model_type'] == 'xgb'

    def test_run_iteration(self, sample_data):
        trainer = AutoMLTrain(*sample_data)
        config = trainer.generate_config()
        result = trainer.run_iteration(config)
        assert 'score' in result
        assert 'metrics' in result
        assert 'config' in result

    def test_run_iteration_returns_dict(self, sample_data):
        trainer = AutoMLTrain(*sample_data)
        config = trainer.generate_config()
        result = trainer.run_iteration(config)
        assert isinstance(result, dict)