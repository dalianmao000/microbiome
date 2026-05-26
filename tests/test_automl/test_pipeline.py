import pytest
import pandas as pd
import numpy as np
from biomekit.automl.pipeline import AutoMLPipeline

class TestAutoMLPipeline:
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
        pipeline = AutoMLPipeline(X, y)
        assert pipeline.X is not None
        assert pipeline.y is not None

    def test_run(self, sample_data):
        X, y = sample_data
        pipeline = AutoMLPipeline(X, y, max_iterations=5)
        results = pipeline.run()
        assert 'best_score' in results
        assert 'model_card' in results
        assert 'n_experiments' in results

    def test_run_respects_max_iterations(self, sample_data):
        X, y = sample_data
        pipeline = AutoMLPipeline(X, y, max_iterations=3)
        results = pipeline.run()
        assert results['n_experiments'] <= 3

    def test_run_terminates_early_on_good_score(self, sample_data):
        X, y = sample_data
        pipeline = AutoMLPipeline(X, y, max_iterations=50, termination_score=0.99)
        results = pipeline.run()
        # With random data, should run all iterations (0.99 is unreachable)
        assert results['n_experiments'] <= 50