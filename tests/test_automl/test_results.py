import pytest
from biomekit.automl.results import ExperimentResults

class TestExperimentResults:
    def test_add_result(self):
        results = ExperimentResults()
        result = {
            'score': 0.75,
            'metrics': {'performance': 0.8, 'stability': 0.7, 'bio_relevance': 0.6},
            'config': {'feature_strategy': 'rf_importance', 'model_type': 'rf'},
        }
        results.add_result(result)
        assert results.n_results == 1

    def test_best_result(self):
        results = ExperimentResults()
        results.add_result({'score': 0.6, 'config': {}, 'metrics': {}})
        results.add_result({'score': 0.8, 'config': {}, 'metrics': {}})
        results.add_result({'score': 0.7, 'config': {}, 'metrics': {}})
        assert results.best_score == 0.8
        assert results.best_result['score'] == 0.8

    def test_generate_model_card(self):
        results = ExperimentResults()
        results.add_result({
            'score': 0.75,
            'metrics': {'performance': 0.8, 'stability': 0.7, 'bio_relevance': 0.6},
            'config': {'feature_strategy': 'rf_importance', 'model_type': 'rf', 'n_features': 15},
        })
        card = results.generate_model_card()
        assert 'score' in card
        assert 'config' in card
        assert 'metrics' in card

    def test_empty_results(self):
        results = ExperimentResults()
        assert results.best_score == 0.0
        assert results.best_result is None