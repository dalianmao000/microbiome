import pytest
from biomekit.automl.search_space import SearchSpace, FeatureStrategy, ModelType

class TestSearchSpace:
    def test_feature_strategies(self):
        ss = SearchSpace()
        assert FeatureStrategy.LEFSE in ss.feature_strategies
        assert FeatureStrategy.RANDOM_FOREST_IMPORTANCE in ss.feature_strategies
        assert FeatureStrategy.LASSO in ss.feature_strategies

    def test_model_types(self):
        ss = SearchSpace()
        assert ModelType.RANDOM_FOREST in ss.model_types
        assert ModelType.XGBOOST in ss.model_types
        assert ModelType.SVM in ss.model_types
        assert ModelType.LOGISTIC_REGRESSION in ss.model_types

    def test_hyperparam_ranges(self):
        ss = SearchSpace()
        rf_params = ss.get_hyperparams(ModelType.RANDOM_FOREST)
        assert 'n_estimators' in rf_params
        assert 'max_depth' in rf_params
        assert rf_params['n_estimators']['type'] == 'int'
        assert rf_params['n_estimators']['min'] >= 10
        assert rf_params['n_estimators']['max'] <= 1000

    def test_search_space_to_dict(self):
        ss = SearchSpace()
        d = ss.to_dict()
        assert 'feature_strategies' in d
        assert 'model_types' in d
        assert 'hyperparams' in d