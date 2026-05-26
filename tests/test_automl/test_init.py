import pytest

class TestModuleExports:
    def test_imports(self):
        from biomekit.automl import (
            AutoMLPipeline,
            Program,
            AutoMLPrepare,
            AutoMLTrain,
            ExperimentResults,
            SearchSpace,
            DomainMetrics,
        )
        assert AutoMLPipeline is not None
        assert Program is not None
        assert SearchSpace is not None

    def test_all_exports_present(self):
        from biomekit import automl
        expected = [
            'AutoMLPipeline',
            'Program',
            'AutoMLPrepare',
            'AutoMLTrain',
            'ExperimentResults',
            'SearchSpace',
            'FeatureStrategy',
            'ModelType',
            'DomainMetrics',
        ]
        for name in expected:
            assert hasattr(automl, name), f"Missing export: {name}"