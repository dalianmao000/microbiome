import pytest
import pandas as pd
import numpy as np
from biomekit.integration.multiomics_pipeline import MultiOmicsPipeline

class TestMultiOmicsPipeline:
    @pytest.fixture
    def sample_omics_data(self):
        np.random.seed(42)
        n_samples = 30
        return {
            '16s': pd.DataFrame(np.random.rand(n_samples, 10), columns=[f'16s_f{i}' for i in range(10)]),
            'metabolomics': pd.DataFrame(np.random.rand(n_samples, 10), columns=[f'meta_f{i}' for i in range(10)]),
            'metagenomics': pd.DataFrame(np.random.rand(n_samples, 10), columns=[f'mgn_f{i}' for i in range(10)]),
        }

    @pytest.fixture
    def sample_labels(self):
        return np.array([0, 1] * 15)

    def test_constructor_defaults(self, sample_omics_data, sample_labels):
        pipeline = MultiOmicsPipeline()
        assert pipeline.biomarker_method == 'diablo'
        assert pipeline.correlation_method == 'cca'
        assert pipeline.fusion == 'early'

    def test_fit_biomarker_discovery(self, sample_omics_data, sample_labels):
        pipeline = MultiOmicsPipeline(biomarker_method='cca')
        results = pipeline.fit_biomarker_discovery(sample_omics_data, sample_labels)
        assert 'method' in results
        assert 'loadings' in results

    def test_fit_correlation(self, sample_omics_data):
        pipeline = MultiOmicsPipeline()
        results = pipeline.fit_correlation(sample_omics_data)
        assert 'method' in results
        assert 'correlations' in results

    def test_fit_classification(self, sample_omics_data, sample_labels):
        pipeline = MultiOmicsPipeline(fusion='late', encode='pca')
        results = pipeline.fit_classification(sample_omics_data, sample_labels)
        assert 'fusion' in results
        assert 'accuracy' in results

    def test_fit_all(self, sample_omics_data, sample_labels):
        pipeline = MultiOmicsPipeline()
        report = pipeline.fit(sample_omics_data, sample_labels)
        # report is a MultiOmicsReport - just check it exists
        assert report is not None

    def test_invalid_fusion_raises(self, sample_omics_data, sample_labels):
        pipeline = MultiOmicsPipeline(fusion='invalid')
        with pytest.raises(ValueError, match="Invalid fusion"):
            pipeline.fit_classification(sample_omics_data, sample_labels)

    def test_missing_omics_key_raises(self, sample_labels):
        pipeline = MultiOmicsPipeline()
        with pytest.raises(ValueError, match="Missing required omics keys"):
            pipeline.fit_biomarker_discovery({'16s': pd.DataFrame(np.random.rand(5, 3))}, sample_labels)

    def test_invalid_biomarker_method(self, sample_omics_data, sample_labels):
        pipeline = MultiOmicsPipeline(biomarker_method='invalid_method')
        with pytest.raises(ValueError, match="Invalid biomarker_method"):
            pipeline.fit_biomarker_discovery(sample_omics_data, sample_labels)

    def test_invalid_correlation_method(self, sample_omics_data):
        pipeline = MultiOmicsPipeline(correlation_method='invalid_method')
        with pytest.raises(ValueError, match="Invalid correlation_method"):
            pipeline.fit_correlation(sample_omics_data)