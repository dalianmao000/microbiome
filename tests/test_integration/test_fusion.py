import pytest
import pandas as pd
import numpy as np
from biomekit.integration.fusion import (
    CCAAnalyzer,
    ProcrustesAnalyzer,
)

class TestCCAAnalyzer:
    def test_fit_transform(self):
        np.random.seed(42)
        df_a = pd.DataFrame(np.random.rand(10, 5), columns=[f'a{i}' for i in range(5)])
        df_b = pd.DataFrame(np.random.rand(10, 5), columns=[f'b{i}' for i in range(5)])
        data_dict = {'omics_a': df_a, 'omics_b': df_b}

        analyzer = CCAAnalyzer(n_components=2)
        result = analyzer.fit_transform(data_dict)

        assert 'loadings' in result
        assert 'correlations' in result
        assert 'scores' in result
        assert result['method'] == 'cca'

    def test_get_loadings(self):
        np.random.seed(42)
        df_a = pd.DataFrame(np.random.rand(10, 3), columns=[f'a{i}' for i in range(3)])
        df_b = pd.DataFrame(np.random.rand(10, 3), columns=[f'b{i}' for i in range(3)])
        analyzer = CCAAnalyzer(n_components=2)
        analyzer.fit({'omics_a': df_a, 'omics_b': df_b})
        loadings = analyzer.get_loadings()
        assert isinstance(loadings, pd.DataFrame)

    def test_wrong_number_of_blocks_raises(self):
        df_a = pd.DataFrame(np.random.rand(10, 3), columns=[f'a{i}' for i in range(3)])
        analyzer = CCAAnalyzer()
        with pytest.raises(ValueError, match="exactly two"):
            analyzer.fit({'omics_a': df_a})

class TestProcrustesAnalyzer:
    def test_fit_transform(self):
        np.random.seed(42)
        coords_a = np.random.rand(10, 5)
        coords_b = np.random.rand(10, 5)
        analyzer = ProcrustesAnalyzer()
        result = analyzer.fit_transform({'pcoa_a': coords_a, 'pcoa_b': coords_b})

        assert result['method'] == 'procrustes'
        assert 'statistic' in result
        assert 'residuals' in result

    def test_get_loadings_raises(self):
        analyzer = ProcrustesAnalyzer()
        with pytest.raises(NotImplementedError):
            analyzer.get_loadings()

    def test_wrong_number_of_matrices_raises(self):
        np.random.seed(42)
        coords_a = np.random.rand(10, 5)
        analyzer = ProcrustesAnalyzer()
        with pytest.raises(ValueError, match="exactly two"):
            analyzer.fit({'pcoa_a': coords_a})