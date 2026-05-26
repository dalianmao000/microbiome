import pytest
import pandas as pd
import numpy as np
from biomekit.integration.report import MultiOmicsReport

class TestMultiOmicsReport:
    def test_constructor(self):
        results = {
            'biomarkers': {
                'method': 'diablo',
                'loadings': pd.DataFrame({'16s': [0.1, 0.2], 'metabolomics': [0.3, 0.4]}),
                'selected_features': {'16s': ['f1'], 'metabolomics': ['f2']},
                'components': np.array([[0.5], [0.6]]),
                'explained_variance': 0.7,
            },
            'correlations': {
                'method': 'cca',
                'loadings': pd.DataFrame(),
                'correlations': np.array([0.8, 0.6]),
                'scores': {},
            },
            'predictions': {
                'fusion': 'early',
                'y_true': np.array([0, 1, 0, 1]),
                'y_pred': np.array([0, 1, 0, 0]),
                'scores': np.array([0.9, 0.7, 0.8, 0.3]),
                'accuracy': (0.75, 0.1),
                'f1': (0.8, 0.15),
                'auc': (0.85, 0.1),
                'cv_results': {},
            },
        }
        report = MultiOmicsReport(results)
        assert report.biomarkers is not None
        assert report.correlations is not None
        assert report.predictions is not None

    def test_biomarkers_parsed(self):
        results = {
            'biomarkers': {
                'method': 'diablo',
                'loadings': pd.DataFrame({'16s': [0.1, 0.2], 'metabolomics': [0.3, 0.4]}),
                'selected_features': {},
                'components': np.array([]),
                'explained_variance': 0.0,
            },
            'correlations': {},
            'predictions': {},
        }
        report = MultiOmicsReport(results)
        assert isinstance(report.biomarkers, pd.DataFrame)

    def test_plot(self):
        results = {
            'biomarkers': {'method': 'diablo', 'loadings': pd.DataFrame(), 'selected_features': {}, 'components': np.array([]), 'explained_variance': 0.0},
            'correlations': {'method': 'cca', 'loadings': pd.DataFrame(), 'correlations': np.array([]), 'scores': {}},
            'predictions': {'fusion': 'early', 'y_true': np.array([]), 'y_pred': np.array([]), 'scores': np.array([]), 'accuracy': (0.0, 0.0), 'f1': (0.0, 0.0), 'auc': (0.0, 0.0), 'cv_results': {}},
        }
        report = MultiOmicsReport(results)
        fig = report.plot()
        assert fig is not None
        assert len(fig.axes) == 2

    def test_to_dict(self):
        results = {
            'biomarkers': {'method': 'diablo', 'loadings': pd.DataFrame(), 'selected_features': {}, 'components': np.array([]), 'explained_variance': 0.0},
            'correlations': {'method': 'cca', 'loadings': pd.DataFrame(), 'correlations': np.array([]), 'scores': {}},
            'predictions': {'fusion': 'early', 'y_true': np.array([]), 'y_pred': np.array([]), 'scores': np.array([]), 'accuracy': (0.0, 0.0), 'f1': (0.0, 0.0), 'auc': (0.0, 0.0), 'cv_results': {}},
        }
        report = MultiOmicsReport(results)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert 'biomarkers' in d
        assert 'correlations' in d
        assert 'predictions' in d

    def test_summary(self):
        results = {
            'biomarkers': {'method': 'diablo', 'loadings': pd.DataFrame(), 'selected_features': {}, 'components': np.array([]), 'explained_variance': 0.0},
            'correlations': {'method': 'cca', 'loadings': pd.DataFrame(), 'correlations': np.array([]), 'scores': {}},
            'predictions': {'fusion': 'early', 'y_true': np.array([]), 'y_pred': np.array([]), 'scores': np.array([]), 'accuracy': (0.75, 0.1), 'f1': (0.8, 0.15), 'auc': (0.85, 0.1), 'cv_results': {}},
        }
        report = MultiOmicsReport(results)
        report.summary()  # Should not raise