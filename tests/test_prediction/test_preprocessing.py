import pytest
import numpy as np
import pandas as pd
from biomekit.prediction.preprocessing import CLRTransformer, VarianceFilter, LogTransformer, PercentTransformer, PreprocessingPipeline

def test_clr_transform():
    """Test CLR transformation on small dataset."""
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
    transformer = CLRTransformer()
    result = transformer.fit_transform(X)
    assert result.shape == X.shape
    assert not np.any(np.isnan(result))

def test_log_transform():
    """Test log transformation."""
    X = np.array([[1, 2], [4, 8]])
    transformer = LogTransformer()
    result = transformer.fit_transform(X)
    assert result.shape == X.shape
    assert np.all(result >= 0)

def test_percent_transform():
    """Test percentile transformation."""
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    transformer = PercentTransformer()
    result = transformer.fit_transform(X)
    assert result.shape == X.shape
    assert np.all(result >= 0) and np.all(result <= 1)

def test_variance_filter():
    """Test variance-based feature filter."""
    X = np.array([[1, 50, 1], [2, 60, 2], [3, 70, 3], [4, 80, 4]], dtype=float)
    filter_obj = VarianceFilter(threshold=10.0)
    result = filter_obj.fit_transform(X)
    assert result.shape[1] < X.shape[1]  # Some features filtered

def test_preprocessing_pipeline_clr():
    """Test full preprocessing pipeline with CLR."""
    X = np.random.rand(50, 100) + 0.1  # Add small offset to avoid zeros
    pipeline = PreprocessingPipeline(transform='clr', filter_low_var=True, variance_threshold=0.01)
    result = pipeline.fit_transform(X)
    assert result.shape[0] == 50
    assert result.shape[1] <= 100
    assert not np.any(np.isnan(result))

def test_preprocessing_pipeline_log():
    """Test preprocessing pipeline with log transform."""
    X = np.random.rand(30, 50) + 0.1
    pipeline = PreprocessingPipeline(transform='log', filter_low_var=False)
    result = pipeline.fit_transform(X)
    assert result.shape == X.shape