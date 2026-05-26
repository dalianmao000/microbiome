"""Tests for alpha diversity."""
import numpy as np
import pandas as pd
from biomekit.diversity.alpha import (
    observed_features, shannon_index, chao1_index,
    simpson_index, alpha_diversity
)


def test_observed_features():
    df = pd.DataFrame({'A': [10, 0, 5], 'B': [0, 20, 0]}, index=['S1', 'S2', 'S3'])
    result = observed_features(df)
    assert result.iloc[0] == 1  # S1 has A=10, B=0 -> 1 non-zero
    assert result.iloc[1] == 1
    assert result.iloc[2] == 1


def test_shannon_index():
    df = pd.DataFrame({'A': [10, 5], 'B': [5, 10]}, index=['S1', 'S2'])
    result = shannon_index(df)
    assert result.iloc[0] == result.iloc[1]  # Should be equal for uniform distribution


def test_chao1_index():
    df = pd.DataFrame({'A': [10, 20], 'B': [5, 0]}, index=['S1', 'S2'])
    result = chao1_index(df)
    assert result.iloc[0] >= 2


def test_simpson_index():
    df = pd.DataFrame({'A': [10, 10], 'B': [10, 10]}, index=['S1', 'S2'])
    result = simpson_index(df)
    assert result.iloc[0] == 0.5  # Maximum diversity for uniform 2-feature distribution


def test_alpha_diversity_all():
    df = pd.DataFrame(np.random.rand(10, 5), columns=[f'OTU_{i}' for i in range(5)])
    result = alpha_diversity(df)
    assert result.shape == (10, 4)  # 4 default metrics
    assert 'shannon' in result.columns
    assert 'observed' in result.columns