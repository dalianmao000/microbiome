"""Tests for beta diversity."""
import numpy as np
import pandas as pd
from biomekit.diversity.beta import (
    bray_curtis_distance, jaccard_distance,
    beta_diversity, pcoa
)


def test_bray_curtis():
    df = pd.DataFrame({'A': [10, 5], 'B': [5, 10]}, index=['S1', 'S2'])
    result = bray_curtis_distance(df)
    assert result.iloc[0, 1] == result.iloc[1, 0]
    assert result.iloc[0, 0] == 0


def test_jaccard():
    df = pd.DataFrame({'A': [10, 0], 'B': [0, 10]}, index=['S1', 'S2'])
    result = jaccard_distance(df)
    assert result.iloc[0, 1] == 1.0


def test_beta_diversity_default():
    df = pd.DataFrame(np.random.rand(10, 5), columns=[f'OTU_{i}' for i in range(5)])
    result = beta_diversity(df)
    assert result.shape == (10, 10)


def test_pcoa():
    df = pd.DataFrame(np.random.rand(10, 5), columns=[f'OTU_{i}' for i in range(5)])
    dist = beta_diversity(df, metric='braycurtis')
    result = pcoa(dist, n_components=2)
    assert result.shape[0] == 10
    assert 'PC1' in result.columns
    assert 'PC2' in result.columns