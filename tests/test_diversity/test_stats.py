"""Tests for diversity statistics."""
import numpy as np
import pandas as pd
from biomekit.diversity.stats import permanova, anosim


def test_permanova_fallback():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(20, 10), columns=[f'OTU_{i}' for i in range(10)])

    from biomekit.diversity.beta import beta_diversity
    dist = beta_diversity(df, metric='braycurtis')

    metadata = pd.DataFrame({
        'sample_id': df.index,
        'group': ['case'] * 10 + ['control'] * 10
    })

    result = permanova(dist, metadata, group_column='group', permutations=99)
    assert 'test_statistic' in result
    assert 'p_value' in result


def test_anosim_fallback():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(20, 10), columns=[f'OTU_{i}' for i in range(10)])

    from biomekit.diversity.beta import beta_diversity
    dist = beta_diversity(df, metric='braycurtis')

    metadata = pd.DataFrame({
        'sample_id': df.index,
        'group': ['case'] * 10 + ['control'] * 10
    })

    result = anosim(dist, metadata, group_column='group', permutations=99)
    assert 'test_statistic' in result
    assert 'p_value' in result