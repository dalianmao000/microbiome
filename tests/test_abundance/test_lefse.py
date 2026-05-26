"""Tests for LEfSe implementation."""
import pandas as pd
import numpy as np
from biomekit.abundance.lefse import run_lefse, kruskal_wallis_test, wilcoxon_test, lda_regression


def test_kruskal_wallis():
    np.random.seed(42)
    data = np.random.negative_binomial(5, 0.5, size=(20, 10))
    df = pd.DataFrame(data, columns=[f'OTU_{i}' for i in range(10)])
    groups = pd.Series(['case'] * 10 + ['control'] * 10)

    result = kruskal_wallis_test(df, groups)
    assert 'feature' in result.columns
    assert 'H_statistic' in result.columns
    assert 'p_value' in result.columns


def test_wilcoxon():
    np.random.seed(42)
    data = np.random.rand(20, 5)
    df = pd.DataFrame(data, columns=[f'OTU_{i}' for i in range(5)])

    result = wilcoxon_test(df, list(range(10)), list(range(10, 20)))
    assert len(result) > 0


def test_lda_regression():
    np.random.seed(42)
    data = np.random.rand(20, 10)
    df = pd.DataFrame(data, columns=[f'OTU_{i}' for i in range(10)])
    groups = pd.Series(['case'] * 10 + ['control'] * 10)

    result = lda_regression(df, groups)
    assert 'feature' in result.columns
    assert 'lda_score' in result.columns


def test_run_lefse():
    np.random.seed(42)
    n_samples = 30
    n_features = 50

    case_data = np.random.negative_binomial(n=5, p=0.5, size=(15, n_features))
    control_data = np.random.negative_binomial(n=5, p=0.3, size=(15, n_features))

    abundance_df = pd.DataFrame(
        np.vstack([case_data, control_data]),
        index=[f'S{i:04d}' for i in range(n_samples)],
        columns=[f'OTU_{i:04d}' for i in range(n_features)]
    )

    metadata = pd.DataFrame({
        'sample_id': abundance_df.index,
        'group': ['case'] * 15 + ['control'] * 15
    })

    results = run_lefse(abundance_df, metadata, group_column='group')

    assert 'lda_scores' in results
    assert 'effect_sizes' in results
    assert 'summary' in results