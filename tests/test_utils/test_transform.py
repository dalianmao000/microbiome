"""Tests for data transformation utilities."""
import numpy as np
import pandas as pd
from biomekit.utils.transform import clr_transform, normalize, rarefaction


def test_clr_transform():
    df = pd.DataFrame({'A': [10, 20], 'B': [30, 40]}, index=['S1', 'S2'])
    clr_df = clr_transform(df)
    assert clr_df.shape == df.shape
    assert np.allclose(clr_df.sum(axis=1), 0, atol=1e-10)


def test_normalize_relative():
    df = pd.DataFrame({'A': [10, 20], 'B': [30, 40]}, index=['S1', 'S2'])
    norm_df = normalize(df, method='relative')
    assert np.allclose(norm_df.sum(axis=1), 1.0)


def test_normalize_log():
    df = pd.DataFrame({'A': [10, 20], 'B': [30, 40]}, index=['S1', 'S2'])
    norm_df = normalize(df, method='log')
    assert norm_df.iloc[0, 0] == np.log1p(10)


def test_rarefaction():
    df = pd.DataFrame(np.random.randint(0, 100, (10, 5)),
                      index=[f'S{i}' for i in range(10)],
                      columns=[f'OTU_{i}' for i in range(5)])
    rarefied = rarefaction(df, random_seed=42)
    assert rarefied.shape == df.shape
    assert (rarefied.sum(axis=1) <= df.sum(axis=1)).all()