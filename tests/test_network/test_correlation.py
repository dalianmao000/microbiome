"""Tests for network correlation analysis."""
import numpy as np
import pandas as pd
from biomekit.network.correlation import (
    spearman_correlation, sparcc_correlation,
    build_correlation_network, sparcc_network, spearman_network
)


def test_spearman_correlation():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(30, 10), columns=[f'OTU_{i}' for i in range(10)])

    corr, pval = spearman_correlation(df)

    assert corr.shape == (10, 10)
    assert pval.shape == (10, 10)
    assert np.allclose(np.diag(corr), 1.0)


def test_sparcc_correlation():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(30, 10), columns=[f'OTU_{i}' for i in range(10)])

    corr, pval = sparcc_correlation(df, threshold=0.1)

    assert corr.shape == (10, 10)


def test_build_correlation_network():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(30, 10), columns=[f'OTU_{i}' for i in range(10)])

    corr, pval = spearman_correlation(df)
    network = build_correlation_network(corr, pval, correlation_threshold=0.3, p_value_threshold=0.05)

    assert 'source' in network.columns
    assert 'target' in network.columns
    assert 'correlation' in network.columns


def test_sparcc_network():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(30, 10), columns=[f'OTU_{i}' for i in range(10)])

    network = sparcc_network(df, threshold=0.3)

    assert len(network) >= 0


def test_spearman_network():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(30, 10), columns=[f'OTU_{i}' for i in range(10)])

    network = spearman_network(df, alpha=0.05)

    assert 'source' in network.columns