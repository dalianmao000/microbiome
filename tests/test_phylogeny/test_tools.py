"""Tests for phylogeny tools."""
import pandas as pd
from biomekit.phylogeny.tools import build_tree, bootstrap_tree


def test_build_tree():
    sequences = pd.Series(['ATCGATCG', 'GCTAGCTA', 'TTTAATTT'] * 3)
    tree = build_tree(sequences)
    assert tree.endswith(';')
    assert '(' in tree


def test_bootstrap_tree():
    sequences = pd.Series(['ATCGATCG', 'GCTAGCTA', 'TTTAATTT'] * 3)
    tree, bootstrap_df = bootstrap_tree(sequences, n_bootstraps=10)
    assert tree.endswith(';')
    assert 'bootstrap_support' in bootstrap_df.columns