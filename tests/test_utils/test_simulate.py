"""Tests for simulated data generator."""
import numpy as np
import pandas as pd
from data.simulate_abundance import simulate_abundance_data, simulate_with_phylogeny


def test_simulate_returns_dataframe():
    abundance_df, metadata = simulate_abundance_data(n_samples=20, n_features=50)
    assert isinstance(abundance_df, pd.DataFrame)
    assert isinstance(metadata, pd.DataFrame)


def test_simulate_shapes():
    abundance_df, metadata = simulate_abundance_data(n_samples=20, n_features=50)
    assert abundance_df.shape == (20, 50)
    assert metadata.shape == (20, 4)  # sample_id, group, age, BMI


def test_simulate_has_diff_abundant_features():
    np.random.seed(42)
    abundance_df, metadata = simulate_abundance_data(
        n_samples=30, n_features=50,
        n_diff_abundant=5,
        diff_effect_size=2.0
    )
    # Check that at least some features show group differences
    case_mask = metadata['group'].values == 'case'
    control_mask = metadata['group'].values == 'control'
    case_mean = abundance_df.values[case_mask].mean(axis=0)
    control_mean = abundance_df.values[control_mask].mean(axis=0)
    diff_ratio = np.abs(case_mean / (control_mean + 1e-10))
    n_different = (diff_ratio > 1.5).sum()
    assert n_different >= 3  # at least 3 features should be different


def test_simulate_with_phylogeny():
    abundance_df, metadata, tree_newick = simulate_with_phylogeny(n_samples=20, n_features=50)
    assert isinstance(abundance_df, pd.DataFrame)
    assert isinstance(metadata, pd.DataFrame)
    assert isinstance(tree_newick, str)
    assert tree_newick.endswith(';')