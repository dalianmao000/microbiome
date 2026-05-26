"""
End-to-end integration tests for the full analysis pipeline.
"""
import numpy as np
import pandas as pd
from data.simulate_abundance import simulate_abundance_data
from biomekit.diversity.alpha import alpha_diversity
from biomekit.diversity.beta import beta_diversity, pcoa
from biomekit.abundance.lefse import run_lefse
from biomekit.function.picrust2 import run_picrust2
from biomekit.network.correlation import spearman_network


def test_full_pipeline():
    """Test complete analysis from simulated data to results."""
    np.random.seed(42)
    abundance_df, metadata = simulate_abundance_data(
        n_samples=50,
        n_features=100,
        n_diff_abundant=10,
        diff_effect_size=2.0
    )

    # Alpha diversity
    alpha_results = alpha_diversity(abundance_df)
    assert alpha_results.shape == (50, 4), f"Expected (50, 4), got {alpha_results.shape}"

    # Beta diversity
    dist_matrix = beta_diversity(abundance_df, metric='braycurtis')
    assert dist_matrix.shape == (50, 50), f"Expected (50, 50), got {dist_matrix.shape}"

    # PCoA
    pcoa_results = pcoa(dist_matrix)
    assert pcoa_results.shape[0] == 50, f"Expected 50 samples, got {pcoa_results.shape[0]}"

    # LEfSe
    lefse_results = run_lefse(abundance_df, metadata, group_column='group')
    assert 'lda_scores' in lefse_results
    assert 'summary' in lefse_results

    print("All pipeline tests passed!")


def test_functional_annotation():
    """Test PICRUSt2 functional annotation."""
    np.random.seed(42)
    abundance_df, _ = simulate_abundance_data(n_samples=20, n_features=50)

    result = run_picrust2(abundance_df)
    assert 'KO_abundance' in result
    assert result['KO_abundance'].shape[0] == 20


def test_network_analysis():
    """Test network correlation analysis."""
    np.random.seed(42)
    abundance_df, _ = simulate_abundance_data(n_samples=30, n_features=30)

    network = spearman_network(abundance_df, alpha=0.05)
    assert 'source' in network.columns
    assert 'target' in network.columns


def test_metadata_consistency():
    """Test that metadata is consistent with abundance data."""
    np.random.seed(123)
    abundance_df, metadata = simulate_abundance_data(n_samples=50, n_features=100)

    assert len(metadata) == len(abundance_df)
    assert abundance_df.index.tolist() == metadata['sample_id'].tolist()