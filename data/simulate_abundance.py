"""
Simulated microbiome abundance data generator.

Generates realistic synthetic data for testing and demonstration purposes.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def simulate_abundance_data(
    n_samples: int = 50,
    n_features: int = 100,
    n_groups: int = 2,
    group_ratios: Optional[list] = None,
    n_diff_abundant: int = 10,
    diff_effect_size: float = 2.0,
    sparsity: float = 0.3,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate simulated microbiome abundance data with known differential features.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of microbial features (e.g., OTUs or species)
    n_groups : int
        Number of groups (default 2 for case/control)
    group_ratios : list
        Ratio of samples in each group, e.g. [0.5, 0.5]
    n_diff_abundant : int
        Number of features with true differential abundance
    diff_effect_size : float
        Effect size multiplier for differential features
    sparsity : float
        Proportion of zeros in the abundance matrix
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    abundance_df : pd.DataFrame
        Shape (n_samples, n_features), sample x feature abundance table
    metadata : pd.DataFrame
        Shape (n_samples, 4), contains sample_id, group, age, BMI columns
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if group_ratios is None:
        group_ratios = [1.0 / n_groups] * n_groups

    # Generate group assignments
    group_names = ['case', 'control'] if n_groups == 2 else [f'group_{i}' for i in range(n_groups)]
    group_assignments = np.random.choice(group_names, size=n_samples, p=group_ratios)

    # Generate base abundance using log-normal distribution
    base_means = np.random.lognormal(mean=2, sigma=2, size=n_features)
    base_abundance = np.zeros((n_samples, n_features))

    for i in range(n_samples):
        sample_abundance = base_means * np.random.lognormal(mean=0, sigma=1, size=n_features)
        base_abundance[i] = sample_abundance

    # Inject differential abundance signal
    diff_indices = np.random.choice(n_features, size=n_diff_abundant, replace=False)

    for idx in diff_indices:
        for i in range(n_samples):
            if group_assignments[i] == 'case':
                base_abundance[i, idx] *= diff_effect_size
            else:
                base_abundance[i, idx] /= diff_effect_size

    # Apply sparsity (add zeros)
    zero_mask = np.random.random((n_samples, n_features)) < sparsity
    base_abundance[zero_mask] = 0

    # Ensure non-negative
    base_abundance = np.maximum(base_abundance, 0)

    # Create DataFrame
    feature_names = [f'OTU_{i:04d}' for i in range(n_features)]
    sample_names = [f'S{i:04d}' for i in range(n_samples)]
    abundance_df = pd.DataFrame(base_abundance, index=sample_names, columns=feature_names)

    # Generate metadata
    metadata = pd.DataFrame({
        'sample_id': sample_names,
        'group': group_assignments,
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'BMI': np.random.normal(24, 4, n_samples)
    })

    return abundance_df, metadata


def simulate_with_phylogeny(
    n_samples: int = 50,
    n_features: int = 100,
    tree_depth: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Generate abundance data with simulated phylogenetic tree.

    Returns
    -------
    abundance_df : pd.DataFrame
    metadata : pd.DataFrame
    tree_newick : str
        Newick formatted tree string
    """
    abundance_df, metadata = simulate_abundance_data(n_samples, n_features)

    # Generate a simple balanced tree
    n_taxa = n_features
    taxa_names = abundance_df.columns.tolist()

    def generate_tree_recursive(names, depth):
        if depth == 0 or len(names) == 1:
            return names[0]
        mid = len(names) // 2
        left = generate_tree_recursive(names[:mid], depth - 1)
        right = generate_tree_recursive(names[mid:], depth - 1)
        return f'({left}:0.1,{right}:0.1)'

    tree_newick = generate_tree_recursive(taxa_names, tree_depth) + ';'

    return abundance_df, metadata, tree_newick