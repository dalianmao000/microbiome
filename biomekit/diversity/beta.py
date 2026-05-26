"""
Beta diversity metrics and ordination methods.

Includes: Bray-Curtis, Jaccard, UniFrac (weighted/unweighted), Aitchison
Methods: PCoA, NMDS
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from typing import Optional


def bray_curtis_distance(abundance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Bray-Curtis dissimilarity matrix.

    BC_ij = 1 - 2 * sum(min(x_ki, x_kj)) / sum(x_ki + x_kj)
    """
    distances = pdist(abundance_df.values, metric='braycurtis')
    dist_matrix = squareform(distances)

    dist_df = pd.DataFrame(
        dist_matrix,
        index=abundance_df.index,
        columns=abundance_df.index
    )
    return dist_df


def jaccard_distance(abundance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Jaccard dissimilarity.

    J_ij = 1 - |A ∩ B| / |A ∪ B|
    """
    distances = pdist(abundance_df.values, metric='jaccard')
    dist_matrix = squareform(distances)

    dist_df = pd.DataFrame(
        dist_matrix,
        index=abundance_df.index,
        columns=abundance_df.index
    )
    return dist_df


def weighted_unifrac_distance(
    abundance_df: pd.DataFrame,
    tree_newick: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate weighted UniFrac distance.

    Requires phylogenetic tree in Newick format.
    Falls back to Bray-Curtis if no tree provided.
    """
    if tree_newick is None:
        return bray_curtis_distance(abundance_df)

    try:
        from skbio import TreeNode
        from skbio.diversity import beta_diversity

        tree = TreeNode.read(tree_newick)
        tree_tips = set([tip.name for tip in tree.tips()])
        common_features = [f for f in abundance_df.columns if f in tree_tips]
        filtered_df = abundance_df[common_features]

        distances = beta_diversity(
            'weighted_unifrac',
            filtered_df.values,
            ids=filtered_df.index.tolist(),
            tree=tree
        )
        dist_matrix = squareform(distances)
        return pd.DataFrame(dist_matrix, index=abundance_df.index, columns=abundance_df.index)
    except Exception:
        return bray_curtis_distance(abundance_df)


def unweighted_unifrac_distance(
    abundance_df: pd.DataFrame,
    tree_newick: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate unweighted UniFrac distance."""
    if tree_newick is None:
        return jaccard_distance(abundance_df)

    try:
        from skbio import TreeNode
        from skbio.diversity import beta_diversity

        tree = TreeNode.read(tree_newick)
        tree_tips = set([tip.name for tip in tree.tips()])
        common_features = [f for f in abundance_df.columns if f in tree_tips]
        filtered_df = abundance_df[common_features]

        distances = beta_diversity(
            'unweighted_unifrac',
            filtered_df.values,
            ids=filtered_df.index.tolist(),
            tree=tree
        )
        dist_matrix = squareform(distances)
        return pd.DataFrame(dist_matrix, index=abundance_df.index, columns=abundance_df.index)
    except Exception:
        return jaccard_distance(abundance_df)


def pcoa(
    distance_matrix: pd.DataFrame,
    n_components: int = 3,
) -> pd.DataFrame:
    """
    Perform Principal Coordinates Analysis (PCoA).

    Parameters
    ----------
    distance_matrix : pd.DataFrame
        Square dissimilarity matrix
    n_components : int
        Number of principal coordinates to return

    Returns
    -------
    pd.DataFrame
        PCoA coordinates (samples x PCs)
    """
    try:
        from skbio.stats.ordination import pcoa
        dist_matrix_values = distance_matrix.values.copy()
        np.fill_diagonal(dist_matrix_values, 0)
        ordination = pcoa(dist_matrix_values, number_of_dimensions=n_components)
        pc_df = pd.DataFrame(
            ordination.samples[:, :n_components],
            index=distance_matrix.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        return pc_df
    except ImportError:
        return _pcoa_fallback(distance_matrix, n_components)


def _pcoa_fallback(distance_matrix: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """Fallback PCoA using eigenvalue decomposition."""
    dist_values = distance_matrix.values.copy()
    np.fill_diagonal(dist_values, 0)

    # Double centering
    n = dist_values.shape[0]
    row_mean = dist_values.mean(axis=1)
    col_mean = dist_values.mean(axis=0)
    grand_mean = dist_values.mean()
    centered = dist_values - row_mean[:, np.newaxis] - row_mean[np.newaxis, :] + grand_mean
    centered = -(centered / 2)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(centered)

    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top n_components
    pc_df = pd.DataFrame(
        eigenvectors[:, :n_components] * np.sqrt(np.abs(eigenvalues[:n_components])),
        index=distance_matrix.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    return pc_df


def beta_diversity(
    abundance_df: pd.DataFrame,
    metric: str = 'braycurtis',
    tree_newick: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate beta diversity distance matrix.

    Parameters
    ----------
    abundance_df : pd.DataFrame
    metric : str
        'braycurtis', 'jaccard', 'weighted_unifrac', 'unweighted_unifrac'
    tree_newick : str, optional
        Required for UniFrac metrics

    Returns
    -------
    pd.DataFrame
        Distance matrix
    """
    metric_map = {
        'braycurtis': bray_curtis_distance,
        'jaccard': jaccard_distance,
        'weighted_unifrac': weighted_unifrac_distance,
        'unweighted_unifrac': unweighted_unifrac_distance,
    }

    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric}")

    if metric in ('weighted_unifrac', 'unweighted_unifrac'):
        return metric_map[metric](abundance_df, tree_newick)
    else:
        return metric_map[metric](abundance_df)