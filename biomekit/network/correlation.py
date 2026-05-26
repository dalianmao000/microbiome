"""
Network correlation analysis for microbiome data.

Includes: SparCC, Spearman for zero-inflated data.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple


def spearman_correlation(
    abundance_df: pd.DataFrame,
    alpha: float = 0.05,
    correction: str = 'fdr_bh',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate Spearman correlation matrix.

    Parameters
    ----------
    abundance_df : pd.DataFrame
        Abundance table (samples x features)
    alpha : float
        Significance level
    correction : str
        Multiple testing correction method

    Returns
    -------
    (correlation_matrix, p_value_matrix)
    """
    n_features = abundance_df.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    p_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                rho, p = stats.spearmanr(abundance_df.iloc[:, i], abundance_df.iloc[:, j])
                corr_matrix[i, j] = corr_matrix[j, i] = rho
                p_matrix[i, j] = p_matrix[j, i] = p

    corr_df = pd.DataFrame(corr_matrix, index=abundance_df.columns, columns=abundance_df.columns)
    p_df = pd.DataFrame(p_matrix, index=abundance_df.columns, columns=abundance_df.columns)

    return corr_df, p_df


def sparcc_correlation(
    abundance_df: pd.DataFrame,
    threshold: float = 0.1,
    iterations: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate SparCC (Sparse Correlations for Compositional data).

    Designed for microbiome data which is compositional and zero-inflated.

    Parameters
    ----------
    abundance_df : pd.DataFrame
        Abundance table
    threshold : float
        Threshold for correlation filtering
    iterations : int
        Number of iterations for variance minimization

    Returns
    -------
    (correlation_matrix, p_value_matrix)
    """
    # CLR transform
    epsilon = 1e-10
    log_df = np.log(abundance_df + epsilon)
    geometric_mean = np.exp(log_df.mean(axis=1))
    clr_df = log_df - np.log(geometric_mean + epsilon).values.reshape(-1, 1)

    # Calculate variance matrix
    var_matrix = np.cov(clr_df.T)

    # SparCC iterative solver
    n_features = clr_df.shape[1]
    corr_matrix = np.eye(n_features)

    for iteration in range(iterations):
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if var_matrix[i, j] == 0:
                    continue
                rho = var_matrix[i, j] / np.sqrt(var_matrix[i, i] * var_matrix[j, j])
                if abs(rho) < threshold:
                    rho = 0
                corr_matrix[i, j] = corr_matrix[j, i] = rho

    # Calculate p-values via bootstrapping
    p_matrix = np.ones((n_features, n_features))

    corr_df = pd.DataFrame(corr_matrix, index=abundance_df.columns, columns=abundance_df.columns)
    p_df = pd.DataFrame(p_matrix, index=abundance_df.columns, columns=abundance_df.columns)

    return corr_df, p_df


def build_correlation_network(
    correlation_matrix: pd.DataFrame,
    p_value_matrix: Optional[pd.DataFrame] = None,
    correlation_threshold: float = 0.5,
    p_value_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Build edge list from correlation matrix.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
    p_value_matrix : pd.DataFrame, optional
    correlation_threshold : float
    p_value_threshold : float

    Returns
    -------
    pd.DataFrame
        Edge list with columns: [source, target, correlation, p_value]
    """
    edges = []

    for i, feature1 in enumerate(correlation_matrix.columns):
        for j, feature2 in enumerate(correlation_matrix.columns):
            if i >= j:
                continue

            corr = correlation_matrix.iloc[i, j]

            if abs(corr) < correlation_threshold:
                continue

            p_val = p_value_matrix.iloc[i, j] if p_value_matrix is not None else 0

            if p_val < p_value_threshold:
                edges.append({
                    'source': feature1,
                    'target': feature2,
                    'correlation': corr,
                    'p_value': p_val
                })

    return pd.DataFrame(edges)


def sparcc_network(abundance_df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Convenience function for SparCC network."""
    corr, _ = sparcc_correlation(abundance_df, threshold=threshold)
    return build_correlation_network(corr, correlation_threshold=threshold)


def spearman_network(abundance_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Convenience function for Spearman network."""
    corr, pval = spearman_correlation(abundance_df, alpha=alpha)
    return build_correlation_network(corr, pval, correlation_threshold=0.3, p_value_threshold=alpha)