"""
Data transformation utilities for microbiome analysis.

Includes CLR transformation, rarefaction, normalization, etc.
"""
import numpy as np
import pandas as pd
from typing import Optional


def clr_transform(df: pd.DataFrame, epsilon: float = 1e-10) -> pd.DataFrame:
    """
    Centered Log-Ratio (CLR) transformation.

    Handles zero values by adding small epsilon before log transformation.

    Parameters
    ----------
    df : pd.DataFrame
        Abundance table (samples x features)
    epsilon : float
        Small value to add before log to handle zeros

    Returns
    -------
    pd.DataFrame
        CLR-transformed abundance
    """
    df_eps = df + epsilon
    log_df = np.log(df_eps)
    geometric_means = np.exp(log_df.mean(axis=1))
    clr_df = log_df.subtract(np.log(geometric_means + epsilon), axis=0)
    return clr_df


def rarefaction(df: pd.DataFrame, depth: Optional[int] = None,
               random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Rarefy abundance table to specified sequencing depth.

    Parameters
    ----------
    df : pd.DataFrame
        Abundance table (samples x features)
    depth : int, optional
        Target sequencing depth. If None, uses minimum sample sum.
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Rarefied abundance table
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sample_sums = df.sum(axis=1)
    if depth is None:
        depth = int(sample_sums.min())

    rarefied_data = []
    for idx, row in df.iterrows():
        sample_depth = int(sample_sums[idx])
        if sample_depth < depth:
            rarefied_data.append(row.values)
            continue

        non_zero_indices = np.where(row.values > 0)[0]
        counts = row.values[non_zero_indices]
        reads = np.repeat(non_zero_indices, counts)
        np.random.shuffle(reads)
        selected = reads[:depth]

        new_row = np.zeros(df.shape[1])
        for i, idx in enumerate(non_zero_indices):
            new_row[idx] = (selected == idx).sum()

        rarefied_data.append(new_row)

    rarefied_df = pd.DataFrame(
        rarefied_data,
        index=df.index,
        columns=df.columns
    )
    return rarefied_df


def normalize(df: pd.DataFrame, method: str = 'relative') -> pd.DataFrame:
    """
    Normalize abundance table.

    Parameters
    ----------
    df : pd.DataFrame
        Abundance table
    method : str
        'relative' - divide by sample sum (relative abundance)
        'percent' - multiply by 100 for percentage
        'log' - log transform after adding 1

    Returns
    -------
    pd.DataFrame
        Normalized abundance
    """
    if method == 'relative':
        sample_sums = df.sum(axis=1)
        return df.div(sample_sums, axis=0)
    elif method == 'percent':
        sample_sums = df.sum(axis=1)
        return df.div(sample_sums, axis=0) * 100
    elif method == 'log':
        return np.log1p(df)
    else:
        raise ValueError(f"Unknown normalization method: {method}")