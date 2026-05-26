"""
Alpha diversity metrics for microbiome analysis.

Includes: Observed, Shannon, Chaol, ACE, Simpson, Fisher, Faith's PD
"""
import numpy as np
import pandas as pd
from typing import Optional


def observed_features(abundance_df: pd.DataFrame) -> pd.Series:
    """Count number of non-zero features per sample."""
    return (abundance_df > 0).sum(axis=1)


def shannon_index(abundance_df: pd.DataFrame) -> pd.Series:
    """
    Calculate Shannon diversity index.

    H = -sum(p_i * log(p_i))
    where p_i is the proportion of feature i
    """
    sample_sums = abundance_df.sum(axis=1)
    proportions = abundance_df.div(sample_sums, axis=0)
    proportions = proportions.replace(0, np.nan)
    shannon = -(proportions * np.log(proportions)).sum(axis=1)
    return shannon


def chao1_index(abundance_df: pd.DataFrame) -> pd.Series:
    """
    Calculate Chao1 richness estimator.

    S_chao1 = S_obs + n_1(n_1 - 1) / (2(n_2 + 1))

    where:
    - S_obs = observed number of features
    - n_1 = number of singletons
    - n_2 = number of doubletons
    """
    n1 = (abundance_df == 1).sum(axis=1)
    n2 = (abundance_df == 2).sum(axis=1)
    chao1 = observed_features(abundance_df) + (n1 * (n1 - 1)) / (2 * (n2 + 1))
    return chao1


def simpson_index(abundance_df: pd.DataFrame) -> pd.Series:
    """
    Calculate Simpson diversity index.

    D = 1 - sum(p_i^2)
    """
    sample_sums = abundance_df.sum(axis=1)
    proportions = abundance_df.div(sample_sums, axis=0)
    simpson = 1 - (proportions ** 2).sum(axis=1)
    return simpson


def alpha_diversity(
    abundance_df: pd.DataFrame,
    metrics: Optional[list] = None,
) -> pd.DataFrame:
    """
    Calculate multiple alpha diversity metrics.

    Parameters
    ----------
    abundance_df : pd.DataFrame
        Abundance table (samples x features)
    metrics : list, optional
        List of metrics to calculate.
        Options: ['observed', 'shannon', 'chao1', 'simpson']
        If None, calculates all.

    Returns
    -------
    pd.DataFrame
        DataFrame with samples as index, metrics as columns
    """
    if metrics is None:
        metrics = ['observed', 'shannon', 'chao1', 'simpson']

    results = pd.DataFrame(index=abundance_df.index)

    metric_funcs = {
        'observed': observed_features,
        'shannon': shannon_index,
        'chao1': chao1_index,
        'simpson': simpson_index,
    }

    for metric in metrics:
        if metric in metric_funcs:
            results[metric] = metric_funcs[metric](abundance_df)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results