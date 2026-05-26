"""
LEfSe (Linear Discriminant Analysis Effect Size) implementation.

Algorithm:
1. Kruskal-Wallis test for multi-group comparison
2. Wilcoxon test for pairwise group comparison
3. Linear Discriminant Analysis (LDA) to estimate effect size

Reference: Segata et al. (2011) Nature Methods
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, Union


def kruskal_wallis_test(
    abundance_df: pd.DataFrame,
    groups: pd.Series,
) -> pd.DataFrame:
    """
    Perform Kruskal-Wallis H-test for multi-group comparison.

    Parameters
    ----------
    abundance_df : pd.DataFrame
        Abundance table (samples x features)
    groups : pd.Series
        Group assignments for each sample

    Returns
    -------
    pd.DataFrame
        DataFrame with 'feature', 'H_statistic', 'p_value'
    """
    unique_groups = groups.unique()
    group_positions = {g: np.where(groups.values == g)[0] for g in unique_groups}

    results = []
    for feature in abundance_df.columns:
        feature_values = [abundance_df.iloc[group_positions[g], abundance_df.columns.get_loc(feature)].values for g in unique_groups]
        h_stat, p_value = stats.kruskal(*feature_values)
        results.append({
            'feature': feature,
            'H_statistic': h_stat,
            'p_value': p_value
        })

    return pd.DataFrame(results)


def wilcoxon_test(
    abundance_df: pd.DataFrame,
    group1_idx: list,
    group2_idx: list,
) -> pd.DataFrame:
    """
    Perform pairwise Wilcoxon rank-sum test.

    Parameters
    ----------
    abundance_df : pd.DataFrame
    group1_idx : list
        Sample indices for group 1
    group2_idx : list
        Sample indices for group 2

    Returns
    -------
    pd.DataFrame
        DataFrame with 'feature', 'W_statistic', 'p_value'
    """
    results = []
    for feature in abundance_df.columns:
        g1_values = abundance_df.iloc[group1_idx, abundance_df.columns.get_loc(feature)].values
        g2_values = abundance_df.iloc[group2_idx, abundance_df.columns.get_loc(feature)].values

        if len(g1_values) < 3 or len(g2_values) < 3:
            continue

        w_stat, p_value = stats.mannwhitneyu(g1_values, g2_values, alternative='two-sided')
        results.append({
            'feature': feature,
            'W_statistic': w_stat,
            'p_value': p_value
        })

    return pd.DataFrame(results)


def lda_regression(
    abundance_df: pd.DataFrame,
    groups: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Linear Discriminant Analysis to estimate effect sizes.

    Parameters
    ----------
    abundance_df : pd.DataFrame
    groups : pd.Series
    alpha : float
        Significance level for KW test

    Returns
    -------
    pd.DataFrame
        Features with significant LDA scores
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import LabelEncoder

    mean_abundance = abundance_df.mean()
    high_abundance_features = mean_abundance[mean_abundance > 1e-5].index
    filtered_df = abundance_df[high_abundance_features]

    if len(filtered_df.columns) == 0:
        return pd.DataFrame(columns=['feature', 'lda_score', 'group'])

    le = LabelEncoder()
    y = le.fit_transform(groups)

    lda = LinearDiscriminantAnalysis()
    lda.fit(filtered_df.values, y)

    lda_scores = np.abs(lda.coef_[0])

    results = []
    for i, feature in enumerate(filtered_df.columns):
        group_effect = {}
        for j, group in enumerate(le.classes_):
            group_positions = np.where(y == j)[0]
            group_mean = filtered_df.iloc[group_positions, filtered_df.columns.get_loc(feature)].mean()
            group_effect[group] = group_mean

        max_group = max(group_effect, key=group_effect.get)
        max_mean = group_effect[max_group]

        results.append({
            'feature': feature,
            'lda_score': lda_scores[i],
            'group': max_group,
            'mean_abundance': max_mean
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('lda_score', ascending=False)

    return results_df


def run_lefse(
    abundance_df: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str = 'group',
    subclass_column: Optional[str] = None,
    min_abundance: float = 1e-6,
    min_prevalence: float = 0.1,
    alpha: float = 0.05,
    lda_threshold: float = 2.0,
) -> Dict:
    """
    Run complete LEfSe analysis pipeline.

    Parameters
    ----------
    abundance_df : pd.DataFrame
        Abundance table (samples x features)
    metadata : pd.DataFrame
        Sample metadata
    group_column : str
        Column name for primary group
    subclass_column : str, optional
        Column name for subclass (for nested analysis)
    min_abundance : float
        Minimum mean abundance threshold
    min_prevalence : float
        Minimum prevalence threshold (fraction of samples)
    alpha : float
        Significance level
    lda_threshold : float
        LDA score threshold for significance

    Returns
    -------
    dict
        {
            'lda_scores': pd.DataFrame,
            'effect_sizes': pd.DataFrame,
            'kw_results': pd.DataFrame,
            'wilcoxon_results': pd.DataFrame,
            'summary': dict
        }
    """
    mean_abundance = abundance_df.mean()
    prevalence = (abundance_df > min_abundance).mean()

    filter_mask = (mean_abundance > min_abundance) & (prevalence >= min_prevalence)
    filtered_df = abundance_df.loc[:, filter_mask]

    if filtered_df.shape[1] == 0:
        return {
            'lda_scores': pd.DataFrame(),
            'effect_sizes': pd.DataFrame(),
            'kw_results': pd.DataFrame(),
            'wilcoxon_results': pd.DataFrame(),
            'summary': {'n_significant': 0, 'message': 'No features passed filtering'}
        }

    groups = metadata[group_column]

    kw_results = kruskal_wallis_test(filtered_df, groups)

    from statsmodels.stats.multitest import multipletests
    kw_results['p_adjusted'] = multipletests(kw_results['p_value'], method='fdr_bh')[1]

    sig_features = kw_results[kw_results['p_adjusted'] < alpha]['feature'].tolist()

    unique_groups = groups.unique()
    if len(unique_groups) == 2:
        group1_idx = groups[groups == unique_groups[0]].index.tolist()
        group2_idx = groups[groups == unique_groups[1]].index.tolist()
        wilcoxon_results = wilcoxon_test(filtered_df[sig_features], group1_idx, group2_idx)
    else:
        wilcoxon_results = pd.DataFrame()

    lda_results = lda_regression(filtered_df, groups, alpha)

    significant_lda = lda_results[lda_results['lda_score'] >= lda_threshold]

    summary = {
        'n_input_features': abundance_df.shape[1],
        'n_filtered_features': filtered_df.shape[1],
        'n_significant': len(significant_lda),
        'lda_threshold': lda_threshold
    }

    return {
        'lda_scores': significant_lda,
        'effect_sizes': lda_results,
        'kw_results': kw_results,
        'wilcoxon_results': wilcoxon_results,
        'summary': summary
    }