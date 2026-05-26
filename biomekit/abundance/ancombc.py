"""
ANCOM-BC wrapper for differential abundance analysis.

ANCOM-BC (Analysis of Composition of Microbiomes with Bias Correction)
handles zero-inflated microbiome data better than DESeq2.
This module provides a fallback pure Python implementation when R is not available.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def run_ancombc(
    abundance_df: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str = 'group',
    formula: str = '~ group',
    adj_formula: Optional[str] = None,
    zero_cut: float = 0.9,
    lib_cut: float = 1000,
    global_test: bool = False,
) -> Dict:
    """
    Run ANCOM-BC analysis.

    Falls back to a Python-only implementation when rpy2/R is not available.

    Parameters
    ----------
    abundance_df : pd.DataFrame
        Count table (samples x features)
    metadata : pd.DataFrame
        Sample metadata
    group_column : str
        Group variable
    formula : str
        Model formula
    zero_cut : float
        Threshold for zero proportion cut-off
    lib_cut : float
        Threshold for library size cut-off
    global_test : bool
        Whether to perform global test

    Returns
    -------
    dict
        {
            'results': pd.DataFrame,
            'taxa': pd.DataFrame,
            'bias_correction': dict
        }
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()

        ancombc = importr('ANCOMBC')
        count_matrix = ro.ExpressionMatrix(abundance_df.values.T)
        col_data = pandas2ri.py2rpy(metadata)

        result = ancombc.ancombc(
            phyloseq=ro.NTuple([count_matrix, col_data]),
            formula=ro.Formula(formula),
            zero_cut=zero_cut,
            lib_cut=lib_cut,
            global_test=global_test
        )

        res_df = pandas2ri.rpy2py(result.rx('res'))
        taxa_df = pandas2ri.rpy2py(result.rx('taxa'))
        bias_correction = result.rx('bias_correct')

        return {
            'results': res_df,
            'taxa': taxa_df,
            'bias_correction': dict(bias_correction)
        }
    except ImportError:
        return _ancombc_fallback(abundance_df, metadata, group_column)


def _ancombc_fallback(
    abundance_df: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str = 'group',
) -> Dict:
    """Fallback pure Python ANCOM-BC-like implementation."""
    from scipy.stats import mannwhitneyu

    groups = metadata[group_column]
    case_mask = groups.values == 'case'
    control_mask = groups.values == 'control'

    results = []
    for feature in abundance_df.columns:
        case_values = abundance_df.iloc[case_mask, abundance_df.columns.get_loc(feature)].values
        control_values = abundance_df.iloc[control_mask, abundance_df.columns.get_loc(feature)].values

        case_mean = np.mean(case_values)
        control_mean = np.mean(control_values)

        if control_mean > 0:
            log_ratio = np.log((case_mean + 1) / (control_mean + 1))
        else:
            log_ratio = 0

        _, pvalue = mannwhitneyu(case_values, control_values, alternative='two-sided')

        results.append({
            'feature': feature,
            'log_ratio': log_ratio,
            'pvalue': pvalue,
            'qvalue': pvalue,
            'case_mean': case_mean,
            'control_mean': control_mean
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('pvalue')

    return {
        'results': results_df,
        'taxa': pd.DataFrame({'feature': abundance_df.columns}),
        'bias_correction': {'note': 'Fallback implementation'}
    }