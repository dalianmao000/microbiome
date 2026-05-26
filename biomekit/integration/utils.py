"""Utility functions for multi-omics integration."""
from typing import Dict, Literal, Union
import pandas as pd
import numpy as np

InputFormat = Literal['separate', 'prealigned']


def detect_input_format(data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> InputFormat:
    """Detect whether input is separate omics dict or pre-aligned DataFrame."""
    if isinstance(data, dict):
        return 'separate'
    return 'prealigned'


def align_omics_data(omics_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Inner join samples across all omics DataFrames."""
    if not omics_dict:
        return {}

    sample_sets = [set(df.index) for df in omics_dict.values()]
    common_samples = sorted(set.intersection(*sample_sets))

    if not common_samples:
        raise ValueError("No common samples found across omics")

    return {name: df.loc[common_samples] for name, df in omics_dict.items()}


def concat_with_labels(omics_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate omics DataFrames with omics-type labels in column names.

    All DataFrames must have the same sample indices. Use align_omics_data()
    first if alignment is needed.
    """
    if not omics_dict:
        return pd.DataFrame()

    frames = []
    for omics_name, df in omics_dict.items():
        labeled_df = df.copy()
        labeled_df.columns = [f"{omics_name}_{col}" for col in df.columns]
        frames.append(labeled_df)

    # Validate that all DataFrames have the same sample indices
    indices = [set(df.index) for df in frames]
    if len(indices) > 1:
        first_idx = indices[0]
        for i, idx in enumerate(indices[1:], start=1):
            if idx != first_idx:
                raise ValueError(
                    f"DataFrame indices do not match. All DataFrames must have the same "
                    f"sample indices for concatenation. Got different indices for "
                    f"{list(omics_dict.keys())[0]} and {list(omics_dict.keys())[i]}."
                )

    return pd.concat(frames, axis=1)


def validate_omics_keys(omics_dict: Dict[str, pd.DataFrame]) -> None:
    """Validate that required omics keys are present."""
    required_keys = {'16s', 'metabolomics', 'metagenomics'}
    missing = required_keys - set(omics_dict.keys())
    if missing:
        raise ValueError(f"Missing required omics keys: {sorted(missing)}. Expected: {sorted(required_keys)}")