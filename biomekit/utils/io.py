"""
I/O utilities for microbiome data formats.

Supports BIOM, TSV, CSV, and QIIME2 artifact formats.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, TextIO

try:
    import biom
    HAS_BIOM = True
except ImportError:
    HAS_BIOM = False


def read_biom(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Read BIOM format file and return abundance DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to .biom file

    Returns
    -------
    pd.DataFrame
        Abundance table with samples as rows, features as columns

    Raises
    ------
    ImportError
        If biom package is not installed
    """
    if not HAS_BIOM:
        raise ImportError(
            "biom package is required to read BIOM files. "
            "Install with: pip install biom"
        )

    with biom.open_json(filepath) as f:
        table = biom.Table.from_json(f.read())

    data = table.matrix_data.toarray().T
    sample_ids = [str(x) for x in table.ids(axis='sample')]
    feature_ids = [str(x) for x in table.ids(axis='observation')]

    df = pd.DataFrame(data, index=sample_ids, columns=feature_ids)
    return df


def read_tsv(input: Union[str, Path, TextIO], index_col: int = 0) -> pd.DataFrame:
    """Read TSV format file."""
    if isinstance(input, (str, Path)):
        return pd.read_csv(input, sep='\t', index_col=index_col)
    else:
        return pd.read_csv(input, sep='\t', index_col=index_col)


def read_csv(filepath: Union[str, Path], index_col: int = 0) -> pd.DataFrame:
    """Read CSV format file."""
    return pd.read_csv(filepath, index_col=index_col)


def write_tsv(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """Write DataFrame to TSV format."""
    df.to_csv(filepath, sep='\t', index=True)


def detect_format(filepath: Union[str, Path]) -> str:
    """Detect file format from extension."""
    suffix = Path(filepath).suffix.lower()
    format_map = {
        '.biom': 'biom',
        '.tsv': 'tsv',
        '.csv': 'csv',
        '.qza': 'qza',
    }
    return format_map.get(suffix, 'unknown')


def read_any(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Auto-detect format and read file."""
    fmt = detect_format(filepath)
    if fmt == 'biom':
        return read_biom(filepath)
    elif fmt == 'tsv':
        return read_tsv(filepath, **kwargs)
    elif fmt == 'csv':
        return read_csv(filepath, **kwargs)
    else:
        raise ValueError(f"Unknown format: {fmt}")