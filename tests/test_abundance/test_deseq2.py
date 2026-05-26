"""Tests for DESeq2 wrapper."""
import pandas as pd
import numpy as np
from biomekit.abundance.deseq2 import run_deseq2


def test_run_deseq2_fallback():
    np.random.seed(42)
    data = np.random.randint(0, 100, (20, 10))
    df = pd.DataFrame(data, index=[f'S{i}' for i in range(20)],
                     columns=[f'OTU_{i}' for i in range(10)])

    metadata = pd.DataFrame({
        'sample_id': df.index,
        'group': ['case'] * 10 + ['control'] * 10
    })

    result = run_deseq2(df, metadata, group_column='group')

    assert 'results' in result
    assert 'normalized_counts' in result
    assert 'dispersion' in result
    assert len(result['results']) == 10