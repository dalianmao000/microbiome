"""Tests for PICRUSt2 wrapper."""
import numpy as np
import pandas as pd
from biomekit.function.picrust2 import run_picrust2, run_faprotax


def test_run_picrust2_fallback():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(10, 20), columns=[f'OTU_{i}' for i in range(20)])

    result = run_picrust2(df)

    assert 'KO_abundance' in result
    assert 'EC_abundance' in result
    assert 'pathway_abundance' in result
    assert result['KO_abundance'].shape[0] == 10


def test_run_faprotax_fallback():
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(10, 20), columns=[f'OTU_{i}' for i in range(20)])

    result = run_faprotax(df)

    assert 'function_abundance' in result
    assert 'function_list' in result
    assert len(result['function_list']) > 0