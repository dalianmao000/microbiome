import pytest
import pandas as pd
import numpy as np
from biomekit.integration.utils import (
    detect_input_format,
    align_omics_data,
    concat_with_labels,
    validate_omics_keys,
)

class TestDetectInputFormat:
    def test_separate_dict(self):
        df_16s = pd.DataFrame([[1, 2], [3, 4]], columns=['f1', 'f2'], index=['s1', 's2'])
        df_meta = pd.DataFrame([[5, 6], [7, 8]], columns=['f3', 'f4'], index=['s1', 's2'])
        data = {'16s': df_16s, 'metabolomics': df_meta}
        assert detect_input_format(data) == 'separate'

    def test_prealigned_dataframe(self):
        df = pd.DataFrame([[1, 2, 5, 6], [3, 4, 7, 8]], columns=['16s_f1', '16s_f2', 'meta_f3', 'meta_f4'], index=['s1', 's2'])
        assert detect_input_format(df) == 'prealigned'

class TestAlignOmicsData:
    def test_inner_join_samples(self):
        df_16s = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['f1', 'f2'], index=['s1', 's2', 's3'])
        df_meta = pd.DataFrame([[5, 6], [7, 8]], columns=['f3', 'f4'], index=['s1', 's2'])
        result = align_omics_data({'16s': df_16s, 'metabolomics': df_meta})
        assert list(result.keys()) == ['16s', 'metabolomics']
        assert list(result['16s'].index) == ['s1', 's2']
        assert list(result['metabolomics'].index) == ['s1', 's2']

    def test_empty_intersection_raises(self):
        df_16s = pd.DataFrame([[1, 2]], columns=['f1', 'f2'], index=['s1'])
        df_meta = pd.DataFrame([[5, 6]], columns=['f3', 'f4'], index=['s99'])
        with pytest.raises(ValueError, match="No common samples"):
            align_omics_data({'16s': df_16s, 'metabolomics': df_meta})

class TestConcatWithLabels:
    def test_concat_with_omics_prefix(self):
        df_16s = pd.DataFrame([[1, 2], [3, 4]], columns=['f1', 'f2'], index=['s1', 's2'])
        df_meta = pd.DataFrame([[5, 6], [7, 8]], columns=['f3', 'f4'], index=['s1', 's2'])
        result = concat_with_labels({'16s': df_16s, 'metabolomics': df_meta})
        assert '16s_f1' in result.columns
        assert 'metabolomics_f3' in result.columns

    def test_mismatched_indices_raises(self):
        df_16s = pd.DataFrame([[1, 2]], columns=['f1', 'f2'], index=['s1', 's2'])
        df_meta = pd.DataFrame([[5, 6]], columns=['f3', 'f4'], index=['s1', 's99'])
        with pytest.raises(ValueError, match="indices do not match"):
            concat_with_labels({'16s': df_16s, 'metabolomics': df_meta})

class TestValidateOmicsKeys:
    def test_valid_keys(self):
        data = {'16s': pd.DataFrame(), 'metabolomics': pd.DataFrame(), 'metagenomics': pd.DataFrame()}
        validate_omics_keys(data)  # should not raise

    def test_missing_key_raises(self):
        data = {'16s': pd.DataFrame()}
        with pytest.raises(ValueError, match="Missing required omics keys"):
            validate_omics_keys(data)