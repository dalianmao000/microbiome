"""Tests for I/O utilities."""
import pandas as pd
import numpy as np
from io import StringIO
from biomekit.utils.io import read_tsv, write_tsv, detect_format


def test_read_tsv():
    test_data = "sample\tOTU_1\tOTU_2\nS1\t100\t200\nS2\t50\t150"
    df = read_tsv(StringIO(test_data))
    assert df.shape == (2, 2)
    assert 'OTU_1' in df.columns


def test_detect_format():
    assert detect_format("data.biom") == "biom"
    assert detect_format("data.tsv") == "tsv"
    assert detect_format("data.csv") == "csv"
    assert detect_format("data.qza") == "qza"


def test_write_and_read_tsv(tmp_path):
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['S1', 'S2'])
    filepath = tmp_path / "test.tsv"
    write_tsv(df, filepath)
    df_read = read_tsv(filepath)
    assert df.equals(df_read)