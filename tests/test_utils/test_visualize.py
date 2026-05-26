"""Tests for visualization utilities."""
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from biomekit.utils.visualize import plot_abundance_bar, plot_heatmap


def test_plot_abundance_bar():
    if not HAS_PLOTTING:
        import pytest
        pytest.skip("matplotlib/seaborn not installed")

    df = pd.DataFrame({
        'OTU_1': np.random.rand(10) * 100,
        'OTU_2': np.random.rand(10) * 100,
    })
    fig = plot_abundance_bar(df, top_n=2)
    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_heatmap():
    if not HAS_PLOTTING:
        import pytest
        pytest.skip("matplotlib/seaborn not installed")

    df = pd.DataFrame(np.random.rand(10, 5),
                      index=[f'S{i}' for i in range(10)],
                      columns=[f'OTU_{i}' for i in range(5)])
    fig = plot_heatmap(df)
    assert fig is not None