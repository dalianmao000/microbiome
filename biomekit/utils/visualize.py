"""
Visualization utilities for microbiome analysis.
"""
import pandas as pd
import numpy as np
from typing import Optional, List

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def plot_abundance_bar(
    abundance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot top N most abundant features as bar chart."""
    if not HAS_PLOTTING:
        raise ImportError(
            "matplotlib and seaborn are required for visualization. "
            "Install with: pip install matplotlib seaborn"
        )
    mean_abundance = abundance_df.mean().sort_values(ascending=False)
    top_features = mean_abundance.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    top_features.plot(kind='bar', ax=ax)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Mean Abundance')
    ax.set_title(f'Top {top_n} Most Abundant Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_pcoa(
    pcoa_df: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    color_map: Optional[dict] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot PCoA ordination colored by group."""
    if not HAS_PLOTTING:
        raise ImportError(
            "matplotlib and seaborn are required for visualization. "
            "Install with: pip install matplotlib seaborn"
        )
    fig, ax = plt.subplots(figsize=figsize)
    groups = metadata[group_column].unique()

    if color_map is None:
        color_map = {g: f'C{i}' for i, g in enumerate(groups)}

    for group in groups:
        mask = metadata[group_column] == group
        ax.scatter(
            pcoa_df.loc[mask, 'PC1'],
            pcoa_df.loc[mask, 'PC2'],
            c=color_map[group],
            label=group,
            alpha=0.7,
            s=50,
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCoA Ordination')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_heatmap(
    abundance_df: pd.DataFrame,
    samples: Optional[List[str]] = None,
    features: Optional[List[str]] = None,
    figsize: tuple = (10, 8),
    cmap: str = 'viridis',
) -> plt.Figure:
    """Plot abundance heatmap."""
    if not HAS_PLOTTING:
        raise ImportError(
            "matplotlib and seaborn are required for visualization. "
            "Install with: pip install matplotlib seaborn"
        )
    data = abundance_df.copy()

    if samples is not None:
        data = data.loc[samples]
    if features is not None:
        data = data[features]

    data_log = np.log1p(data)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data_log, cmap=cmap, ax=ax)
    ax.set_title('Abundance Heatmap (log-transformed)')
    plt.tight_layout()
    return fig