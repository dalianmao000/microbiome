"""Multi-omics integration report class."""
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MultiOmicsReport:
    """Container for multi-omics analysis results with plotting and export."""

    def __init__(self, results: Dict):
        """
        results: dict with keys 'biomarkers', 'correlations', 'predictions'
        """
        self.results = results
        self.biomarkers = self._parse_biomarkers(results.get('biomarkers', {}))
        self.correlations = self._parse_correlations(results.get('correlations', {}))
        self.predictions = results.get('predictions', {})

    def _parse_biomarkers(self, biomarkers: Dict) -> pd.DataFrame:
        """Parse biomarkers into a merged DataFrame."""
        if not biomarkers or 'loadings' not in biomarkers:
            return pd.DataFrame()
        loadings = biomarkers.get('loadings', {})
        if isinstance(loadings, pd.DataFrame):
            return loadings
        return pd.DataFrame()

    def _parse_correlations(self, correlations: Dict) -> pd.DataFrame:
        """Parse correlations into a DataFrame."""
        if not correlations:
            return pd.DataFrame()
        if 'loadings' in correlations:
            loadings = correlations['loadings']
            if isinstance(loadings, pd.DataFrame):
                return loadings
        return pd.DataFrame()

    def plot(self) -> plt.Figure:
        """Generate matplotlib figure with biomarker heatmap and classification metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Biomarker loadings heatmap
        if not self.biomarkers.empty:
            axes[0].clear()
            im = axes[0].imshow(self.biomarkers.values, cmap='RdBu_r', aspect='auto')
            axes[0].set_xticks(range(len(self.biomarkers.columns)))
            axes[0].set_xticklabels(self.biomarkers.columns, rotation=45, ha='right')
            axes[0].set_yticks(range(len(self.biomarkers.index)))
            axes[0].set_yticklabels(self.biomarkers.index)
            axes[0].set_title('Biomarker Loadings')
            plt.colorbar(im, ax=axes[0])
        else:
            axes[0].text(0.5, 0.5, 'No biomarker data', ha='center', va='center')
            axes[0].set_title('Biomarker Loadings')

        # Subplot 2: Classification metrics bar plot
        if self.predictions:
            metrics = {
                'Accuracy': self.predictions.get('accuracy', (0, 0)),
                'F1': self.predictions.get('f1', (0, 0)),
                'AUC': self.predictions.get('auc', (0, 0)),
            }
            names = list(metrics.keys())
            means = [m[0] for m in metrics.values()]
            stds = [m[1] for m in metrics.values()]
            axes[1].bar(names, means, yerr=stds, capsize=5, color=['steelblue', 'seagreen', 'coral'])
            axes[1].set_ylim(0, 1.2)
            axes[1].set_title('Classification Metrics')
            axes[1].set_ylabel('Score')
        else:
            axes[1].text(0.5, 0.5, 'No prediction data', ha='center', va='center')
            axes[1].set_title('Classification Metrics')

        plt.tight_layout()
        return fig

    def to_dict(self) -> Dict:
        """Export all results as nested dict."""
        return self.results

    def summary(self) -> None:
        """Print human-readable summary."""
        print("=== Multi-Omics Report Summary ===")
        print(f"\nBiomarker Discovery ({self.biomarkers.shape if not self.biomarkers.empty else 'No data'})")
        print(f"Correlation Analysis ({self.correlations.shape if not self.correlations.empty else 'No data'})")
        if self.predictions:
            print(f"\nClassification (fusion={self.predictions.get('fusion', 'N/A')})")
            print(f"  Accuracy: {self.predictions.get('accuracy', ('N/A', ''))}")
            print(f"  F1: {self.predictions.get('f1', ('N/A', ''))}")
            print(f"  AUC: {self.predictions.get('auc', ('N/A', ''))}")