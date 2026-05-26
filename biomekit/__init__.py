"""
biomekit - A comprehensive Python toolkit for microbiome data analysis.
"""
__version__ = "0.1.0"

from biomekit.utils.io import read_biom, read_tsv, read_csv, read_any, detect_format
from biomekit.utils.transform import clr_transform, rarefaction, normalize
from biomekit.utils.visualize import plot_abundance_bar, plot_pcoa, plot_heatmap

try:
    from biomekit.abundance import run_lefse, run_deseq2, run_ancombc
except ImportError:
    pass

try:
    from biomekit.diversity import alpha_diversity, beta_diversity, permanova
except ImportError:
    pass

try:
    from biomekit.function import run_picrust2, run_faprotax
except ImportError:
    pass

try:
    from biomekit.phylogeny import build_tree, bootstrap_tree
except ImportError:
    pass

try:
    from biomekit.network import sparcc_network, spearman_network
except ImportError:
    pass

try:
    from biomekit.prediction import (
        MicrobiomePipeline, MicrobiomeClassifier, SurvivalRegressor,
        SHAPExplainer, PermutationImportance,
        extract_top_markers, plot_feature_importance, plot_shap_summary
    )
except ImportError:
    pass

__all__ = [
    # Utils
    "read_biom", "read_tsv", "read_csv", "read_any", "detect_format",
    "clr_transform", "rarefaction", "normalize",
    "plot_abundance_bar", "plot_pcoa", "plot_heatmap",
    # Abundance (if available)
    "run_lefse", "run_deseq2", "run_ancombc",
    # Diversity (if available)
    "alpha_diversity", "beta_diversity", "permanova",
    # Function (if available)
    "run_picrust2", "run_faprotax",
    # Phylogeny (if available)
    "build_tree", "bootstrap_tree",
    # Network (if available)
    "sparcc_network", "spearman_network",
    # Prediction (if available)
    "MicrobiomePipeline", "MicrobiomeClassifier", "SurvivalRegressor",
    "SHAPExplainer", "PermutationImportance",
    "extract_top_markers", "plot_feature_importance", "plot_shap_summary",
]