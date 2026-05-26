"""
biomekit - A comprehensive Python toolkit for microbiome data analysis.
"""
__version__ = "0.1.0"

from biomekit.abundance import run_lefse, run_deseq2, run_ancombc
from biomekit.diversity import alpha_diversity, beta_diversity, permanova
from biomekit.function import run_picrust2, run_faprotax
from biomekit.phylogeny import build_tree, bootstrap_tree
from biomekit.network import sparcc_network, spearman_network

__all__ = [
    "run_lefse", "run_deseq2", "run_ancombc",
    "alpha_diversity", "beta_diversity", "permanova",
    "run_picrust2", "run_faprotax",
    "build_tree", "bootstrap_tree",
    "sparcc_network", "spearman_network",
]