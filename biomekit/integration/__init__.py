"""Multi-omics integration module for biomarker discovery, functional correlation, and integrated classification."""
from .multiomics_pipeline import MultiOmicsPipeline
from .report import MultiOmicsReport
from .fusion import CCAAnalyzer, ProcrustesAnalyzer
from .utils import detect_input_format, align_omics_data, concat_with_labels, validate_omics_keys

__all__ = [
    'MultiOmicsPipeline',
    'MultiOmicsReport',
    'CCAAnalyzer',
    'ProcrustesAnalyzer',
    'detect_input_format',
    'align_omics_data',
    'concat_with_labels',
    'validate_omics_keys',
]