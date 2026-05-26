"""Prediction module for microbiome-based disease classification and prognosis."""
from .preprocessing import CLRTransformer, LogTransformer, PercentTransformer, VarianceFilter, PreprocessingPipeline
from .encoders import AutoencoderEncoder, VAEEncoder
from .classifiers import MicrobiomeClassifier
from .regressors import SurvivalRegressor
from .pipeline import MicrobiomePipeline
from .explainer import SHAPExplainer, PermutationImportance, PartialDependencePlot
from .utils import extract_top_markers, plot_feature_importance, plot_shap_summary, learning_curve

__all__ = [
    # Preprocessing
    'CLRTransformer', 'LogTransformer', 'PercentTransformer', 'VarianceFilter', 'PreprocessingPipeline',
    # Encoders
    'AutoencoderEncoder', 'VAEEncoder',
    # Classifiers
    'MicrobiomeClassifier',
    # Regressors
    'SurvivalRegressor',
    # Pipeline
    'MicrobiomePipeline',
    # Explainers
    'SHAPExplainer', 'PermutationImportance', 'PartialDependencePlot',
    # Utils
    'extract_top_markers', 'plot_feature_importance', 'plot_shap_summary', 'learning_curve',
]