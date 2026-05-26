"""Multi-omics integration pipeline."""
from typing import Dict, Literal, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from .utils import detect_input_format, align_omics_data, concat_with_labels, validate_omics_keys
from .fusion import CCAAnalyzer, ProcrustesAnalyzer
from .report import MultiOmicsReport

FusionStrategy = Literal['early', 'late']
BiomarkerMethod = Literal['diablo', 'splsda']
CorrelationMethod = Literal['cca', 'procrustes']
ClassifierType = Literal['rf', 'svm', 'xgb', 'mlp', 'gb']


class AutoDetectingInputFormat:
    """Utility to detect and normalize input format."""

    @staticmethod
    def process(data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        fmt = detect_input_format(data)
        if fmt == 'prealigned':
            raise NotImplementedError("Pre-aligned format parsing not yet implemented")
        validate_omics_keys(data)
        return align_omics_data(data)


class MultiOmicsPipeline:
    """Main pipeline for multi-omics integration analysis."""

    def __init__(
        self,
        biomarker_method: BiomarkerMethod = 'diablo',
        correlation_method: CorrelationMethod = 'cca',
        fusion: FusionStrategy = 'early',
        classifier: ClassifierType = 'rf',
        encode: Optional[Literal['autoencoder', 'pca']] = None,
        latent_dim: int = 16,
        cv: int = 5,
        n_components: int = 2,
        n_selected_features: int = 50,
    ):
        self.biomarker_method = biomarker_method
        self.correlation_method = correlation_method
        self.fusion = fusion
        self.classifier = classifier
        self.encode = encode
        self.latent_dim = latent_dim
        self.cv = cv
        self.n_components = n_components
        self.n_selected_features = n_selected_features
        self._results: Dict = {}

    def fit_biomarker_discovery(self, omics_data: Dict[str, pd.DataFrame], y: np.ndarray) -> Dict:
        """Run biomarker discovery (CCA or Procrustes as fallback since DIABLO/sPLS-DA need mixOmics)."""
        if self.biomarker_method not in ('diablo', 'splsda', 'cca', 'procrustes'):
            raise ValueError(f"Invalid biomarker_method: {self.biomarker_method}. Must be one of: diablo, splsda, cca, procrustes")
        validate_omics_keys(omics_data)
        keys = list(omics_data.keys())
        # CCA/Procrustes work on pairs; use first two blocks
        pair_data = {keys[0]: omics_data[keys[0]], keys[1]: omics_data[keys[1]]}
        # DIABLO/sPLS-DA require mixOmics; fall back to CCA/Procrustes for now
        if self.biomarker_method in ('cca', 'diablo', 'splsda'):
            analyzer = CCAAnalyzer(n_components=self.n_components)
        else:
            analyzer = ProcrustesAnalyzer()
        return analyzer.fit_transform(pair_data)

    def fit_correlation(self, omics_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run functional correlation (CCA or Procrustes)."""
        if self.correlation_method not in ('cca', 'procrustes'):
            raise ValueError(f"Invalid correlation_method: {self.correlation_method}. Must be one of: cca, procrustes")
        validate_omics_keys(omics_data)
        keys = list(omics_data.keys())
        # CCA/Procrustes work on pairs; use first two blocks
        pair_data = {keys[0]: omics_data[keys[0]], keys[1]: omics_data[keys[1]]}
        if self.correlation_method == 'cca':
            analyzer = CCAAnalyzer(n_components=self.n_components)
        else:
            analyzer = ProcrustesAnalyzer()
        return analyzer.fit_transform(pair_data)

    def _get_classifier(self):
        """Return a fresh classifier instance."""
        classifiers = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'xgb': GradientBoostingClassifier(random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
            'gb': GradientBoostingClassifier(random_state=42),
        }
        if self.classifier not in classifiers:
            raise ValueError(f"Invalid classifier: {self.classifier}")
        return classifiers[self.classifier]

    def fit_classification(self, omics_data: Dict[str, pd.DataFrame], y: np.ndarray) -> Dict:
        """Run integrated classification with early or late fusion."""
        if self.fusion not in ('early', 'late'):
            raise ValueError(f"Invalid fusion: {self.fusion}. Must be 'early' or 'late'.")

        if self.fusion == 'early':
            concatenated = concat_with_labels(omics_data)
            X = concatenated.values
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            latents = []
            for name, df in omics_data.items():
                if self.encode == 'pca':
                    pca = PCA(n_components=min(5, df.shape[1]))
                    encoded = pca.fit_transform(df)
                else:
                    encoded = df.values
                latents.append(encoded)
            X = np.hstack(latents)

        clf = self._get_classifier()
        y_pred = cross_val_predict(clf, X, y, cv=self.cv)
        scores = cross_val_score(clf, X, y, cv=self.cv, scoring='accuracy')
        accuracy = (float(scores.mean()), float(scores.std()))

        f1_scores = cross_val_score(clf, X, y, cv=self.cv, scoring='f1')
        f1 = (float(f1_scores.mean()), float(f1_scores.std()))

        auc_scores = None
        try:
            auc_scores = cross_val_score(clf, X, y, cv=self.cv, scoring='roc_auc')
            auc = (float(auc_scores.mean()), float(auc_scores.std()))
        except Exception:
            auc = (0.0, 0.0)

        return {
            'fusion': self.fusion,
            'y_true': y,
            'y_pred': y_pred,
            'scores': scores,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'cv_results': {},
        }

    def fit(self, omics_data: Union[Dict[str, pd.DataFrame], pd.DataFrame], y: np.ndarray) -> 'MultiOmicsReport':
        """Run all three scenarios and return MultiOmicsReport."""
        processed_data = AutoDetectingInputFormat.process(omics_data)

        self._results = {
            'biomarkers': self.fit_biomarker_discovery(processed_data, y),
            'correlations': self.fit_correlation(processed_data),
            'predictions': self.fit_classification(processed_data, y),
        }
        return MultiOmicsReport(self._results)

    def transform(self, omics_data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> np.ndarray:
        """Transform new data through the pipeline."""
        raise NotImplementedError("transform not yet implemented")

    def fit_transform(self, omics_data: Union[Dict[str, pd.DataFrame], pd.DataFrame], y: np.ndarray) -> 'MultiOmicsReport':
        """Fit and transform in one call."""
        return self.fit(omics_data, y)