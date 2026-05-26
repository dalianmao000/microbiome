"""Complete prediction pipeline for microbiome analysis."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

from .preprocessing import CLRTransformer, PreprocessingPipeline
from .encoders import AutoencoderEncoder
from .classifiers import MicrobiomeClassifier


class MicrobiomePipeline:
    """Complete prediction pipeline with preprocessing, encoding, and classification.

    Parameters:
        preprocess: Preprocessing transform ('clr', 'log', 'percent', or None)
        encode: Encoding method ('autoencoder', 'vae', or None)
        classifier: Classifier type ('rf', 'svm', 'xgb', 'mlp', 'gb')
        latent_dim: Latent dimension for autoencoder
        variance_threshold: Variance threshold for filtering
        **classifier_kwargs: Additional arguments for classifier
    """

    def __init__(self, preprocess='clr', encode=None, classifier='rf',
                 latent_dim=16, variance_threshold=0.01, **classifier_kwargs):
        self.preprocess = preprocess
        self.encode = encode
        self.classifier = classifier
        self.latent_dim = latent_dim
        self.variance_threshold = variance_threshold
        self.classifier_kwargs = classifier_kwargs

        self.preprocessor = None
        self.encoder = None
        self.clf = None
        self.fitted = False

    def _init_components(self):
        # Preprocessing
        if self.preprocess == 'clr':
            self.preprocessor = CLRTransformer()
        elif self.preprocess:
            self.preprocessor = PreprocessingPipeline(
                transform=self.preprocess,
                filter_low_var=True,
                variance_threshold=self.variance_threshold
            )
        else:
            self.preprocessor = None

        # Encoding
        if self.encode == 'autoencoder':
            self.encoder = AutoencoderEncoder(latent_dim=self.latent_dim)
        elif self.encode == 'vae':
            from .encoders import VAEEncoder
            self.encoder = VAEEncoder(latent_dim=self.latent_dim)
        else:
            self.encoder = None

        # Classifier
        self.clf = MicrobiomeClassifier(model=self.classifier, **self.classifier_kwargs)

    def fit(self, X, y):
        self._init_components()

        # Preprocess
        if self.preprocessor is not None:
            X_t = self.preprocessor.fit_transform(X)
        else:
            X_t = np.asarray(X)

        # Encode
        if self.encoder is not None:
            X_t = self.encoder.fit_transform(X_t)

        # Classify
        self.clf.fit(X_t, y)
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        X_t = np.asarray(X)
        if self.preprocessor is not None:
            X_t = self.preprocessor.transform(X_t)
        if self.encoder is not None:
            X_t = self.encoder.transform(X_t)
        return self.clf.predict(X_t)

    def predict_proba(self, X):
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        X_t = np.asarray(X)
        if self.preprocessor is not None:
            X_t = self.preprocessor.transform(X_t)
        if self.encoder is not None:
            X_t = self.encoder.transform(X_t)
        return self.clf.predict_proba(X_t)

    def evaluate(self, X, y, cv=5):
        """Evaluate pipeline with cross-validation.

        Returns:
            Dictionary with metric means and standard deviations
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'auc_roc': [], 'auc_pr': []
        }

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.fit(X_train, y_train)
            y_pred = self.predict(X_val)

            # Binary vs multi-class handling
            n_classes = len(np.unique(y))
            if n_classes == 2:
                y_proba = self.predict_proba(X_val)[:, 1]
            else:
                y_proba = None

            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
            metrics['recall'].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
            metrics['f1'].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))

            if y_proba is not None:
                try:
                    metrics['auc_roc'].append(roc_auc_score(y_val, y_proba))
                    metrics['auc_pr'].append(average_precision_score(y_val, y_proba))
                except ValueError:
                    metrics['auc_roc'].append(np.nan)
                    metrics['auc_pr'].append(np.nan)

        # Return mean and std
        results = {k: (np.nanmean(v), np.nanstd(v)) for k, v in metrics.items()}
        return results

    def get_feature_importance(self):
        if self.clf and hasattr(self.clf, 'feature_importances_'):
            return self.clf.feature_importances_
        return None

    def get_params(self, deep=True):
        return {
            'preprocess': self.preprocess,
            'encode': self.encode,
            'classifier': self.classifier,
            'latent_dim': self.latent_dim,
            'variance_threshold': self.variance_threshold,
            **self.classifier_kwargs
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self