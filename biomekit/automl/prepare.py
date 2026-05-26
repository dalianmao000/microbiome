"""Fixed evaluator for microbiome AutoML (analogous to autoresearch prepare.py)."""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from .evaluator import DomainMetrics


class AutoMLPrepare:
    """
    Fixed evaluator that loads data, runs cross-validation, and computes domain metrics.
    This is NOT modified by the agent.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int = 5,
        n_features: int = 20,
        disease_label: Optional[str] = None,
    ):
        if cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.n_features = n_features
        self.disease_label = disease_label
        self.domain_metrics = DomainMetrics()

    def get_fold_indices(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        return list(cv.split(self.X, self.y))

    def _apply_feature_selection(
        self,
        strategy: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_features: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if strategy == 'rf_importance':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[-n_features:]
        elif strategy == 'lasso':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            lr = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=1000, random_state=42)
            lr.fit(X_train_scaled, y_train)
            importances = np.abs(lr.coef_[0])
            top_indices = np.argsort(importances)[-n_features:]
        else:
            top_indices = np.arange(min(n_features, X_train.shape[1]))

        feature_names = [f'f{i}' for i in top_indices]
        return X_train[:, top_indices], X_test[:, top_indices], feature_names

    def _get_model(self, model_type: str, hyperparams: Dict[str, Any]):
        if model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 5),
                random_state=42,
            )
        elif model_type == 'lr':
            return LogisticRegression(
                C=hyperparams.get('C', 1.0),
                penalty=hyperparams.get('penalty', 'l2'),
                solver='saga',
                max_iter=1000,
                random_state=42,
            )
        elif model_type == 'svm':
            return SVC(
                C=hyperparams.get('C', 1.0),
                kernel=hyperparams.get('kernel', 'rbf'),
                probability=True,
                random_state=42,
            )
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)

    def evaluate_config(
        self,
        config: Dict[str, Any],
        compute_stability: bool = False,
    ) -> Dict[str, Any]:
        feature_strategy = config.get('feature_strategy', 'rf_importance')
        model_type = config.get('model_type', 'rf')
        n_features = config.get('n_features', self.n_features)
        hyperparams = config.get('hyperparams', {})

        fold_scores = []
        selected_features_per_fold = []

        for train_idx, test_idx in self.get_fold_indices():
            X_train, X_test = self.X.iloc[train_idx].values, self.X.iloc[test_idx].values
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            X_train_sel, X_test_sel, features = self._apply_feature_selection(
                feature_strategy, X_train, y_train, X_test, n_features
            )

            selected_features_per_fold.append(features)

            model = self._get_model(model_type, hyperparams)
            model.fit(X_train_sel, y_train)
            score = model.score(X_test_sel, y_test)
            fold_scores.append(score)

        performance = float(np.mean(fold_scores))

        stability = 0.0
        bio_relevance = 0.0
        selected_features = []

        if selected_features_per_fold:
            stability = float(self.domain_metrics.feature_stability(selected_features_per_fold))

            if self.disease_label:
                # Compute stable features: appearing in at least 50% of folds
                n_folds = len(selected_features_per_fold)
                feature_counts: Dict[str, int] = {}
                for feats in selected_features_per_fold:
                    for f in feats:
                        feature_counts[f] = feature_counts.get(f, 0) + 1
                threshold = n_folds * 0.5
                stable_features = [f for f, count in feature_counts.items() if count >= threshold]
                if stable_features:
                    bio_relevance = float(self.domain_metrics.biological_relevance(stable_features, self.disease_label))

            # selected_features is the last fold's features (for consistency)
            selected_features = selected_features_per_fold[-1]

        score = self.domain_metrics.compute_composite(
            performance=performance,
            stability=stability,
            bio_relevance=bio_relevance,
        )

        return {
            'score': score,
            'metrics': {
                'performance': performance,
                'stability': stability,
                'bio_relevance': bio_relevance,
            },
            'selected_features': selected_features,
        }