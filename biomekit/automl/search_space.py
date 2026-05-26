"""Search space definitions for microbiome AutoML."""
from enum import Enum
from typing import Dict, List, Any


class FeatureStrategy(Enum):
    """Allowed feature selection strategies."""
    LEFSE = "lefse"
    RANDOM_FOREST_IMPORTANCE = "rf_importance"
    LASSO = "lasso"
    ANCOM = "ancom"
    WILCOXON = "wilcoxon"
    ZERO_INFLATED = "zero_inflated"


class ModelType(Enum):
    """Allowed model types."""
    RANDOM_FOREST = "rf"
    XGBOOST = "xgb"
    SVM = "svm"
    LOGISTIC_REGRESSION = "lr"
    GRADIENT_BOOSTING = "gb"
    MLP = "mlp"


class SearchSpace:
    """Defines the searchable space for AutoML experiments."""

    def __init__(self):
        self.feature_strategies: List[FeatureStrategy] = [
            FeatureStrategy.LEFSE,
            FeatureStrategy.RANDOM_FOREST_IMPORTANCE,
            FeatureStrategy.LASSO,
            FeatureStrategy.ANCOM,
            FeatureStrategy.WILCOXON,
            FeatureStrategy.ZERO_INFLATED,
        ]
        self.model_types: List[ModelType] = [
            ModelType.RANDOM_FOREST,
            ModelType.XGBOOST,
            ModelType.SVM,
            ModelType.LOGISTIC_REGRESSION,
            ModelType.GRADIENT_BOOSTING,
            ModelType.MLP,
        ]
        self._hyperparams = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': {'type': 'int', 'min': 50, 'max': 500, 'default': 200},
                'max_depth': {'type': 'int', 'min': 3, 'max': 20, 'default': 10},
                'min_samples_split': {'type': 'int', 'min': 2, 'max': 20, 'default': 5},
            },
            ModelType.XGBOOST: {
                'n_estimators': {'type': 'int', 'min': 50, 'max': 500, 'default': 200},
                'max_depth': {'type': 'int', 'min': 3, 'max': 15, 'default': 6},
                'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3, 'default': 0.1},
            },
            ModelType.SVM: {
                'C': {'type': 'float', 'min': 0.01, 'max': 10.0, 'default': 1.0},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf'], 'default': 'rbf'},
            },
            ModelType.LOGISTIC_REGRESSION: {
                'C': {'type': 'float', 'min': 0.01, 'max': 10.0, 'default': 1.0},
                'penalty': {'type': 'categorical', 'choices': ['l1', 'l2'], 'default': 'l2'},
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': {'type': 'int', 'min': 50, 'max': 500, 'default': 200},
                'max_depth': {'type': 'int', 'min': 3, 'max': 10, 'default': 5},
                'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3, 'default': 0.1},
            },
            ModelType.MLP: {
                'hidden_layer_sizes': {'type': 'tuple', 'default': (100,)},
                'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'logistic'], 'default': 'relu'},
                'alpha': {'type': 'float', 'min': 0.0001, 'max': 1.0, 'default': 0.0001},
                'learning_rate': {'type': 'categorical', 'choices': ['constant', 'invscaling', 'adaptive'], 'default': 'constant'},
            },
        }

    def get_hyperparams(self, model_type: ModelType) -> Dict[str, Any]:
        """Get hyperparam ranges for a model type."""
        if model_type not in self._hyperparams:
            raise KeyError(f"No hyperparams defined for model type: {model_type}")
        return self._hyperparams[model_type]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize search space to dict."""
        return {
            'feature_strategies': [s.value for s in self.feature_strategies],
            'model_types': [m.value for m in self.model_types],
            'hyperparams': {
                m.value: h for m, h in self._hyperparams.items()
            },
        }