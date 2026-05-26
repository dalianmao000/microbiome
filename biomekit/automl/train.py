"""Agent-modifiable training pipeline (analogous to autoresearch train.py)."""
from typing import Dict, Any, Optional
import random
import pandas as pd
import numpy as np

from .prepare import AutoMLPrepare
from .search_space import SearchSpace, FeatureStrategy, ModelType


class AutoMLTrain:
    """
    Training pipeline that the agent modifies.
    Analogous to autoresearch's train.py.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int = 5,
        search_space: Optional[SearchSpace] = None,
    ):
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.search_space = search_space or SearchSpace()
        self.prepare = AutoMLPrepare(X, y, cv_folds=cv_folds)

    def generate_config(
        self,
        feature_strategy: Optional[str] = None,
        model_type: Optional[str] = None,
        n_features: Optional[int] = None,
    ) -> Dict[str, Any]:
        if feature_strategy is None:
            feature_strategy = random.choice(self.search_space.feature_strategies).value

        if model_type is None:
            model_type = random.choice(self.search_space.model_types).value

        if n_features is None:
            n_features = random.choice([10, 15, 20, 25])

        hyperparams = self._sample_hyperparams(ModelType(model_type))

        return {
            'feature_strategy': feature_strategy,
            'model_type': model_type,
            'n_features': n_features,
            'hyperparams': hyperparams,
        }

    def _sample_hyperparams(self, model_type: ModelType) -> Dict[str, Any]:
        hyperparams = {}
        ranges = self.search_space.get_hyperparams(model_type)

        for param_name, param_spec in ranges.items():
            param_type = param_spec['type']
            if param_type == 'int':
                hyperparams[param_name] = random.randint(
                    param_spec['min'], param_spec['max']
                )
            elif param_type == 'float':
                hyperparams[param_name] = random.uniform(
                    param_spec['min'], param_spec['max']
                )
            elif param_type == 'categorical':
                hyperparams[param_name] = random.choice(param_spec['choices'])
            elif param_type == 'tuple':
                # Sample hidden_layer_sizes as a tuple of ints, e.g., (100,) or (64, 32)
                layers = []
                n_layers = random.randint(1, 3)
                for _ in range(n_layers):
                    layers.append(random.randint(16, 256))
                hyperparams[param_name] = tuple(layers)

        return hyperparams

    def run_iteration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        result = self.prepare.evaluate_config(config, compute_stability=True)
        result['config'] = config
        return result