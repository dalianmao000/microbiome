"""AutoML Pipeline: orchestrates program + prepare + train + results."""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from .program import Program
from .train import AutoMLTrain
from .results import ExperimentResults


class AutoMLPipeline:
    """
    Main AutoML pipeline orchestrating the autoresearch-style workflow:
    1. Program defines constraints and goals
    2. Prepare provides fixed evaluation
    3. Train generates and tests configurations
    4. Results tracks experiment history
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
        termination_score: float = 0.85,
        max_iterations: int = 50,
        cv_folds: int = 5,
        disease_label: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.program = Program(
            weights=weights,
            termination_score=termination_score,
            max_iterations=max_iterations,
            disease_label=disease_label,
        )
        self.trainer = AutoMLTrain(X, y, cv_folds=cv_folds)
        self.experiment_results = ExperimentResults()

    def run(self) -> Dict[str, Any]:
        """
        Run AutoML experiments until termination criteria are met.
        """
        n_iterations = 0

        while n_iterations < self.program.max_iterations:
            config = self.trainer.generate_config()
            result = self.trainer.run_iteration(config)
            self.experiment_results.add_result(result)

            n_iterations += 1

            if self.program.should_terminate(result['score'], n_iterations):
                break

        return {
            'best_score': self.experiment_results.best_score,
            'model_card': self.experiment_results.generate_model_card(),
            'n_experiments': self.experiment_results.n_results,
            'all_results': self.experiment_results.to_dict(),
        }