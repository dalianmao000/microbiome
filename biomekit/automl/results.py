"""Experiment results tracking and model card generation."""
from typing import Dict, List, Any, Optional
from datetime import datetime


class ExperimentResults:
    """Tracks experiment history and generates model cards."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    @property
    def n_results(self) -> int:
        return len(self.results)

    @property
    def best_score(self) -> float:
        if not self.results:
            return 0.0
        return max(r['score'] for r in self.results)

    @property
    def best_result(self) -> Optional[Dict[str, Any]]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r['score'])

    def add_result(self, result: Dict[str, Any]) -> None:
        result['timestamp'] = datetime.now().isoformat()
        self.results.append(result)

    def generate_model_card(self) -> Dict[str, Any]:
        best = self.best_result
        if not best:
            return {}
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            'score': best['score'],
            'config': best['config'],
            'metrics': best['metrics'],
            'selected_features': best.get('selected_features', []),
            'duration_seconds': duration,
            'n_experiments': self.n_results,
            'timestamp': best.get('timestamp', ''),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_results': self.n_results,
            'best_score': self.best_score,
            'best_result': self.best_result,
            'all_results': self.results,
        }