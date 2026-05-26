"""Program class: loads and validates program.md-style constraints."""
from typing import Dict, Any, Optional
from .search_space import SearchSpace


class Program:
    """
    Represents the autoresearch-style 'program.md' — human-defined constraints
    that guide the AutoML search.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        termination_score: float = 0.85,
        max_iterations: int = 50,
        disease_label: Optional[str] = None,
    ):
        self.weights = weights or {
            'performance': 0.5,
            'stability': 0.25,
            'bio_relevance': 0.25,
        }
        self.termination_score = termination_score
        self.max_iterations = max_iterations
        self.disease_label = disease_label
        self.search_space = SearchSpace()

    def should_terminate(self, score: float, n_iterations: int) -> bool:
        if n_iterations >= self.max_iterations:
            return True
        if score >= self.termination_score:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'weights': self.weights,
            'termination_score': self.termination_score,
            'max_iterations': self.max_iterations,
            'disease_label': self.disease_label,
            'search_space': self.search_space.to_dict(),
        }