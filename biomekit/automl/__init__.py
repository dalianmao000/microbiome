"""AutoML module for microbiome data analysis using autoresearch methodology."""
import os
from .pipeline import AutoMLPipeline
from .program import Program
from .prepare import AutoMLPrepare
from .train import AutoMLTrain
from .results import ExperimentResults
from .search_space import SearchSpace, FeatureStrategy, ModelType
from .evaluator import DomainMetrics

PROGRAMS_DIR = os.path.join(os.path.dirname(__file__), 'programs')

__all__ = [
    'AutoMLPipeline',
    'Program',
    'AutoMLPrepare',
    'AutoMLTrain',
    'ExperimentResults',
    'SearchSpace',
    'FeatureStrategy',
    'ModelType',
    'DomainMetrics',
    'PROGRAMS_DIR',
    'list_programs',
    'load_program',
]


def list_programs():
    """List available program.md templates."""
    return [f.replace('.md', '') for f in os.listdir(PROGRAMS_DIR) if f.endswith('.md')]


def load_program(name: str) -> str:
    """Load a program.md template by name."""
    path = os.path.join(PROGRAMS_DIR, f'{name}.md')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Program '{name}' not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()