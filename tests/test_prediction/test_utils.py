import pytest
import numpy as np
from biomekit.prediction.utils import extract_top_markers

def test_extract_top_markers():
    """Test marker extraction from importance scores."""
    importance = np.random.rand(50)
    feature_names = [f'feature_{i}' for i in range(50)]

    markers = extract_top_markers(importance, feature_names, top_n=10)
    assert len(markers) == 10
    assert all(isinstance(m, tuple) for m in markers)
    assert all(isinstance(m[0], str) for m in markers)
    assert all(isinstance(m[1], (int, float)) for m in markers)

def test_extract_markers_direction():
    """Test marker extraction with different directions."""
    importance = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    feature_names = ['a', 'b', 'c', 'd', 'e']

    markers_pos = extract_top_markers(importance, feature_names, top_n=3, direction='positive')
    assert markers_pos[0][0] == 'd'  # Highest

    markers_neg = extract_top_markers(importance, feature_names, top_n=3, direction='negative')
    assert markers_neg[0][0] == 'a'  # Lowest