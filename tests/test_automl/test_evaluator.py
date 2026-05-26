import pytest
import numpy as np
from biomekit.automl.evaluator import DomainMetrics

class TestDomainMetrics:
    def test_feature_stability_score(self):
        selected_list = [
            ['f1', 'f2', 'f3'],
            ['f1', 'f2', 'f4'],
            ['f1', 'f2', 'f3'],
            ['f1', 'f3', 'f5'],
            ['f2', 'f3', 'f4'],
        ]
        score = DomainMetrics.feature_stability(selected_list)
        assert 0.0 <= score <= 1.0

    def test_feature_stability_perfect(self):
        selected_list = [['f1', 'f2', 'f3']] * 5
        score = DomainMetrics.feature_stability(selected_list)
        assert score == 1.0

    def test_feature_stability_empty(self):
        score = DomainMetrics.feature_stability([])
        assert score == 0.0

    def test_biological_relevance_mock(self):
        selected = ['Bifidobacterium', 'Faecalibacterium', 'Lactobacillus']
        disease = 'IBD'
        score = DomainMetrics.biological_relevance(selected, disease)
        assert score > 0.5

    def test_biological_relevance_unknown_disease(self):
        selected = ['unknown_bacterium_xyz']
        score = DomainMetrics.biological_relevance(selected, 'unknown_disease_xyz')
        assert score == 0.0

    def test_composite_score(self):
        metrics = DomainMetrics()
        result = metrics.compute_composite(
            performance=0.8,
            stability=0.7,
            bio_relevance=0.6,
            weights={'performance': 0.5, 'stability': 0.25, 'bio_relevance': 0.25}
        )
        assert 0.0 <= result <= 1.0
        expected = 0.5 * 0.8 + 0.25 * 0.7 + 0.25 * 0.6
        assert abs(result - expected) < 0.001