"""Domain metrics for microbiome AutoML: stability and biological relevance."""
from typing import Dict, List, Any, Optional
import numpy as np

BIOLOGICAL_RELEVANCE_DB = {
    'IBD': ['Bifidobacterium', 'Faecalibacterium', 'Lactobacillus', 'Roseburia', 'Eubacterium'],
    'CDI': ['Bacteroides', 'Ruminococcus', 'Blautia', 'Collinsella'],
    'metabolic_syndrome': ['Akkermansia', 'Bacteroides', 'Prevotella', 'Alistipes'],
    'respiratory': ['Streptococcus', 'Haemophilus', 'Neisseria', 'Fusobacterium'],
}


class DomainMetrics:
    @staticmethod
    def feature_stability(selected_features_list: List[List[str]]) -> float:
        if not selected_features_list or len(selected_features_list) < 2:
            return 0.0
        n = len(selected_features_list)
        jaccard_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                set_i = set(selected_features_list[i])
                set_j = set(selected_features_list[j])
                if not set_i or not set_j:
                    continue
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccard_scores.append(intersection / union if union > 0 else 0.0)
        return float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

    @staticmethod
    def biological_relevance(selected_features: List[str], disease: str) -> float:
        if not selected_features or disease not in BIOLOGICAL_RELEVANCE_DB:
            return 0.0
        known_relevant = set(BIOLOGICAL_RELEVANCE_DB.get(disease, []))
        if not known_relevant:
            return 0.0
        matched = sum(1 for f in selected_features if f in known_relevant)
        return float(matched / len(selected_features)) if selected_features else 0.0

    def compute_composite(
        self,
        performance: float,
        stability: float,
        bio_relevance: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        if weights is None:
            weights = {'performance': 0.5, 'stability': 0.25, 'bio_relevance': 0.25}
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        score = (
            normalized_weights.get('performance', 0.5) * performance +
            normalized_weights.get('stability', 0.25) * stability +
            normalized_weights.get('bio_relevance', 0.25) * bio_relevance
        )
        return float(score)