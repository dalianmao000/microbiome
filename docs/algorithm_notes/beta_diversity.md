# Beta Diversity Algorithm Notes

## Overview

Beta diversity measures the similarity or dissimilarity between samples. Key metrics include Bray-Curtis, Jaccard, and UniFrac distances.

## Key Metrics

### Bray-Curtis Dissimilarity
$$BC_{ij} = 1 - \frac{2\sum_k \min(x_{ki}, x_{kj})}{\sum_k x_{ki} + \sum_k x_{kj}}$$

### Jaccard Distance
$$J_{ij} = 1 - \frac{|A \cap B|}{|A \cup B|}$$

### UniFrac Distances
UniFrac uses phylogenetic information:
- **Weighted UniFrac**: Considers branch lengths and abundances
- **Unweighted UniFrac**: Only considers presence/absence

## PCoA (Principal Coordinates Analysis)

PCoA transforms a distance matrix into a set of orthogonal axes that maximize variance.

## Which Metric to Use?

| Metric | Considers | Best for |
|--------|----------|----------|
| Bray-Curtis | Abundance | General comparison |
| Jaccard | Presence/absence | Turnover |
| Weighted UniFrac | Abundance + phylogeny | Functional diversity |
| Unweighted UniFrac | Phylogeny | Structural diversity |

## Interview Talking Points

- "Beta diversity measures between-sample diversity"
- "Bray-Curtis is widely used because it considers abundance differences"
- "UniFrac incorporates phylogenetic relationships, making it useful for understanding evolutionary divergence"