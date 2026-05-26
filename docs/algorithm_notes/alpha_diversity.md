# Alpha Diversity Algorithm Notes

## Overview

Alpha diversity measures the diversity within a single sample. Key metrics include observed features, Shannon index, Chao1, and Simpson index.

## Key Metrics

### Observed Features
Simply counts the number of non-zero features (OTUs/species) in a sample.

### Shannon Index
$$H = -\sum_{i=1}^{S} p_i \log(p_i)$$

Where $p_i$ is the proportion of feature $i$.

### Chao1 Richness Estimator
$$S_{chao1} = S_{obs} + \frac{n_1(n_1-1)}{2(n_2+1)}$$

Where $n_1$ is the number of singletons and $n_2$ is the number of doubletons.

### Simpson Index
$$D = 1 - \sum_{i=1}^{S} p_i^2$$

## Which Metric to Use?

| Metric | Sensitive to | Best for |
|--------|-------------|----------|
| Observed | Sample size | Richness |
| Shannon | Rare species | General diversity |
| Chao1 | Low frequency | Rare species |
| Simpson | Dominant species | Evenness |

## Interview Talking Points

- "Alpha diversity measures within-sample diversity"
- "Chao1 is a richness estimator that accounts for unobserved species"
- "Simpson index gives more weight to dominant species, while Shannon is more sensitive to rare species"