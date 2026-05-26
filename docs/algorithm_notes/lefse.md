# LEfSe Algorithm Notes

## Overview

LEfSe (Linear Discriminant Analysis Effect Size) is a tool for high-dimensional biomarker discovery and explanation. It identifies features that are differentially abundant between groups.

## Algorithm Steps

### 1. Kruskal-Wallis Test (Multi-group comparison)

The Kruskal-Wallis H-test is a non-parametric method for testing whether samples originate from the same distribution.

**Why Kruskal-Wallis instead of ANOVA?**
- Microbiome data is typically non-normal (zero-inflated, skewed)
- Kruskal-Wallis does not assume normality

**Mathematical formulation:**
$$H = \\frac{12}{N(N+1)} \\sum_{j=1}^{g} \\frac{R_j^2}{n_j} - 3(N+1)$$

Where:
- $N$ = total number of samples
- $g$ = number of groups
- $R_j$ = sum of ranks in group $j$
- $n_j$ = number of samples in group $j$

### 2. Wilcoxon Test (Pairwise comparison)

For binary group comparison, Wilcoxon rank-sum test assesses whether the distributions differ.

### 3. Linear Discriminant Analysis (LDA)

LDA estimates the effect size of each feature's contribution to group separation.

**Key insight:** LDA provides both statistical significance AND biological relevance through effect size.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| alpha | 0.05 | Significance level for KW test |
| lda_threshold | 2.0 | Minimum LDA score for reporting |
| min_abundance | 1e-6 | Minimum mean abundance |
| min_prevalence | 0.1 | Minimum prevalence across samples |

## Limitations

1. Zero-inflated data may affect p-values
2. LDA assumes multivariate normality (often violated)
3. Multiple testing correction is critical

## Interview Talking Points

- "LEfSe uses a two-stage approach: first tests statistical significance, then estimates biological relevance via LDA"
- "The choice of Kruskal-Wallis over ANOVA reflects the non-normal nature of microbiome count data"
- "LDA threshold of 2.0 means the feature's log-ratio between groups is at least 2 on log10 scale"