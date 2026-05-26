# PERMANOVA Algorithm Notes

## Overview

PERMANOVA (Permutational Multivariate Analysis of Variance) is a non-parametric multivariate statistical test for comparing groups in ecological data.

## Mathematical Formulation

$$F = \frac{SS_{between}}{SS_{within}} \cdot \frac{df_{within}}{df_{between}}$$

Where:
- $SS_{between}$ = sum of squared distances between group centroids
- $SS_{within}$ = sum of squared distances within groups

## How PERMANOVA Works

1. Calculate observed F-statistic from the distance matrix
2. Permute group assignments randomly
3. Recalculate F-statistic for each permutation
4. Compare observed F to null distribution

## Advantages for Microbiome

1. **Non-parametric** - Does not assume normality
2. **Works with any distance** - Can use Bray-Curtis, UniFrac, etc.
3. **Multivariate** - Tests overall community differences

## Limitations

1. **Requires sufficient permutations** - For accurate p-values
2. **Can be affected by dispersion** - Similar to ANOVA assumptions
3. **May have inflated Type I error** - With heterogeneous dispersions

## Interview Talking Points

- "PERMANOVA tests whether groups differ in overall community composition"
- "It works by comparing within-group to between-group distances"
- "The p-value is obtained by permutation, making it non-parametric"