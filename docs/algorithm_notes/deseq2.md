# DESeq2 Algorithm Notes

## Overview

DESeq2 is a differential expression analysis tool originally developed for RNA-seq data, now widely used for microbiome differential abundance analysis. It uses negative binomial generalized linear models.

## Key Concepts

### 1. Normalization (Size Factors)

DESeq2 estimates size factors to account for differences in sequencing depth across samples.

$$\text{size factor}_s = \text{median}_{g} \frac{x_{gs}}{\left(\prod_{v=1}^{n} x_{vs}\right)^{1/n}}$$

### 2. Dispersion Estimation

Dispersion is estimated as a function of the mean expression:

$$\hat{\alpha}_g = \alpha_0 + \frac{\alpha_1}{\bar{x}_g}$$

### 3. Negative Binomial GLM

The model for counts $Y_{gs}$ is:

$$Y_{gs} \sim \text{NB}(\mu_{gs}, \alpha_g)$$
$$\mu_{gs} = s_g \cdot q_g$$

Where $s_g$ is the size factor and $q_g$ is the expected expression level.

## Why DESeq2 for Microbiome?

1. **Handles library size differences** - Critical for microbiome sequencing
2. **Models count data directly** - Avoids log-transformation issues
3. **Shrinkage estimators** - Improves stability with small samples

## Limitations

1. **Assumes NB distribution** - May not fit zero-inflated microbiome data
2. **Reference selection matters** - Results can vary based on reference level
3. **Sensitive to outliers** - Some OTUs can dominate the analysis

## Interview Talking Points

- "DESeq2 was designed for RNA-seq but works well for microbiome count data"
- "The negative binomial distribution handles overdispersion better than Poisson"
- "DESeq2's shrinkage estimators help with small sample sizes common in microbiome studies"