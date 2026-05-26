# FAPROTAX Algorithm Notes

## Overview

FAPROTAX is a database and Python script for functional annotation of prokaryotic taxa based on published cultures and functional references.

## Key Features

### 1. Culture-Based Annotation

FAPROTAX builds functional assignments from published bacterial and archaeal cultures, providing more reliable functional predictions for well-characterized taxa.

### 2. Function Categories

FAPROTAX covers key biogeochemical cycles:
- **Nitrogen cycling**: nitrification, denitrification, nitrogen fixation
- **Carbon cycling**: fermentation, methanogenesis
- **Sulfur cycling**: sulfate reduction

### 3. Taxonomic Assignment

Unlike PICRUSt2 which uses phylogenetic placement, FAPROTAX requires taxonomic annotation of input sequences.

## Limitations

1. **Culture bias** - Only includes taxa with cultured representatives
2. **Limited scope** - Mostly environmental bacteria
3. **Taxonomy dependent** - Requires accurate taxonomic classification

## Interview Talking Points

- "FAPROTAX is based on published cultures, so it's well-validated for known bacteria"
- "It covers important functional groups like nitrogen and carbon cyclers"
- "The trade-off is that it may miss uncultured or poorly characterized taxa"