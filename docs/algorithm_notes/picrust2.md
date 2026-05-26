# PICRUSt2 Algorithm Notes

## Overview

PICRUSt2 (Phylogenetic Investigation of Communities by Reconstruction of Unobserved States) predicts functional potential of microbial communities from marker gene data (e.g., 16S rRNA).

## Key Concepts

### 1. Hidden State Prediction

PICRUSt2 places 16S sequences onto a reference phylogenetic tree and infers the functional gene content of unobserved ancestors.

### 2. Reference Databases

PICRUSt2 uses reference databases like:
- **Greengenes** (for 16S sequences)
- **KOs** (KEGG Orthologs)
- **ECs** (Enzyme Commission numbers)
- **MetaCyc pathways**

### 3. Output Types

- **KO abundances**: KEGG Ortholog gene families
- **EC abundances**: Enzyme Commission numbers
- **Pathway abundances**: MetaCyc pathways

## Pipeline Steps

1. **Place 16S sequences** onto reference tree (via EPA-NG or pplacer)
2. **Infer gene families** from ancestor states
3. **Collapse gene families** to functional categories
4. ** Stratify by sample**

## Limitations

1. **Reference bias** - Limited to genes in reference databases
2. **16S only** - Cannot capture strain-level variation
3. **Predicted function** - Not direct metagenomics

## Interview Talking Points

- "PICRUSt2 uses the evolutionary relationships between organisms to predict their functional potential"
- "It infers what genes an organism likely has based on what similar organisms have"
- "The accuracy depends on the quality and completeness of the reference database"