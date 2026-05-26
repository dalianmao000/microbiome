# biomekit

A comprehensive Python toolkit for microbiome data analysis.

## Features

- Differential abundance analysis (LEfSe, DESeq2, ANCOM-BC)
- Alpha and Beta diversity analysis
- Functional prediction (PICRUSt2, FAPROTAX)
- Phylogenetic analysis
- Network correlation analysis

## Quick Start

```python
from biomekit.abundance import run_lefse
from biomekit.utils import simulate_abundance_data

# Generate simulated data
abundance_df, metadata = simulate_abundance_data(n_samples=50, n_features=100)

# Run LEfSe analysis
results = run_lefse(abundance_df, metadata, group_column="group")
```

## Installation

```bash
pip install biomekit
# or
docker-compose up -d
```

## Modules

See [docs/](docs/) for detailed documentation.