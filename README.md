# biomekit

A comprehensive Python toolkit for microbiome data analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## Features

- **Differential Abundance Analysis**: LEfSe, DESeq2, ANCOM-BC
- **Alpha Diversity**: Observed, Shannon, Chaol, Simpson indices
- **Beta Diversity**: Bray-Curtis, Jaccard, UniFrac distances with PCoA
- **Statistical Tests**: PERMANOVA, ANOSIM
- **Functional Prediction**: PICRUSt2, FAPROTAX wrappers
- **Network Analysis**: Spearman, SparCC correlation networks
- **Phylogenetic Tools**: Tree building with bootstrap support

## Installation

### Via pip
```bash
pip install biomekit
```

### Via Docker
```bash
docker-compose up -d
```

### From source
```bash
git clone https://github.com/dalianmao000/biomekit.git
cd biomekit
pip install -e .
```

## Quick Start

```python
from biomekit.utils.simulate import simulate_abundance_data
from biomekit.abundance import run_lefse
from biomekit.diversity import alpha_diversity, beta_diversity, pcoa
from biomekit.function import run_picrust2

# Generate simulated microbiome data
abundance_df, metadata = simulate_abundance_data(
    n_samples=50,
    n_features=100,
    n_diff_abundant=10
)

# Calculate alpha diversity
alpha_div = alpha_diversity(abundance_df)
print(f"Alpha diversity:\n{alpha_div.head()}")

# Beta diversity and ordination
dist_matrix = beta_diversity(abundance_df, metric='braycurtis')
coords = pcoa(dist_matrix)
print(f"\nPCoA coordinates:\n{coords.head()}")

# Differential abundance analysis
results = run_lefse(abundance_df, metadata, group_column='group')
print(f"\nSignificant features: {results['summary']['n_significant']}")

# Functional prediction
func_results = run_picrust2(abundance_df)
print(f"\nKO abundance shape: {func_results['KO_abundance'].shape}")
```

## Project Structure

```
biomekit/
├── biomekit/              # Main Python package
│   ├── abundance/        # Differential abundance (LEfSe, DESeq2, ANCOM-BC)
│   ├── diversity/        # Alpha/Beta diversity and stats
│   ├── function/        # Functional prediction (PICRUSt2, FAPROTAX)
│   ├── network/          # Correlation network analysis
│   ├── phylogeny/         # Phylogenetic tree tools
│   └── utils/            # I/O, transforms, visualization
├── tests/                # Unit and integration tests (41 tests)
├── docs/algorithm_notes/ # Algorithm documentation (8 docs)
├── data/                 # Data simulation utilities
├── pyproject.toml        # Poetry package configuration
├── Dockerfile           # Docker image definition
└── docker-compose.yml   # Docker Compose configuration
```

## Module Overview

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `abundance` | Differential abundance analysis | `run_lefse`, `run_deseq2`, `run_ancombc` |
| `diversity` | Alpha/Beta diversity & statistics | `alpha_diversity`, `beta_diversity`, `permanova` |
| `function` | Functional prediction from 16S data | `run_picrust2`, `run_faprotax` |
| `network` | Microbial correlation networks | `sparcc_network`, `spearman_network` |
| `phylogeny` | Phylogenetic tree construction | `build_tree`, `bootstrap_tree` |
| `utils` | Data I/O, transforms, visualization | `read_tsv`, `clr_transform`, `plot_pcoa` |

## API Examples

### LEfSe Analysis

```python
from biomekit.abundance import run_lefse

results = run_lefse(
    abundance_df,
    metadata,
    group_column='group',
    alpha=0.05,
    lda_threshold=2.0
)
print(f"Found {results['summary']['n_significant']} significant features")
```

### Diversity Analysis

```python
from biomekit.diversity import alpha_diversity, beta_diversity, permanova

# Alpha diversity
alpha_div = alpha_diversity(abundance_df, metrics=['shannon', 'chao1'])

# Beta diversity
dist_matrix = beta_diversity(abundance_df, metric='braycurtis')

# Statistical test
result = permanova(dist_matrix, metadata, group_column='group')
print(f"PERMANOVA p-value: {result['p_value']:.4f}")
```

### Network Analysis

```python
from biomekit.network import sparcc_network

network = sparcc_network(abundance_df, threshold=0.3)
print(f"Network edges: {len(network)}")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=biomekit --cov-report=html

# Run specific module
pytest tests/test_abundance/ -v
```

## Documentation

Algorithm documentation is available in `docs/algorithm_notes/`:
- [LEfSe](docs/algorithm_notes/lefse.md)
- [DESeq2](docs/algorithm_notes/deseq2.md)
- [ANCOM-BC](docs/algorithm_notes/ancombc.md)
- [Alpha Diversity](docs/algorithm_notes/alpha_diversity.md)
- [Beta Diversity](docs/algorithm_notes/beta_diversity.md)
- [PERMANOVA](docs/algorithm_notes/permanova.md)
- [PICRUSt2](docs/algorithm_notes/picrust2.md)
- [FAPROTAX](docs/algorithm_notes/faprotax.md)

## Roadmap

| Priority | Module/Feature | Status | Description |
|:--------:|:---------------|:------:|:------------|
| P0 | Algorithm Module Library | ✅ Complete | Core differential abundance, diversity, and network algorithms |
| P1 | Prediction Model Suite | 🔜 Planned | Disease classification,疗效预测 models with ML/DL |
| P2 | Multi-omics Integration | 🔜 Planned | 16S + metabolome + metagenome fusion analysis |
| P3 | Snakemake Pipeline | 🔜 Planned | Production-grade workflow with Snakemake/Nextflow |
| P4 | Web Dashboard | 🔜 Planned | Interactive visualization dashboard |
| P5 | AutoML Module | 🔜 Planned | Automated hyperparameter optimization for microbiome data |

### Status Legend
- ✅ Complete - Ready for use
- 🔜 Planned - On development roadmap
- 🔄 In Progress - Currently being developed

## Use Cases

This project demonstrates proficiency in:

1. **Algorithm implementation**: Understanding and implementing published algorithms (LEfSe, alpha/beta diversity)
2. **Bioinformatics pipelines**: Building modular, reproducible analysis workflows
3. **Statistical analysis**: Applying appropriate statistical tests for microbiome data
4. **Data visualization**: Creating publication-quality plots
5. **Software engineering**: Packaging, testing, documentation best practices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

dalianmao000