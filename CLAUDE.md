# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

biomekit is a Python toolkit for microbiome data analysis. It provides algorithms for differential abundance analysis, diversity metrics, functional prediction, network analysis, and ML-based disease classification.

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=biomekit --cov-report=html

# Run specific module tests
pytest tests/test_abundance/ -v
pytest tests/test_prediction/ -v

# Run a single test
pytest tests/test_prediction/test_preprocessing.py::test_clr_transform -v

# Lint and type check
mypy biomekit/
black --check biomekit/
isort --check biomekit/
```

## Architecture

### Module Structure

The package is organized into domain-specific modules under `biomekit/`:

- **abundance/** - Differential abundance analysis (LEfSe, DESeq2, ANCOM-BC)
- **diversity/** - Alpha/Beta diversity metrics and statistical tests (PERMANOVA, ANOSIM)
- **function/** - Functional prediction wrappers (PICRUSt2, FAPROTAX)
- **network/** - Correlation network analysis (SparCC, Spearman)
- **phylogeny/** - Phylogenetic tree building
- **prediction/** - ML pipeline for disease classification and prognosis (Autoencoder, RF, SVM, XGBoost, SHAP)
- **utils/** - I/O, transforms (CLR), rarefaction, visualization

### Core Data Flow

Microbiome analysis typically follows this pipeline:
```
Raw Abundance Matrix → Preprocessing (CLR/log transform) → Analysis → Visualization
                                           ↓
                              ┌─────────────┼─────────────┐
                              ↓             ↓             ↓
                         diversity    abundance      prediction
                              ↓             ↓             ↓
                         beta/pcoa    lefse/deseq2   pipeline/clf
```

### Key Entry Points

- `biomekit.abundance.run_lefse()` - LEfSe differential abundance
- `biomekit.diversity.beta_diversity()` - Beta diversity distance matrix
- `biomekit.prediction.MicrobiomePipeline` - End-to-end classification pipeline
- `biomekit.utils.simulate.simulate_abundance_data()` - Generate synthetic data for testing

### Optional Dependencies

Some modules require external libraries that may not be installed:
- `rpy2` - For DESeq2 and ANCOM-BC (R-based analysis)
- `lifelines` - For Cox proportional hazards survival analysis
- `shap` - For SHAP-based model explanation
- `torch` - For autoencoder dimensionality reduction

All imports use try/except pattern in `__init__.py` so missing dependencies don't break the entire package.

## Code Conventions

- Each module has its own `__init__.py` exposing key functions
- sklearn-compatible classes inherit from `BaseEstimator, TransformerMixin`
- Tests live in `tests/test_<module>/test_<component>.py`
- Algorithm documentation in `docs/algorithm_notes/`

## Roadmap Status

| Priority | Module | Status |
|:--------:|:-------|:------:|
| P0 | Algorithm Module Library | ✅ Complete |
| P1 | Prediction Model Suite | ✅ Complete |
| P2 | Multi-omics Integration | 🔜 Planned |
| P3 | Snakemake Pipeline | 🔜 Planned |
| P4 | Web Dashboard | 🔜 Planned |
| P5 | AutoML Module | 🔜 Planned |

## Git Workflow

- Branch: `main`
- Tags: `v0.1.0`, `v0.2.0`
- Remote: `origin` → `https://github.com/dalianmao000/microbiome.git`
- Excluded from push: `chat-*.txt` files