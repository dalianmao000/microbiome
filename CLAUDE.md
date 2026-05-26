# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

biomekit is a Python toolkit for microbiome data analysis. It provides algorithms for differential abundance analysis, diversity metrics, functional prediction, network analysis, ML-based disease classification, multi-omics integration, and AutoML with autoresearch methodology.

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
pytest tests/test_automl/ -v
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
- **integration/** - Multi-omics integration (DIABLO, CCA, Procrustes, Early/Late Fusion)
- **automl/** - Automated hyperparameter optimization following autoresearch methodology
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
- `biomekit.automl.AutoMLPipeline` - AutoML pipeline with autoresearch design
- `biomekit.integration.MultiOmicsPipeline` - Multi-omics fusion analysis
- `biomekit.utils.simulate.simulate_abundance_data()` - Generate synthetic data for testing

### Optional Dependencies

Some modules require external libraries that may not be installed:
- `rpy2` - For DESeq2 and ANCOM-BC (R-based analysis)
- `lifelines` - For Cox proportional hazards survival analysis
- `shap` - For SHAP-based model explanation
- `torch` - For autoencoder dimensionality reduction

All imports use try/except pattern in `__init__.py` so missing dependencies don't break the entire package.

### AutoML (autoresearch methodology)

The `automl/` module follows karpathy's autoresearch design:
- `program.md` - Human-authored constraints defining search space and goals
- `prepare.py` - Fixed evaluator (data loading, CV, metric computation)
- `train.py` - Agent-modifiable code
- `pipeline.py` - Orchestration combining all components
- `programs/` - Scenario-specific templates (biomarker_discovery, lung_gut_axis, nine_constitution, fmt_matching, multiomics_integration)

Multi-objective scoring: performance (0.5) + stability (0.25) + biological_relevance (0.25)

## Code Conventions

- Each module has its own `__init__.py` exposing key functions
- sklearn-compatible classes inherit from `BaseEstimator, TransformerMixin`
- Tests live in `tests/test_<module>/test_<component>.py`
- Algorithm documentation in `docs/algorithm_notes/` (EN) and `docs/algorithm_notes_zh/` (ZH)
- Technical memos in `docs/memo-*.md`

## Roadmap Status

| Priority | Module | Status |
|:--------:|:-------|:------:|
| P0 | Algorithm Module Library | ✅ Complete |
| P1 | Prediction Model Suite | ✅ Complete |
| P2 | Multi-omics Integration | ✅ Complete |
| P3 | Snakemake Pipeline | 🔜 Planned |
| P4 | Web Dashboard | 🔜 Planned |
| P5 | AutoML Module | ✅ Complete |

## Git Workflow

- Branch: `main`
- Tags: `v0.1.0`, `v0.2.0`
- Remote: `origin` → `https://github.com/dalianmao000/microbiome.git`
- Excluded from push: `chat-*.txt` files