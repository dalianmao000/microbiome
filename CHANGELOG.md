# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-26

### Added
- Prediction model module (biomekit.prediction)
  - Preprocessing: CLR, log, percent transforms, variance filtering
  - Autoencoders: AE and VAE for dimensionality reduction
  - Classifiers: RF, SVM, XGBoost, MLP, GradientBoosting
  - Survival: Cox proportional hazards, linear, logistic regression
  - Pipeline: Complete fit/predict/evaluate workflow
  - Explainers: SHAP, permutation importance, partial dependence
  - Utils: Marker extraction, visualization helpers

## [0.1.0] - 2026-05-26

### Added
- Initial release
- Differential abundance analysis (LEfSe, DESeq2, ANCOM-BC)
- Alpha diversity metrics (Observed, Shannon, Chaol, Simpson)
- Beta diversity metrics (Bray-Curtis, Jaccard, UniFrac)
- PCoA ordination
- Statistical tests (PERMANOVA, ANOSIM)
- Functional prediction (PICRUSt2, FAPROTAX wrappers)
- Network correlation analysis (Spearman, SparCC)
- Phylogenetic tree building with bootstrap
- Comprehensive test suite (41 tests)
- Algorithm documentation (8 documents)
- Simulated data generator
- Docker support
- MIT License