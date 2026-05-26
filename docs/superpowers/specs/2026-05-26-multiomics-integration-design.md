# Multi-omics Integration — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a comprehensive multi-omics integration module that fuses 16S, metabolomics, and metagenomics data for biomarker discovery, functional correlation, and integrated classification.

**Architecture:** A `MultiOmicsPipeline` class with swappable fusion strategies (early/late) and selectable methods per scenario (DIABLO/sPLS-DA for biomarkers, CCA/Procrustes for correlation). Outputs a `MultiOmicsReport` object. Reuses existing biomekit components (CLR transform, MicrobiomePipeline, plotting utilities).

**Tech Stack:** Python, pandas, numpy, scikit-learn, mixOmics (optional), scipy

---

## Module Structure

```
biomekit/integration/
├── __init__.py
├── multiomics_pipeline.py   # MultiOmicsPipeline class
├── fusion.py               # Fusion methods (DIABLO, sPLS-DA, CCA, Procrustes)
├── report.py               # MultiOmicsReport class
└── utils.py                # Input format detection, omics alignment
```

---

## Input Format Auto-Detection

Two accepted input formats:

1. **Separate dict** — `{'16s': df_16s, 'metabolomics': df_meta, 'metagenomics': df_mgn}` where each value is a DataFrame with samples as rows and features as columns
2. **Pre-aligned single DataFrame** — All omics merged into one matrix with omics-type labels embedded in column names or row metadata

Detection logic: if input is a dict with DataFrame values → separate omics mode; if a single DataFrame → pre-aligned mode.

Internal alignment: sample-based inner join ensures only common samples across omics are analyzed. Feature matching by name (exact or fuzzy via Levenshtein distance threshold ≥0.8).

---

## Scenario 1: Biomarker Discovery

### Methods

**DIABLO (Data Integration Analysis for Biomarker discovery using Latent cOmponents)**
- Sparse multi-block discriminant analysis
- Selects features from each omics that best discriminate groups
- Output: canonical loadings per omics, discriminant components, selected features per block

**sPLS-DA (sparse Partial Least Squares Discriminant Analysis)**
- Works on concatenated omics matrix
- L1 penalization for feature selection
- Output: loading vectors, selected feature list

### Parameters
- `biomarker_method`: `'diablo'` (default) or `'splsda'`
- `n_components`: Number of discriminant components (default: 2)
- `n_selected_features`: Features per omics to select (default: 50)

### Output
`biomarkers` key in results dict:
```python
{
    'method': 'diablo',
    'loadings': {omics_name: pd.DataFrame},   # canonical loadings
    'selected_features': {omics_name: list}, # feature names
    'components': np.ndarray,                  # discriminant components
    'explained_variance': float
}
```

---

## Scenario 2: Functional Correlation

### Methods

**CCA (Canonical Correlation Analysis)**
- Finds max-correlation linear combinations between two omics sets
- Output: canonical loadings, correlations,_scores per omics pair

**Procrustes Analysis**
- Rotates/scale-translates two PCoA coordinate matrices to maximize alignment
- Output: Procrustes statistic, transformed coordinates, residual sum of squares

### Parameters
- `correlation_method`: `'cca'` (default) or `'procrustes'`
- `n_components`: Number of canonical components (default: 2)

### Output
`correlations` key in results dict:
```python
{
    'method': 'cca',
    'loadings': {omics_pair: pd.DataFrame},   # canonical loadings per pair
    'correlations': np.ndarray,                # canonical correlations
    'scores': {omics_name: np.ndarray}       # canonical scores
}
# or for procrustes:
{
    'method': 'procrustes',
    'statistic': float,                        # Procrustes statistic
    'residuals': float,                        # residual sum of squares
    'transformed_coords': {ref: np.ndarray}   # transformed PCoA coords
}
```

---

## Scenario 3: Integrated Classification

### Fusion Strategies

**Early fusion** — All omics concatenated into single feature matrix → CLR transform → MicrobiomePipeline (encode → classify)
```python
# Conceptual flow
concatenated = pd.concat([df_16s, df_meta, df_mgn], axis=1)
pipeline = MicrobiomePipeline(preprocess='clr', classifier='rf')
pipeline.fit(concatenated, y)
```

**Late fusion** — Each omics independently preprocessed (CLR + optional PCA/autoencoder) → latent representations concatenated → joint classifier
```python
# Conceptual flow
latents = {name: autoencoder.encode(df) for name, df in omics_dict.items()}
fused = pd.concat(latents.values(), axis=1)
pipeline = MicrobiomePipeline(preprocess=None, classifier='rf')  # already encoded
pipeline.fit(fused, y)
```

### Parameters
- `fusion`: `'early'` (default) or `'late'`
- `classifier`: One of `'rf'`, `'svm'`, `'xgb'`, `'mlp'`, `'gb'` (default: `'rf'`)
- `encode`: For late fusion, encoder per omics: `'autoencoder'`, `'pca'`, or `None` (passthrough)
- `latent_dim`: Latent dimension per omics for autoencoder (default: 16)
- `cv`: Cross-validation folds for evaluation (default: 5)

### Output
`predictions` key in results dict:
```python
{
    'fusion': 'early',
    'y_true': np.ndarray,
    'y_pred': np.ndarray,
    'scores': np.ndarray,        # decision function or proba
    'accuracy': (mean, std),
    'f1': (mean, std),
    'auc': (mean, std),          # if binary
    'cv_results': dict           # per-fold details
}
```

---

## MultiOmicsReport Class

```python
from biomekit.integration import MultiOmicsReport

report = MultiOmicsReport(results)
report.biomarkers      # DataFrame: features × omics loadings
report.correlations    # DataFrame: canonical correlations per pair
report.predictions     # dict: y_true, y_pred, scores
report.plot()          # matplotlib.Figure
```

### Attributes
- `biomarkers`: Merged DataFrame of selected features and their loadings
- `correlations`: Canonical loadings matrix with omics pair labels
- `predictions`: Classification results dict

### Methods
- `plot()`: Generates a matplotlib figure with 2 subplots:
  1. Heatmap of biomarker loadings across omics
  2. Bar plot of classification metrics (accuracy, F1, AUC)
- `to_dict()`: Export all results as nested dict (for programmatic access)
- `summary()`: Print human-readable summary of all three scenarios

### Constructor
```python
def __init__(self, results: dict):
    """
    results: dict with keys 'biomarkers', 'correlations', 'predictions'
    Each value is the output from the respective analysis method.
    """
```

---

## Method Selection Matrix

| Scenario | Method | When to use |
|----------|--------|-------------|
| Biomarker discovery | DIABLO | Known group labels, need multi-omics discriminative features |
| Biomarker discovery | sPLS-DA | High-dimensional omics, want sparse feature selection |
| Functional correlation | CCA | Want linear relationships between two omics blocks |
| Functional correlation | Procrustes | Aligning ordination results (PCoA) across omics |
| Integrated classification | Early fusion | Omics strongly correlated, shared variance is informative |
| Integrated classification | Late fusion | Each omics has independent predictive signal |

---

## Integration with Existing Modules

| Existing Component | Used In | Purpose |
|-------------------|---------|---------|
| `MicrobiomePipeline` | Scenario 3 | Classification engine |
| `clr_transform` | Scenario 3 early fusion | Preprocessing |
| `AutoencoderEncoder` | Scenario 3 late fusion | Per-omics latent encoding |
| `plot_heatmap` | `MultiOmicsReport.plot()` | Visualize loadings |
| `beta_diversity`, `pcoa` | Procrustes | PCoA coordinates for Procrustes input |

---

## Error Handling

- **Missing omics**: If an omics key is missing from input dict, raise `ValueError` with list of expected keys
- **Empty intersection**: If no common samples across omics, raise `ValueError` with sample counts per omics
- **Feature name conflicts**: Warn and suffix duplicate feature names with omics prefix
- **Optional dependency missing** (`mixOmics`): If DIABLO/sPLS-DA requested but `mixOmics` not installed, raise `ImportError` with install hint. CCA and Procrustes use scipy only.
- **Invalid fusion mode**: Raise `ValueError` if `fusion` not in `{'early', 'late'}`
- **Invalid method**: Raise `ValueError` if `biomarker_method` or `correlation_method` not in valid options

---

## Optional Dependencies

- `mixOmics`: Required for DIABLO and sPLS-DA. Install via `pip install mixOmics`
- `scipy`: Required for CCA and Procrustes (already a common dependency)
- `scikit-learn`: Required for all classifiers and preprocessing

---

## File Responsibilities

### `fusion.py`
- `DIABLOAnalyzer`: Class wrapping mixOmics implementation of DIABLO
- `SPLSDAAnalyzer`: Class wrapping mixOmics sPLS-DA
- `CCAAnalyzer`: Class using scipy.stats.cca
- `ProcrustesAnalyzer`: Class using scipy.spatial.procrustes

Each analyzer is a self-contained class with:
- `fit(data_dict, y)` — fit the model
- `transform(data_dict)` — apply to new data
- `get_loadings()` — return canonical/sparse loadings DataFrame
- `get_selected_features()` — return feature selection list (DIABLO/sPLS-DA only)

### `multiomics_pipeline.py`
- `MultiOmicsPipeline`: Main class with `fit`, `transform`, `fit_transform`, `evaluate` methods
- `AutoDetectingInputFormat`: Utility class that detects and normalizes input format
- Internally routes to correct fusion and method classes

### `report.py`
- `MultiOmicsReport`: Result container class with attributes and plotting
- `MultiOmicsReportFactory`: Factory that constructs report from pipeline results

### `utils.py`
- `detect_input_format(data)` — Returns `'separate'` or `'prealigned'`
- `align_omics_data(omics_dict)` — Inner join samples, warn on feature conflicts
- `concat_with_labels(omics_dict)` — Concatenate with omics-type labels
- `validate_omics_keys(omics_dict)` — Check required omics keys are present
