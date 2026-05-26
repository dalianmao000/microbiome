# Traditional AutoML vs Autoresearch Methodology

**Date**: 2026-05-27
**Topic**: Core differences between traditional AutoML frameworks and autoresearch approach

---

## Overview

| Aspect | Traditional AutoML | Autoresearch |
|-------|-------------------|-------------|
| **Core Idea** | Hyperparameter optimization + architecture search | Program-guided agent-driven research |
| **Search Strategy** | Bayesian, RL, Evolutionary algorithms | Human-authored constraints + agent code generation |
| **Evaluation** | Single/multi-objective metric | Fixed evaluator + composite score |
| **Code Modifiability** | Black-box optimizer, user controls config | Human writes constraints, agent modifies train.py |
| **Iteration Speed** | Minutes to hours per trial | Fast iterations (5-min cycles) |
| **Result Reproducibility** | Config-based | Git-versioned experiments |

---

## Traditional AutoML Frameworks

**Examples**: AutoGluon, TPOT, Auto-sklearn, H2O AutoML, NASBench

### Strengths
- **Proven effectiveness**: Widely used in Kaggle competitions, AutoML benchmarks
- **Comprehensive search**: Bayesian optimization, evolutionary algorithms, reinforcement learning
- **Broad algorithm coverage**: Classification, regression, time-series, recommendation
- **Production-ready**: Enterprise support, deployment tools, monitoring

### Limitations
- **Black-box tuning**: User configures search space, but underlying algorithm is opaque
- **Single-objective focus**: Optimizes for accuracy/AUC, less emphasis on domain-specific constraints
- **Domain-agnostic**: Designed for tabular data, not specialized for microbiome/biological data
- **No human-in-the-loop**: Hard to inject domain knowledge during search
- **Compute intensive**: Requires significant GPU/CPU resources for large search spaces

---

## Autoresearch Methodology

**Origin**: karpathy/autoresearch - a minimal AutoML framework for code generation research

### Core Design
```
┌─────────────────────────────────────┐
│  program.md  ←  Human-authored constraints  │
├─────────────────────────────────────┤
│  train.py    ←  Agent-modifiable code      │
├─────────────────────────────────────┤
│  prepare.py ←  Fixed evaluator (不可变)    │
└─────────────────────────────────────┘
```

### Strengths
- **Transparent constraints**: program.md explicitly encodes what to explore and what to avoid
- **Human-in-the-loop**: Domain experts write constraints, agent explores solutions
- **Fast iteration**: Lightweight experiments, git-versioned results
- **Domain-adaptable**: Can encode biological/clinical constraints directly
- **Multi-objective**: Performance + stability + biological relevance out-of-the-box

### Limitations
- **Not a full AutoML system**: Focuses on code generation research, not production deployment
- **Limited algorithm support**: Simple search (random), no Bayesian optimization
- **Manual result tracking**: Relies on git for experiment versioning (no automatic logging dashboard)
- **No AutoEnsemble**: Doesn't automatically combine multiple models
- **Scalability**: Designed for single-node execution

---

## Key Differences

### 1. Role of Human Expert

| Traditional AutoML | Autoresearch |
|-------------------|--------------|
| User provides search space + metric | Expert writes program.md constraints |
| Human cannot intervene mid-search | Human can adjust constraints between iterations |
| Domain knowledge injected via config | Domain knowledge encoded as rules |
| Expert role: initial setup | Expert role: continuous feedback |

### 2. Search Mechanism

| Traditional AutoML | Autoresearch |
|-------------------|--------------|
| Bayesian optimization (TPE, Parzen estimator) | Random search + agent exploration |
| Evolutionary algorithms (genetic programming) | Agent modifies train.py code |
| Reinforcement learning (Neural Architecture Search) | Agent writes new code strategies |
| Search space is fixed during optimization | Search space can be refined via program.md |

### 3. Evaluation Philosophy

| Traditional AutoML | Autoresearch |
|-------------------|--------------|
| Fixed evaluation function | prepare.py is immutable (fixed evaluator) |
| Single objective (accuracy, AUC) | Composite score: performance + stability + bio_relevance |
| Test set evaluation | Cross-validation + stability tracking |
| Overfitting risk on validation set | Explicit stability measure to combat overfitting |

### 4. Microbiome-Specific Adaptations

| Aspect | Traditional AutoML | Autoresearch |
|--------|-------------------|--------------|
| Feature selection | Generic methods (no domain knowledge) | Domain-aware strategies (LEfSe, ANCOM, LASSO) |
| Biological relevance | Not considered | DomainMetrics.biological_relevance() |
| Feature stability | Not measured | Jaccard similarity across CV folds |
| Domain constraints | Hard to inject | program.md encode gut-lung axis, TCM constitution rules |

---

## When to Use Which

### Use Traditional AutoML When:
- ✅ Working with standard tabular data (customer, financial, sensor data)
- ✅ Need production-ready deployment with minimal effort
- ✅ Want automatic ensemble of multiple models
- ✅ Have limited domain expertise to write constraints
- ✅ Need Bayesian optimization or neural architecture search

### Use Autoresearch When:
- ✅ Working with specialized domain (microbiome, genomics, clinical)
- ✅ Want to inject domain knowledge explicitly
- ✅ Need multi-objective optimization with domain-specific metrics
- ✅ Planning human-in-the-loop research workflows
- ✅ Interested in exploring algorithm design space, not just hyperparameter tuning

---

## Hybrid Approach (Recommended for Microbiome)

```
┌──────────────────────────────────────────────────────┐
│           Hybrid AutoML for Microbiome                │
├──────────────────────────────────────────────────────┤
│  1. Autoresearch-style:                               │
│     - program.md with microbiome constraints          │
│     - Multi-objective scoring (performance + stability │
│       + biological relevance)                        │
│     - Feature selection strategies (LEfSe, LASSO)    │
├──────────────────────────────────────────────────────┤
│  2. Traditional AutoML-style:                        │
│     - Bayesian optimization (Optuna) for hyperparams  │
│     - AutoGluon-style ensemble for final model        │
│     - Production deployment pipeline                  │
└──────────────────────────────────────────────────────┘
```

### Implementation Plan

1. **Phase 1**: Use autoresearch for exploration (train.py iterations)
2. **Phase 2**: Apply Bayesian optimization (Optuna) on top-performing configs
3. **Phase 3**: Ensemble best models using AutoGluon or custom stacking

---

## References

- karpathy/autoresearch: https://github.com/karpathy/autoresearch
- AutoGluon: https://github.com/autogluon/autogluon
- TPOT: https://github.com/EpistasisLab/tpot
- LEfSe: https://doi.org/10.1093/nar/gkdr932
- ANCOM-BC: https://doi.org/10.1371/journal.pbio.3001122