# AutoML Module - P1/P2 Coverage Analysis

**Date**: 2026-05-27
**Status**: AutoML P5 complete, but P1/P2 coverage is partial

---

## P1: Prediction Model Suite — Partial Coverage

| P1需求 | AutoML现状 | 差距 |
|--------|-----------|------|
| 疾病分类 (RF/XGB/SVM/LR/MLP) | ✅ 6种模型全覆盖 | — |
| 超参数自动搜索 | ✅ random search | 缺少贝叶斯优化 |
| 特征选择策略 | ✅ 6种策略 | — |
| Autoencoder降维 | ❌ 未接入 | `preprocessing.py`有但不在search_space |
| 预后生存模型 (Cox) | ❌ 未支持 | 需新增ModelType |
| 可解释性 (SHAP) | ✅ `explainer.py`独立模块 | 需集成到pipeline |

**P1 Gap Summary**:
- Autoencoder encoding 和 Cox生存分析 需补充到search_space
- Random search可用，贝叶斯优化可选升级

---

## P2: Multi-omics Integration — Framework Only

| P2需求 | AutoML现状 | 差距 |
|--------|-----------|------|
| program.md模板 | ✅ `multiomics_integration.md` | — |
| Early/Late Fusion | ❌ 未实现 | 需在train.py/fusion.py |
| DIABLO/CCA/Procrustes | ❌ 未实现 | 在`biomekit/integration/`不在AutoML |
| 多组学超参数搜索 | ✅ 搜索空间已定义 | — |

**P2 Gap Summary**:
- AutoML搜索空间框架 ✅ 可用
- 多组学融合算法(DIABLO/CCA/Procrustes)需依赖`biomekit/integration/`模块
- AutoML不替代算法实现，只负责自动化调参

---

## Conclusion

- **AutoML是"自动化调参框架"，不是"算法实现"**
- P1的分类/回归自动化 ✅ 可用
- P2的搜索空间框架 ✅ 可用，但多组学融合算法需依赖`biomekit/integration/`模块
- Autoencoder encoding 和 Cox生存分析 需补充到search_space

---

## Potential Enhancements

### High Priority
- [ ] Add `autoencoder` to ModelType/encoding options in search_space
- [ ] Add `cox` survival model type for prognosis tasks
- [ ] Integrate SHAP explainer into AutoML pipeline output

### Medium Priority
- [ ] Add Bayesian optimization (Optuna) as alternative to random search
- [ ] Add Early/Late Fusion strategies to train.py

### Low Priority (depends on integration module)
- [ ] Wire DIABLO/CCA/Procrustes from `biomekit/integration/` into AutoML prepare.py
- [ ] Multi-omics feature selection strategies