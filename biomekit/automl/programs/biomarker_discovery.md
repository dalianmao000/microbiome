# program.md - Biomarker Discovery for IBD Classification

> **场景**: 疾病相关菌群标志物筛选 (autoresearch微生物组场景1)
> **适用**: 炎症性肠病(IBD)等慢性肠道疾病的微生物标志物自动发现

---

## 🔬 研究目标

在16S rRNA测序数据中，筛选出与炎症性肠病(IBD)显著相关的微生物标志物组合，
构建可解释、可泛化的二分类预测模型。

**评价指标**: 多目标优化 - 性能(0.5) + 稳定性(0.25) + 生物相关性(0.25)

---

## 📊 数据规范

- 输入: OTU/ASV表格 (samples × features), 已稀有化+CLR转换
- 标签: binary (0=健康, 1=IBD)
- 元数据: age, sex, BMI, antibiotic_use (作为协变量)
- 数据分割: 70%训练 / 15%验证 / 15%测试, 分层抽样, 固定random_state=42

---

## 🧭 搜索空间 (agent可修改部分)

### 特征工程策略
- [x] `lefse` - LEfSe差异丰度分析 (LDA>2.0)
- [x] `rf_importance` - RandomForest特征重要性
- [x] `lasso` - LASSO正则化特征选择
- [x] `ancom` - ANCOM-BC差异检验
- [x] `wilcoxon` - Wilcoxon秩和检验

### 模型候选池
- [x] `rf` - RandomForest (n_estimators∈[50,200], max_depth∈[3,10])
- [x] `xgb` - XGBoost (n_estimators∈[100,300], max_depth∈[3,8], learning_rate∈[0.01,0.3])
- [x] `svm` - SVM (C∈[0.1,10], kernel∈['rbf','linear'])
- [x] `lr` - LogisticRegression (C∈[0.01,10], penalty∈['l1','l2'])
- [x] `mlp` - MLP (hidden_layer_sizes∈[(64,),(128,),(64,32)], alpha∈[0.0001,0.01])

### 超参数范围
- `n_features`: [10, 20, 50, 100]  # 最终入选标志物数量
- `cv_folds`: 5  # 固定,用于内部验证

---

## 📏 评估函数 (prepare.py实现, agent不可修改)

### 主指标 (权重0.5)
- 5折交叉验证的平均 ROC-AUC (测试集)

### 辅助指标
- **特征稳定性** (权重0.25): 在10次bootstrap中,入选特征的重合率 (Jaccard similarity)
- **生物学相关性** (权重0.25): 入选特征中,有至少30%在Disbiome/KEGG数据库有IBD关联文献

### 综合得分
```
score = 0.5×AUC + 0.25×Stability + 0.25×Biological_Relevance
```

---

## 🧬 领域约束规则 (硬约束,违反则实验直接失败)

1. **[必须]** 最终模型特征数 ≤ 50,避免过拟合小样本
2. **[必须]** 若使用树模型,必须输出feature_importance
3. **[禁止]** 特征选择过程泄露测试集信息 (必须在CV折内独立完成)
4. **[建议]** 优先保留有明确功能注释的菌属 (如Faecalibacterium, Roseburia, Akkermansia)

---

## 🔄 迭代策略

- 每次迭代只修改1个组件: 要么特征选择策略,要么模型类型,要么超参数
- 若连续3次迭代综合得分提升 < 0.01,则尝试"探索模式": 随机组合新策略
- 保留历史最佳配置到 `results/best_config.json`

---

## 📤 输出规范

每次实验必须记录:
- `selected_features`: List[str] + 对应LDA值/p值
- `model_config`: 模型类型+关键超参数
- `metrics`: {auc, stability_score, bio_relevance}
- `explanation`: 简要说明选择该策略的reasoning

---

## 🚫 终止条件

- 达到最大迭代次数: 50次
- 或 测试集AUC ≥ 0.85 且 特征稳定性 ≥ 0.6
- 或 连续10次迭代无显著改进 (Δscore < 0.005)