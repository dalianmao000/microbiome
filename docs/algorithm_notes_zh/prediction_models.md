# 预测模型文档

## 概述

预测模块提供了一套完整的机器学习工具，用于基于微生物组的疾病分类、疗效预测和患者预后建模。

## 架构

```
原始丰度矩阵
        ↓
预处理 (CLR/Log/百分比变换 + 方差过滤)
        ↓
编码 (可选：Autoencoder/VAE 用于降维)
        ↓
分类 (RF, SVM, XGBoost, MLP, GradientBoosting)
        或
生存分析 (Cox Proportional Hazards)
        ↓
评估 (多指标交叉验证)
        ↓
解释 (SHAP, Permutation Importance, Partial Dependence Plots)
```

## 核心类

### MicrobiomePipeline

集预处理、编码和分类于一体的完整预测流程。

```python
from biomekit.prediction import MicrobiomePipeline

pipeline = MicrobiomePipeline(
    preprocess='clr',     # 变换方式：'clr', 'log', 'percent'
    encode='autoencoder', # 编码方式：'autoencoder', 'vae' 或 None
    classifier='rf',      # 分类器：'rf', 'svm', 'xgb', 'mlp', 'gb'
    latent_dim=16,        # Autoencoder 的潜在维度
    n_estimators=100      # 树的数量（基于树的模型）
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
results = pipeline.evaluate(X_test, y_test, cv=5)
```

### 预处理

- `CLRTransformer`：针对组成型数据的中心化对数比变换
- `LogTransformer`：带伪计数的对数变换
- `PercentTransformer`：行方向百分比/秩变换
- `VarianceFilter`：移除低方差特征
- `PreprocessingPipeline`：可配置的处理流程，组合多种变换和过滤方法

### 编码器

- `AutoencoderEncoder`：用于降维的神经网络
- `VAEEncoder`：用于概率表示的变分自编码器

### 分类器

- `MicrobiomeClassifier`：RF、SVM、XGBoost、MLP、GB 的统一接口

### 解释器

- `SHAPExplainer`：用于特征重要性的 SHAP 值
- `PermutationImportance`：基于置换的重要性
- `PartialDependencePlot`：特征与预测值的关系

## 评估指标

该流程使用多个指标进行评估：
- 准确率
- 精确率（加权）
- 召回率（加权）
- F1 分数（加权）
- AUC-ROC（二分类）
- AUC-PR（二分类）

交叉验证（默认 5 折分层）提供每个指标的均值和标准差。

## 示例：疾病分类

```python
from biomekit.prediction import MicrobiomePipeline
from biomekit.utils.simulate import simulate_abundance_data

# 生成数据
X, metadata = simulate_abundance_data(n_samples=200, n_features=500)
y = (metadata['group'] == 'disease').astype(int).values

# 运行流程
pipeline = MicrobiomePipeline(preprocess='clr', classifier='rf', n_estimators=100)
pipeline.fit(X, y)

# 评估
results = pipeline.evaluate(X, y, cv=5)
print(f"准确率: {results['accuracy'][0]:.3f} ± {results['accuracy'][1]:.3f}")
print(f"AUC-ROC: {results['auc_roc'][0]:.3f} ± {results['auc_roc'][1]:.3f}")

# 获取重要特征
importance = pipeline.get_feature_importance()
```

## 示例：预后建模

```python
from biomekit.prediction import SurvivalRegressor

# 生存时间和事件
y_time = np.random.rand(100) * 100 + 10  # 生存时间
y_event = np.random.randint(0, 2, 100)  # 1=死亡, 0=截尾

regressor = SurvivalRegressor(model='cox')
regressor.fit(X, y_time, y_event)
risk_scores = regressor.predict_risk(X)
```

## 依赖

- scikit-learn：基础机器学习工具
- PyTorch：神经网络编码器
- xgboost：梯度提升
- shap：SHAP 解释
- lifelines：生存分析
- matplotlib/seaborn：可视化