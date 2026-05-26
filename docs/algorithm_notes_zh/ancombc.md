# ANCOM-BC 算法笔记

## 概述

ANCOM-BC (Analysis of Composition of Microbiomes with Bias Correction，专著偏差校正的微生物组组成分析) 是专为 microbiome 数据设计的分析方法，用于解决基于组成性分析中的偏差问题。

## 核心概念

### 1. 偏差校正 (Bias Correction)

ANCOM-BC 校正由 microbiome 数据的组成性特性所引入的偏差：

$$W_{ij} = \log\left(\frac{\tau_{ij}}{\tau_{i+}}\right) - E\left[\log\left(\frac{\tau_{ij}}{\tau_{i+}}\right)\right]$$

### 2. 对数比值分析 (Log-Ratio Analysis)

ANCOM-BC 使用特征的对数比值进行运算，避免了直接比较丰度时的问题：

$$\text{log-ratio}_{ij} = \log(x_i) - \log(x_j)$$

### 3. 结构零的处理 (Structural Zero Handling)

ANCOM-BC 可以处理结构零（某些组中不存在但在其他组中存在的特征）。

## 相对于 DESeq2 的优势

1. **专为 microbiome 设计** - 无需调整组成性
2. **显式处理零值** - 更适合稀疏的 microbiome 数据
3. **对异常值不敏感**

## 局限性

1. **计算密集** - 大规模数据集需要优化
2. **解释复杂度** - 对数比值不够直观

## 面试要点

- "ANCOM-BC 专为处理 microbiome 数据的组成性特性而设计"
- "偏差校正项解释了我们无法观察到绝对丰度这一事实"
- "结构零处理非常重要，因为许多微生物在某些样本中确实不存在"