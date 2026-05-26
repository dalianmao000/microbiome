# DESeq2 算法笔记

## 概述

DESeq2 是一款差异表达分析工具，最初为 RNA-seq 数据开发，现已广泛用于微生物组差异丰度分析。它采用负二项分布广义线性模型（negative binomial GLM）。

## 核心概念

### 1. 归一化（Size Factors）

DESeq2 估计 size factors 以校正样本间测序深度的差异。

$$\text{size factor}_s = \text{median}_{g} \frac{x_{gs}}{\left(\prod_{v=1}^{n} x_{vs}\right)^{1/n}}$$

### 2. 离散估计

离散度（dispersion）作为平均表达量的函数进行估计：

$$\hat{\alpha}_g = \alpha_0 + \frac{\alpha_1}{\bar{x}_g}$$

### 3. 负二项分布 GLM

计数数据 $Y_{gs}$ 的模型为：

$$Y_{gs} \sim \text{NB}(\mu_{gs}, \alpha_g)$$
$$\mu_{gs} = s_g \cdot q_g$$

其中 $s_g$ 为 size factor，$q_g$ 为期望表达水平。

## 为什么用 DESeq2 分析微生物组？

1. **处理文库大小差异** — 对微生物组测序至关重要
2. **直接对计数数据建模** — 避免对数转换的问题
3. **收缩估计器** — 提高小样本量研究的稳定性

## 局限性

1. **假设 NB 分布** — 可能不适合零inflation的微生物组数据
2. **参考组选择很重要** — 结果可能因参考水平不同而变化
3. **对异常值敏感** — 某些 OTU 可能主导分析结果

## 面试要点

- "DESeq2 最初为 RNA-seq 设计，但对微生物组计数数据同样适用"
- "负二项分布比泊松分布更好地处理过度离散问题"
- "DESeq2 的收缩估计器有助于改善微生物组研究中常见的小样本量问题"