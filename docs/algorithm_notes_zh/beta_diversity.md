# Beta 多样性算法笔记

## 概述

Beta 多样性用于衡量样本之间的相似性或差异性。主要指标包括 Bray-Curtis、Jaccard 和 UniFrac 距离。

## 主要指标

### Bray-Curtis 差异度
$$BC_{ij} = 1 - \frac{2\sum_k \min(x_{ki}, x_{kj})}{\sum_k x_{ki} + \sum_k x_{kj}}$$

### Jaccard 距离
$$J_{ij} = 1 - \frac{|A \cap B|}{|A \cup B|}$$

### UniFrac 距离
UniFrac 利用系统发育信息：
- **加权 UniFrac**：考虑分支长度和丰度
- **非加权 UniFrac**：仅考虑存在/缺席

## PCoA（主坐标分析）

PCoA 将距离矩阵转换为一组正交轴，以最大化方差。

## 如何选择指标？

| 指标 | 考虑因素 | 适用场景 |
|--------|----------|----------|
| Bray-Curtis | 丰度 | 一般比较 |
| Jaccard | 存在/缺席 | 更替 |
| 加权 UniFrac | 丰度 + 系统发育 | 功能多样性 |
| 非加权 UniFrac | 系统发育 | 结构多样性 |

## 面试要点

- "Beta 多样性衡量样本间多样性"
- "Bray-Curtis 因其考虑丰度差异而被广泛使用"
- "UniFrac 结合了系统发育关系，对于理解进化分歧很有帮助"