# PICRUSt2 算法笔记

## 概述

PICRUSt2（Phylogenetic Investigation of Communities by Reconstruction of Unobserved States，即通过重建未观测状态进行群落系统发育研究）用于从标记基因数据（如 16S rRNA）预测微生物群落的功能潜力。

## 核心概念

### 1. 隐藏状态预测

PICRUSt2 将 16S 序列放置于参考系统发育树中，并推断未观测祖先的功能基因含量。

### 2. 参考数据库

PICRUSt2 使用的参考数据库包括：
- **Greengenes**（用于 16S 序列）
- **KO**（KEGG Orthologs，KEGG 直系同源基因）
- **EC**（Enzyme Commission numbers，酶委员会编号）
- **MetaCyc 代谢途径**

### 3. 输出类型

- **KO 丰度**：KEGG 直系同源基因家族
- **EC 丰度**：酶委员会编号
- **代谢途径丰度**：MetaCyc 代谢途径

## 流程步骤

1. **放置 16S 序列**到参考树（通过 EPA-NG 或 pplacer）
2. **推断基因家族**从祖先状态
3. **折叠基因家族**到功能类别
4. **按样本分层**

## 局限性

1. **参考偏倚**——仅限于参考数据库中的基因
2. **仅限 16S**——无法捕获菌株水平变异
3. **预测功能**——不是直接宏基因组学

## 面试要点

- "PICRUSt2 利用生物之间的进化关系来预测它们的功能潜力"
- "它基于相似生物拥有的基因来推断一个生物可能具有的基因"
- "准确性取决于参考数据库的质量和完整性"