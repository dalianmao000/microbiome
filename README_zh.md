# biomekit

微生物组数据分析综合Python工具包。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 功能特性

- **差异丰度分析**: LEfSe, DESeq2, ANCOM-BC
- **Alpha多样性**: Observed, Shannon, Chaol, Simpson 指数
- **Beta多样性**: Bray-Curtis, Jaccard, UniFrac 距离及 PCoA 排序
- **统计检验**: PERMANOVA, ANOSIM
- **功能预测**: PICRUSt2, FAPROTAX 封装
- **网络分析**: Spearman, SparCC 相关性网络
- **系统发育**: 系统发育树构建与Bootstrap支持

## 安装

### 通过 pip 安装
```bash
pip install biomekit
```

### 通过 Docker 安装
```bash
docker-compose up -d
```

### 从源码安装
```bash
git clone https://github.com/dalianmao000/biomekit.git
cd biomekit
pip install -e .
```

## 快速开始

```python
from biomekit.utils.simulate import simulate_abundance_data
from biomekit.abundance import run_lefse
from biomekit.diversity import alpha_diversity, beta_diversity, pcoa
from biomekit.function import run_picrust2

# 生成模拟微生物组数据
abundance_df, metadata = simulate_abundance_data(
    n_samples=50,
    n_features=100,
    n_diff_abundant=10
)

# 计算Alpha多样性
alpha_div = alpha_diversity(abundance_df)
print(f"Alpha多样性:\n{alpha_div.head()}")

# Beta多样性与排序
dist_matrix = beta_diversity(abundance_df, metric='braycurtis')
coords = pcoa(dist_matrix)
print(f"\nPCoA坐标:\n{coords.head()}")

# 差异丰度分析
results = run_lefse(abundance_df, metadata, group_column='group')
print(f"\n显著特征数: {results['summary']['n_significant']}")

# 功能预测
func_results = run_picrust2(abundance_df)
print(f"\nKO丰度形状: {func_results['KO_abundance'].shape}")
```

## 项目结构

```
biomekit/
├── biomekit/              # 主Python包
│   ├── abundance/         # 差异丰度分析 (LEfSe, DESeq2, ANCOM-BC)
│   ├── diversity/         # Alpha/Beta多样性及统计
│   ├── function/          # 功能预测 (PICRUSt2, FAPROTAX)
│   ├── network/           # 相关性网络分析
│   ├── phylogeny/         # 系统发育树工具
│   └── utils/            # 数据IO、转换、可视化
├── tests/                 # 单元测试与集成测试 (41个测试)
├── docs/algorithm_notes/ # 算法文档 (8篇)
├── data/                  # 数据模拟工具
├── pyproject.toml         # Poetry包配置
├── Dockerfile             # Docker镜像定义
└── docker-compose.yml     # Docker Compose配置
```

## 模块概览

| 模块 | 描述 | 核心函数 |
|------|------|----------|
| `abundance` | 差异丰度分析 | `run_lefse`, `run_deseq2`, `run_ancombc` |
| `diversity` | Alpha/Beta多样性及统计 | `alpha_diversity`, `beta_diversity`, `permanova` |
| `function` | 基于16S的功能预测 | `run_picrust2`, `run_faprotax` |
| `network` | 微生物相关性网络 | `sparcc_network`, `spearman_network` |
| `phylogeny` | 系统发育树构建 | `build_tree`, `bootstrap_tree` |
| `utils` | 数据IO、转换、可视化 | `read_tsv`, `clr_transform`, `plot_pcoa` |

## API 示例

### LEfSe 分析

```python
from biomekit.abundance import run_lefse

results = run_lefse(
    abundance_df,
    metadata,
    group_column='group',
    alpha=0.05,
    lda_threshold=2.0
)
print(f"发现 {results['summary']['n_significant']} 个显著特征")
```

### 多样性分析

```python
from biomekit.diversity import alpha_diversity, beta_diversity, permanova

# Alpha多样性
alpha_div = alpha_diversity(abundance_df, metrics=['shannon', 'chao1'])

# Beta多样性
dist_matrix = beta_diversity(abundance_df, metric='braycurtis')

# 统计检验
result = permanova(dist_matrix, metadata, group_column='group')
print(f"PERMANOVA p值: {result['p_value']:.4f}")
```

### 网络分析

```python
from biomekit.network import sparcc_network

network = sparcc_network(abundance_df, threshold=0.3)
print(f"网络边数: {len(network)}")
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行并生成覆盖率报告
pytest tests/ -v --cov=biomekit --cov-report=html

# 运行特定模块测试
pytest tests/test_abundance/ -v
```

## 文档

算法文档位于 `docs/algorithm_notes/`:
- [LEfSe](docs/algorithm_notes/lefse.md)
- [DESeq2](docs/algorithm_notes/deseq2.md)
- [ANCOM-BC](docs/algorithm_notes/ancombc.md)
- [Alpha多样性](docs/algorithm_notes/alpha_diversity.md)
- [Beta多样性](docs/algorithm_notes/beta_diversity.md)
- [PERMANOVA](docs/algorithm_notes/permanova.md)
- [PICRUSt2](docs/algorithm_notes/picrust2.md)
- [FAPROTAX](docs/algorithm_notes/faprotax.md)

## 开发路线图

| 优先级 | 模块/功能 | 状态 | 描述 |
|:------:|:----------|:----:|:------|
| P0 | 算法模块库 | ✅ 完成 | 核心差异丰度、多样性、网络算法 |
| P1 | 预测模型集 | 🔜 计划中 | 疾病分类、疗效预测模型（ML/DL） |
| P2 | 多组学整合 | 🔜 计划中 | 16S + 代谢组 + 宏基因组融合分析 |
| P3 | Snakemake流程 | 🔜 计划中 | 基于Snakemake/Nextflow的生产级工作流 |
| P4 | Web仪表盘 | 🔜 计划中 | 交互式可视化仪表盘 |
| P5 | AutoML模块 | 🔜 计划中 | 微生物组数据自动超参数优化 |

### 状态图例
- ✅ 完成 - 可直接使用
- 🔜 计划中 - 列入开发计划
- 🔄 进行中 - 正在开发中

## 应用场景

本项目展示以下能力:

1. **算法实现**: 理解并实现已发表算法 (LEfSe, alpha/beta多样性)
2. **生物信息流程**: 构建模块化、可复现的分析工作流
3. **统计分析**: 应用适合微生物组数据的统计检验方法
4. **数据可视化**: 创建可发表质量的图表
5. **软件工程**: 打包、测试、文档最佳实践

## 贡献

欢迎贡献代码！请提交 Pull Request。

## 许可

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 作者

dalianmao000