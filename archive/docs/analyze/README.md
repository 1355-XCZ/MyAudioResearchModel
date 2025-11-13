# 实验分析文档目录

本目录包含所有训练实验的详细分析和论文材料。

---

## 目录结构

```
analyze/
├── README.md                    # 本文件
│
├── rvq_training/                # RVQ训练分析
│   ├── EXPERIMENTAL_REPORT.md   # 完整实验报告
│   ├── PAPER_TABLES.md          # 论文用表格
│   ├── training_summary.csv     # 训练指标（CSV格式）
│   ├── training_log.err         # 完整训练日志
│   └── training_log.out         # 标准输出
│
├── entropy_training/            # 熵模型训练分析（待添加）
│   └── （RVQ完成后训练）
│
├── evaluation/                  # 评估结果分析（待添加）
│   └── （训练完成后评估）
│
└── paper_materials/             # 论文材料汇总
    └── （后续汇总所有关键数据）
```

---

## 文件说明

### RVQ训练（已完成）

#### EXPERIMENTAL_REPORT.md
**用途**: 完整的实验报告，包含所有细节  
**内容**:
- 实验配置（模型、数据、训练参数）
- 训练过程记录
- 最终结果分析
- 讨论和结论

**适用**: 实验文档、技术报告

#### PAPER_TABLES.md
**用途**: 论文撰写用的表格和数据  
**内容**:
- LaTeX表格模板
- 关键数值（可直接引用）
- 图表绘制建议

**适用**: 论文Results部分

#### training_summary.csv
**用途**: 结构化数据，可用于绘图  
**格式**: CSV（可导入Excel/Python/R）  
**内容**: 每个epoch的还原率、困惑度等指标

#### training_log.err/out
**用途**: 原始训练日志  
**内容**: 完整的训练输出（包含所有192层统计）

---

## 快速索引

### 论文撰写需要的关键数据

#### Methods部分

- 模型配置: `EXPERIMENTAL_REPORT.md` 第1节
- 训练参数: `EXPERIMENTAL_REPORT.md` 第1.2节

#### Results部分

- 主要结果: `PAPER_TABLES.md` Table 3
- 训练曲线: `training_summary.csv` + Figure 1
- 码本质量: `PAPER_TABLES.md` Table 5

#### Discussion部分

- 结果讨论: `EXPERIMENTAL_REPORT.md` 第8节
- 对比分析: `PAPER_TABLES.md` Table 3

---

## 关键数值速查

| 指标 | 值 | 位置 |
|------|----|----|
| 还原率 | 94.72% | 所有文档 |
| 训练epochs | 22 (best) / 32 (total) | EXPERIMENTAL_REPORT.md |
| 码本利用率 | 100% | PAPER_TABLES.md Table 3 |
| 困惑度 | 125.0/128 | PAPER_TABLES.md Table 3 |
| 训练数据 | 100h中英文 | EXPERIMENTAL_REPORT.md 第1.3节 |

---

## 后续计划

### 熵模型训练（下一步）

**预计内容**:
- 训练配置记录
- 索引提取统计
- Loss曲线
- bits/frame分析

**时间**: RVQ完成后立即进行

### 评估结果

**预计内容**:
- 码率扫描结果
- 层数扫描结果
- 准确率/置信度曲线
- 混淆矩阵

**时间**: 所有训练完成后

---

## 使用指南

### 查看训练过程

```bash
# 查看所有epoch的还原率
cat rvq_training/training_summary.csv

# 查看特定epoch的详细日志
grep "Epoch 22" rvq_training/training_log.err -A 200
```

### 提取数据用于绘图

```python
import pandas as pd

# 读取训练数据
df = pd.read_csv('analyze/rvq_training/training_summary.csv')

# 绘制还原率曲线
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['restoration_rate'])
plt.xlabel('Epoch')
plt.ylabel('Restoration Rate (%)')
plt.savefig('restoration_curve.png')
```

---

**最后更新**: 2025-11-12  
**状态**: RVQ训练已完成并分析，熵模型和评估待进行

