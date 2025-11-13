# 论文用表格和数据

## Table 1: RVQ模型配置

| Parameter | Value | Description |
|-----------|-------|-------------|
| Feature Dimension | 768 | emotion2vec特征维度 |
| Number of Groups | 12 | 分组数 |
| Group Dimension | 64 | 每组维度 |
| Layers per Group | 16 | 每组残差量化层数 |
| **Total Layers** | **192** | 总层数 (12×16) |
| Codebook Size | 128 | 每层码本大小 |
| **Theoretical Capacity** | **1344 bits/frame** | 理论最大容量 |
| Bitrate (@ 50Hz) | 67,200 bps | 最大码率 |
| SKIP Mechanism | Enabled | 启用层级跳过 |
| EMA Decay | 0.99 | 指数移动平均衰减率 |
| Commitment Weight | 0.25 | Commitment loss权重 |

---

## Table 2: 训练数据统计

| Dataset | Samples | Duration | Language | Split |
|---------|---------|----------|----------|-------|
| Training Set | 37,722 | ~90h | EN+ZH | 90% |
| Validation Set | 4,192 | ~10h | EN+ZH | 10% |
| **Total** | **41,914** | **~100h** | **Bilingual** | **100%** |

**Normalization**: Z-score normalization using 100h statistics  
**Random Seed**: 1344871 (for reproducibility)

---

## Table 3: 训练结果摘要

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Restoration Rate** | **94.72%** | >90% | ✅ Exceeded |
| Cosine Similarity | 0.9472 | >0.90 | ✅ Exceeded |
| Validation Loss (MSE) | 0.0999 | - | - |
| Codebook Utilization | 100.00% | >70% | ✅ Perfect |
| Perplexity | 125.0/128 | ~128 | ✅ Near-optimal |
| Best Epoch | 22 | - | - |
| Total Epochs Trained | 32 | max 100 | Early stopped |
| Training Time | 80 minutes | <72h | ✅ Efficient |

---

## Table 4: 早停机制效果

| Aspect | Configuration | Result |
|--------|--------------|--------|
| Early Stop Enabled | True | ✅ |
| Patience | 10 epochs | ✅ |
| Min Delta | 0.01% | ✅ |
| Trigger Epoch | 32 | ✅ |
| Best Epoch | 22 | ✅ |
| Epochs Saved | 68 | ✅ |
| Time Saved | ~170 minutes | ✅ |

**Effectiveness**: 早停机制节省了68%的计划训练时间

---

## Table 5: 逐层码本统计（采样）

### 早期层（学习粗糙特征）

| Layer | Utilization | Perplexity | SKIP Rate |
|-------|-------------|------------|-----------|
| 0-15 | 100.0% | 80-100 | 0.0% |
| 16-31 | 100.0% | 100-115 | 0.0% |

### 中期层（学习中等细节）

| Layer | Utilization | Perplexity | SKIP Rate |
|-------|-------------|------------|-----------|
| 32-95 | 100.0% | 115-120 | 0.0% |
| 96-127 | 100.0% | 120-125 | 0.0% |

### 后期层（学习细微残差）

| Layer | Utilization | Perplexity | SKIP Rate |
|-------|-------------|------------|-----------|
| 128-159 | 100.0% | 125-127.5 | 0.0% |
| 160-191 | 100.0% | 127.5-127.9 | 0.0% |

**观察**: 后期层困惑度更高（接近128），符合理论预期

---

## Figure 1: 训练曲线（建议绘制）

### 数据点

```python
epochs = [1,2,3,4,5,10,15,20,22,25,30,32]
restoration_rates = [94.41, 94.51, 94.57, 94.60, 94.62, 94.68, 94.70, 94.71, 94.72, 94.72, 94.73, 94.73]
```

### 绘图建议

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(epochs, restoration_rates, 'b-o', linewidth=2)
plt.axvline(x=22, color='r', linestyle='--', label='Best Epoch (22)')
plt.axhline(y=94.72, color='g', linestyle='--', label='Best Rate (94.72%)')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Restoration Rate (%)', fontsize=12)
plt.title('RVQ Training: Restoration Rate vs Epoch', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('restoration_rate_curve.png', dpi=300)
```

---

## Figure 2: 困惑度分布（建议绘制）

### 数据

192层的困惑度值（见training_log.err Epoch 32详细统计）

### 绘图建议

```python
# 按层绘制困惑度
plt.figure(figsize=(12, 6))
plt.bar(range(192), perplexity_values, alpha=0.7)
plt.axhline(y=128, color='r', linestyle='--', label='Ideal (128)')
plt.axhline(y=125, color='g', linestyle='--', label='Average (125.0)')
plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('Perplexity', fontsize=12)
plt.title('Codebook Perplexity Distribution (192 Layers)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('perplexity_distribution.png', dpi=300)
```

---

## 论文引用格式

### LaTeX表格示例

```latex
\begin{table}[ht]
\centering
\caption{Grouped RVQ Training Results}
\begin{tabular}{lcc}
\hline
Metric & Value & Target \\
\hline
Restoration Rate & 94.72\% & >90\% \\
Codebook Utilization & 100.00\% & >70\% \\
Perplexity & 125.0/128 & $\approx$128 \\
Best Epoch & 22 & - \\
\hline
\end{tabular}
\label{tab:rvq_results}
\end{table}
```

---

## 关键数值（直接引用）

**用于论文Results部分**：

- 还原率: **94.72%**
- 余弦相似度: **0.9472**
- 码本利用率: **100.00%**
- 困惑度: **125.0** (out of 128)
- 训练epochs: **22** (best) / **32** (total)
- 训练时间: **80分钟**
- 数据量: **100小时**（中英文双语）
- 随机种子: **1344871**（可重现）

---

**文件列表**：
- training_summary.csv - 训练指标（可导入Excel/Python）
- EXPERIMENTAL_REPORT.md - 完整实验报告
- PAPER_TABLES.md - 论文用表格（本文件）
- training_log.err - 完整训练日志
- training_log.out - 标准输出

