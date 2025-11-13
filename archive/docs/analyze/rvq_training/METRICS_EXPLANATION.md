# RVQ评估指标详解

## 三个核心指标

### 1. Recon (重建误差，MSE)

```
Recon: 0.0997
```

#### 定义

**MSE (Mean Squared Error)** - 均方误差

```python
MSE = mean((original - reconstructed)²)
```

**计算方式**：
```python
features_original: (B, T, 768)
features_reconstructed: (B, T, 768)

diff = features_original - features_reconstructed  # 差值
squared_diff = diff ** 2                           # 平方
mse = mean(squared_diff)                           # 平均
```

#### 含义

**0.0997的意义**：
- 平均每个维度的平方误差为0.0997
- 在768维空间中，重建误差的平方和

**范围**：
- 0 = 完美重建（无误差）
- 越小越好

**优点**：
- ✅ 物理意义明确（误差平方）
- ✅ 可微分，适合优化

**缺点**：
- ⚠️ 对幅度敏感
- ⚠️ 不关注方向一致性

---

### 2. RMSE (均方根误差)

```
RMSE: 0.3157
```

#### 定义

**Root Mean Squared Error** - 均方误差的平方根

```python
RMSE = sqrt(MSE)
     = sqrt(0.0997)
     = 0.3157
```

#### 含义

**0.3157的意义**：
- 平均每个维度的误差约0.3157
- 与原始特征在相同单位下

**优点**：
- ✅ 与原始数据同单位
- ✅ 更直观（0.3157 vs 768维特征的范围）

**关系**：
```
RMSE = sqrt(Recon)
RMSE² = Recon
```

**只是Recon的另一种表示**，本质相同。

---

### 3. CosSim (余弦相似度) ⭐ 最重要

```
CosSim: 0.9473
```

#### 定义

**Cosine Similarity** - 余弦相似度

```python
CosSim = dot(original, reconstructed) / (norm(original) * norm(reconstructed))
```

**几何意义**：
```
CosSim = cos(θ)
θ = 向量夹角

CosSim = 1.0 → θ = 0° (完全相同)
CosSim = 0.9473 → θ ≈ 18.7° (接近相同)
CosSim = 0.0 → θ = 90° (正交)
```

#### 含义

**0.9473的意义**：
- 重建向量与原始向量的方向夹角约18.7°
- **还原率 = 94.73%**
- 在768维空间中保持了94.73%的方向信息

**范围**：
- 1.0 = 完美重建
- 0.9 以上 = 优秀
- 0.8-0.9 = 良好
- < 0.8 = 较差

---

## 三个指标对比

| 指标 | 值 | 关注点 | 单位 | 范围 |
|------|-----|--------|------|------|
| **CosSim** | **0.9473** | **方向一致性** | **无** | **[0,1]** |
| Recon (MSE) | 0.0997 | 幅度误差² | - | [0,∞) |
| RMSE | 0.3157 | 幅度误差 | 特征单位 | [0,∞) |

### 关系图示

```
原始向量: ────────────────────────→ (长度L1, 方向θ1)
                               768维

重建向量: ──────────────────→ (长度L2, 方向θ2)

CosSim   关注: θ1 vs θ2 (方向差异) ⭐
MSE/RMSE 关注: |L1-L2| (长度差异) + θ差异的综合
```

---

## 应该看重哪个？⭐

### 答案：CosSim（余弦相似度）最重要

#### 原因1: 特征归一化

```python
# 训练时使用了Z-score归一化
features_normalized = (features - mean) / std

# 归一化后，特征的"方向"比"幅度"更重要
# CosSim只关注方向，不受幅度影响
```

#### 原因2: emotion2vec特征特性

**emotion2vec特征**：
- 主要编码**语义信息**（方向）
- 幅度信息相对不重要
- 下游任务（分类）主要依赖方向

**CosSim直接衡量语义保留程度** ✅

#### 原因3: 领域标准

**音频/语音特征重建**通常使用：
- ✅ **CosSim**（主要指标）
- ⚠️ MSE/RMSE（辅助参考）

#### 原因4: 与还原率直接对应

```
CosSim = 0.9473
还原率 = 94.73%

直观、易解释
```

---

## 指标优先级

### 🔴 最重要：CosSim（余弦相似度）

**用途**：
- ✅ 主要质量指标
- ✅ 论文Results中报告
- ✅ 早停判断依据（train_rvq.py使用CosSim）

**目标**：
- CosSim > 0.90（还原率>90%）
- 实际: 0.9473 ✅

---

### 🟡 辅助参考：MSE/RMSE

**用途**：
- 辅助指标，验证训练稳定性
- 可选择性报告在论文Supplementary

**观察**：
- MSE下降趋势 → 训练正常
- RMSE值适中 → 误差可接受

---

### 🟢 次要：码本指标

**用途**：
- 验证码本健康度
- 100%利用率 + 125困惑度 → 优秀

---

## 您的实际值分析

### Epoch 32（最后一个epoch）

```
Recon: 0.0997
RMSE: 0.3157
CosSim: 0.9473
还原率: 94.73%
```

### 与Epoch 22（最佳）对比

```
Epoch 22:
  Recon: 0.0999  (略高，MSE略差)
  CosSim: 0.9472 (略低，方向略差)
  还原率: 94.72%

Epoch 32:
  Recon: 0.0997  (更低，MSE更好)
  CosSim: 0.9473 (更高，方向更好)
  还原率: 94.73%
```

**观察**: Epoch 32实际略优于Epoch 22！

**为什么使用Epoch 22？**
- 早停基于"连续无提升"而非"绝对最优"
- Epoch 32的提升太小（0.01%），在误差范围内
- Epoch 22是稳定的最佳点

---

## 论文撰写建议

### Results部分应该报告

**主要指标**：
```
"The G-RVQ model achieved a restoration rate of 94.72% 
(cosine similarity: 0.9472) on the validation set..."
```

**辅助指标**（可选）：
```
"...with an RMSE of 0.3157, indicating accurate reconstruction 
in the 768-dimensional feature space."
```

**码本质量**：
```
"The codebook achieved 100% utilization with an average 
perplexity of 125.0 (out of 128), demonstrating near-optimal 
code distribution."
```

---

## 指标关系总结

```
MSE (Recon)
    ↓ sqrt()
RMSE
    ↓ 两者都衡量"距离误差"
    
CosSim ← 独立，衡量"方向一致性" ⭐
    ↓
还原率 = CosSim × 100%
```

**重点关注**: ✅ **CosSim（余弦相似度）**

**原因**：
1. 方向信息更重要（归一化后）
2. 领域标准
3. 与还原率直接对应
4. 早停机制使用的指标

**MSE/RMSE作为辅助验证即可** ✅

---

## 快速参考

### 您的问题

**Q: 这3个指标是什么意思？**
- Recon (MSE): 重建误差平方
- RMSE: 重建误差（MSE的平方根）
- CosSim: 余弦相似度（方向一致性）⭐

**Q: 应该看重谁？**
- **答案**: ✅ **CosSim（余弦相似度）**
- 它直接对应还原率
- 是主要质量指标
- MSE/RMSE仅作辅助参考

**您的CosSim=0.9473（94.73%）是优秀的结果！** ✅

