# RVQ配置决策记录

记录所有关键参数的选择理由和调整过程

---

## 参数演变历史

### num_fine_layers（每组层数）

| 版本 | 值 | 理由 | 结果 |
|------|-----|------|------|
| 初始 | 3 | 默认配置，快速测试 | 未测试 |
| 测试 | 8 | 提升容量，测试流程 | 84.32% (3 epochs) |
| 尝试 | 12 | 进一步提升 | 已取消 |
| **最终** | **16** | **追求高质量** | **94.72%** ✅ |

**决策**: 16层提供1344 bpf容量，达到94.72%还原率

---

### batch_size

| 环境 | 值 | 理由 |
|------|-----|------|
| H100 | 64 | 大显存，可用较大batch |
| **L40S** | **64** | **测试验证未OOM，保持64** |

**决策**: 64 batch size在L40S上安全运行

---

### num_epochs

| 阶段 | 值 | 理由 |
|------|-----|------|
| 初始 | 15 | 标准配置 |
| 调整 | 100 | 用户要求充分训练 |
| **实际** | **22** | **早停机制自动停止** |

**决策**: 设置100上限+早停（patience=10），实际训练32 epochs

---

### 早停机制

| 参数 | 值 | 效果 |
|------|-----|------|
| enable | True | 启用自动停止 |
| patience | 10 | 容忍10个epoch |
| min_delta | 0.0001 | 0.01%还原率阈值 |

**效果**: 
- 在Epoch 32触发（连续10个epoch提升<0.01%）
- 使用Epoch 22的最佳模型
- 节省68个epoch（约170分钟）

---

### 随机种子

| 参数 | 值 | 理由 |
|------|-----|------|
| 初始 | 42 | 默认值 |
| **最终** | **1344871** | **用户指定，统一随机性** |

**决策**: 所有实验使用统一seed=1344871确保可重现

---

## 关键决策总结

### 1. 为什么选择16层？

**理由**：
- 3层容量不足（252 bpf）
- 8层测试显示潜力（84% @ 3 epochs）
- 12层可能不够
- **16层提供1344 bpf容量，足够高质量重建**

**结果**: 94.72%还原率，达到预期

---

### 2. 为什么使用EMA模式？

**配置**: decay=0.99, commitment_weight=0.25

**理由**：
- EMA自动更新码本，无需优化器
- 更稳定，不易过拟合
- vector-quantize-pytorch库推荐方式

**结果**: 训练稳定，码本质量优秀

---

### 3. 为什么早停patience=10？

**理由**：
- 验证集有自然波动
- patience=10给模型"恢复"机会
- 区分"暂时波动"和"真正收敛"

**结果**: 
- Epoch 22达到最佳
- Epoch 23-32波动但无提升
- patience=10正确识别收敛

---

### 4. 为什么使用100h中英文数据？

**理由**：
- 更大数据量支持更多层
- 中英文双语提高泛化
- 与用户现有数据匹配

**结果**: 100%码本利用率，无过拟合

---

## 未达到98-99%的原因分析

### 预期vs实际

- 预期: 98-99%
- 实际: 94.72%
- 差距: 3.3-4.3%

### 可能原因

1. **容量与数据不匹配**
   - 16层提供高容量（1344 bpf）
   - 但100h数据可能不足以充分学习192层的所有码本

2. **EMA保守性**
   - decay=0.99每次只更新1%
   - 32 epochs可能还不够充分

3. **特征复杂度**
   - emotion2vec特征可能包含难以量化的信息
   - 某些维度可能天然难以精确重建

4. **配置权衡**
   - commitment_weight=0.25可能偏小
   - 可以尝试0.5-1.0

### 进一步提升方向

| 方法 | 预期效果 | 代价 |
|------|---------|------|
| 增加epochs（32→100） | +0.5-1% | 训练时间×3 |
| 增加层数（16→20） | +1-2% | 训练时间+25% |
| 增加码本（128→256） | +0.5-1% | 训练时间+15% |
| 调整commitment_weight | +0.5-1.5% | 需要重新训练 |

**建议**: 94.72%已足够发表，无需进一步优化

---

## 配置文件存档

### config.py (Epoch 22时的配置)

```python
@dataclass
class GroupedRVQConfig:
    feature_dim: int = 768
    num_groups: int = 12
    group_dim: int = 64
    num_fine_layers: int = 16          # ✅ 最终选择
    fine_codebook_size: int = 128
    enable_skip: bool = True
    decay: float = 0.99
    commitment_weight: float = 0.25
    kmeans_init: bool = True
    kmeans_iters: int = 10
    threshold_ema_dead_code: float = 2.0
    eval_interval: int = 1
    save_interval: int = 5
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001

@dataclass
class DataConfig:
    train_data_dir: str = "emilia_vevo_training_50h"
    mean_std_path: str = "ev2_mean_std_100h_EN_ZH.npz"
    supported_languages: List[str] = ['EN', 'ZH']
    train_split: float = 0.9
    seed: int = 1344871              # ✅ 统一种子

@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 100             # ✅ 上限（实际早停32）
    # ... 其他标准参数
```

---

**记录日期**: 2025-11-12  
**实验状态**: ✅ 完成  
**用途**: 论文撰写参考

