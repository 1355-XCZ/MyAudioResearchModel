# Issue #6: 训练集/验证集划分说明

**优先级**: 🟢 信息性  
**状态**: ✅ 已确认  
**影响**: 了解实际训练数据量

## 数据划分情况

### 当前配置

```python
# config.py 第196行
@dataclass
class DataConfig:
    train_split: float = 0.9  # 90%训练，10%验证
    seed: int = 42            # 随机种子（可重复）
```

### 实际划分

```python
# data_loader.py 第304-312行
train_size = int(data_config.train_split * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(data_config.seed)
)
```

### 数据量

**100h中英文数据**：
- 总样本数：41,914个
- 训练集：37,723个（90%）
- 验证集：4,191个（10%）

**划分方式**：
- 随机划分（seed=42，可重复）
- 无分层采样（不保证语言/说话人均匀）

## 为什么需要验证集？

### 用途
1. **RVQ训练**：监控重建质量，防止过拟合
2. **熵模型训练**：监控bits/frame，选择最佳checkpoint
3. **超参数选择**：基于验证集性能调整

### 注意
- ✅ 验证集**不用于**最终评估
- ✅ 最终评估使用**独立的测试集**（IEMOCAP/RAVDESS/ESD）

## 是否使用全部100h？

### 实际训练数据量

**RVQ训练**：
- 实际使用：90h（37,723个样本）
- 验证监控：10h（4,191个样本）
- 码本学习：基于90h数据

**熵模型训练**：
- 实际使用：90h（37,723个样本）
- 验证监控：10h（4,191个样本）

### 结论

✅ **使用了全部100h数据，但分为训练集和验证集**
- 训练集：90h（用于学习）
- 验证集：10h（用于监控，防止过拟合）

这是标准的机器学习实践，**不是浪费数据**。

## 如果需要使用全部100h训练

可以修改配置：
```python
# config.py
train_split: float = 1.0  # 使用全部数据训练
```

但这样会失去验证集监控，**不推荐**（科学实验需要验证集）。

## 用户确认

根据用户说"由于研究目前已经进入了发表阶段，我们训练好的模型不需要在pipeline重新训练直接使用就好"：

- ✅ 已有训练好的checkpoint
- ✅ 不需要重新训练
- ✅ 直接加载checkpoint进行评估

**建议**: 保持当前train_split=0.9配置（标准实践）

