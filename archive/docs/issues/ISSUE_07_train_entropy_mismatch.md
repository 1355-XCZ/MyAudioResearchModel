# Issue #7: train_entropy.py包含条件模型训练（与entropy_model.py不匹配）

**优先级**: 🔴 高  
**状态**: ⚠️ 需要修改  
**影响**: 运行train_entropy.py会报错

## 问题描述

`train_entropy.py` 是从 `emotion_infobottleneck` 完整复制的，包含：
1. ✅ 阶段2.1：训练无条件 q(z)
2. ❌ 阶段2.2：训练条件 q(z|y)（**会报错**）

但 `entropy_model.py` 已移除条件模型相关代码：
- ❌ ConditionalEntropyModel类不存在
- ❌ make_conditional_from_unconditional函数不存在

**运行会报错**：
```python
# train_entropy.py 第525行
from entropy_model import make_conditional_from_unconditional
# ImportError: cannot import name 'make_conditional_from_unconditional'
```

## 解决方案

### 方案A: 修改train_entropy.py，只保留q(z)训练（推荐）

删除或注释掉阶段2.2的所有代码（约100行）：

```python
# train_entropy.py 删除/注释第525-608行
# 即从"阶段2.2: 从 q(z) 派生条件熵模型"开始到文件末尾

# 修改后的main函数结尾：
def main():
    # ... 前面的代码不变 ...
    
    # 4. 训练无条件熵模型 q(z)
    entropy_model_q_z = create_entropy_model(
        entropy_model_config, 
        grouped_rvq_config
    )
    
    train_entropy_model(
        entropy_model_q_z,
        train_indices,
        train_labels,
        train_masks,
        training_config,
        entropy_model_config,
        model_name="q_z"
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 无条件熵模型 q(z) 训练完成!")
    logger.info("=" * 80)
    # ✅ 结束，不再训练q(z|y)
```

### 方案B: 添加条件判断跳过q(z|y)

```python
# train_entropy.py 在阶段2.2之前添加
TRAIN_CONDITIONAL = False  # 设置为False跳过q(z|y)训练

if TRAIN_CONDITIONAL:
    # 阶段2.2: 训练q(z|y)
    ...
else:
    logger.info("⏭️  跳过条件熵模型训练（发布版本只需要q(z)）")
```

## 修改后需要做的

### 1. 添加seed设置

在main函数开头添加：
```python
def main():
    """主函数：训练无条件熵模型 q(z)"""
    
    # 设置随机种子（确保可重现）
    import random
    import numpy as np
    
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 加载配置
    (grouped_rvq_config, entropy_model_config, ...) = get_config_from_env()
    
    # 设置种子
    set_seed(data_config.seed)  # ✅ 使用1344871
    logger.info(f"✅ 已设置随机种子: {data_config.seed}")
    
    # 后续训练逻辑...
```

### 2. 修改checkpoint保存名称

确保保存为评估脚本期望的名称：
```python
# train_entropy.py（修改train_entropy_model函数中的保存逻辑）
save_path = checkpoint_dir / f"{model_name}_best.pt"
# 对于q(z)：save_path = checkpoints/q_z_best.pt

# 但评估脚本期望：
# checkpoints/entropy_model_best.pt

# 需要统一命名
```

## 推荐修改

创建一个简化版的train_entropy.py（只训练q(z)），保存为：
```python
# 建议文件名：train_entropy_uncond_only.py
# 或直接修改 train_entropy.py
```

## 快速修复（临时）

如果急需运行，可以手动停止训练：
```python
# train_entropy.py 第527行后添加
logger.info("✅ q(z) 训练完成，跳过q(z|y)（发布版本）")
return  # 提前退出main函数
```

## 修复优先级

🔴 **高优先级** - 必须修复才能运行train_entropy.py

建议立即采用方案A：修改train_entropy.py，只保留q(z)训练部分。

