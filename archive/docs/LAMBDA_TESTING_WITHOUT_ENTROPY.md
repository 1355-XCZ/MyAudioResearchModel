# 提前测试λ值与码率关系

**问题**: 能不能不等熵模型训练完，就测试λ值对应的码率？  
**答案**: ✅ **可以！有两种方式**

---

## 方式1: 简化ECVQ测试（无需熵模型）

### 原理

**grouped_rvq.py支持简化ECVQ**:
```python
# 当entropy_model=None时
_, indices, _, _ = rvq_model(
    features,
    lambda_rate=torch.tensor(lambda_val),
    entropy_model=None,  # ✅ 不需要熵模型
    valid_mask=valid_mask
)

# 简化判决：只基于失真
J(k) = D(k) + λ·C  # C是常数（假设均匀分布）
```

**作用**:
- 可以测试不同λ产生的SKIP率
- 可以估算对应的码率范围
- **但不是最终的精确码率**

### 快速测试脚本

```python
# test_lambda_bitrate.py（新建文件）

import torch
from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from data_loader import create_dataloaders

config = get_default_config()
device = 'cuda'

# 加载RVQ
rvq_model = GroupedResidualVQ(config['grouped_rvq']).to(device)
ckpt = torch.load('checkpoints/grouped_rvq_best.pt')
rvq_model.load_state_dict(ckpt['model_state_dict'])
rvq_model.eval()

# 加载少量测试数据
test_config = replace(config['data'], max_samples=100)
train_loader, _ = create_dataloaders(test_config, config['training'], config['grouped_rvq'])

# 测试不同λ值
test_lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0]

for lam in test_lambdas:
    total_bits = 0
    total_frames = 0
    
    for batch in train_loader:
        features = batch['features'].to(device)
        lengths = batch['lengths'].to(device)
        
        with torch.no_grad():
            # 简化ECVQ（无熵模型）
            _, indices, _, stats = rvq_model(
                features,
                lambda_rate=torch.tensor(lam, device=device),
                entropy_model=None,  # ✅ 关键：无熵模型
                valid_mask=(torch.arange(features.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1))
            )
        
        # 统计SKIP
        skip_count = stats.get('total_skips', 0)
        total_count = stats.get('total_codes', 0)
        total_frames += lengths.sum().item()
    
    skip_rate = skip_count / total_count if total_count > 0 else 0
    approx_bitrate = 1344 * (1 - skip_rate)  # 粗略估算
    
    print(f"λ={lam:7.1f}: SKIP率={skip_rate:.1%}, 粗略码率≈{approx_bitrate:.0f} bpf")
```

**运行**:
```bash
python test_lambda_bitrate.py
```

**输出示例**:
```
λ=0.1:    SKIP率=5%,   粗略码率≈1277 bpf
λ=1.0:    SKIP率=25%,  粗略码率≈1008 bpf
λ=10.0:   SKIP率=60%,  粗略码率≈538 bpf
λ=100.0:  SKIP率=85%,  粗略码率≈202 bpf
λ=1000.0: SKIP率=95%,  粗略码率≈67 bpf
```

**作用**: ✅ **快速了解λ值范围，无需等熵模型**

---

## 方式2: 真实ECVQ测试（需要熵模型）

### 原理

**使用训练好的熵模型q(z)**:
```python
_, indices, _, _ = rvq_model(
    features,
    lambda_rate=torch.tensor(lambda_val),
    entropy_model=entropy_model,  # ✅ 使用训练好的熵模型
    valid_mask=valid_mask
)

# 真实ECVQ判决
J(k) = D(k) + λ·(-log₂ q(k|ctx))

# 计算真实码率
bits = entropy_model.compute_bits(indices, valid_mask)
bitrate = bits / (frames / 50)
```

**作用**:
- 精确测试λ值对应的码率
- 这是最终评估时使用的方法
- **需要熵模型训练完成**

---

## 对比

| 方式 | 需要熵模型 | 准确度 | 用途 |
|------|-----------|--------|------|
| **简化ECVQ** | ❌ 不需要 | 粗略估计 | **提前测试λ范围** ⭐ |
| **真实ECVQ** | ✅ 需要 | 精确 | 最终评估 |

---

## 您可以现在就做

### 创建测试脚本

我可以帮您创建 `test_lambda_range.py`:
```python
# 使用简化ECVQ测试当前25个λ值
# 输出每个λ的SKIP率和粗略码率
# 验证是否覆盖0-1344范围
```

**运行时间**: 约5-10分钟（只用100个样本）

**输出**:
```
λ值范围测试结果:
λ=0.01  → SKIP ~1%  → 码率 ~1330 bpf
λ=0.05  → SKIP ~3%  → 码率 ~1300 bpf
...
λ=2048  → SKIP ~96% → 码率 ~54 bpf

覆盖范围: 54-1330 bpf
建议: 如果要更低，增加更大的λ
```

**需要我帮您创建这个测试脚本吗？**

---

## 关于当前训练

**Job 18560114**: 已提交  
**配置**: batch=16, frames=128  
**状态**: batch=16应该绝对安全

**建议**: 
- 让它继续运行
- 同时可以测试λ值范围（互不影响）

---

**总结**: 
1. ✅ lambda_grid已移到config.py
2. ✅ 可以用简化ECVQ提前测试λ范围（无需等熵模型）
3. ✅ batch=16应该解决OOM

**需要我创建快速测试脚本吗？**
