# Issue #4: Checkpoint路径配置

**优先级**: 🟡 中  
**状态**: ⚠️ 待确认策略  
**影响**: 模型加载的便捷性

## 问题描述

### 当前配置（相对路径）

```python
# config.py 第178行
@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints"  # ⚠️ 相对路径
```

### 实际checkpoint位置

根据用户确认，checkpoint应该在：
```
src/Amphion/models/vc/my_publish_emo_rvq_bottleneck/checkpoints/
├── grouped_rvq_best.pt
└── entropy_model_best.pt
```

## 潜在问题

### 问题1: 相对路径的歧义
如果从不同目录运行脚本：
```bash
# 从项目根目录运行
cd /path/to/my_publish_emo_rvq_bottleneck
python train_rvq.py  
# ✅ checkpoint保存到: ./checkpoints/

# 从其他目录运行
cd /some/other/dir
python /path/to/my_publish_emo_rvq_bottleneck/train_rvq.py
# ❌ checkpoint保存到: /some/other/dir/checkpoints/ （错误位置）
```

### 问题2: 评估脚本的硬编码路径
```python
# run_evaluation.py 第29行
rvq_checkpoint_path = Path('checkpoints/grouped_rvq_best.pt')  # ⚠️ 相对路径
```

## 解决方案

### 方案A: 使用项目根目录的相对路径（推荐）
```python
# 在每个脚本开头
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)  # 切换到项目根目录

# 然后使用相对路径
checkpoint_path = 'checkpoints/grouped_rvq_best.pt'
```

### 方案B: 使用绝对路径
```python
# config.py
output_dir: str = "/data/.../my_publish_emo_rvq_bottleneck/checkpoints"
```

### 方案C: 基于__file__的动态路径
```python
# 各脚本中
CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'
```

## 建议

**发表版本**: 采用方案A
- 在各训练和评估脚本开头添加 `os.chdir(PROJECT_ROOT)`
- 保持配置中的相对路径
- 在README中说明"必须从项目根目录运行"

## 修复位置

需要修改：
1. `train_rvq.py` - 开头添加chdir
2. `train_entropy.py` - 开头添加chdir  
3. `run_evaluation.py` - 开头添加chdir
4. README.md - 添加使用说明

