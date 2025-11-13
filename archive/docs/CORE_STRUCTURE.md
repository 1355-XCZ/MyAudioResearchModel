# 核心代码结构说明

本文档清晰说明项目的核心代码组织，帮助快速理解和维护。

## 📦 核心模型代码（必需）

### 1. 模型定义
```
config.py                    # 统一配置管理（所有超参数）
grouped_rvq.py              # 分组RVQ模型（12组×16层，ECVQ+SKIP）
entropy_model.py            # 无条件熵模型 q(z)
rate_controller.py          # 码率控制器（二分搜索）
```

### 2. 训练流程
```
train_rvq.py               # RVQ训练主脚本
train_entropy.py           # 熵模型训练主脚本
data_loader.py             # 100h训练数据加载器
bucket_sampler.py          # 长度分桶采样器
compute_normalization.py   # 计算归一化参数（已完成，可归档）
```

### 3. 数据集抽象层
```
datasets/
├── __init__.py
├── base_dataset.py        # 基类（含随机采样逻辑）
├── iemocap_dataset.py     # IEMOCAP数据集（4类情感）
├── ravdess_dataset.py     # RAVDESS数据集（7类情感）
└── esd_dataset.py         # ESD数据集（5类情感）
```

### 4. 评估模块
```
evaluation/
├── __init__.py
├── emotion_classifier.py  # emotion2vec分类器封装
├── method_rate_sweep.py   # 码率扫描方法（核心评估）
├── method_layer_sweep.py  # 层数扫描方法
├── analyzer.py            # 结果分析和可视化
└── eval_dataset.py        # 评估数据集封装
```

## 🚀 主要入口脚本

### 训练入口
```
run_train_rvq.slurm        # RVQ训练任务（~6小时）
run_train_entropy.slurm    # 熵模型训练任务（~30小时）
```

### 评估入口
```
run_evaluation_iemocap.py  # IEMOCAP评估（400样本）
run_evaluation_ravdess.py  # RAVDESS评估（700样本）
run_evaluation_esd.py      # ESD评估（500样本）

run_eval_iemocap.slurm     # 对应的slurm任务
run_eval_ravdess.slurm
run_eval_esd.slurm
```

### 特征提取
```
extract_evaluation_features.py  # 提取评估数据集的emotion2vec特征
run_extract_features.slurm      # 对应的slurm任务
```

## 🗑️ 可删除的文件（重复/废弃）

### 重复文件
```
extract_eval_features.py        # 与extract_evaluation_features.py重复
run_evaluation.py               # 旧版串行评估（已被分数据集并行版本替代）
run_evaluation.slurm
run_evaluation_test.py          # 测试版本（可删除）
run_evaluation_test.slurm
```

### 已撤回的优化代码（README明确说明）
```
evaluation/lambda_calibration.py    # λ标定表优化（收益<1%，已撤回）
evaluation/method_rate_sweep_batch.py  # 批处理优化（反而变慢，已撤回）
```

### 快速测试脚本（可归档）
```
run_evaluation_esd_quick.py         # 快速测试（开发时使用）
run_evaluation_esd_quick_v2.py
run_eval_esd_quick.slurm
run_eval_esd_quick_v2.slurm
```

### 临时测试脚本（可归档）
```
test_setup.py                   # 初始设置测试
test_classification_accuracy.py # 分类准确率测试
test_entropy_quick.py           # 熵模型快速测试
test_lambda_bitrate.py          # Lambda码率测试
test_lambda_range.py            # Lambda范围测试
run_test_entropy_quick.slurm
run_test_lambda.slurm
run_train_rvq_test.slurm
```

### 验证脚本（功能完成后可归档）
```
verify_extracted_features.py    # 验证提取的特征
verify_features_classification.py
verify_sampled_accuracy.py      # 验证采样准确性
run_verify_features.slurm
run_verify_sampled.slurm
```

### 数据生成脚本（一次性使用，可归档）
```
generate_iemocap_labels.py      # 生成IEMOCAP标签（已完成）
regenerate_ravdess_labels.py    # 重新生成RAVDESS标签（已完成）
run_compute_normalization.slurm # 计算归一化（已完成）
```

## 📂 数据和结果目录

### 模型检查点
```
checkpoints/
├── grouped_rvq_best.pt        # 最佳RVQ模型（必需）
├── entropy_model_best.pt      # 最佳熵模型（必需）
└── grouped_rvq_epoch*.pt      # 中间epoch检查点（可删除，保留best即可）
```

### 评估结果
```
evaluation_results/            # 正式评估结果（100样本/情感）
evaluation_quick_v2_results/   # 快速测试结果（20样本/情感，可归档）
evaluation_quick_results/      # 旧版快速测试（可删除）
evaluation_test_results/       # 测试结果（可删除）
```

### 缓存数据
```
indices_cache/                 # 熵模型训练的索引缓存（必需）
ev2_mean_std_100h_EN_ZH.npz   # 归一化参数（必需）
```

### 分析和文档
```
analyze/                       # 分析结果和图表（可归档）
issues/                        # 已解决的问题记录（可归档到docs/）
HANDOVER.md                    # 交接文档（可归档）
LAMBDA_TESTING_WITHOUT_ENTROPY.md  # 测试文档（可归档）
```

## 🎯 核心代码依赖关系

```
训练流程:
data_loader.py → train_rvq.py → grouped_rvq.py
                                      ↓
                               (生成indices_cache)
                                      ↓
                           train_entropy.py → entropy_model.py

评估流程:
datasets/*.py → run_evaluation_*.py → evaluation/method_*.py → evaluation/emotion_classifier.py
                                            ↓
                                    evaluation/analyzer.py
```

## 📋 推荐的整理操作

### 立即可删除（安全）
```bash
# 删除Python缓存
rm -rf **/__pycache__/

# 删除重复的中间epoch检查点（保留best）
rm checkpoints/grouped_rvq_epoch*.pt

# 删除旧的评估结果目录
rm -rf evaluation_quick_results/
rm -rf evaluation_test_results/
```

### 可归档（移到archive/目录）
```bash
mkdir -p archive/

# 归档已撤回的优化代码
mv evaluation/lambda_calibration.py archive/
mv evaluation/method_rate_sweep_batch.py archive/

# 归档测试和验证脚本
mv test_*.py archive/
mv verify_*.py archive/
mv run_test_*.slurm archive/
mv run_verify_*.slurm archive/

# 归档快速测试脚本
mv run_evaluation_*_quick*.py archive/
mv run_eval_*_quick*.slurm archive/

# 归档一次性数据生成脚本
mv generate_iemocap_labels.py archive/
mv regenerate_ravdess_labels.py archive/
mv run_compute_normalization.slurm archive/

# 归档issues和分析
mv issues/ archive/
mv analyze/ archive/
mv HANDOVER.md archive/
mv LAMBDA_TESTING_WITHOUT_ENTROPY.md archive/
```

### 可删除的重复文件
```bash
# 删除重复的特征提取脚本
rm extract_eval_features.py  # 使用extract_evaluation_features.py

# 删除旧版串行评估
rm run_evaluation.py run_evaluation.slurm
rm run_evaluation_test.py run_evaluation_test.slurm
```

## 📖 最小核心文件清单

如果只保留核心功能，最小文件集合为：

```
核心模型（6个文件）:
- config.py
- grouped_rvq.py
- entropy_model.py
- rate_controller.py
- bucket_sampler.py
- data_loader.py

训练脚本（2个文件）:
- train_rvq.py
- train_entropy.py

数据集（4个文件）:
- datasets/__init__.py
- datasets/base_dataset.py
- datasets/iemocap_dataset.py
- datasets/ravdess_dataset.py
- datasets/esd_dataset.py

评估模块（5个文件）:
- evaluation/__init__.py
- evaluation/emotion_classifier.py
- evaluation/method_rate_sweep.py
- evaluation/method_layer_sweep.py
- evaluation/analyzer.py
- evaluation/eval_dataset.py

评估脚本（3个文件）:
- run_evaluation_iemocap.py
- run_evaluation_ravdess.py
- run_evaluation_esd.py

特征提取（1个文件）:
- extract_evaluation_features.py

Slurm任务（7个文件）:
- run_train_rvq.slurm
- run_train_entropy.slurm
- run_extract_features.slurm
- run_eval_iemocap.slurm
- run_eval_ravdess.slurm
- run_eval_esd.slurm

配置和文档（3个文件）:
- requirements.txt
- README.md
- CORE_STRUCTURE.md（本文件）

总计: ~35个核心文件
```

---
最后更新: 2025-11-13

