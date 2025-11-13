# 项目结构总览

本文档提供项目的简化结构说明。

## 📁 当前主目录结构（已整理）

```
my_publish_emo_rvq_bottleneck/
│
├── 📦 核心模型代码
│   ├── config.py                    # 统一配置管理
│   ├── grouped_rvq.py              # 分组RVQ模型（12组×16层）
│   ├── entropy_model.py            # 无条件熵模型
│   ├── rate_controller.py          # 码率控制器
│   ├── bucket_sampler.py           # 长度分桶采样器
│   └── data_loader.py              # 训练数据加载器
│
├── 📂 数据集抽象层
│   └── datasets/
│       ├── __init__.py
│       ├── base_dataset.py         # 基类（含随机采样）
│       ├── iemocap_dataset.py      # IEMOCAP（4类情感）
│       ├── ravdess_dataset.py      # RAVDESS（7类情感）
│       └── esd_dataset.py          # ESD（5类情感）
│
├── 📊 评估模块
│   └── evaluation/
│       ├── __init__.py
│       ├── emotion_classifier.py   # emotion2vec分类器
│       ├── method_rate_sweep.py    # 码率扫描评估
│       ├── method_layer_sweep.py   # 层数扫描评估
│       ├── analyzer.py             # 结果分析和可视化
│       └── eval_dataset.py         # 评估数据集封装
│
├── 🚀 训练脚本
│   ├── train_rvq.py                # RVQ训练
│   ├── train_entropy.py            # 熵模型训练
│   ├── run_train_rvq.slurm         # RVQ训练任务
│   └── run_train_entropy.slurm     # 熵模型训练任务
│
├── 🔬 评估脚本
│   ├── extract_evaluation_features.py  # 特征提取
│   ├── run_extract_features.slurm      # 特征提取任务
│   ├── run_evaluation_iemocap.py       # IEMOCAP评估
│   ├── run_evaluation_ravdess.py       # RAVDESS评估
│   ├── run_evaluation_esd.py           # ESD评估
│   ├── run_eval_iemocap.slurm          # IEMOCAP评估任务
│   ├── run_eval_ravdess.slurm          # RAVDESS评估任务
│   └── run_eval_esd.slurm              # ESD评估任务
│
├── 💾 模型和数据
│   ├── checkpoints/
│   │   ├── grouped_rvq_best.pt         # 最佳RVQ模型
│   │   ├── entropy_model_best.pt       # 最佳熵模型
│   │   └── logs/                       # 训练日志
│   ├── indices_cache/                  # 熵模型训练缓存
│   └── ev2_mean_std_100h_EN_ZH.npz    # 归一化参数
│
├── 📈 评估结果
│   └── evaluation_results/             # 正式评估结果
│       ├── rate_sweep/                 # 码率扫描结果
│       ├── layer_sweep/                # 层数扫描结果
│       └── analysis/                   # 分析图表
│
├── 🗄️ 归档文件（已整理）
│   └── archive/
│       ├── deprecated/                 # 已废弃的优化代码
│       ├── tests/                      # 临时测试脚本
│       ├── verification/               # 验证脚本
│       ├── quick_tests/                # 快速测试脚本和结果
│       ├── data_preparation/           # 数据生成脚本
│       ├── old_results/                # 旧评估结果
│       └── docs/                       # 历史文档和分析
│
└── 📖 文档
    ├── README.md                       # 项目主文档
    ├── PROJECT_STRUCTURE.md            # 本文件（项目结构）
    ├── CORE_STRUCTURE.md               # 核心代码详细说明
    ├── DEPRECATED.md                   # 废弃文件说明
    └── requirements.txt                # 依赖包列表
```

## 🎯 核心文件说明

### 模型定义（6个文件）
- `config.py` - 所有配置和超参数的统一管理
- `grouped_rvq.py` - 12组×16层的分组RVQ模型，支持ECVQ和SKIP
- `entropy_model.py` - 无条件自回归熵模型 q(z)
- `rate_controller.py` - 基于二分搜索的精确码率控制
- `bucket_sampler.py` - 按序列长度分桶的采样器
- `data_loader.py` - 100h训练数据的加载器

### 数据集（4个文件）
- `datasets/base_dataset.py` - 基类，提供随机采样功能
- `datasets/iemocap_dataset.py` - IEMOCAP数据集（4类情感）
- `datasets/ravdess_dataset.py` - RAVDESS数据集（7类情感）
- `datasets/esd_dataset.py` - ESD数据集（5类情感，中英文）

### 评估模块（5个文件）
- `evaluation/emotion_classifier.py` - emotion2vec分类器封装
- `evaluation/method_rate_sweep.py` - 码率扫描方法（核心）
- `evaluation/method_layer_sweep.py` - 层数扫描方法
- `evaluation/analyzer.py` - 结果分析和可视化
- `evaluation/eval_dataset.py` - 评估数据集封装

## 🚀 快速开始

### 1. 训练流程
```bash
# 步骤1: 训练RVQ（~6小时）
sbatch run_train_rvq.slurm

# 步骤2: 训练熵模型（~30小时）
sbatch run_train_entropy.slurm

# 步骤3: 提取评估特征（~1小时）
sbatch run_extract_features.slurm
```

### 2. 评估流程（并行）
```bash
# 3个数据集并行评估
sbatch run_eval_iemocap.slurm    # ~2小时
sbatch run_eval_ravdess.slurm    # ~3小时
sbatch run_eval_esd.slurm        # ~2.5小时
```

### 3. 查看结果
```bash
# 评估结果
ls evaluation_results/rate_sweep/
ls evaluation_results/layer_sweep/
ls evaluation_results/analysis/

# 分析图表
# - {dataset}_rate_vs_accuracy.png
# - {dataset}_layer_vs_accuracy.png
```

## 📦 归档目录说明

### archive/ 目录包含：

- **deprecated/** - 已撤回的优化代码（lambda_calibration, method_rate_sweep_batch）
- **tests/** - 临时测试脚本（test_*.py）
- **verification/** - 验证脚本（verify_*.py）
- **quick_tests/** - 快速测试脚本和结果
- **data_preparation/** - 一次性数据生成脚本
- **old_results/** - 旧版评估结果
- **docs/** - 历史文档（issues, analyze, HANDOVER.md等）

这些文件已归档保留，不影响主目录的清晰度，但可以随时查阅。

## 📊 项目统计

- **核心Python文件**: 17个
- **Slurm任务脚本**: 7个
- **文档文件**: 4个
- **总计主目录文件**: ~30个（整理前 >60个）

## 🔗 相关文档

- **详细核心代码说明**: 查看 `CORE_STRUCTURE.md`
- **废弃文件清单**: 查看 `DEPRECATED.md`
- **项目详细说明**: 查看 `README.md`

---
最后更新: 2025-11-13
整理状态: ✅ 已完成

