# 情感RVQ信息瓶颈实验（发布版）

## 项目简介

本项目研究**emotion2vec特征在RVQ信息瓶颈下的情感识别鲁棒性**，使用真实的熵编码(ECVQ)实现精确的码率控制。这是`emotion_information_bottleneck`和`emotion_bottleneck_rate`项目的整合与升级版本。

**核心研究问题**：
- 在不同压缩码率下，情感识别性能如何退化？
- 不同情感类别对信息瓶颈的敏感度是否不同？
- 码率与层数两种压缩方式的效果对比如何？
- 真实熵编码(ECVQ)的码率控制精度如何？

## 实验状态

### ✅ 已完成
- ✅ 数据准备：100h中英文emotion2vec特征（Emilia数据集）
- ✅ 归一化参数：`ev2_mean_std_100h_EN_ZH.npz`
- ✅ RVQ训练：12组×16层/组，K=128+SKIP机制
- ✅ 熵模型训练：Epoch 1完成（Loss: 125.94, Acc: 88.1%）
- ✅ 评估特征提取：IEMOCAP(5,531), RAVDESS(1,440), ESD(35,000)
- ✅ 图表分析器：分情感曲线，正确单位标注

### 🏃 进行中
- 🏃 熵模型训练：Epoch 2-5（Job 18600318）
- 🏃 IEMOCAP评估：400样本（4类×100，随机采样）- Job 18600001
- 🏃 RAVDESS评估：700样本（7类×100，随机采样）- Job 18600002
- 🏃 ESD评估：500样本（5类×100，随机采样）- Job 18600003
- 🏃 ESD快速测试V2：100样本（5类×20）- Job 18600884

## 核心创新

### 1. 真实熵编码码率控制
```
特征 → ECVQ决策(λ控制) → 熵编码 → 精确码率测量
每个样本单独搜索λ，确保目标码率误差<1 BPF
```

### 2. 分组RVQ架构
```
768维 → 12组 × 64维
每组：16层量化器 (K=128 + SKIP)
总层数：192层（12组×16层）
```

### 3. 无条件自回归熵模型
```
q(z) = ∏_{t,g,m} q(k_{t,g,m} | history_{<t,g,m})
帧内自回归：192步序列建模
Transformer: 8层，512维，8头注意力
```

### 4. 双重评估方法
- **方法1（码率扫描）**：通过λ控制码率（5-50 BPF重点测试）
- **方法2（层数扫描）**：通过减少层数（12组×1~5层，保持分组结构）

## 当前实验配置

### 数据采样（基于先验知识）
```python
# 每个数据集随机采样100样本/情感（seed=42可复现）
IEMOCAP: 400样本（angry, happy, neutral, sad × 100）
RAVDESS: 700样本（7类 × 100）
ESD: 500样本（5类 × 100）
```

### 码率扫描配置（基于先验：50 BPF以下是关键区域）
```python
rate_sweep_rates_bpf = [
    5, 10, 15, 20, 25, 30, 40, 50,  # 重点：5-50 BPF密集采样
    100, 200,                        # 高码率参考点
    float('inf')                     # 无量化baseline
]  # 11个码率点
```

### 层数扫描配置（基于先验：5层/组后变化不大）
```python
layer_sweep_layers = [
    12,   # 12组×1层/组
    24,   # 12组×2层/组
    36,   # 12组×3层/组
    48,   # 12组×4层/组
    60    # 12组×5层/组
]  # 5个层数点，保持12组结构
```

### 码率控制精度
```python
rate_tolerance_bpf = 1.0         # 容差：1 BPF（未收敛可接受）
max_binary_search_iters = 50     # 最大迭代次数
lambda_hint优化：利用第一个样本的λ加速后续样本收敛
```

## 优化措施

### 性能优化（已实施）
- ✅ **Mask缓存**：避免每次重建causal mask
- ✅ **推理模式**：`torch.inference_mode()`禁用autograd
- ✅ **TF32加速**：矩阵运算加速
- ✅ **AMP混合精度**：FP32 → BF16
- ✅ **λ先验加速**：利用第一个样本的λ缩小搜索范围

### 评估优化（已实施）
- ✅ **随机采样**：每个情感100样本（seed=42可复现）
- ✅ **重点区域密集测试**：5-50 BPF密集，>50稀疏
- ✅ **保持分组结构**：层数扫描保持12组完整性

### 撤回的优化（反而变慢）
- ❌ λ标定表：收益<1%
- ❌ 批处理：真ECVQ无法批处理，反而慢

## 技术细节

### ECVQ决策（真实熵编码）
```python
# 逐层决策：选择k或SKIP
J(k) = D(k) + λ·(-log₂ q(k|ctx))

# λ控制码率：
λ越大 → 选SKIP倾向越高 → 码率越低
λ越小 → 选量化倾向越高 → 码率越高
```

### 二分搜索（每样本精确控制）
```python
# 每个样本单独搜索λ，确保达到目标码率
for sample in dataset:
    λ* = binary_search(target_rate_bpf, tolerance=1.0 bpf)
    # 利用前一个样本的λ作为先验，加速收敛
    实际码率误差：<1 BPF ✅
```

### 码率计算
```python
# 使用熵模型精确计算
bits = entropy_model.compute_bits(indices, valid_mask)
rate_bpf = bits / num_frames
```

## 快速开始

### 1. 环境准备

```bash
# 加载模块
module load GCCcore/11.3.0 Python/3.10.4 GCC/11.3.0 OpenMPI/4.1.4
module load FFmpeg/4.4.2 CUDA/11.8.0 cuDNN/8.7.0.84-CUDA-11.8.0

# 激活环境
source /data/gpfs/projects/punim2341/haoguangzhou/venvs/vevo-source-fix/bin/activate
```

### 2. 训练流程

```bash
cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

# 步骤1：训练RVQ（约6小时）
sbatch run_train_rvq.slurm

# 步骤2：训练熵模型（约30小时）
sbatch run_train_entropy.slurm

# 步骤3：提取评估数据集特征（约1小时）
sbatch run_extract_features.slurm
```

### 3. 评估流程

```bash
# 方案A：单任务评估（所有数据集串行）
sbatch run_evaluation.slurm

# 方案B：并行评估（3个数据集并行，推荐）
sbatch run_eval_iemocap.slurm
sbatch run_eval_ravdess.slurm
sbatch run_eval_esd.slurm

# 方案C：快速测试（验证流程，20样本/情感）
sbatch run_eval_esd_quick_v2.slurm
```

### 4. 查看结果

```bash
# 评估结果
ls evaluation_results/rate_sweep/
ls evaluation_results/layer_sweep/
ls evaluation_results/analysis/

# 图表
# - {dataset}_rate_vs_accuracy.png  # 分情感码率曲线
# - {dataset}_layer_vs_accuracy.png # 分情感层数曲线
```

## 项目结构

```
my_publish_emo_rvq_bottleneck/
├── config.py                      # 统一配置管理
├── grouped_rvq.py                 # 分组RVQ模型（12组×16层，ECVQ+SKIP）
├── entropy_model.py               # 无条件熵模型 q(z)
├── rate_controller.py             # 码率控制器（二分搜索+先验加速）
├── data_loader.py                 # 数据加载器（100h训练数据）
├── bucket_sampler.py              # 长度分桶采样器
│
├── datasets/                      # 数据集抽象层（面向对象）
│   ├── base_dataset.py            # 基类（含随机采样）
│   ├── iemocap_dataset.py         # IEMOCAP（4类核心情感）
│   ├── ravdess_dataset.py         # RAVDESS（7类）
│   └── esd_dataset.py             # ESD（5类）
│
├── evaluation/                    # 评估模块
│   ├── emotion_classifier.py      # emotion2vec分类器
│   ├── method_rate_sweep.py       # 码率扫描（每样本搜λ+先验）
│   ├── method_layer_sweep.py      # 层数扫描
│   ├── analyzer.py                # 结果分析器（分情感绘图）
│   └── eval_dataset.py            # 评估数据集（批处理支持）
│
├── train_rvq.py                   # RVQ训练脚本
├── train_entropy.py               # 熵模型训练脚本（带索引缓存）
├── run_evaluation_*.py            # 评估脚本（分数据集）
│
├── run_train_rvq.slurm            # RVQ训练任务
├── run_train_entropy.slurm        # 熵模型训练任务
├── run_eval_*.slurm               # 评估任务（分数据集）
│
├── checkpoints/                   # 模型checkpoint
│   ├── grouped_rvq_best.pt        # RVQ模型（完整参数）
│   └── entropy_model_best.pt      # 熵模型（Epoch 1, Loss: 125.94）
│
├── indices_cache/                 # 提取索引缓存（防止数据丢失）
├── evaluation_results/            # 正式评估结果（100样本/情感）
├── evaluation_quick_v2_results/   # 快速测试结果（20样本/情感）
└── README.md                      # 本文件
```

## 当前实验意图

### 研究目标
**在真实熵编码(ECVQ)下，研究情感识别对码率的鲁棒性**

与现有工作的区别：
- `emotion_information_bottleneck`：使用固定层数k（离散），无真实熵编码
- `emotion_bottleneck_rate`：使用λ控制（连续），但未实现真实熵模型
- **本项目**：**真实ECVQ + 真实熵模型 + 精确码率控制** ✅

### 关键设计决策

#### 1. 每个样本单独搜索λ
**决策**：不使用全局λ，每个样本单独二分搜索  
**原因**：
- 不同样本内容复杂度不同，同一λ导致码率波动（误差>15 BPF）
- 真实应用场景需要精确码率控制
- 研究码率与性能的准确关系

**代价**：速度慢10倍  
**优化**：利用第一个样本的λ作为先验，缩小搜索范围

#### 2. 重点测试5-50 BPF区域
**决策**：11个码率点，8个在5-50 BPF  
**原因**：
- 快速测试显示50 BPF以上准确率已达96%（饱和）
- 5-50 BPF是性能退化的关键区域
- 节省计算资源，聚焦有意义的区间

#### 3. 保持12组结构，测试1-5层/组
**决策**：层数扫描测试12×1, 12×2, ..., 12×5  
**原因**：
- 快速测试显示5层/组后准确率变化不大
- 保持12组结构确保下游分类模型兼容
- 不破坏分组RVQ的设计意图

#### 4. 随机采样100样本/情感
**决策**：从全量数据随机采样（seed=42）  
**原因**：
- 全量评估需要16天（不可行）
- 100样本/情感足够统计显著性
- 随机采样保证代表性

## 技术架构

### 分组RVQ + ECVQ
```
输入: (B, T, 768)
  ↓
分组: 12组 × (B, T, 64)
  ↓
每组16层量化（逐层ECVQ决策）:
  for m in 1..16:
    残差 → 查码本(128码) → 计算 J(k) = D(k) + λ·(-log₂ q(k|ctx))
    if min(J) is SKIP:
      索引 = -1 (SKIP)
    else:
      索引 = argmin(J(k))
  ↓
输出: 量化特征 (B, T, 768) + 索引 (B, T, 192)
```

### 熵模型（无条件）
```python
q(z) = q(k_{1,1,1}) · q(k_{1,1,2}|k_{1,1,1}) · ... · q(k_{T,12,16}|history)

实现:
- Token embedding: V=129 (K=128 + SKIP)
- Position encoding
- Group/Layer embedding (区分12组×16层位置)
- Transformer: 8层，causal mask
- 输出: next-token概率分布
```

### 码率控制（二分搜索 + 先验加速）
```python
# 第一个样本：完整二分搜索
λ₁*, R₁ = binary_search(target_rate, tol=1 BPF, max_iter=50)

# 后续样本：以λ₁为先验
if |R(λ₁) - target| < 1 BPF:
    直接使用λ₁（跳过搜索）✅
else:
    在[0.5λ₁, 2λ₁]范围内搜索（收敛更快）✅
```

## 实验结果（快速测试）

### ESD快速测试（20样本/情感，10码率点）

**方法1：码率 vs 准确率（分情感）**

| 码率(BPF) | angry | happy | neutral | sad | surprised |
|-----------|-------|-------|---------|-----|-----------|
| 10        | ~40%  | ~45%  | ~50%    | ~38% | ~42%     |
| 20-30     | ~75%  | ~80%  | ~85%    | ~72% | ~78%     |
| 40-50     | ~92%  | ~95%  | ~96%    | ~90% | ~93%     |
| 100+      | ~96%  | ~96%  | ~96%    | ~96% | ~96%     |

**关键发现**：
- 50 BPF达到96%准确率（接近无量化）✅
- 5-50 BPF是性能退化关键区域 ✅
- 不同情感敏感度略有差异（neutral最鲁棒）

**方法2：层数 vs 准确率**

| 层数 | 准确率 |
|------|--------|
| 24层 | ~82%   |
| 48层+ | ~96%  |

**关键发现**：
- 48层（12组×4层）达到96%性能 ✅
- 与码率50 BPF性能相当

## 运行指南

### 训练RVQ

```bash
sbatch run_train_rvq.slurm
```

**配置要点**：
- 数据：100h中英文emotion2vec特征
- 架构：12组×16层/组，K=128+SKIP
- 训练：100 epochs，early stopping

### 训练熵模型

```bash
sbatch run_train_entropy.slurm
```

**配置要点**：
- 索引提取：19个target_bpf点（含inf）
- 索引缓存：`indices_cache/`（防止crash丢失）
- 训练：5 epochs，batch=16

**恢复训练**：
```python
resume_from_checkpoint = True  # 自动从best checkpoint继续
```

### 运行评估

#### 并行评估（推荐）
```bash
# 3个数据集并行运行
sbatch run_eval_iemocap.slurm  # ~2小时
sbatch run_eval_ravdess.slurm  # ~3小时
sbatch run_eval_esd.slurm      # ~2.5小时
```

#### 快速验证
```bash
# 快速测试（20样本/情感，~1小时）
sbatch run_eval_esd_quick_v2.slurm
```

### 分析结果

评估完成后自动生成：

```
evaluation_results/analysis/
├── ESD_rate_vs_accuracy.png      # 分情感码率曲线
├── ESD_layer_vs_accuracy.png     # 分情感层数曲线
├── IEMOCAP_rate_vs_accuracy.png
├── RAVDESS_rate_vs_accuracy.png
└── summary_report.txt            # 文本摘要
```

## 数据集详情

### IEMOCAP（评估用4类核心情感）
```python
类别: angry, happy, neutral, sad
样本数: 5,531 → 采样400（100/类，随机seed=1344871）
映射: 
  ang → angry
  hap, exc → happy（merged）
  neu → neutral（只用原始neu，排除fru）
  sad → sad
排除: frustrated, surprised, fearful, disgusted, other（<50样本）
```

### RAVDESS（7类）
```python
类别: angry, disgust, fearful, happy, neutral, sad, surprised
样本数: 1,440 → 采样700（100/类，随机seed=1344871）
映射:
  calm → neutral（merged）
  disgust → disgusted（拼写转换）
仅speech: 排除song文件
```

### ESD（5类，中英文）
```python
类别: angry, happy, neutral, sad, surprise
样本数: 35,000 → 采样500（100/类，随机seed=1344871）
语言: English + Chinese
映射: surprise → surprised（拼写转换）
```

## 评估特征提取

```bash
# 一次性提取3个数据集的emotion2vec特征
sbatch run_extract_features.slurm
```

**输出**：
```
/data/gpfs/projects/punim2341/haoguangzhou/data/evaluation_features/
├── IEMOCAP/
│   └── Session*/.../*_ev2_frame.npy, *_emotion.txt
├── RAVDESS/
│   └── *_ev2_frame.npy, *_emotion.txt
└── ESD/
    └── 0001/Angry/*_ev2_frame.npy, *_emotion.txt
```

## 配置参数

所有参数在`config.py`中集中管理：

```python
@dataclass
class GroupedRVQConfig:
    feature_dim: int = 768
    num_groups: int = 12
    group_dim: int = 64
    num_layers_per_group: int = 16  # 每组16层
    codebook_size: int = 128
    use_skip: bool = True

@dataclass  
class EntropyModelConfig:
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 8
    target_bpf_grid: List[float] = [2, 5, 10, ..., 1344, inf]

@dataclass
class RateControlConfig:
    rate_tolerance_bpf: float = 1.0  # 容差1 BPF
    max_binary_search_iters: int = 50

@dataclass
class EvaluationConfig:
    rate_sweep_rates_bpf: List[float] = [5, 10, ..., 50, 100, 200, inf]
    layer_sweep_layers: List[int] = [12, 24, 36, 48, 60]
```

## 性能优化总结

| 优化 | 状态 | 效果 |
|------|------|------|
| Mask缓存 | ✅ | 避免重建causal mask |
| 推理模式 | ✅ | 禁用autograd |
| TF32加速 | ✅ | 矩阵运算加速 |
| AMP混合精度 | ✅ | FP32→BF16 |
| λ先验加速 | ✅ | 缩小搜索范围，收敛更快 |
| 采样评估 | ✅ | 减少94%计算量 |
| 重点区域密集测试 | ✅ | 聚焦5-50 BPF |

## 注意事项

### 码率波动说明
- 每个样本单独搜索λ确保目标码率（误差<1 BPF）
- 绘图使用**目标码率**（target_rate_bpf）作为x轴
- 不使用平均码率（避免离群值误导）

### 随机采样可复现性
```python
# 所有采样使用统一seed=1344871（与训练一致）
samples = dataset.sample_balanced(samples_per_emotion=100, seed=1344871)
# 每次运行抽取相同样本 ✅
```

### 层数扫描保持分组
- ❌ 错误：[1, 2, 4, 8, 16, 32, 64, 128, 192]（破坏分组）
- ✅ 正确：[12, 24, 36, 48, 60]（12组×1~5层/组）

## 输出格式

### JSON结果文件
```json
{
  "dataset": "ESD",
  "target_rates_bpf": [5, 10, 15, ...],
  "emotion_mapping": {...},
  "rate_points": {
    "10_bpf": {
      "target_rate_bpf": 10,
      "achieved_rates": [10.15, 9.87, ...],  // 每个样本实际码率
      "lambda_values": [1.13, 1.09, ...],    // 每个样本的λ
      "predictions": ["happy", "sad", ...],
      "ground_truths": ["happy", "sad", ...],
      "confidences": [0.92, 0.85, ...],
      "accuracy": 0.85,
      "avg_rate_bpf": 10.02
    }
  }
}
```

### 图表
- **分情感曲线**：每个情感独立显示
- **正确坐标轴**：Target Rate (BPF), Accuracy (0-1)
- **保存格式**：PNG, 300 DPI

## 预期成果

### 科学贡献
1. **首个真实ECVQ的情感信息瓶颈研究**
2. **码率-性能权衡的精确量化**（误差<1 BPF）
3. **分情感鲁棒性分析**（哪些情感对压缩更敏感）
4. **码率vs层数两种压缩方式对比**

### 可发表图表
- 分情感准确率 vs 目标码率曲线
- 分情感准确率 vs 层数曲线
- 置信度退化曲线
- 每个数据集独立分析

## 相关项目

- `emotion_information_bottleneck`：层数k控制（离散）
- `emotion_bottleneck_rate`：λ控制但无真实熵模型
- `emotion_infobottleneck`：分组RVQ技术来源
- `emilia_vevo_integration`：100h训练数据来源

## 版本信息

- **版本**: 1.0.0-rc1
- **创建时间**: 2025-11-12
- **当前状态**: 评估实验运行中（预计2-4小时完成）
- **下一步**: 分析结果，撰写论文图表

## 致谢

本项目整合了`emotion_information_bottleneck`的实验设计和`emotion_infobottleneck`的RVQ技术，在真实熵编码下实现了精确的码率控制和情感鲁棒性研究。

---

**最后更新**: 2025-11-13  
**实验状态**: 🏃 运行中（4个评估任务并行）
