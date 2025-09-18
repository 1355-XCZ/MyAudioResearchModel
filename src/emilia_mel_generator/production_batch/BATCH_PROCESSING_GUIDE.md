# Emilia数据集批处理系统使用指南

## 概述

这是一个基于成功的 `test_emilia_correct.py` 脚本开发的正式批处理系统，用于大规模处理Emilia数据集。

## 核心功能

- ✅ **使用BigVGAN提取原始音频的mel频谱图**
- ✅ **使用Vevo TTS生成中性语音的mel频谱图**  
- ✅ **提取emotion2vec情感特征**
- ✅ **使用Vevo声码器重建音频（确保完全对齐）**
- ✅ **支持大规模批处理和进度管理**
- ✅ **检查点恢复机制**
- ✅ **SLURM集群调度支持**

## 文件结构

```
src/emilia_mel_generator/
├── emilia_batch_processor.py    # 主批处理脚本
├── batch_config.yaml           # 配置文件
├── submit_batch_job.slurm      # SLURM作业脚本
├── submit_batch.sh             # 便捷提交脚本
└── BATCH_PROCESSING_GUIDE.md   # 本文档
```

## 快速开始

### 1. 基本用法

```bash
# 最简单的使用方式
./submit_batch.sh -i /path/to/emilia/dataset -o /path/to/output

# 带更多参数的使用方式
./submit_batch.sh \
    -i /data/emilia \
    -o /output/results \
    -b 200 \
    -r \
    --log-level DEBUG
```

### 2. 直接使用Python脚本

```bash
python emilia_batch_processor.py \
    --input_data_path /path/to/emilia/dataset \
    --output_base_path /path/to/output \
    --batch_size 100 \
    --device cuda \
    --resume \
    --checkpoint_interval 50 \
    --log_level INFO
```

### 3. 使用SLURM集群

```bash
# 设置环境变量
export INPUT_DATA_PATH="/data/emilia"
export OUTPUT_BASE_PATH="/output/results"
export BATCH_SIZE=100

# 提交作业
sbatch submit_batch_job.slurm
```

## 命令行参数

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--input_data_path` | 输入数据路径（Emilia数据集路径） | - | ✅ |
| `--output_base_path` | 输出基础路径 | - | ✅ |
| `--batch_size` | 批处理大小 | 100 | ❌ |
| `--device` | 计算设备 | cuda | ❌ |
| `--resume` | 从检查点恢复处理 | false | ❌ |
| `--checkpoint_interval` | 检查点保存间隔（批次数） | 50 | ❌ |
| `--log_level` | 日志级别 | INFO | ❌ |
| `--log_file` | 日志文件路径 | None | ❌ |

## 输出结构

```
output_base_path/
├── mels/                       # mel频谱图文件
│   ├── sample_001_mel_original_bigvgan.npy
│   └── sample_001_mel_neutral_vevo.npy
├── emotion_features/           # emotion2vec特征
│   └── sample_001_ev2.npz
├── verification_audio/         # 验证音频
│   ├── sample_001_s1_original.wav
│   ├── sample_001_original_reconstructed.wav
│   └── sample_001_neutral_reconstructed.wav
├── reports/                    # 处理报告
│   ├── final_report_20250918_143022.json
│   └── processing_summary.json
├── checkpoints/                # 检查点文件
│   └── checkpoint_batch_000001.json
└── logs/                       # 日志文件
    └── emilia_batch_12345.log
```

## 配置管理

### 修改 `batch_config.yaml`

```yaml
# 数据路径配置
data_paths:
  input_data_root: "/your/emilia/dataset/path"
  output_root: "/your/output/path"

# 处理配置
processing:
  batch_size: 100
  device: "cuda"
  
# 模型配置
models:
  vevo_tts:
    flow_matching_steps: 16
```

## 监控和管理

### 查看作业状态

```bash
# 查看SLURM作业状态
squeue -u $USER

# 查看特定作业
squeue -j JOB_ID

# 取消作业
scancel JOB_ID
```

### 查看日志

```bash
# 实时查看输出日志
tail -f logs/emilia_batch_12345.out

# 查看错误日志
tail -f logs/emilia_batch_12345.err

# 查看Python日志
tail -f output_path/logs/emilia_batch_12345.log
```

### 检查进度

```bash
# 查看处理进度
cat output_path/reports/processing_summary.json

# 查看最新检查点
ls -la output_path/checkpoints/
```

## 错误处理和恢复

### 从检查点恢复

如果任务中断，可以使用 `--resume` 参数从最新检查点恢复：

```bash
./submit_batch.sh -i /data/emilia -o /output/results -r
```

### 常见问题

1. **GPU内存不足**
   - 减少 `batch_size`
   - 使用 `device=cpu`

2. **磁盘空间不足**
   - 检查输出路径的可用空间
   - 调整 `checkpoint_interval`

3. **模型下载失败**
   - 检查网络连接
   - 手动下载Vevo模型

4. **权限问题**
   - 确保对输入和输出路径有读写权限
   - 检查SLURM作业的用户权限

## 性能优化

### GPU使用优化

```bash
# 使用多GPU（如果可用）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 调整批处理大小
--batch_size 200  # 增加批大小提高GPU利用率
```

### 内存优化

```bash
# 减少内存使用
--batch_size 50   # 减少批大小
--checkpoint_interval 25  # 更频繁的检查点
```

## 集群环境配置

### SLURM参数调整

修改 `submit_batch_job.slurm` 中的资源请求：

```bash
#SBATCH --mem=64G              # 增加内存
#SBATCH --gres=gpu:2           # 使用多GPU
#SBATCH --time=48:00:00        # 延长时间限制
```

### 环境模块

如果使用环境模块系统：

```bash
# 在SLURM脚本中添加
module load cuda/11.8
module load python/3.9
module load conda
```

## 故障排除

### 检查系统状态

```bash
# 检查GPU状态
nvidia-smi

# 检查磁盘空间
df -h

# 检查内存使用
free -h

# 检查Python环境
conda list | grep torch
```

### 调试模式

```bash
# 启用详细日志
--log_level DEBUG

# 小批量测试
--batch_size 10
```

## 联系和支持

如果遇到问题，请检查：
1. 日志文件中的详细错误信息
2. 检查点文件中的处理状态
3. 系统资源使用情况

---

**注意**: 这个批处理系统基于成功的测试脚本 `test_emilia_correct.py` 开发，继承了其所有的技术优势和稳定性。
