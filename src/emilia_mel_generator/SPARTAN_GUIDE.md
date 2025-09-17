# 🖥️ Spartan 集群使用指南

## 📋 概述

使用 [墨尔本大学 Spartan HPC 集群](https://dashboard.hpc.unimelb.edu.au/) 进行大规模 Emilia 数据集处理。

## 🚀 使用流程

### 1. 本地测试 (必需)
```bash
# 在本地先运行小规模测试
cd src/emilia_mel_generator
python test_local_small.py

# 验证生成的音频文件
# 听取 test_output_small/verification_audio/ 中的音频
# 确认中性mel重建音频保持音色但去除情感
```

### 2. 上传到 Spartan
```bash
# 上传项目到集群
scp -r MyAudioResearchModel username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2341/haoguangzhou/
```

### 3. 环境准备
```bash
# 登录Spartan
ssh username@spartan.hpc.unimelb.edu.au

# 进入项目目录
cd /data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel/src/emilia_mel_generator

# 创建日志目录
mkdir -p /data/gpfs/projects/punim2341/haoguangzhou/logs

# 设置执行权限
chmod +x submit_spartan.slurm
chmod +x submit_multiple_jobs.sh
```

### 4. 提交作业

#### 单批次测试
```bash
# 提交单个测试批次
BATCH_ID=0 TOTAL_BATCHES=1 TARGET_HOURS=1 K_VARIANTS=1 sbatch submit_spartan.slurm
```

#### 多批次生产运行
```bash
# 修改参数 (在 submit_multiple_jobs.sh 中)
# TOTAL_BATCHES=10    # 分10个批次
# TARGET_HOURS=50     # 每语言50小时
# K_VARIANTS=1        # K=1 (可根据需要调整)

# 提交所有批次
./submit_multiple_jobs.sh
```

## 📊 资源配置

### 推荐配置 (基于您的参考)
```bash
#SBATCH --partition=gpu-a100-short    # A100 GPU (短时间)
#SBATCH --gres=gpu:1                  # 1个GPU
#SBATCH -c 8                          # 8个CPU核心 (增加了)
#SBATCH --mem=32G                     # 32GB内存 (增加了)
#SBATCH -t 04:00:00                   # 4小时时间限制 (增加了)
```

### 分区选择建议
- **gpu-a100-short**: 适合4小时内的快速处理
- **gpu-a100**: 适合更长时间的大批次处理
- **gpu**: 通用GPU分区，如果A100不可用

### 内存和时间估算
```
每小时数据处理需求:
- 内存: ~2-4GB
- 时间: ~10-20分钟
- GPU: 加速mel提取和特征计算

50小时数据 (单批次):
- 内存: ~32GB (安全余量)
- 时间: ~3-4小时
- 建议: 分10个批次，每批次5小时数据
```

## 📁 文件结构 (Spartan)

### 项目路径
```
/data/gpfs/projects/punim2341/haoguangzhou/
├── MyAudioResearchModel/
│   └── src/emilia_mel_generator/
├── emilia_dataset/                    # 输出数据
│   ├── batch_00/
│   ├── batch_01/
│   └── ...
│   └── merged_dataset/                # 合并后数据
└── logs/                              # 作业日志
    ├── emilia-data-extract-*.out
    └── emilia-data-extract-*.err
```

## 🔧 监控和管理

### 查看作业状态
```bash
# 查看所有作业
squeue -u $USER

# 查看特定作业
squeue -j <job_id>

# 查看作业详情
scontrol show job <job_id>
```

### 查看日志
```bash
# 实时查看输出日志
tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-data-extract-*.out

# 查看错误日志
tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-data-extract-*.err
```

### 取消作业
```bash
# 取消特定作业
scancel <job_id>

# 取消所有相关作业
scancel -u $USER --name=emilia-data-extract
```

## 📊 批次处理策略

### 推荐分批方案
```
总数据: 50小时 × 2语言 = 100小时
批次数: 10个批次
每批次: 5小时 × 2语言 = 10小时
预计时间: 每批次30-60分钟
```

### 批次提交顺序
```bash
# 方案1: 顺序提交 (保守)
for i in {0..9}; do
    sbatch --export=BATCH_ID=$i submit_spartan.sh
    sleep 300  # 等待5分钟
done

# 方案2: 并行提交 (激进)
./submit_multiple_jobs.sh  # 同时提交所有批次
```

## 🧪 测试验证

### 本地测试检查清单
- [ ] 运行 `test_local_small.py` 成功
- [ ] 生成的mel文件形状正确 [128, frames]
- [ ] emotion2vec特征维度正确 [768], [frames, 768]
- [ ] 验证音频可以播放且质量可接受
- [ ] 中性重建音频保持音色但去除情感表达

### Spartan测试检查清单
- [ ] 单批次测试作业成功完成
- [ ] GPU和内存使用正常
- [ ] 输出文件生成正确
- [ ] 日志无严重错误
- [ ] 数据质量符合预期

## ⚠️ 注意事项

### 数据集访问
- 确保HuggingFace访问权限已配置
- Emilia和Emo-Emilia数据集访问权限已获得
- 网络连接稳定，能够下载大型数据集

### 存储管理
- 监控磁盘空间使用 (`df -h`)
- 及时清理临时文件
- 备份重要中间结果

### 作业管理
- 合理设置时间限制
- 监控内存使用避免OOM
- 使用信号处理优雅退出

## 🎯 完成后验证

### 数据完整性检查
```bash
# 检查生成的文件数量
find /data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset -name "*.npy" | wc -l

# 检查CSV文件
head /data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset/merged_dataset/csv_lists/all_tuples.csv

# 验证数据质量
python -c "
import numpy as np
mel = np.load('path/to/sample.npy')
print(f'Mel shape: {mel.shape}')
print(f'Mel range: {mel.min():.3f} to {mel.max():.3f}')
"
```

### 下载结果
```bash
# 下载处理后的数据到本地
scp -r username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset ./
```

## 🎉 使用总结

1. **本地测试**: 验证10条音频处理正确
2. **Spartan提交**: 分批处理50小时数据
3. **结果验证**: 检查数据完整性和质量
4. **开始训练**: 使用处理后数据训练模型

这个流程确保了从小规模验证到大规模生产的平滑过渡！
