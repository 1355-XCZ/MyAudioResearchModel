# Emilia数据集批处理系统 - 生产环境

这个文件夹包含了基于成功的 `test_emilia_correct.py` 开发的正式批处理系统，专门用于生产环境的大规模数据处理。

## 📁 文件结构

```
production_batch/
├── emilia_batch_processor.py    # 主批处理脚本
├── batch_config.yaml           # 通用配置文件
├── spartan_config.yaml         # Spartan集群专用配置
├── submit_batch.sh             # 通用提交脚本
├── submit_batch_job.slurm      # 通用SLURM脚本
├── submit_spartan.sh           # Spartan集群专用提交脚本
├── submit_spartan.slurm        # Spartan集群专用SLURM脚本
├── test_batch_processor.py     # 系统测试脚本
├── BATCH_PROCESSING_GUIDE.md   # 详细使用指南
└── README.md                   # 本文档
```

## 🚀 快速开始 (Spartan集群)

### 1. 使用默认配置

```bash
cd production_batch
./submit_spartan.sh
```

### 2. 自定义配置

```bash
./submit_spartan.sh -b 200 -r -t 08:00:00
```

### 3. 检查作业状态

```bash
# 查看作业队列
squeue -u haoguangz

# 查看日志
tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-vevo-batch-JOBID.out
```

## 🔧 核心特性

基于成功的 `test_emilia_correct.py` 实现：

- ✅ **使用 BigVGAN 提取原始音频 mel**
- ✅ **使用 Vevo TTS 生成中性 mel** 
- ✅ **提取 emotion2vec 情感特征**
- ✅ **完全使用 Vevo 声码器重建音频**
- ✅ **解决了音频质量和重复问题**
- ✅ **支持大规模批处理**
- ✅ **检查点恢复机制**
- ✅ **Spartan 集群优化**

## 📊 Spartan 集群配置

### 资源配置
- **分区**: gpu-a100-short (4小时) / gpu-a100 (长时间)
- **GPU**: 1x A100
- **CPU**: 8核
- **内存**: 32GB
- **账号**: punim2341

### 路径配置
- **项目根目录**: `/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel`
- **数据集路径**: `/data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset`
- **输出路径**: `/data/gpfs/projects/punim2341/haoguangzhou/emilia_output`
- **日志路径**: `/data/gpfs/projects/punim2341/haoguangzhou/logs`

### 环境配置
- **Conda环境**: marm5
- **Python路径**: `/data/gpfs/projects/punim2341/haoguangzhou/miniconda3/envs/marm5/bin/python`

## 📈 监控和管理

### 查看进度
```bash
# 查看处理进度
cat /data/gpfs/projects/punim2341/haoguangzhou/emilia_output/reports/processing_summary.json

# 查看最新检查点
ls -la /data/gpfs/projects/punim2341/haoguangzhou/emilia_output/checkpoints/
```

### 作业管理
```bash
# 提交作业
./submit_spartan.sh

# 查看作业状态
squeue -j JOBID

# 取消作业
scancel JOBID

# 查看作业详情
scontrol show job JOBID
```

## 🛠️ 故障排除

### 常见问题

1. **权限问题**
   ```bash
   # 确保有正确的文件权限
   chmod +x submit_spartan.sh
   chmod +x submit_spartan.slurm
   ```

2. **环境问题**
   ```bash
   # 检查conda环境
   conda activate marm5
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **存储空间**
   ```bash
   # 检查可用空间
   df -h /data/gpfs/projects/punim2341/haoguangzhou/
   ```

### 调试模式

```bash
# 启用详细日志
./submit_spartan.sh -l DEBUG -b 10
```

## 📧 联系信息

- **用户**: haoguangz@student.unimelb.edu.au
- **项目**: punim2341
- **集群**: Spartan (University of Melbourne)

---

**注意**: 这个系统完全基于成功测试的 `test_emilia_correct.py` 脚本，继承了所有的技术优势和稳定性。
