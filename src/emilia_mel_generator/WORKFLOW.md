# 🔄 完整工作流程

## 📋 两阶段处理流程

### 🧪 Phase 1: 本地测试验证
```bash
cd src/emilia_mel_generator

# 运行10条音频的小规模测试
python main.py --test-local --k-variants 1

# 检查验证音频
# 听取 test_output_small/verification_audio/ 中的音频文件:
# - *_s1_original.wav (原始s1音频)
# - *_original_reconstructed.wav (原始mel重建音频)  
# - *_neutral_reconstructed.wav (中性mel重建音频)

# 确认: 中性重建音频保持s1音色但去除情感表达
```

### 🖥️ Phase 2: Spartan 集群大规模处理
```bash
# 1. 上传到Spartan
scp -r MyAudioResearchModel username@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2341/haoguangzhou/

# 2. 登录集群
ssh username@spartan.hpc.unimelb.edu.au

# 3. 进入项目目录
cd /data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel/src/emilia_mel_generator

# 4. 设置权限
chmod +x submit_spartan.slurm submit_multiple_jobs.sh

# 5. 提交批量作业 (分10批处理50小时数据)
./submit_multiple_jobs.sh

# 6. 监控作业
squeue -u $USER
```

## 📊 关键配置

### 默认设置 (已修正)
- **K值**: 1 (默认，可根据需要增大)
- **目标数据**: 50小时中英文
- **批次数**: 10个 (每批次5小时)
- **存储**: ~50-80GB

### Spartan资源配置
```bash
#SBATCH --partition=gpu-a100-short    # A100 GPU短时间分区
#SBATCH --gres=gpu:1                  # 1个GPU
#SBATCH -c 8                          # 8个CPU核心
#SBATCH --mem=32G                     # 32GB内存
#SBATCH -t 04:00:00                   # 4小时时间限制
```

## 🎯 数据验证

### 本地测试验证点
1. ✅ **mel形状**: [128, frames] 正确
2. ✅ **emotion2vec维度**: [768], [frames, 768] 正确
3. ✅ **音频重建**: 可播放且质量可接受
4. ✅ **中性化效果**: 保持音色，去除情感

### Spartan输出验证
1. ✅ **文件完整性**: 所有.npy, .npz, .json文件生成
2. ✅ **CSV文件**: train_tuples.csv, val_tuples.csv正确
3. ✅ **数据量**: 符合预期的样本数量
4. ✅ **质量**: 随机抽样验证数据质量

## 🔧 故障排除

### 本地测试问题
- **GPU内存不足**: 减少测试样本数
- **音频质量差**: 检查mel重建参数
- **特征维度错误**: 检查emotion2vec配置

### Spartan集群问题
- **作业排队**: 选择合适的分区和时间
- **内存不足**: 增加内存申请或减少批次大小
- **网络问题**: 检查HuggingFace访问权限

## 📁 最终文件结构

```
src/emilia_mel_generator/ (12个文件)
├── 核心代码:
│   ├── corrected_data_generator.py   # 主生成器
│   ├── config.py                     # 配置管理
│   ├── utils.py                      # 工具函数
│   └── main.py                       # 命令行接口
├── 测试脚本:
│   ├── test_local_small.py           # 本地测试 ⭐
│   ├── spartan_batch_job.py          # Spartan批处理 ⭐
│   ├── submit_spartan.sh             # Slurm提交脚本 ⭐
│   └── submit_multiple_jobs.sh       # 多批次提交 ⭐
├── 文档:
│   ├── README.md                     # 主要说明
│   ├── USAGE.md                      # 使用指南
│   ├── SPARTAN_GUIDE.md              # Spartan使用指南 ⭐
│   └── WORKFLOW.md                   # 完整工作流程 ⭐
└── __init__.py                       # 包初始化
```

## 🎉 立即开始

### 1. 本地验证
```bash
cd src/emilia_mel_generator
python main.py --test-local
```

### 2. 集群处理 (验证通过后)
```bash
# 上传并运行
./submit_multiple_jobs.sh
```

**工作流程已完全准备就绪，可以开始处理您的数据！** 🚀
