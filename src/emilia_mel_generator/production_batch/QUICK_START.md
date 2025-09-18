# Emilia训练数据生成系统 - 快速开始指南

## 🎯 Spartan集群快速部署

### 1. 准备部署包
```bash
cd production_batch
./deploy_to_spartan.sh
```

### 2. 上传到Spartan
```bash
scp emilia_batch_system_*.tar.gz haoguangz@spartan.hpc.unimelb.edu.au:~/
```

### 3. 在Spartan上部署
```bash
ssh haoguangz@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel
tar -xzf ~/emilia_batch_system_*.tar.gz -C src/emilia_mel_generator/production_batch/
chmod +x src/emilia_mel_generator/production_batch/*.sh
```

### 4. 提交训练数据生成作业
```bash
cd src/emilia_mel_generator/production_batch

# 生成英文和中文各50小时训练数据
./submit_training_data.sh

# 或自定义时长
./submit_training_data.sh -H 25.0  # 每种语言25小时

# 测试模式
./submit_training_data.sh -m 100   # 只处理100个样本
```

## 📊 监控作业

```bash
# 查看作业状态
squeue -u haoguangz

# 查看实时日志和进度条
tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-training-data-JOBID.out

# 验证生成的数据格式
python verify_training_data.py --output_path /data/gpfs/projects/punim2341/haoguangzhou/emilia_training_data

# 查看生成报告
cat /data/gpfs/projects/punim2341/haoguangzhou/emilia_training_data/reports/generation_report.json
```

## 🔧 核心技术

基于成功的 `test_emilia_correct.py`：
- ✅ **源mel和中性mel都使用Vevo提取器**（确保参数一致）
- ✅ **完全兼容Vevo声码器**（hop_size=480, n_mels=128）
- ✅ **使用真实音频文本**（解决重复问题）
- ✅ **英文和中文各50小时数据**
- ✅ **14个样本测试成功**

## 📁 输出结构

```
/data/gpfs/projects/punim2341/haoguangzhou/emilia_training_data/
├── mels/
│   ├── *_mel_original_vevo.npz     # 源音频mel [T, 128] Vevo兼容
│   └── *_mel_neutral_vevo.npz      # 中性mel [T, 128] Vevo兼容
├── emotion_features/
│   └── *_ev2.npz                   # 情感特征
└── reports/
    └── generation_report.json      # 生成报告

✅ 所有mel文件都使用相同的Vevo参数：
  - hop_size: 480
  - n_mels: 128  
  - sample_rate: 24000
  - 格式: [T, 128]
```

## ⚡ 快速命令

```bash
# 生成训练数据（英文+中文各50小时）
./submit_training_data.sh

# 自定义时长
./submit_training_data.sh -H 25.0  # 每种语言25小时

# 查看状态
squeue -u haoguangz

# 查看进度
tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-training-data-*.out

# 验证数据格式
python verify_training_data.py --output_path /path/to/output

# 取消作业
scancel JOBID
```
