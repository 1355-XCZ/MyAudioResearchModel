# 服务器测试指南 - 小规模验证

## 🎯 **测试目标**

在Spartan集群上进行小规模测试，验证完整的训练数据生成流程：

- **规模**: 中英文各10条样本 (共20条)
- **时间**: 30-60分钟
- **资源**: 1x GPU, 4 CPU, 16GB RAM

## 🚀 **快速开始**

### **1. 提交测试作业**
```bash
cd production_batch

# 使用默认配置
./submit_server_test.sh

# 或自定义配置
./submit_server_test.sh -s 10 -o /path/to/test/output
```

### **2. 监控测试进度**
```bash
# 查看作业状态
squeue -u haoguangz

# 实时查看日志
tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-JOBID.out
```

## 📋 **测试内容验证**

### **1. Emo-Emilia数据集集成**
```bash
# 在服务器上验证
python -c "
from datasets import load_dataset
ds = load_dataset('ASLP-lab/Emo-Emilia')
print(f'✅ 加载成功: {len(ds[\"train\"])} 个样本')

# 统计neutral样本
neutral_samples = [item for item in ds['train'] if item['emotion'] == 'neutral']
print(f'📊 Neutral样本: {len(neutral_samples)} 个')

# 语言分布
lang_count = {}
for item in neutral_samples:
    lang = item['language']
    lang_count[lang] = lang_count.get(lang, 0) + 1
print(f'📊 Neutral语言分布: {lang_count}')
"
```

### **2. 语言匹配验证**
```bash
# 测试完成后运行
python verify_language_matching.py --output_path /data/gpfs/projects/punim2341/haoguangzhou/emilia_test/test_small_*

# 预期输出:
# ✅ EN: 100% 使用英文neutral参考
# ✅ ZH: 100% 使用中文neutral参考
```

### **3. 数据格式验证**
```bash
# 验证生成的数据格式
python verify_training_data.py --output_path /data/gpfs/projects/punim2341/haoguangzhou/emilia_test/test_small_*

# 预期输出:
# ✅ mel格式: [T, 128] Vevo兼容
# ✅ hop_size: 480, n_mels: 128
```

## 📊 **预期测试结果**

### **目录结构**
```
/data/gpfs/projects/punim2341/haoguangzhou/emilia_test/test_small_20250918_153000/
├── original_mels/              # 20个源mel文件
│   ├── en_000001_mel_original_vevo.npz
│   ├── zh_000001_mel_original_vevo.npz
│   └── ...
├── neutral_mels/               # 20个中性mel文件
│   ├── en_000001_mel_neutral_vevo.npz
│   ├── zh_000001_mel_neutral_vevo.npz
│   └── ...
├── emotion_features/           # 20个情感特征文件
│   ├── en_000001_emotion_features.npz
│   ├── zh_000001_emotion_features.npz
│   └── ...
├── metadata/                   # 元信息文件
│   ├── training_dataset_metadata.csv
│   ├── dataset_statistics.json
│   └── README.md
└── reports/
    └── final_generation_report.json
```

### **CSV元信息示例**
```csv
sample_id,language,duration_seconds,text,speaker,original_mel_file,neutral_mel_file,emotion_features_file,processing_timestamp
en_000001,en,3.2,"This is English test sample...",en_speaker_1,en_000001_mel_original_vevo.npz,en_000001_mel_neutral_vevo.npz,en_000001_emotion_features.npz,2025-09-18T15:30:00
zh_000001,zh,2.8,"这是中文测试样本...",zh_speaker_1,zh_000001_mel_original_vevo.npz,zh_000001_mel_neutral_vevo.npz,zh_000001_emotion_features.npz,2025-09-18T15:30:01
```

### **成功指标**
- ✅ **成功率**: ≥80% 样本处理成功
- ✅ **语言匹配**: ≥90% 使用正确语言的neutral参考
- ✅ **格式兼容**: 100% Vevo兼容格式
- ✅ **文件完整**: 每个样本3个文件 + CSV元信息

## 🔧 **故障排除**

### **常见问题**

1. **Emo-Emilia加载失败**
   ```bash
   # 检查HuggingFace登录
   hf auth whoami
   
   # 如果未登录
   hf auth login
   ```

2. **GPU内存不足**
   ```bash
   # 减少样本数
   ./submit_server_test.sh -s 10
   ```

3. **网络问题**
   ```bash
   # 检查网络连接
   ping huggingface.co
   
   # 检查代理设置
   echo $HTTP_PROXY
   ```

### **调试命令**

```bash
# 查看详细错误日志
tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-*.err

# 检查作业状态
scontrol show job JOBID

# 取消作业
scancel JOBID

# 重新提交
./submit_server_test.sh
```

## 🎉 **测试成功后的下一步**

如果小规模测试成功，可以进行大规模处理：

```bash
# 提交大规模分批处理
./submit_batch_manager.sh

# 或单批处理
./submit_training_data.sh -H 50.0  # 每种语言50小时
```

## 📋 **测试检查清单**

- [ ] 提交测试作业成功
- [ ] Emo-Emilia数据集加载成功  
- [ ] 中英文neutral参考音频正确匹配
- [ ] 生成Vevo兼容的mel频谱图
- [ ] emotion2vec特征从源音频提取
- [ ] CSV元信息文件正确生成
- [ ] 目录结构清晰易懂
- [ ] 数据格式验证通过

完成这个检查清单后，就可以放心地进行大规模数据处理了！🚀
