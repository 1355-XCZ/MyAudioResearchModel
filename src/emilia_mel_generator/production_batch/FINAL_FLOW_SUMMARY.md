# 🎯 主脚本完整处理流程 - 最终确认

## 📋 **每个音频的详细处理过程**

### **输入示例:**
```
音频文件: 中文财经播报，3.2秒
sample_id: "zh_001234"
text: "具体而言，上半周持续上涨的逻辑在于部分资金交易供给进一步短缺的预期。"
language: "zh"
speaker: "财经播报员A"
```

### **处理步骤:**

#### **Step 1: 准备源音频**
```python
# 加载音频数据
audio_array = sample['audio']['array']  # [76800] 采样点
sample_rate = 24000  # 重采样到24kHz
duration = 3.2  # 秒

# 转换为tensor
audio_tensor = torch.tensor(audio_array).unsqueeze(0).to('cuda')  # [1, 76800]
```

#### **Step 2: 提取源音频的Vevo兼容mel**
```python
# 使用Vevo mel提取器（确保兼容Vevo声码器）
mel_original_vevo = vevo_pipeline.extract_mel_feature(audio_tensor)
# 输出: [1, 160, 128]  (160帧 = 76800/480, 128个mel通道)

# 验证Vevo兼容性
assert mel_original_vevo.shape[-1] == 128  # 128个mel通道
assert mel_original_vevo.dim() == 3        # [batch, time, mel_channels]
# hop_size=480, n_mels=128, sample_rate=24000
```

#### **Step 3: 获取中文中性参考音频**
```python
# 严格语言匹配：中文音频必须使用中文中性参考
chinese_neutral_refs = [
    ref for ref in emo_emilia_neutral_pool 
    if ref['language'].lower() == 'zh'  # 只选择中文neutral
]

# 随机选择一个中文中性参考
selected_neutral = random.choice(chinese_neutral_refs)
# 例如:
# {
#   'audio': [72000] 采样点, 3.0秒中文neutral音频,
#   'text': "这是一个关于市场情况的中性陈述。",
#   'language': "zh",
#   'emotion': "neutral"
# }

# 确保语言匹配
assert selected_neutral['language'].lower() == 'zh'
```

#### **Step 4: 使用Vevo TTS生成中性mel**
```python
# 保存临时文件
sf.write("temp/source_zh.wav", audio_array, 24000)          # 源音频（音色参考）
sf.write("temp/neutral_zh.wav", selected_neutral['audio'], 24000)  # 中文中性参考

# Vevo TTS调用
gen_audio = vevo_pipeline.inference_ar_and_fm(
    src_wav_path=None,                                       # TTS模式
    src_text="具体而言，上半周持续上涨的逻辑...",              # 源音频真实文本
    style_ref_wav_path="temp/neutral_zh.wav",               # 中文中性参考音频
    style_ref_wav_text="这是一个关于市场情况的中性陈述。",     # 中文中性参考文本
    timbre_ref_wav_path="temp/source_zh.wav",               # 源音频音色参考
    src_text_language="zh",                                 # 中文
    style_ref_wav_text_language="zh",                       # 中文（匹配！）
    flow_matching_steps=16,
    use_global_guided_inference=False
)

# 从生成的音频提取mel（使用相同的Vevo提取器）
gen_audio_tensor = torch.tensor(gen_audio).unsqueeze(0).to('cuda')
mel_neutral_vevo = vevo_pipeline.extract_mel_feature(gen_audio_tensor)
# 输出: [1, T', 128]  (T'可能不同，但格式相同)

# 验证格式一致性
assert mel_neutral_vevo.shape[-1] == 128  # 与源mel相同参数
assert mel_neutral_vevo.dim() == 3
```

#### **Step 5: 从源音频提取emotion2vec情感表征**
```python
# 重要：从原始源音频提取情感特征，不是中性化后的
original_audio_tensor = torch.tensor(audio_array).unsqueeze(0).to('cuda')  # 源音频
ev2_features = emotion2vec_extractor.extract_features(original_audio_tensor)

# 输出（源音频的真实情感表征）:
# - utterance: [1, 768] 源音频语句级情感特征  
# - frame: [1, 160, 768] 源音频帧级情感特征

# 这些特征反映的是原始音频的真实情感状态
# 用于训练情感转换模型：从源情感转换到中性情感
```

#### **Step 6: 保存训练数据到清晰目录结构**
```python
# 文件1: original_mels/zh_001234_mel_original_vevo.npz
{
    'mel': [160, 128],           # 源音频mel数据
    'format': 'vevo_compatible',
    'hop_size': 480,
    'n_mels': 128,
    'sample_rate': 24000,
    'language': 'zh',
    'duration': 3.2,
    'source_type': 'original_audio'
}

# 文件2: neutral_mels/zh_001234_mel_neutral_vevo.npz
{
    'mel': [T', 128],            # 中性化mel数据
    'format': 'vevo_compatible', 
    'hop_size': 480,             # 与源mel相同参数
    'n_mels': 128,
    'sample_rate': 24000,
    'language': 'zh',
    'source_type': 'neutral_transformed',
    'neutral_reference_source': 'emo_emilia',
    'neutral_reference_language': 'zh'  # 确认使用中文中性参考
}

# 文件3: emotion_features/zh_001234_emotion_features.npz  
{
    'utterance': [768],          # 源音频语句级情感特征
    'frame': [160, 768],         # 源音频帧级情感特征
    'language': 'zh',
    'source_type': 'original_audio_emotion',  # 明确标记来源
    'duration': 3.2
}

# 文件4: metadata/training_dataset_metadata.csv (追加记录)
sample_id,language,duration_seconds,text,speaker,original_mel_file,neutral_mel_file,emotion_features_file,original_mel_shape,neutral_mel_shape,emotion_source,processing_timestamp
zh_001234,zh,3.2,"具体而言，上半周持续上涨...",财经播报员A,zh_001234_mel_original_vevo.npz,zh_001234_mel_neutral_vevo.npz,zh_001234_emotion_features.npz,160x128,T'x128,original_audio,2025-09-18T15:30:00
```

## 🔧 **关键确认点**

### **1. emotion2vec来源确认**
- ✅ **从源音频提取**: `emotion2vec_extractor.extract_features(original_audio_tensor)`
- ✅ **不是中性化后**: 特征反映原始情感状态
- ✅ **用途**: 训练情感转换模型（源情感→中性情感）

### **2. 语言匹配严格保证**
- ✅ **中文音频**: 使用中文Emo-Emilia neutral参考
- ✅ **英文音频**: 使用英文Emo-Emilia neutral参考
- ✅ **验证机制**: 代码中有严格的语言匹配检查

### **3. mel参数完全一致**
- ✅ **提取器**: 源mel和中性mel都用 `vevo_pipeline.extract_mel_feature()`
- ✅ **参数**: hop_size=480, n_mels=128, sample_rate=24000
- ✅ **格式**: [T, 128] Vevo兼容
- ✅ **用途**: 可直接用于Vevo声码器

### **4. 清晰的目录结构**
- ✅ **分类存储**: 源mel、中性mel、情感特征分别存储
- ✅ **CSV元信息**: 完整的样本信息和文件映射
- ✅ **易于访问**: 提供dataset_accessor.py工具

## 📊 **最终数据规模**

```
英文: 50小时 ≈ 约10,000个样本
中文: 50小时 ≈ 约10,000个样本
总计: 100小时 ≈ 约20,000个样本

每个样本3个文件:
- 20,000个源mel文件
- 20,000个中性mel文件  
- 20,000个情感特征文件
总计: 60,000个训练数据文件 + 1个CSV元信息文件
```

## 🎯 **训练用途**

生成的数据可用于训练：

1. **情感转换模型**:
   - 输入: 源mel + 源情感特征  
   - 输出: 中性mel
   - 学习: 如何将有情感的mel转换为中性mel

2. **情感识别模型**:
   - 输入: 源mel
   - 输出: 情感特征
   - 学习: 从mel预测情感

3. **语音合成模型**:
   - 使用Vevo兼容的mel进行训练
   - 可直接用Vevo声码器生成音频

这个流程是否完全符合你的预期？特别是emotion2vec从源音频提取这一点？🤔
