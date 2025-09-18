# Emilia训练数据生成 - 详细处理流程

## 🎯 **完整处理流程概览**

每个音频样本经过以下步骤处理，生成3个训练数据文件：

```
输入: 原始音频 (sample.wav)
  ↓
步骤1: 使用Vevo提取源mel频谱图
  ↓  
步骤2: 从Emo-Emilia获取匹配语言的中性参考音频
  ↓
步骤3: 使用Vevo TTS生成中性mel频谱图
  ↓
步骤4: 提取emotion2vec情感特征
  ↓
输出: 3个训练数据文件
```

## 📋 **详细处理步骤**

### **阶段0: 系统初始化**

```python
# 1. 加载Vevo TTS流水线
vevo_pipeline = VevoInferencePipeline(...)

# 2. 加载Emotion2Vec提取器  
emotion2vec_extractor = Emotion2VecExtractor(...)

# 3. 从Emo-Emilia加载中性参考音频池
emo_dataset = load_dataset("ASLP-lab/Emo-Emilia")
neutral_reference_pool = []
for item in emo_dataset['train']:
    if item['emotion'] == 'neutral':
        neutral_reference_pool.append({
            'audio': item['audio'],
            'text': item['text'], 
            'language': item['language'],  # 'en' 或 'zh'
            'speaker': item['speaker']
        })

# 结果: EN约100个 + ZH约100个 = 200个中性参考音频
```

### **阶段1: 数据集筛选**

```python
# 1. 加载Emilia数据集
emilia_dataset = load_dataset("amphion/Emilia-Dataset")

# 2. 按语言和时长筛选
target_hours_per_lang = 50.0  # 每种语言50小时
lang_duration = {'en': 0.0, 'zh': 0.0}
selected_samples = []

for item in emilia_dataset:
    language = item['language'].lower()  # 'en' 或 'zh'
    duration = len(item['audio']['array']) / item['audio']['sampling_rate']
    
    if lang_duration[language] < 50.0 * 3600:  # 50小时 = 180000秒
        selected_samples.append({
            'id': f"{language}_{len(selected_samples):06d}",
            'audio': item['audio'],
            'text': item['text'],
            'language': language,
            'duration': duration
        })
        lang_duration[language] += duration

# 结果: 约10,000-20,000个样本，英文和中文各50小时
```

### **阶段2: 单个音频处理** 

对每个音频样本 `sample` 执行以下处理：

#### **步骤2.1: 准备音频数据**
```python
sample_id = sample['id']           # 例如: "en_001234" 或 "zh_005678"  
audio_array = sample['audio']['array']     # 音频波形数据
sample_rate = sample['audio']['sampling_rate']  # 采样率
text = sample['text']              # 音频对应的文本
language = sample['language']      # 'en' 或 'zh'

# 重采样到24kHz（如果需要）
if sample_rate != 24000:
    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=24000)

# 转换为tensor
audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).to(device)
```

#### **步骤2.2: 提取源音频的Vevo兼容mel频谱图**
```python
# 使用Vevo的mel提取器（确保与Vevo声码器兼容）
mel_original_vevo = vevo_pipeline.extract_mel_feature(audio_tensor)

# 验证格式
assert mel_original_vevo.shape[-1] == 128  # Vevo格式: [1, T, 128]
assert mel_original_vevo.dim() == 3

# mel参数:
# - hop_size: 480
# - n_mels: 128  
# - sample_rate: 24000
# - 格式: [1, T_frames, 128]
```

#### **步骤2.3: 获取匹配语言的中性参考音频**
```python
# 严格语言匹配
if language == 'zh':
    # 中文音频 -> 从中文neutral样本中随机选择
    neutral_candidates = [
        ref for ref in neutral_reference_pool 
        if ref['language'].lower() == 'zh'
    ]
elif language == 'en':
    # 英文音频 -> 从英文neutral样本中随机选择  
    neutral_candidates = [
        ref for ref in neutral_reference_pool
        if ref['language'].lower() == 'en'
    ]

# 随机选择一个中性参考
import random
selected_neutral_ref = random.choice(neutral_candidates)

neutral_audio = selected_neutral_ref['audio']['array']
neutral_text = selected_neutral_ref['text']
neutral_language = selected_neutral_ref['language']

# 确保语言匹配
assert neutral_language.lower() == language.lower()
```

#### **步骤2.4: 使用Vevo TTS生成中性mel频谱图**
```python
# 保存临时音频文件
sf.write("temp/source_timbre.wav", audio_array, 24000)
sf.write("temp/neutral_style.wav", neutral_audio, 24000)

# 文本处理
clean_source_text = text.strip()[:200]  # 限制长度
clean_neutral_text = neutral_text.strip()

# Vevo TTS调用
gen_audio = vevo_pipeline.inference_ar_and_fm(
    src_wav_path=None,                    # TTS模式
    src_text=clean_source_text,           # 源音频的真实文本
    style_ref_wav_path="temp/neutral_style.wav",    # 中性风格参考音频
    style_ref_wav_text=clean_neutral_text,          # 中性风格参考文本
    timbre_ref_wav_path="temp/source_timbre.wav",   # 源音频作为音色参考
    src_text_language=language,           # 源文本语言
    style_ref_wav_text_language=language, # 风格文本语言（匹配）
    flow_matching_steps=16,
    use_global_guided_inference=False
)

# 保存生成的音频并提取mel
sf.write("temp/vevo_output.wav", gen_audio, 24000)
vevo_audio = librosa.load("temp/vevo_output.wav", sr=24000)[0]
vevo_audio_tensor = torch.from_numpy(vevo_audio).float().unsqueeze(0).to(device)

# 关键：使用相同的Vevo mel提取器
mel_neutral_vevo = vevo_pipeline.extract_mel_feature(vevo_audio_tensor)

# 验证格式一致性
assert mel_neutral_vevo.shape[-1] == 128  # 与源mel相同格式
assert mel_neutral_vevo.dim() == 3
```

#### **步骤2.5: 提取emotion2vec情感特征**
```python
# 从源音频提取情感特征
ev2_features = emotion2vec_extractor.extract_features(audio_tensor)

# 包含:
# - utterance: 语句级情感特征 [1, D]
# - frame: 帧级情感特征 [1, T, D]
```

#### **步骤2.6: 保存训练数据**
```python
# 保存源mel（Vevo兼容格式）
np.savez_compressed(
    f"{sample_id}_mel_original_vevo.npz",
    mel=mel_original_vevo.squeeze().cpu().numpy(),  # [T, 128]
    shape=mel_original_vevo.shape,
    format="vevo_compatible",
    hop_size=480,
    n_mels=128,
    sample_rate=24000,
    source_type="original_audio",
    language=language
)

# 保存中性mel（相同格式）
np.savez_compressed(
    f"{sample_id}_mel_neutral_vevo.npz", 
    mel=mel_neutral_vevo.squeeze().cpu().numpy(),   # [T, 128]
    shape=mel_neutral_vevo.shape,
    format="vevo_compatible", 
    hop_size=480,
    n_mels=128,
    sample_rate=24000,
    source_type="neutral_transformed",
    neutral_reference_source="emo_emilia",
    neutral_reference_language=neutral_language,
    language=language
)

# 保存情感特征
np.savez_compressed(
    f"{sample_id}_ev2.npz",
    utterance=ev2_features['utterance'].cpu().numpy(),
    frame=ev2_features['frame'].cpu().numpy(),
    language=language
)
```

## 📊 **具体示例：处理一个中文音频**

### **输入:**
```
sample_id: "zh_001234"
audio: [48000个采样点] @ 24kHz = 2秒音频
text: "具体而言，上半周持续上涨的逻辑在于部分资金交易供给进一步短缺的预期。"
language: "zh"
```

### **处理过程:**

#### **Step 1: 提取源mel**
```python
# 输入: 中文音频 [48000] @ 24kHz
audio_tensor = torch.tensor(audio).unsqueeze(0).to('cuda')  # [1, 48000]

# 使用Vevo mel提取器
mel_original_vevo = vevo_pipeline.extract_mel_feature(audio_tensor)
# 输出: [1, 100, 128] (T=100帧，hop_size=480，所以100*480=48000采样点)
```

#### **Step 2: 获取中文中性参考**
```python
# 从Emo-Emilia中文neutral样本中随机选择
neutral_ref = random.choice([
    ref for ref in neutral_reference_pool 
    if ref['language'] == 'zh'  # 严格匹配中文
])

# 例如选中:
neutral_audio = [72000个采样点] @ 24kHz = 3秒中文neutral音频
neutral_text = "这是一个中性的陈述关于市场情况。"
neutral_language = "zh"
```

#### **Step 3: Vevo TTS生成中性mel**
```python
# Vevo TTS调用
gen_audio = vevo_pipeline.inference_ar_and_fm(
    src_text="具体而言，上半周持续上涨的逻辑在于部分资金交易供给进一步短缺的预期。",  # 源文本
    style_ref_wav_path="neutral_zh_ref.wav",      # 中文中性参考音频
    style_ref_wav_text="这是一个中性的陈述关于市场情况。",  # 中文中性参考文本  
    timbre_ref_wav_path="source_zh.wav",          # 源音频作为音色参考
    src_text_language="zh",                       # 中文
    style_ref_wav_text_language="zh",             # 中文（匹配）
    # ... 其他参数
)

# 提取中性mel（使用相同的Vevo提取器）
neutral_audio_tensor = torch.tensor(gen_audio).unsqueeze(0).to('cuda')
mel_neutral_vevo = vevo_pipeline.extract_mel_feature(neutral_audio_tensor)
# 输出: [1, T', 128] (T'可能与T不同，但格式相同)
```

#### **Step 4: 提取情感特征**
```python
# 从源音频提取
ev2_features = emotion2vec_extractor.extract_features(audio_tensor)
# 输出: 
# - utterance: [1, 768] 语句级特征
# - frame: [1, T_frame, 768] 帧级特征
```

#### **Step 5: 保存训练数据**
```python
# 文件1: zh_001234_mel_original_vevo.npz
{
    'mel': [100, 128],           # 源音频mel
    'format': 'vevo_compatible',
    'hop_size': 480,
    'n_mels': 128,
    'language': 'zh',
    'source_type': 'original_audio'
}

# 文件2: zh_001234_mel_neutral_vevo.npz  
{
    'mel': [T', 128],            # 中性化mel
    'format': 'vevo_compatible',
    'hop_size': 480,
    'n_mels': 128,
    'language': 'zh',
    'source_type': 'neutral_transformed',
    'neutral_reference_source': 'emo_emilia',
    'neutral_reference_language': 'zh'  # 确认使用中文中性参考
}

# 文件3: zh_001234_ev2.npz
{
    'utterance': [768],          # 语句级情感特征
    'frame': [T_frame, 768],     # 帧级情感特征
    'language': 'zh'
}
```

## 🔧 **关键技术点**

### **1. 参数一致性保证**
```python
# 源mel和中性mel都使用相同的Vevo提取器
mel_original = vevo_pipeline.extract_mel_feature(source_audio)
mel_neutral = vevo_pipeline.extract_mel_feature(neutral_audio)

# 确保参数完全相同:
# - hop_size: 480 (不是BigVGAN的256)
# - n_mels: 128 (不是BigVGAN的100)  
# - sample_rate: 24000
# - 格式: [T, 128] (Vevo兼容)
```

### **2. 语言严格匹配**
```python
if source_language == 'zh':
    # 中文音频必须使用中文中性参考
    neutral_refs = [ref for ref in emo_emilia_pool if ref['language'] == 'zh']
    
elif source_language == 'en':
    # 英文音频必须使用英文中性参考
    neutral_refs = [ref for ref in emo_emilia_pool if ref['language'] == 'en']

selected_neutral = random.choice(neutral_refs)
assert selected_neutral['language'] == source_language  # 严格验证
```

### **3. 真实文本使用**
```python
# 不使用固定文本，使用真实文本
source_text = sample['text']           # 源音频的真实转录文本
neutral_text = neutral_ref['text']     # Emo-Emilia中性样本的真实文本

# Vevo TTS使用真实文本，避免重复问题
```

## 📊 **数据规模和分布**

### **输入数据（Emilia）:**
- **总时长**: 100小时 (英文50h + 中文50h)
- **样本数**: 约15,000-20,000个样本
- **平均时长**: 每个样本3-6秒

### **中性参考数据（Emo-Emilia）:**
- **总样本**: 1400个 (专家验证)
- **中性样本**: 约200个 (英文100个 + 中文100个)
- **使用方式**: 每个样本随机选择匹配语言的中性参考

### **输出训练数据:**
```
每个样本生成3个文件:
- sample_id_mel_original_vevo.npz    (源mel，Vevo兼容)
- sample_id_mel_neutral_vevo.npz     (中性mel，Vevo兼容)  
- sample_id_ev2.npz                  (情感特征)

总文件数: 约45,000-60,000个文件
总大小: 约50-100GB
```

## ✅ **质量保证检查点**

### **1. 格式一致性**
```python
assert mel_original.shape[-1] == 128
assert mel_neutral.shape[-1] == 128
assert mel_original.shape[0] == mel_neutral.shape[0] == 1
```

### **2. 语言匹配验证**
```python
assert neutral_ref_language == source_language
print(f"✅ {source_language.upper()} 音频 -> {neutral_ref_language.upper()} 中性参考")
```

### **3. Vevo兼容性验证**
```python
# 所有mel都可以直接用于Vevo声码器
reconstructed_audio = vevo_vocoder(mel.transpose(1, 2))
```

## 🎯 **最终训练数据用途**

生成的数据可用于训练：
1. **情感转换模型**: 学习从源mel+情感特征生成中性mel
2. **情感识别模型**: 使用emotion2vec特征训练分类器
3. **语音合成模型**: 使用Vevo兼容的mel进行训练

这个流程确保了：
- **✅ 参数完全一致** (源mel和中性mel)
- **✅ 语言严格匹配** (中文配中文，英文配英文)
- **✅ 高质量中性参考** (Emo-Emilia专家验证)
- **✅ Vevo声码器兼容** (可直接用于推理)

请确认这个流程是否符合你的预期？🤔
