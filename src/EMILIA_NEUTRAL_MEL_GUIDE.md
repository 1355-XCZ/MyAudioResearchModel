# Emilia 数据集中性频谱图生成指南

## 📋 概述

这个模块用于从 [Emilia 数据集](https://huggingface.co/datasets/amphion/Emilia-Dataset) 生成大量中性频谱图，用于训练 FlowSE-Emo2Vec 模型。

## 🔧 Vevo TTS 模式配置确认

### 📊 关键参数匹配

根据 Vevo 的配置文件分析，确定了以下关键参数：

| 参数 | Vevo 配置 | FlowSE 配置 | 说明 |
|------|-----------|-------------|------|
| **sample_rate** | 24000 | 24000 | ✅ 采样率匹配 |
| **n_fft** | 1920 | 1024 | ⚠️ 需要调整 |
| **hop_size** | 480 | 256 | ⚠️ 需要调整 |
| **win_size** | 1920 | 1024 | ⚠️ 需要调整 |
| **num_mels** | 128 | 100 | ⚠️ 需要调整 |
| **fmax** | 12000 | - | ✅ 新增配置 |

### 🎯 推荐的统一配置

为了确保 Vevo → FlowSE → Vocoder 的完整兼容性，建议使用以下配置：

```yaml
# 统一的 mel 频谱图配置
mel_config:
  sample_rate: 24000     # 24kHz 采样率
  n_fft: 1920           # Vevo 的 FFT 大小
  hop_length: 480       # Vevo 的 hop 大小  
  win_length: 1920      # Vevo 的窗口大小
  n_mel_channels: 128   # Vevo 的 mel 通道数
  f_min: 0              # 最小频率
  f_max: 12000          # 最大频率 (采样率的一半)
  power: 2.0            # 功率谱
  normalized: false     # 不标准化
```

## 🚀 使用方法

### 1. 环境准备

```bash
# 安装依赖
pip install datasets huggingface_hub torchaudio librosa soundfile

# 登录 Hugging Face (需要访问 Emilia 数据集)
huggingface-cli login
```

### 2. 基本使用

```python
from emilia_neutral_mel_generator import EmiliaNeutralMelGenerator

# 创建生成器
generator = EmiliaNeutralMelGenerator(
    output_dir="data/emilia_neutral_mels",
    balance_languages=True,
    max_samples_per_lang=10000
)

# 运行生成流程
generator.run()
```

### 3. 自定义配置

```python
# 自定义 Vevo 配置
custom_vevo_config = {
    "sample_rate": 24000,
    "hop_size": 480,
    "n_fft": 1920,
    "win_size": 1920,
    "num_mels": 128,
    "fmin": 0,
    "fmax": 12000,
    "mel_mean": -4.92,
    "mel_var": 8.14,
    "max_length": 36000,
    "min_length": 2400,
}

generator = EmiliaNeutralMelGenerator(
    vevo_config=custom_vevo_config
)
```

## 📁 输出结构

```
data/emilia_neutral_mels/
├── EN/                          # 英文样本
│   ├── EN_000001.npy           # mel 频谱图
│   ├── EN_000001.json          # 元数据
│   └── ...
├── ZH/                          # 中文样本  
│   ├── ZH_000001.npy
│   ├── ZH_000001.json
│   └── ...
└── file_lists/                  # 文件列表
    ├── train.json              # 训练集列表
    ├── val.json                # 验证集列表
    └── stats.json              # 统计信息
```

## 📊 数据格式

### Mel 频谱图 (.npy)
- 形状: `[128, frames]` (num_mels, time_frames)
- 数据类型: `float32`
- 范围: 标准化后的 log-mel 值

### 元数据 (.json)
```json
{
  "id": "EN_000001",
  "language": "EN",
  "text": "示例文本内容",
  "duration": 1.23,
  "mel_shape": [128, 59],
  "original_duration": 1.25,
  "speaker": "EN_B00000_S00000",
  "vevo_config": {...}
}
```

## 🔗 与音码器的匹配

### Vocos 声码器匹配
根据 FlowSE 使用的 `vocos-mel-24khz` 配置：

```yaml
# 需要确保的参数匹配
vocos_config:
  sample_rate: 24000
  n_fft: 1920      # 与 Vevo 一致
  hop_length: 480  # 与 Vevo 一致  
  n_mel_channels: 128  # 与 Vevo 一致
```

### 更新 FlowSE 配置
需要修改 `src/FlowSE-emo2vec/config/train.yaml`:

```yaml
model:
  mel_spec:
    target_sample_rate: 24000
    n_mel_channels: 128      # 从 100 改为 128
    hop_length: 480          # 从 256 改为 480
    win_length: 1920         # 从 1024 改为 1920
    n_fft: 1920             # 从 1024 改为 1920
    f_min: 0                # 新增
    f_max: 12000            # 新增
```

## 📈 数据平衡策略

### 中英文平衡
- **英文 (EN)**: 目标 10,000 样本
- **中文 (ZH)**: 目标 10,000 样本
- **总计**: 20,000 个平衡样本

### 质量过滤
- 最小长度: 0.1秒 (2,400 samples at 24kHz)
- 最大长度: 1.5秒 (36,000 samples at 24kHz)
- 音频标准化: 防止过载和失真
- 元数据完整性检查

## 🎯 与训练流程的集成

### 1. 生成中性频谱图
```bash
python src/emilia_neutral_mel_generator.py
```

### 2. 准备 emotion2vec 特征
```python
# 从原始音频提取 emotion2vec 特征
# (需要单独实现)
```

### 3. 训练 FlowSE-Emo2Vec
```bash
cd src/FlowSE-emo2vec
python train.py -conf config/train.yaml
```

## ⚠️ 重要注意事项

### 1. 配置一致性
确保整个管道中的 mel 频谱图参数完全一致：
- Vevo 生成中性频谱
- FlowSE 处理频谱  
- Vocoder 重建音频

### 2. 数据许可
- Emilia 数据集使用 `CC BY-NC 4.0` 许可
- 仅可用于非商业研究用途
- 需要同意数据集的使用条款

### 3. 存储需求
- 每个 mel 频谱图约 50KB
- 20,000 样本约需要 1GB 存储空间
- 建议使用 SSD 以提高 I/O 性能

### 4. 内存优化
- 使用流式加载避免内存溢出
- 批处理大小根据可用内存调整
- 及时释放不需要的数据

## 🛠️ 故障排除

### 常见问题

1. **HuggingFace 访问权限**
   ```bash
   huggingface-cli login
   # 输入你的 HF token
   ```

2. **内存不足**
   ```python
   # 减少 max_samples_per_lang
   generator = EmiliaNeutralMelGenerator(max_samples_per_lang=1000)
   ```

3. **配置不匹配**
   - 检查 Vevo, FlowSE, Vocoder 的参数一致性
   - 确保采样率和频谱图维度匹配

这个模块为您提供了一个完整的解决方案，用于从 Emilia 数据集生成与 Vevo 兼容的中性频谱图！
