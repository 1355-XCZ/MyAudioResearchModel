# 🚀 使用指南

## 快速开始

### 生成增强数据集 (推荐)
```bash
cd src/emilia_mel_generator

# 基本使用 (50小时中英文 × 5倍增强)
python main.py --generate

# 自定义参数
python main.py --generate --target-hours 25 --k-variants 3 --output-dir "data/my_dataset"
```

### Python API
```python
from src.emilia_mel_generator import CorrectedDataGenerator

generator = CorrectedDataGenerator(
    target_hours_per_lang=50,  # S1: 50小时/语言
    k_neutral_variants=5       # K=5个中性变体
)

result = generator.run()
```

## 📊 数据说明

### 输入数据源
- **S1**: 50小时中英文原始音频 (Emilia主数据集) - 提供内容和音色
- **S2**: ~200个neutral音频 (Emo-Emilia) - 提供中性风格参考

### Vevo TTS 生成策略 ⭐
```python
# 关键：风格参考使用s2，音色参考使用s1
neutral_mel = vevo_tts.generate(
    content=extract_content(s1),     # s1的内容
    timbre=extract_timbre(s1),       # s1的音色 (保持说话人特色)
    style=extract_style(s2),         # s2的风格 (中性表达方式)
    emotion=None                     # 去除情感
)
```

### 输出训练数据
- **格式**: (mel_原始, mel_中性, ev2_表征) 元组
- **数量**: ~300,000个 (60k × 5)
- **管理**: CSV文件 + 元数据

### 存储需求
- **总计**: ~50-80GB
- **推荐**: 100GB+ SSD

## 🔧 配置

### 关键参数
```yaml
sample_rate: 24000      # 24kHz
n_mel_channels: 128     # 128通道
hop_length: 480         # 50Hz帧率
k_variants: 5           # 5倍增强
```

### 集成要点
- **Vevo TTS**: 需要集成真实模型生成中性mel
- **Emotion2Vec**: 需要集成真实模型提取特征

## 📁 输出文件

```
data/user_specified_dataset/
├── EN/ZH/                          # 语言目录
│   ├── *_mel_original.npy         # 原始mel (目标)
│   ├── *_mel_neutral.npy          # 中性mel (输入)
│   ├── *_ev2_features.npz         # emotion2vec (条件)
│   └── *_metadata.json            # 元数据
└── csv_lists/
    ├── train_tuples.csv           # 训练集
    ├── val_tuples.csv             # 验证集
    └── all_tuples.csv             # 完整列表
```

完成后可直接用于 FlowSE-Emo2Vec 训练！
