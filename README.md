# 情感音频建模研究项目

这是一个基于多阶段架构的情感音频建模研究项目，支持通过VQ-VAE码本量化实验来研究情感细节的保留程度。

## 项目架构

### 核心思想

本项目基于以下假设进行实验：
- 在不应用VQ-VAE码本或码本很大的情况下，重建的音频应该和原始音频相似
- 随着VQ-VAE码本大小K逐渐减小，细腻情感会逐渐丢失，只保留主要或基础情感

### 系统架构

```
原始音频 P
    ↓
数据处理阶段: Whisper提取文字 + Emotion2Vec提取情感特征
    ↓
内容音素 Pc + 情感特征 Pe
    ↓
阶段A: 音素 → M0 (中性Mel频谱)
    ↓
[可选] VQ-VAE量化层: Pe → Pe' (量化后的情感特征)
    ↓
阶段B1: (M0, Pe/Pe') → M1 (带情感的Mel频谱)
    ↓
[可选] 阶段B2: (M1, Pe/Pe') → M2 (扩散模型细化)
    ↓
声码器: Mel频谱 → 重建音频
```

### 关键设计特点

1. **VQ-VAE量化层位置**: 位于整个B阶段之前，对输入的emotion2vec表征进行量化
2. **B阶段分为两个子阶段**:
   - **B1**: 情感适配器，将M0和情感特征融合生成M1
   - **B2**: 扩散模型细化器，对M1进行进一步优化生成M2（可选）
3. **统一的情感输入**: B1和B2都使用相同的（量化或原始）情感表征
4. **可选开关**: VQ-VAE量化和B2阶段都可以通过配置开关控制

## 模块化设计

项目采用设计模式实现模块化架构，支持灵活切换不同的模型实现：

### 核心接口
- `DataProcessor`: 数据处理接口
- `StageAModel`: 阶段A模型接口 (音素→M0)
- `StageBModel`: 阶段B模型接口 (包含B1和B2)
- `EmotionQuantizer`: 情感量化器接口 (VQ-VAE)
- `Vocoder`: 声码器接口
- `Pipeline`: 完整流水线接口

### 当前实现
- **数据处理**: WhisperEmotionProcessor (Whisper + Emotion2Vec)
- **阶段A**: FastSpeech2StageA (可扩展其他TTS模型)
- **阶段B**: TwoStageEmotionModel (B1情感适配器 + B2扩散细化器)
- **情感量化**: VQVAEEmotionQuantizer (支持多种码本大小)
- **声码器**: HiFiGANVocoder, MelGANVocoder

## 安装和使用

### 环境要求
```bash
pip install -r requirements.txt
```

### 配置文件
编辑 `config/base_config.yaml` 来配置实验参数：

```yaml
# VQ-VAE情感量化器配置 (整个B阶段的前置处理)
emotion_quantizer:
  enabled: true  # 开关：是否启用VQ-VAE量化实验
  type: "vqvae"
  codebook_sizes: [64, 128, 256, 512, 1024]

# 阶段B配置 (包含B1和B2)
stage_b:
  model_type: "two_stage_emotion"
  model_config:
    enable_b2: true  # 是否启用B2阶段
    diffusion_steps: 20  # B2扩散步数
```

### 运行训练
```python
from src.pipeline.training_pipeline import EmotionAudioPipeline

# 创建流水线
pipeline = EmotionAudioPipeline('config/base_config.yaml')

# 训练模型
pipeline.train(pipeline.config)
```

### 运行推理
```python
from src.data_processing.audio_processor import AudioLoader

# 加载测试音频
audio_loader = AudioLoader()
test_audio = audio_loader.load_audio('path/to/test.wav')

# 不同模式的推理
# 1. 基线推理（不量化，仅B1）
baseline_audio = pipeline.inference(test_audio, use_quantizer=False, use_b2=False)

# 2. 使用VQ-VAE量化（B1）
quantized_audio = pipeline.inference(test_audio, use_quantizer=True, codebook_size=256, use_b2=False)

# 3. 使用B2扩散细化
refined_audio = pipeline.inference(test_audio, use_quantizer=False, use_b2=True)

# 4. 完整流水线（量化 + B2）
full_audio = pipeline.inference(test_audio, use_quantizer=True, codebook_size=256, use_b2=True)
```

### VQ-VAE码本实验
```python
# 运行完整的码本大小对比实验
vq_results = pipeline.run_vq_vae_experiment(test_audio)

# 结果包含：
# - baseline: 不使用量化的结果
# - k64, k128, k256, k512, k1024: 不同码本大小的结果
# - 每个结果都包含B1和B2阶段的输出
```

## 实验设计

### VQ-VAE码本大小实验

项目支持系统性地测试不同VQ-VAE码本大小对情感保留的影响：

1. **基线对比**: 不使用量化的原始结果
2. **大码本** (K=1024): 应该能保留细腻的情感细节
3. **中等码本** (K=256, K=512): 保留主要情感，丢失部分细节
4. **小码本** (K=64, K=128): 只保留基础情感特征

### 实验输出

运行实验后，系统会自动生成：

```
outputs/
├── baseline_b1.wav          # 基线B1结果
├── baseline_b2.wav          # 基线B2结果
├── vq_k64_b1.wav           # 码本64的B1结果
├── vq_k64_b2.wav           # 码本64的B2结果
├── vq_k128_b1.wav          # 码本128的B1结果
├── vq_k128_b2.wav          # 码本128的B2结果
├── ...                     # 其他码本大小
└── vq_vae_experiment_report.txt  # 实验报告
```

### 评估指标

- **Mel损失**: 重建Mel频谱与原始Mel频谱的差异
- **VQ损失**: 量化过程的重建损失
- **情感相似度**: 重建音频与原始音频的情感特征相似度
- **内容保留度**: 语音内容的保留程度

## 数据集

### 训练数据
- 支持大量无标签音频数据
- 自动使用Whisper提取文本内容
- 使用Emotion2Vec提取情感特征

### 测试数据
- 推荐使用ESD情感语音数据集
- 支持有标签的情感数据进行定量评估

## 扩展性

项目设计支持轻松扩展新的模型实现：

### 添加新的阶段A模型
```python
from src.core.interfaces import StageAModel

class NewTTSModel(StageAModel):
    def forward(self, phonemes: List[str]) -> ModelOutput:
        # 实现新的TTS模型
        pass
```

### 添加新的阶段B模型
```python
from src.core.interfaces import StageBModel

class NewEmotionModel(StageBModel):
    def forward_b1(self, mel_input: torch.Tensor, emotion_features: np.ndarray) -> ModelOutput:
        # 实现新的B1模型
        pass
    
    def forward_b2(self, mel_input: torch.Tensor, emotion_features: np.ndarray) -> ModelOutput:
        # 实现新的B2模型
        pass
```

### 添加新的量化器
```python
from src.core.interfaces import EmotionQuantizer

class NewQuantizer(EmotionQuantizer):
    def quantize(self, emotion_features: np.ndarray, codebook_size: int) -> Tuple[np.ndarray, float]:
        # 实现新的量化方法
        pass
```

## 项目结构

```
MyAudioResearchModel/
├── config/
│   └── base_config.yaml          # 配置文件
├── src/
│   ├── core/
│   │   └── interfaces.py         # 核心接口定义
│   ├── data_processing/
│   │   └── audio_processor.py    # 音频数据处理
│   ├── models/
│   │   ├── stage_a.py           # 阶段A模型实现
│   │   ├── stage_b.py           # 阶段B模型实现（B1+B2）
│   │   ├── emotion_quantizer.py # VQ-VAE情感量化器
│   │   └── vocoder.py           # 声码器实现
│   ├── pipeline/
│   │   └── training_pipeline.py # 训练和推理流水线
│   └── utils/
│       └── model_factory.py     # 模型工厂
├── examples/
│   └── basic_usage.py           # 使用示例
├── requirements.txt              # 依赖包列表
└── README.md                    # 项目说明
```

## 使用示例

### 基础推理示例
```python
# 运行基础示例
python examples/basic_usage.py
```

### 自定义实验
```python
from src.pipeline.training_pipeline import EmotionAudioPipeline

# 创建流水线
pipeline = EmotionAudioPipeline('config/base_config.yaml')

# 加载测试音频
test_audio = pipeline.dataset_loader.load_test_data()[0]

# 对比实验：测试量化的影响
results = {}

# 1. 基线（无量化，仅B1）
results['baseline_b1'] = pipeline.inference(test_audio, use_quantizer=False, use_b2=False)

# 2. 基线（无量化，B1+B2）
results['baseline_b2'] = pipeline.inference(test_audio, use_quantizer=False, use_b2=True)

# 3. 不同码本大小的量化实验
for k in [64, 128, 256, 512, 1024]:
    results[f'vq_k{k}_b1'] = pipeline.inference(test_audio, use_quantizer=True, codebook_size=k, use_b2=False)
    results[f'vq_k{k}_b2'] = pipeline.inference(test_audio, use_quantizer=True, codebook_size=k, use_b2=True)

# 保存结果进行对比分析
for name, audio in results.items():
    pipeline._save_audio(audio, f'outputs/{name}.wav', test_audio.sample_rate)
```

## 核心创新点

1. **分层量化设计**: VQ-VAE量化作为B阶段的前置处理，而不是独立的后处理步骤
2. **两阶段情感建模**: B1负责基础情感融合，B2负责细节优化
3. **统一情感输入**: 确保B1和B2使用一致的情感表征
4. **可控实验设计**: 通过开关控制不同组件的启用，便于对比实验
5. **模块化架构**: 支持灵活替换各个组件，便于模型迭代

## 开发计划

- [x] 完成核心架构设计
- [x] 实现VQ-VAE情感量化器
- [x] 实现两阶段B模型（B1+B2）
- [x] 实现完整的实验流水线
- [ ] 完善emotion2vec模型集成
- [ ] 添加更多TTS模型支持
- [ ] 实现更多评估指标
- [ ] 添加可视化分析工具
- [ ] 支持分布式训练
- [ ] 添加Web界面演示

## 贡献

欢迎提交Issue和Pull Request来改进项目！

## 许可证

本项目采用MIT许可证。