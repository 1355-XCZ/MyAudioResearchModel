# 🚀 快速部署和测试指南

## 📋 环境准备 (5分钟)

### 1. 创建Conda环境
```bash
# 创建Python 3.9环境
conda create -n audio_tts python=3.9 -y
conda activate audio_tts

# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install soundfile  # 音频文件处理（必需）
```

### 2. 项目结构检查
```bash
# 确保在项目根目录
ls -la
# 应该看到: src/ examples/ config/ requirements.txt

# 检查关键文件
ls src/models/
ls examples/
```

---

## ⚡ 快速功能测试 (2分钟)

### 测试1: 最快验证
```bash
python examples/quick_bypass_test.py
```
**预期结果**: 生成 `quick_test_output.wav` 文件

### 测试2: 完整流程测试
```bash
python examples/audio_to_audio_demo.py
```
选择选项2创建演示音频，然后测试完整流程

---

## 🎯 核心功能使用

### 文本转语音 (最简单)
```python
from src.models.stage_a import TTSStageAModel
from src.models.vocoder import VocoderFactory

# 创建模型
config = {
    'tts_architecture': 'fastspeech2',
    'model_config': {'d_model': 128, 'n_layers': 2, 'n_heads': 4, 'dropout': 0.1},
    'phoneme_vocab': {'source': 'pypinyin_auto'}
}
tts_model = TTSStageAModel(config)

# 文本转Mel频谱
text = "你好世界"
mel_result = tts_model.forward_from_text(text)

# Mel频谱转音频
vocoder = VocoderFactory.create_vocoder('hifigan', {'model_config': {}})
audio = vocoder.synthesize(mel_result.mel_spectrogram)

# 保存音频
import soundfile as sf
sf.write("output.wav", audio, vocoder.sampling_rate)
```

### 音频转音频 (完整流程)
```bash
# 使用现有音频文件
python examples/audio_to_audio_demo.py input.wav output.wav

# 或交互式运行
python examples/audio_to_audio_demo.py
```

---

## 🔧 可选高级功能

### 安装NVIDIA HiFi-GAN (更高音质)
```bash
pip install nemo_toolkit[all]
python examples/nvidia_hifigan_example.py
```

### 安装BigVGAN (最高音质)
```bash
pip install bigvgan
python examples/bigvgan_usage_example.py
```

---

## ✅ 成功标志

### 测试成功的标志:
- [ ] `quick_bypass_test.py` 运行无错误
- [ ] 生成音频文件可以播放
- [ ] 控制台显示绿色 ✅ 成功标志
- [ ] 音频内容是清晰的中文语音

### 如果遇到问题:
```bash
# 检查Python版本
python --version  # 应该是3.9

# 检查关键包
python -c "import torch; print('PyTorch OK')"
python -c "import soundfile; print('SoundFile OK')"

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

---

## 🎯 当前系统能力

### ✅ 已完成功能:
- 中文文本 → 语音合成
- 音频文件 → 文本提取 → 语音重建
- 6种不同质量声码器选择
- 完整的模块化架构

### 🔄 待开发功能:
- 情感处理 (阶段B)
- VQ-VAE情感量化训练
- 端到端情感音频建模

---

## 📞 故障排除

### 常见问题:
1. **导入错误**: 确保在项目根目录运行
2. **音频无声**: 检查soundfile是否安装
3. **模型加载慢**: 首次运行需下载模型，耐心等待
4. **CUDA错误**: 添加 `export CUDA_VISIBLE_DEVICES=""`

### 测试命令:
```bash
# 最小测试
python -c "
from src.models.stage_a import TTSStageAModel
print('✅ 阶段A导入成功')
from src.models.vocoder import VocoderFactory  
print('✅ 声码器导入成功')
print('🎉 基础组件正常')
"
```

---

## 🎉 部署完成

**成功标志**: 运行 `python examples/quick_bypass_test.py` 生成音频文件

**下一步**: 专注开发阶段B，基础架构已就绪！
