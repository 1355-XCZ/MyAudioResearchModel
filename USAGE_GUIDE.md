ce's# 🚀 使用指南

## 📖 快速开始

### **完整流水线测试**

运行完整的流水线测试，验证B阶段透传功能：

```bash
python test_complete_pipeline.py
```

**输入要求**：
- ✅ **只支持音频输入**：`.wav`, `.mp3`, `.flac`, `.ogg` 格式
- ✅ **文字必须从音频中提取**：使用Whisper进行语音识别

### **测试音频文件**

脚本会自动检测以下音频文件：
- `test_audio.wav` （推荐的测试音频）
- `test.wav`
- `sample.wav` 
- `demo.wav`

如果检测到音频文件，会自动进行音频输入测试。

---

## 🔄 完整处理流程

**唯一支持的完整流程**：
```
音频文件 → Whisper(文字提取) → Emotion2Vec(情感特征) → 阶段A → BigVGAN提取Mel → 阶段B(透传) → BigVGAN合成 → 最终音频
```

**流程说明**：
1. **预处理阶段**：从音频中提取文字和情感特征
2. **阶段A**：使用提取的文字生成中性音频和Mel
3. **BigVGAN提取**：从音频提取标准化Mel频谱
4. **阶段B**：处理情感（当前为透传模式）
5. **BigVGAN合成**：生成最终音频

---

## 🎯 测试验证内容

### **✅ 基础架构验证**
- 预处理阶段正常工作（Whisper + Emotion2Vec）
- 阶段A可以从文本生成音频和Mel
- BigVGAN可以提取和合成Mel频谱
- 参数一致性得到保证

### **✅ B阶段透传验证**
- B阶段即使为空也不影响整体流程
- Mel频谱完美透传（无修改）
- 端到端流程完全跑通

### **✅ 模块化架构验证**
- 各组件独立工作
- 支持灵活的输入模式
- 错误处理和降级机制

---

## 📁 输出结果

测试完成后，结果保存在：
- `outputs/complete_pipeline_test/` - 测试音频和元数据
- `outputs/complete_pipeline_test/test_report.json` - 详细测试报告

**生成的文件**：
- `*_stage_a_audio.wav` - 阶段A生成的音频
- `*_final_audio.wav` - 最终合成的音频
- `*_metadata.json` - 处理过程的详细信息

---

## ⚙️ 配置说明

### **预处理器配置**
- **Whisper模型**: `base` (可选: `tiny`, `small`, `medium`, `large`)
- **语言**: `zh` (中文)
- **Emotion2Vec**: `iic/emotion2vec_base`

### **BigVGAN配置**
- **版本**: 22kHz (80通道Mel频谱)
- **采样率**: 22050 Hz
- **Hop长度**: 256

---

## 🔧 故障排除

### **常见问题**

1. **预处理器初始化失败**
   - ⚠️ 会自动降级到简化模式
   - 使用默认文本和模拟情感特征

2. **BigVGAN不可用**
   - ⚠️ 会使用简化的Mel处理
   - 建议安装: `pip install bigvgan`

3. **音频文件无法加载**
   - 检查文件格式是否支持
   - 确保文件路径正确

### **依赖安装**

```bash
# 基础依赖
pip install torch numpy librosa soundfile

# 预处理依赖
pip install whisper transformers

# BigVGAN依赖  
pip install bigvgan

# Emotion2Vec依赖
pip install modelscope
```

---

## 🎊 成功标准

测试成功的标准：
- ✅ 成功率 ≥ 75%
- ✅ B阶段透传验证通过
- ✅ 参数一致性验证通过
- ✅ 至少一种输入模式工作正常

**成功后可以开始开发B阶段的具体实现！**

---

## 🚀 下一步

1. **开发B阶段**: 基础架构已稳定，可以专注B阶段逻辑
2. **训练模型**: 实现B1和B2的训练逻辑
3. **VQ-VAE训练**: 完善码本学习机制
4. **性能优化**: GPU加速和批处理优化
