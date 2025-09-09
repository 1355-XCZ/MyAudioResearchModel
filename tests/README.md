# 测试模块说明

这个目录包含了系统各个模块的测试脚本，用于验证功能正确性和调试问题。

## 测试文件结构

```
tests/
├── __init__.py                 # 测试模块初始化
├── test_data_processing.py     # 预处理模块测试
├── test_models.py              # 各个模型测试
├── test_pipeline.py            # 完整流水线测试
├── run_tests.py                # 统一测试入口
├── outputs/                    # 测试输出目录
└── README.md                   # 本说明文件
```

## 快速开始

### 1. 准备测试环境
```bash
# 安装依赖
pip install -r requirements.txt

# 创建测试音频目录
mkdir -p data/test

# 放置一个测试音频文件
# 将您的测试音频文件放到: data/test/sample.wav
```

### 2. 运行测试
```bash
# 运行统一测试入口
python tests/run_tests.py

# 或者单独运行特定测试
python tests/test_data_processing.py    # 预处理测试
python tests/test_models.py             # 模型测试
python tests/test_pipeline.py           # 流水线测试
```

## 测试说明

### test_data_processing.py - 预处理模块测试
测试内容：
- ✅ 音频加载功能
- ✅ Whisper语音识别
- ✅ Emotion2Vec情感特征提取
- ✅ 音素转换
- ✅ 完整预处理流程
- ✅ 数据集批量加载

### test_models.py - 模型测试
测试内容：
- ✅ 阶段A模型 (音素 → M0)
- ✅ 阶段B模型 (M0 + 情感 → M1/M2)
- ✅ VQ-VAE量化器
- ✅ 声码器 (Mel → 音频)

### test_pipeline.py - 流水线测试
测试内容：
- ✅ 流水线组件创建
- ✅ 端到端推理
- ✅ VQ-VAE码本实验
- ✅ 数据流完整性验证

## 测试输出

测试运行后会在 `tests/outputs/` 目录生成：
- 测试音频文件
- VQ-VAE实验结果
- 各阶段的中间输出

## 故障排除

### 常见问题

1. **模块导入错误**
   ```bash
   # 确保在项目根目录运行测试
   cd /path/to/MyAudioResearchModel
   python tests/run_tests.py
   ```

2. **依赖缺失**
   ```bash
   pip install -r requirements.txt
   pip install modelscope funasr  # Emotion2Vec支持
   ```

3. **测试音频缺失**
   ```bash
   # 将测试音频放到正确位置
   cp your_audio.wav data/test/sample.wav
   ```

4. **模型下载失败**
   - Whisper模型会自动下载到 `~/.cache/whisper/`
   - Emotion2Vec模型会自动下载到 `~/.cache/modelscope/`
   - 首次运行需要网络连接

## 测试策略

### 推荐的测试顺序
1. **快速测试** - 验证核心功能是否工作
2. **预处理测试** - 确保数据预处理正常
3. **模型测试** - 验证各个模型组件
4. **流水线测试** - 测试完整的端到端流程

### 调试建议
- 从简单测试开始，逐步增加复杂度
- 如果某个测试失败，先解决该问题再继续
- 查看详细的错误信息和堆栈跟踪
- 检查配置文件参数是否正确
