# VEVO项目环境配置指南

## 概述

本指南帮助您配置运行VEVO项目所需的完整环境。VEVO是一个零样本语音模仿框架，需要多个深度学习和音频处理库的支持。

## 系统要求

- **Python**: >= 3.9
- **操作系统**: Linux/macOS/Windows
- **GPU**: 建议使用CUDA兼容的GPU (可选但推荐)
- **内存**: 建议16GB以上
- **存储空间**: 建议20GB以上可用空间

## 快速开始

### 方法1: 使用自动化脚本

**Linux/macOS:**
```bash
chmod +x setup_vevo_environment.sh
./setup_vevo_environment.sh
```

**Windows:**
```cmd
setup_vevo_environment.bat
```

### 方法2: 手动安装

1. **创建虚拟环境**
   ```bash
   python -m venv vevo_env
   source vevo_env/bin/activate  # Linux/macOS
   # 或
   vevo_env\Scripts\activate     # Windows
   ```

2. **安装最小依赖**
   ```bash
   pip install -r requirements_minimal.txt
   ```

3. **安装完整依赖**
   ```bash
   pip install -r vevo_complete_requirements.txt
   ```

## 系统依赖安装

### 1. espeak-ng (必需)
用于文本到语音的音素转换。

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install espeak-ng
```

**macOS:**
```bash
brew install espeak
```

**Windows:**
- 下载并安装 [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases)
- 将安装路径添加到系统PATH

### 2. ffmpeg (必需)
用于音频文件处理。

**使用Conda (推荐):**
```bash
conda install -c conda-forge ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
- 下载 [ffmpeg](https://ffmpeg.org/download.html)
- 解压并将bin目录添加到PATH

## 核心依赖说明

### 深度学习框架
- **PyTorch 2.0.1**: 主要深度学习框架
- **Transformers 4.41.2**: HuggingFace模型库
- **Accelerate**: 分布式训练支持

### 音频处理
- **librosa**: 音频分析和处理
- **soundfile**: 音频文件读写
- **encodec**: Facebook的神经音频编解码器
- **pyworld**: F0提取和语音分析

### 文本处理
- **phonemizer**: 文本到音素转换
- **g2p_en**: 英文音素转换
- **pypinyin**: 中文拼音转换

### 语音识别
- **openai-whisper**: OpenAI的语音识别模型
- **fairseq**: Facebook的序列建模工具包

## 特殊安装说明

### 1. 从Git安装的包
```bash
# WhisperX (改进的Whisper)
pip install git+https://github.com/m-bain/whisperx.git

# Lhotse (语音数据处理)
pip install git+https://github.com/lhotse-speech/lhotse

# PESQ评估指标
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
```

### 2. 编译Cython模块
```bash
cd ../../../../../modules/monotonic_align
python setup.py build_ext --inplace
cd ../../models/vc/vevo/my/tool/env1
```

## 常见问题解决

### 1. CUDA版本冲突
如果遇到CUDA相关错误，尝试：
```bash
pip uninstall nvidia-cublas-cu11
```

### 2. phonemizer安装失败
确保已安装espeak-ng，然后：
```bash
pip install phonemizer --no-cache-dir
```

### 3. fairseq安装问题
```bash
pip install fairseq --no-deps
# 然后手动安装fairseq的依赖
```

### 4. 内存不足
如果在安装或运行时遇到内存问题：
- 关闭其他应用程序
- 使用交换文件
- 考虑使用更小的批量大小

## 验证安装

运行以下命令验证关键库是否正确安装：

```python
# 测试脚本
import torch
import librosa
import transformers
import encodec
import whisper

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"Librosa版本: {librosa.__version__}")
print(f"Transformers版本: {transformers.__version__}")
print("所有核心库安装成功!")
```

## 模型下载

VEVO需要预训练模型，这些模型会在首次运行时自动下载：

1. **HuggingFace模型**: 通过transformers库自动下载
2. **Whisper模型**: 通过openai-whisper自动下载
3. **VEVO预训练模型**: 从HuggingFace Hub下载

确保网络连接良好，模型下载可能需要较长时间。

## 性能优化建议

### GPU设置
```python
# 检查GPU
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("使用CPU")
```

### 内存优化
- 使用混合精度训练
- 适当调整batch size
- 启用梯度累积

## 开发环境设置

如果您计划修改代码，建议安装开发工具：

```bash
# 代码格式化
pip install black isort

# 类型检查
pip install mypy

# 代码质量检查
pip install flake8 pylint
```

## 故障排除

### 日志查看
大多数错误信息会在控制台输出，注意查看：
- 导入错误
- 模型加载错误
- CUDA相关错误

### 环境隔离
强烈建议使用虚拟环境避免包冲突：
```bash
# 创建新环境
python -m venv vevo_clean_env
source vevo_clean_env/bin/activate
```

## 更新和维护

定期更新依赖包：
```bash
pip list --outdated
pip install --upgrade package_name
```

## 支持和帮助

如果遇到问题：
1. 检查GitHub Issues
2. 查看项目文档
3. 确认系统依赖是否正确安装
4. 验证Python和包版本兼容性

---

**注意**: 环境配置可能因系统差异而有所不同，请根据实际情况调整。建议在配置前备份现有环境。
