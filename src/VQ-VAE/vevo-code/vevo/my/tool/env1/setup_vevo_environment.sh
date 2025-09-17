#!/bin/bash

# VEVO项目环境配置脚本
# 使用方法: bash setup_vevo_environment.sh

set -e

echo "=== VEVO项目环境配置开始 ==="

# 检查Python版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 创建虚拟环境 (可选)
read -p "是否创建新的虚拟环境? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "创建虚拟环境 vevo_env..."
    python -m venv vevo_env
    source vevo_env/bin/activate
    echo "虚拟环境已激活"
fi

# 更新pip
echo "更新pip..."
pip install --upgrade pip setuptools wheel

# 检查并安装系统依赖
echo "=== 检查系统依赖 ==="

# 检查espeak-ng
if ! command -v espeak-ng &> /dev/null; then
    echo "警告: espeak-ng 未安装"
    echo "请根据您的系统安装:"
    echo "  Ubuntu/Debian: sudo apt-get install espeak-ng"
    echo "  macOS: brew install espeak"
    echo "  Windows: 请下载并安装 espeak-ng"
    read -p "是否继续? (y/n): " continue_setup
    if [ "$continue_setup" != "y" ]; then
        exit 1
    fi
fi

# 检查ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "警告: ffmpeg 未安装"
    echo "尝试使用conda安装ffmpeg..."
    if command -v conda &> /dev/null; then
        conda install -c conda-forge ffmpeg -y
    else
        echo "请手动安装 ffmpeg"
        echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
        echo "  macOS: brew install ffmpeg"
        echo "  Windows: 下载并安装 ffmpeg"
    fi
fi

# 安装核心依赖
echo "=== 安装核心深度学习框架 ==="
pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2

echo "=== 安装Transformers和相关库 ==="
pip install transformers==4.41.2 accelerate==0.24.1 diffusers safetensors

echo "=== 安装数值计算库 ==="
pip install numpy==1.26.0 scipy==1.12.0 pandas matplotlib scikit-learn

echo "=== 安装音频处理库 ==="
pip install librosa soundfile audiomentations pyworld praat-parselmouth
pip install diffsptk pysptk resampy soxr

echo "=== 安装编解码器 ==="
pip install encodec
pip install vocos speechtokenizer descript-audio-codec

echo "=== 安装语音识别库 ==="
pip install openai-whisper fairseq
pip install git+https://github.com/m-bain/whisperx.git

echo "=== 安装文本处理库 ==="
pip install phonemizer==3.2.1 g2p_en pypinyin==0.48.0
pip install jieba cn2an unidecode pyopenjtalk pykakasi

echo "=== 安装配置和工具库 ==="
pip install omegaconf hydra-core ruamel.yaml json5 easydict
pip install datasets huggingface_hub

echo "=== 安装评估指标库 ==="
pip install torchmetrics frechet_audio_distance
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip install pystoi pymcd mir_eval jiwer

echo "=== 安装训练工具 ==="
pip install pytorch-lightning tensorboard tensorboardX wandb

echo "=== 安装其他依赖 ==="
pip install tqdm loguru einops vector-quantize-pytorch
pip install gradio fastapi uvicorn
pip install black ruff Cython
pip install tgt typeguard humanfriendly munch
pip install nnAudio ptwt ffmpeg-python==0.2.0
pip install PyYAML ipython

echo "=== 安装特殊依赖 ==="
pip install git+https://github.com/lhotse-speech/lhotse

echo "=== 编译Cython模块 ==="
cd ../../../../../modules/monotonic_align
python setup.py build_ext --inplace
cd ../../models/vc/vevo/my/tool/env1

echo "=== 环境配置完成! ==="
echo ""
echo "请确认以下事项:"
echo "1. espeak-ng 已正确安装"
echo "2. ffmpeg 已正确安装"  
echo "3. CUDA环境配置正确 (如果使用GPU)"
echo ""
echo "测试安装:"
echo "python -c \"import torch; print('PyTorch版本:', torch.__version__)\""
echo "python -c \"import librosa; print('Librosa版本:', librosa.__version__)\""
echo "python -c \"import transformers; print('Transformers版本:', transformers.__version__)\""
echo ""
echo "如果遇到CUDA相关问题，可能需要运行:"
echo "pip uninstall nvidia-cublas-cu11"
