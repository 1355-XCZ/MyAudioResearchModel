#!/bin/bash

# VEVO项目环境配置脚本 (修复版)
# 修复了版本冲突和安装顺序问题

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

# 安装核心依赖 - 按正确顺序安装避免版本冲突
echo "=== 安装核心深度学习框架 (指定兼容版本) ==="
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

echo "=== 安装Transformers和相关库 ==="
pip install transformers==4.41.2 accelerate==0.24.1 diffusers safetensors

echo "=== 安装数值计算库 ==="
pip install numpy==1.26.0 scipy==1.12.0 pandas matplotlib scikit-learn

echo "=== 安装配置库 (指定兼容版本) ==="
pip install omegaconf==2.3.0 hydra-core ruamel.yaml json5 easydict

echo "=== 安装音频处理库 ==="
pip install librosa soundfile audiomentations pyworld praat-parselmouth
pip install diffsptk==1.0.1 pysptk resampy soxr
pip install nnAudio ptwt ffmpeg-python==0.2.0

echo "=== 安装编解码器 ==="
pip install encodec
pip install vocos speechtokenizer descript-audio-codec

echo "=== 安装文本处理库 ==="
pip install phonemizer==3.2.1 g2p_en pypinyin==0.48.0
pip install jieba cn2an unidecode pyopenjtalk pykakasi
pip install tgt typeguard humanfriendly munch

echo "=== 安装数据处理库 ==="
pip install datasets huggingface_hub

echo "=== 安装评估指标库 ==="
pip install torchmetrics frechet_audio_distance
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip install pystoi pymcd mir_eval jiwer

echo "=== 安装训练工具 ==="
pip install tensorboard tensorboardX wandb

echo "=== 安装语音识别库 ==="
pip install openai-whisper

echo "=== 安装fairseq (可能需要较长时间) ==="
echo "尝试从git安装fairseq..."
if ! pip install git+https://github.com/pytorch/fairseq.git; then
    echo "Git安装失败，尝试备选方法..."
    pip install fairseq --no-deps
    echo "fairseq已安装(无依赖模式)"
fi

echo "=== 安装其他依赖 ==="
pip install tqdm loguru einops vector-quantize-pytorch==1.12.5
pip install gradio fastapi uvicorn
pip install black==24.1.1 ruff Cython
pip install PyYAML ipython onnxruntime

echo "=== 安装特殊依赖 ==="
pip install git+https://github.com/lhotse-speech/lhotse
pip install git+https://github.com/m-bain/whisperx.git

echo "=== 编译Cython模块 ==="
cd ../../../../../modules/monotonic_align
if [ -f "setup.py" ]; then
    python setup.py build_ext --inplace
    echo "✅ Cython模块编译完成"
else
    echo "⚠️  警告: 找不到monotonic_align/setup.py"
fi
cd ../../models/vc/vevo/my/tool/env1

echo "=== 验证安装 ==="
python -c "
try:
    import torch
    print(f'✅ PyTorch版本: {torch.__version__}')
    import transformers  
    print(f'✅ Transformers版本: {transformers.__version__}')
    import librosa
    print(f'✅ Librosa版本: {librosa.__version__}')
    from huggingface_hub import snapshot_download
    print('✅ huggingface_hub导入成功')
    import whisper
    print('✅ OpenAI Whisper导入成功')
    print('🎉 所有核心库验证通过!')
except ImportError as e:
    print(f'❌ 验证失败: {e}')
"

echo "=== 环境配置完成! ==="
echo ""
echo "✅ 修复内容:"
echo "1. 使用兼容的PyTorch版本组合"
echo "2. 修复omegaconf版本冲突"  
echo "3. 改进fairseq安装流程"
echo "4. 优化包安装顺序"
echo ""
echo "请确认以下事项:"
echo "1. espeak-ng 已正确安装"
echo "2. ffmpeg 已正确安装"  
echo "3. CUDA环境配置正确 (如果使用GPU)"
echo ""
echo "如果遇到问题，可以运行修复脚本:"
echo "./fix_environment.sh"
