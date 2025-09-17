#!/bin/bash

# VEVO项目环境配置脚本 (Spartan集群专用版)
# 适配学校集群环境，避免修改系统conda

set -e

echo "=== VEVO项目环境配置开始 (Spartan集群版) ==="

# 检查Python版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 设置环境路径
ENV_BASE_DIR="/data/gpfs/projects/punim2341/haoguangzhou/envs"
ENV_NAME="vevo_env"
ENV_PATH="$ENV_BASE_DIR/$ENV_NAME"

echo "环境将创建在: $ENV_PATH"

# 询问是否创建新环境
read -p "是否在 $ENV_PATH 创建新的VEVO环境? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    # 如果环境已存在，询问是否删除
    if [ -d "$ENV_PATH" ]; then
        echo "环境 $ENV_PATH 已存在"
        read -p "是否删除现有环境并重新创建? (y/n): " recreate
        if [ "$recreate" = "y" ]; then
            echo "删除现有环境..."
            rm -rf "$ENV_PATH"
        else
            echo "使用现有环境..."
        fi
    fi
    
    # 创建环境
    if [ ! -d "$ENV_PATH" ]; then
        echo "创建虚拟环境 $ENV_PATH..."
        python -m venv "$ENV_PATH"
    fi
    
    # 激活环境
    source "$ENV_PATH/bin/activate"
    echo "虚拟环境已激活: $ENV_PATH"
else
    echo "使用当前环境"
fi

# 更新pip
echo "更新pip..."
pip install --upgrade pip setuptools wheel

# 检查系统依赖 - 集群版本
echo "=== 检查系统依赖 (集群环境) ==="

# 检查espeak-ng (集群可能已安装)
if ! command -v espeak-ng &> /dev/null; then
    echo "⚠️  espeak-ng 未找到"
    echo "在Spartan集群上，可能需要加载模块或联系管理员安装"
    echo "暂时跳过，继续安装Python包..."
else
    echo "✅ espeak-ng 已可用"
fi

# 检查ffmpeg (集群可能已安装)
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg 未找到"
    echo "在Spartan集群上，可能需要加载模块:"
    echo "  module avail ffmpeg"
    echo "  module load ffmpeg"
    echo "暂时跳过，继续安装Python包..."
else
    echo "✅ ffmpeg 已可用"
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
pip install librosa soundfile audiomentations pyworld
# praat-parselmouth 可能需要系统依赖，先跳过
echo "尝试安装 praat-parselmouth..."
if ! pip install praat-parselmouth; then
    echo "⚠️  praat-parselmouth 安装失败，跳过"
fi

pip install diffsptk==1.0.1 pysptk resampy soxr
pip install nnAudio ptwt

# ffmpeg-python 可能需要系统ffmpeg，先尝试安装
echo "尝试安装 ffmpeg-python..."
if ! pip install ffmpeg-python==0.2.0; then
    echo "⚠️  ffmpeg-python 安装失败，跳过"
fi

echo "=== 安装编解码器 ==="
pip install encodec
pip install vocos speechtokenizer descript-audio-codec

echo "=== 安装文本处理库 ==="
# phonemizer 需要espeak-ng，先尝试安装
echo "尝试安装 phonemizer..."
if ! pip install phonemizer==3.2.1; then
    echo "⚠️  phonemizer 安装失败，可能需要espeak-ng"
fi

pip install g2p_en pypinyin==0.48.0
pip install jieba cn2an unidecode

# 这些包可能需要额外系统依赖
echo "尝试安装其他文本处理库..."
pip install pyopenjtalk || echo "⚠️  pyopenjtalk 安装失败"
pip install pykakasi || echo "⚠️  pykakasi 安装失败"
pip install tgt typeguard humanfriendly munch

echo "=== 安装数据处理库 ==="
pip install datasets huggingface_hub

echo "=== 安装评估指标库 ==="
pip install torchmetrics frechet_audio_distance
echo "安装 python-pesq..."
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
    if ! pip install fairseq --no-deps; then
        echo "⚠️  fairseq 安装失败，跳过"
    else
        echo "fairseq已安装(无依赖模式)"
    fi
fi

echo "=== 安装其他依赖 ==="
pip install tqdm loguru einops vector-quantize-pytorch==1.12.5
pip install gradio fastapi uvicorn
pip install black==24.1.1 ruff Cython
pip install PyYAML ipython onnxruntime

echo "=== 安装特殊依赖 ==="
echo "安装 lhotse..."
pip install git+https://github.com/lhotse-speech/lhotse || echo "⚠️  lhotse 安装失败"

echo "安装 whisperx..."
pip install git+https://github.com/m-bain/whisperx.git || echo "⚠️  whisperx 安装失败"

echo "=== 编译Cython模块 ==="
# 获取当前脚本的绝对路径，然后计算到项目根目录的路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
MONOTONIC_ALIGN_DIR="$PROJECT_ROOT/modules/monotonic_align"

echo "项目根目录: $PROJECT_ROOT"
echo "Monotonic align目录: $MONOTONIC_ALIGN_DIR"

if [ -f "$MONOTONIC_ALIGN_DIR/setup.py" ]; then
    cd "$MONOTONIC_ALIGN_DIR"
    python setup.py build_ext --inplace
    echo "✅ Cython模块编译完成"
    cd "$SCRIPT_DIR"
else
    echo "⚠️  警告: 找不到 $MONOTONIC_ALIGN_DIR/setup.py"
fi

echo "=== 验证安装 ==="
python -c "
import sys
print('Python路径:', sys.executable)
try:
    import torch
    print(f'✅ PyTorch版本: {torch.__version__}')
    import transformers  
    print(f'✅ Transformers版本: {transformers.__version__}')
    import librosa
    print(f'✅ Librosa版本: {librosa.__version__}')
    from huggingface_hub import snapshot_download
    print('✅ huggingface_hub导入成功')
    try:
        import whisper
        print('✅ OpenAI Whisper导入成功')
    except:
        print('⚠️  OpenAI Whisper导入失败')
    print('🎉 核心库验证通过!')
except ImportError as e:
    print(f'❌ 验证失败: {e}')
"

echo "=== 环境配置完成! ==="
echo ""
echo "✅ 环境信息:"
echo "  环境路径: $ENV_PATH"
echo "  Python版本: $python_version"
echo ""
echo "✅ 激活环境命令:"
echo "  source $ENV_PATH/bin/activate"
echo ""
echo "⚠️  集群环境注意事项:"
echo "1. 某些包可能需要系统依赖，已自动跳过失败的安装"
echo "2. 如需 espeak-ng 或 ffmpeg，请联系管理员或加载相应模块"
echo "3. 在作业脚本中使用时，确保激活正确的环境"
echo ""
echo "🔧 如果遇到问题，可以检查:"
echo "  module avail  # 查看可用模块"
echo "  module load <module_name>  # 加载需要的模块"
