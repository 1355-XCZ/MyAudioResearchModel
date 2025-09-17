#!/bin/bash

# VEVO完整环境配置脚本 (Conda版本)
# 在conda环境中安装所有VEVO依赖

set -e

ENV_NAME="vevo_env"

echo "=== VEVO完整环境配置 (Conda版本) ==="

# 检查环境是否存在
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "❌ Conda环境 '$ENV_NAME' 不存在"
    echo "请先运行: ./create_vevo_conda.sh"
    exit 1
fi

# 初始化并激活环境
echo "初始化conda..."
eval "$(conda shell.bash hook)"

echo "激活conda环境: $ENV_NAME"
conda activate $ENV_NAME

# 验证环境激活
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "❌ 环境激活失败"
    echo "当前环境: $CONDA_DEFAULT_ENV"
    echo "请手动激活环境后再运行脚本:"
    echo "conda activate $ENV_NAME"
    echo "./setup_vevo_conda.sh"
    exit 1
fi

echo "✅ 当前环境: $CONDA_DEFAULT_ENV"

# 检查系统依赖 - 集群版本
echo "=== 检查系统依赖 (集群环境) ==="

# 检查espeak-ng
if ! command -v espeak-ng &> /dev/null; then
    echo "⚠️  espeak-ng 未找到，某些文本处理功能可能受限"
else
    echo "✅ espeak-ng 已可用"
fi

# 检查ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg 未找到，某些音频处理功能可能受限"
    echo "提示: 尝试 'module avail ffmpeg' 查看可用模块"
else
    echo "✅ ffmpeg 已可用"
fi

# 安装完整依赖
echo "=== 安装音频处理库 ==="
pip install audiomentations pyworld
echo "尝试安装 praat-parselmouth..."
pip install praat-parselmouth || echo "⚠️  praat-parselmouth 安装失败，跳过"

pip install diffsptk==1.0.1 pysptk resampy soxr
pip install nnAudio ptwt

echo "尝试安装 ffmpeg-python..."
pip install ffmpeg-python==0.2.0 || echo "⚠️  ffmpeg-python 安装失败，跳过"

echo "=== 安装编解码器 ==="
pip install encodec
pip install vocos speechtokenizer descript-audio-codec

echo "=== 安装文本处理库 ==="
echo "尝试安装 phonemizer..."
pip install phonemizer==3.2.1 || echo "⚠️  phonemizer 安装失败，可能需要espeak-ng"

pip install g2p_en pypinyin==0.48.0
pip install jieba cn2an unidecode

echo "尝试安装其他文本处理库..."
pip install pyopenjtalk || echo "⚠️  pyopenjtalk 安装失败"
pip install pykakasi || echo "⚠️  pykakasi 安装失败"
pip install tgt typeguard humanfriendly munch

echo "=== 安装数据处理库 ==="
pip install datasets pandas matplotlib scikit-learn

echo "=== 安装评估指标库 ==="
pip install torchmetrics frechet_audio_distance
echo "安装 python-pesq..."
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip install pystoi pymcd mir_eval jiwer

echo "=== 安装训练和可视化工具 ==="
pip install tensorboard tensorboardX wandb
pip install gradio

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

echo "=== 安装其他工具库 ==="
pip install einops vector-quantize-pytorch==1.12.5
pip install black==24.1.1 ruff Cython
pip install PyYAML ipython onnxruntime
pip install loguru colorama tabulate

echo "=== 安装特殊依赖 ==="
echo "安装 lhotse..."
pip install git+https://github.com/lhotse-speech/lhotse || echo "⚠️  lhotse 安装失败"

echo "安装 whisperx..."
pip install git+https://github.com/m-bain/whisperx.git || echo "⚠️  whisperx 安装失败"

echo "=== 编译Cython模块 ==="
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

echo "=== 最终验证 ==="
python -c "
import sys
print('Python路径:', sys.executable)
print('Conda环境:', sys.prefix)

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    import transformers  
    print(f'✅ Transformers: {transformers.__version__}')
    import librosa
    print(f'✅ Librosa: {librosa.__version__}')
    from huggingface_hub import snapshot_download
    print('✅ HuggingFace Hub: OK')
    
    # 测试VEVO特定依赖
    try:
        import encodec
        print('✅ Encodec: OK')
    except:
        print('⚠️  Encodec: 失败')
        
    try:
        import whisper
        print('✅ OpenAI Whisper: OK')
    except:
        print('⚠️  OpenAI Whisper: 失败')
        
    print('🎉 VEVO环境配置完成!')
    
except ImportError as e:
    print(f'❌ 验证失败: {e}')
"

echo "=== 环境配置完成! ==="
echo ""
echo "✅ 完整VEVO环境已配置"
echo "  环境名称: $ENV_NAME"
echo "  激活命令: conda activate $ENV_NAME"
echo "  查看环境: conda env list"
echo ""
echo "🧪 测试VEVO推理:"
echo "  conda activate $ENV_NAME"
echo "  cd /path/to/vevo/inference/script"
echo "  python infer_vevo*.py"
echo ""
echo "⚠️  注意事项:"
echo "1. 某些包可能因系统依赖而安装失败，但核心功能应该正常"
echo "2. 在SLURM作业中使用 'conda activate $ENV_NAME'"
echo "3. 如需额外模块，使用 'module load <module_name>'"
