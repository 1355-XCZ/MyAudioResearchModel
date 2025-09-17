#!/bin/bash

# VEVO环境修复脚本
# 修复PyTorch版本冲突和包安装问题

set -e

echo "=== VEVO环境修复开始 ==="

# 激活虚拟环境
if [ -d "vevo_env" ]; then
    source vevo_env/bin/activate
    echo "已激活 vevo_env 虚拟环境"
else
    echo "错误: 找不到 vevo_env 目录"
    exit 1
fi

echo "=== 修复PyTorch版本冲突 ==="
echo "卸载冲突的PyTorch组件..."
pip uninstall torch torchvision torchaudio triton -y

echo "重新安装兼容的PyTorch版本..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

echo "=== 修复配置库版本 ==="
pip uninstall omegaconf -y
pip install omegaconf==2.3.0

echo "=== 修复protobuf版本 ==="
pip uninstall protobuf -y
pip install "protobuf>=3.20.0,<4.0.0"

echo "=== 安装剩余的核心依赖 ==="
pip install transformers==4.41.2 accelerate==0.24.1
pip install numpy==1.26.0 scipy==1.12.0
pip install librosa soundfile
pip install huggingface_hub datasets

echo "=== 安装音频处理库 ==="
pip install phonemizer==3.2.1 g2p_en pypinyin==0.48.0
pip install jieba cn2an unidecode pyopenjtalk pykakasi

echo "=== 安装评估指标库 ==="
pip install torchmetrics pymcd
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip

echo "=== 安装语音识别库 ==="
pip install openai-whisper

echo "=== 尝试安装fairseq (可能需要较长时间) ==="
echo "方法1: 从git安装..."
if ! pip install git+https://github.com/pytorch/fairseq.git; then
    echo "Git安装失败，尝试方法2..."
    pip install fairseq --no-deps
    echo "fairseq已安装(无依赖模式)"
fi

echo "=== 安装其他工具库 ==="
pip install tqdm loguru einops vector-quantize-pytorch==1.12.5
pip install black==24.1.1 ruamel.yaml json5
pip install gradio onnxruntime

echo "=== 安装特殊依赖 ==="
echo "安装lhotse..."
pip install git+https://github.com/lhotse-speech/lhotse

echo "安装whisperx..."
pip install git+https://github.com/m-bain/whisperx.git

echo "=== 验证关键包安装 ==="
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
    print('🎉 核心依赖验证通过!')
except ImportError as e:
    print(f'❌ 导入错误: {e}')
    exit(1)
"

echo "=== 编译Cython模块 ==="
cd ../../../../../modules/monotonic_align
if [ -f "setup.py" ]; then
    python setup.py build_ext --inplace
    echo "✅ Cython模块编译完成"
else
    echo "⚠️  警告: 找不到monotonic_align/setup.py"
fi
cd ../../models/vc/vevo/my/tool/env1

echo "=== 环境修复完成! ==="
echo ""
echo "修复内容:"
echo "1. ✅ 修复了PyTorch版本冲突 (torch==2.0.1, torchvision==0.15.2)"
echo "2. ✅ 修复了omegaconf版本问题"
echo "3. ✅ 修复了protobuf版本冲突"
echo "4. ✅ 重新安装了fairseq"
echo "5. ✅ 安装了所有核心依赖"
echo ""
echo "测试环境:"
echo "source vevo_env/bin/activate"
echo "python -c \"from huggingface_hub import snapshot_download; print('环境正常')\""
