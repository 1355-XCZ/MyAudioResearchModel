#!/bin/bash

# Spartan集群 - 使用conda创建VEVO环境
# 更好的包管理和环境隔离

set -e

# 环境配置
ENV_NAME="vevo_env"

echo "=== 使用Conda创建VEVO环境 ==="
echo "环境名称: $ENV_NAME"

# 检查现有环境
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda环境 '$ENV_NAME' 已存在"
    read -p "是否删除并重新创建? (y/n): " recreate
    if [ "$recreate" = "y" ]; then
        echo "删除现有conda环境..."
        conda env remove -n $ENV_NAME -y
    else
        echo "激活现有环境..."
        conda activate $ENV_NAME
        echo "✅ 环境已激活"
        exit 0
    fi
fi

# 创建新的conda环境
echo "创建新的conda环境: $ENV_NAME"
conda create -n $ENV_NAME python=3.11 -y

# 初始化并激活环境
echo "初始化conda..."
eval "$(conda shell.bash hook)"

echo "激活conda环境..."
conda activate $ENV_NAME

# 验证环境激活
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "❌ 环境激活失败，请手动激活:"
    echo "conda activate $ENV_NAME"
    exit 1
fi

echo "✅ Conda环境已创建并激活: $ENV_NAME"

# 更新pip
echo "更新pip..."
pip install --upgrade pip setuptools wheel

# 安装核心依赖
echo "=== 安装核心深度学习框架 ==="
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

echo "=== 安装Transformers和相关库 ==="
pip install transformers==4.41.2 accelerate==0.24.1

echo "=== 安装数值计算库 ==="
pip install numpy==1.26.0 scipy==1.12.0

echo "=== 安装音频和配置库 ==="
pip install librosa soundfile
pip install huggingface_hub
pip install omegaconf==2.3.0
pip install tqdm

echo "=== 测试核心功能 ==="
python -c "
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
    from huggingface_hub import snapshot_download
    print('✅ HuggingFace Hub: OK')
    import librosa
    print(f'✅ Librosa: {librosa.__version__}')
    print('🎉 核心环境创建成功!')
except Exception as e:
    print(f'❌ 错误: {e}')
"

echo "=== 环境创建完成! ==="
echo ""
echo "✅ Conda环境信息:"
echo "  环境名称: $ENV_NAME"
echo "  激活命令: conda activate $ENV_NAME"
echo "  查看环境: conda env list"
echo ""
echo "📋 下一步选择:"
echo "1. 测试基础功能: python -c \"import torch; print('OK')\""
echo "2. 安装完整VEVO依赖: ./setup_vevo_conda.sh"
echo ""
echo "🔧 使用提醒:"
echo "- 使用 'conda activate $ENV_NAME' 激活环境"
echo "- 使用 'conda deactivate' 退出环境"
echo "- 在SLURM作业脚本中确保激活此环境"
