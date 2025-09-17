#!/bin/bash

# Spartan集群快速创建VEVO环境
# 避免系统权限问题

set -e

# 环境配置
ENV_BASE_DIR="/data/gpfs/projects/punim2341/haoguangzhou/envs"
ENV_NAME="vevo_env"
ENV_PATH="$ENV_BASE_DIR/$ENV_NAME"

echo "=== Spartan集群 - 快速创建VEVO环境 ==="
echo "目标路径: $ENV_PATH"

# 检查目录是否存在
if [ ! -d "$ENV_BASE_DIR" ]; then
    echo "创建基础目录: $ENV_BASE_DIR"
    mkdir -p "$ENV_BASE_DIR"
fi

# 处理现有环境
if [ -d "$ENV_PATH" ]; then
    echo "环境已存在: $ENV_PATH"
    read -p "是否删除并重新创建? (y/n): " recreate
    if [ "$recreate" = "y" ]; then
        echo "删除现有环境..."
        rm -rf "$ENV_PATH"
    else
        echo "使用现有环境"
        source "$ENV_PATH/bin/activate"
        echo "环境已激活"
        exit 0
    fi
fi

# 创建新环境
echo "创建Python虚拟环境..."
python -m venv "$ENV_PATH"

# 激活环境
source "$ENV_PATH/bin/activate"
echo "✅ 环境已创建并激活: $ENV_PATH"

# 更新pip
echo "更新pip..."
pip install --upgrade pip setuptools wheel

# 安装最核心的依赖 - 避免系统依赖问题
echo "=== 安装核心依赖 (集群安全版本) ==="
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install transformers==4.41.2 accelerate==0.24.1
pip install numpy==1.26.0 scipy==1.12.0
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
echo "✅ 基础环境已就绪"
echo "  环境路径: $ENV_PATH"
echo "  激活命令: source $ENV_PATH/bin/activate"
echo ""
echo "📋 下一步选择:"
echo "1. 测试基础功能是否正常"
echo "2. 运行完整安装: ./setup_vevo_spartan.sh"
echo ""
echo "🔧 集群使用提醒:"
echo "- 避免修改系统conda环境"
echo "- 某些包可能需要模块加载"
echo "- 在SLURM作业中使用此环境"
