#!/bin/bash

# 快速创建VEVO环境脚本
# 直接在指定路径创建环境

set -e

# 环境配置
ENV_BASE_DIR="/data/gpfs/projects/punim2341/haoguangzhou/envs"
ENV_NAME="vevo_env"
ENV_PATH="$ENV_BASE_DIR/$ENV_NAME"

echo "=== 快速创建VEVO环境 ==="
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

# 安装最核心的依赖
echo "=== 安装核心依赖 ==="
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install transformers==4.41.2 accelerate==0.24.1
pip install numpy==1.26.0 scipy==1.12.0
pip install librosa soundfile
pip install huggingface_hub
pip install omegaconf==2.3.0

echo "=== 环境创建完成! ==="
echo ""
echo "激活环境: source $ENV_PATH/bin/activate"
echo "验证安装: python -c \"from huggingface_hub import snapshot_download; print('成功')\""
echo ""
echo "如需安装完整依赖，请运行:"
echo "  cd $(pwd)"
echo "  ./setup_vevo_custom_path.sh"
