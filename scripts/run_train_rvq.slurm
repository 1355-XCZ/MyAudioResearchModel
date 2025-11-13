#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --job-name=train_rvq
#SBATCH --output=./logs/train_rvq_%j.out
#SBATCH --error=./logs/train_rvq_%j.err

# 加载本地配置（用户需要根据 local_config.sh.example 创建）
if [ -f ./local_config.sh ]; then
    source ./local_config.sh
fi

# 加载模块
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load GCC/11.3.0
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# 激活虚拟环境（如果配置了）
if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

# 进入项目目录（使用相对于脚本的路径）
cd "$(dirname "$0")/.."

# 设置环境变量（项目现在是自包含的）
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 运行RVQ训练
echo "开始RVQ训练（使用100h中英文数据）..."
echo "时间: $(date)"

python train_rvq.py

echo "训练完成！"
echo "时间: $(date)"

