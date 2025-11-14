#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --job-name=train_entropy
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/train_entropy_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/train_entropy_%j.err

# 加载本地配置
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

# 进入项目目录
PROJECT_DIR="/data/gpfs/projects/punim2341/haoguangzhou/voice/my-research/to_ready/MyAudioResearchModel"
cd "$PROJECT_DIR"

# 设置环境变量
export PROJECT_ROOT="$PROJECT_DIR"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 运行熵模型训练
echo "开始无条件熵模型 q(z) 训练（使用100h中英文数据）..."
echo "时间: $(date)"

python train_entropy.py

echo "训练完成！"
echo "时间: $(date)"

