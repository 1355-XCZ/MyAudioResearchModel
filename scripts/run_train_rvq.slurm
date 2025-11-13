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
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/train_rvq_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/train_rvq_%j.err

# 加载模块
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load GCC/11.3.0
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# 激活虚拟环境
source /data/gpfs/projects/punim2341/haoguangzhou/venvs/vevo-source-fix/bin/activate

# 进入项目目录
cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

# 设置环境变量
export PYTHONPATH=/data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src:$PYTHONPATH

# 运行RVQ训练
echo "开始RVQ训练（使用100h中英文数据）..."
echo "时间: $(date)"

python train_rvq.py

echo "训练完成！"
echo "时间: $(date)"

