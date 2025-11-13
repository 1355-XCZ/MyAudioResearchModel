#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=test_lambda
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/test_lambda_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/test_lambda_%j.err

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

# 运行λ值测试
echo "=========================================="
echo "Lambda值范围测试（简化ECVQ，无需熵模型）"
echo "=========================================="
echo "时间: $(date)"

python test_lambda_range.py

echo ""
echo "=========================================="
echo "完成时间: $(date)"
echo "=========================================="

