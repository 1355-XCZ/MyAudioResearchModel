#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=test_entropy_quick
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/test_entropy_quick_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/test_entropy_quick_%j.err

module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load GCC/11.3.0
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source /data/gpfs/projects/punim2341/haoguangzhou/venvs/vevo-source-fix/bin/activate

cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

export PYTHONPATH=/data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src:$PYTHONPATH

echo "=========================================="
echo "快速测试：0.01数据 + 1个目标bpf"
echo "=========================================="
echo "时间: $(date)"

python test_entropy_quick.py

echo ""
echo "=========================================="
echo "完成时间: $(date)"
echo "=========================================="


