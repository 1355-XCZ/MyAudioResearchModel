#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=extract_features
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/extract_features_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/extract_features_%j.err

# Load local configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_DIR/local_config.sh" ]; then
    source "$PROJECT_DIR/local_config.sh"
fi

cd "$PROJECT_ROOT" || exit 1

# Load modules
module purge
module load GCCcore/11.3.0 Python/3.10.4 GCC/11.3.0 OpenMPI/4.1.4
module load FFmpeg/4.4.2 espeak-ng/1.52 CUDA/11.8.0 cuDNN/8.7.0.84-CUDA-11.8.0

source "$VENV_PATH/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/Amphion:${PYTHONPATH}"

cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

echo "=========================================="
echo "提取评估数据集的emotion2vec特征"
echo "=========================================="
echo "时间: $(date)"

python extract_eval_features.py

echo ""
echo "=========================================="
echo "完成时间: $(date)"
echo "=========================================="


