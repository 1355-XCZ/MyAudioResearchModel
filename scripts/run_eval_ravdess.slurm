#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_ravdess
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/eval_ravdess_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/eval_ravdess_%j.err

PROJECT_ROOT="/data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel"
cd "$PROJECT_ROOT" || exit 1

module purge
module load GCCcore/11.3.0 Python/3.10.4 GCC/11.3.0 OpenMPI/4.1.4
module load FFmpeg/4.4.2 espeak-ng/1.52 CUDA/11.8.0 cuDNN/8.7.0.84-CUDA-11.8.0

source /data/gpfs/projects/punim2341/haoguangzhou/venvs/vevo-source-fix/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/Amphion:${PYTHONPATH}"
export CACHE_PATH="/data/gpfs/projects/punim2341/haoguangzhou/cache"
export HF_HOME="${CACHE_PATH}/huggingface"
export TRANSFORMERS_CACHE="${CACHE_PATH}/transformers"
export HF_DATASETS_CACHE="${CACHE_PATH}/datasets"

cd ${PROJECT_ROOT}/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

echo "=========================================="
echo "RAVDESS评估（100样本/情感，22码率点）"
echo "=========================================="
echo "开始时间: $(date)"

python run_evaluation_ravdess.py

echo ""
echo "=========================================="
echo "完成时间: $(date)"
echo "=========================================="

