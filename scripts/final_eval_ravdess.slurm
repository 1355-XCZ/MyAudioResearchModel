#!/bin/bash
#SBATCH --job-name=final_eval_ravdess
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/final_eval_ravdess_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/final_eval_ravdess_%j.err

# ===========================
# 最终版本评估 - RAVDESS数据集
# ===========================
# 码率点: 5,10,15,20,25,30,35,40,45,50,60,70,80,100,150,200,300
# 每个情感: 150个样本
# 随机种子: 1344871（可复现）
# ===========================

echo "================================================================"
echo "最终评估: RAVDESS数据集"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================================================"

# 项目目录
PROJECT_DIR="/data/gpfs/projects/punim2341/haoguangzhou/voice/my-research/to_ready/MyAudioResearchModel"
cd "$PROJECT_DIR" || exit 1

# 加载local_config
if [ -f "local_config.sh" ]; then
    source local_config.sh
else
    echo "错误: local_config.sh 不存在"
    exit 1
fi

# 加载模块
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load GCC/11.3.0
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0

# 激活虚拟环境
source "$VENV_PATH/bin/activate"

# 设置Python路径
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 检查GPU
nvidia-smi

echo ""
echo "================================================================"
echo "配置信息"
echo "================================================================"
echo "数据集: RAVDESS"
echo "每个情感样本数: 150"
echo "随机种子: 1344871"
echo "码率点: 5,10,15,20,25,30,35,40,45,50,60,70,80,100,150,200,300"
echo "输出目录: evaluation_results/final"
echo "================================================================"
echo ""

# 运行评估
python run_evaluation.py \
    --dataset ravdess \
    --samples 150 \
    --rates "5,10,15,20,25,30,35,40,45,50,60,70,80,100,150,200,300" \
    --output-dir evaluation_results/final \
    --rvq-checkpoint checkpoints/grouped_rvq_best.pt \
    --entropy-checkpoint checkpoints/entropy_model_best.pt

exit_code=$?

echo ""
echo "================================================================"
echo "任务结束"
echo "Exit Code: $exit_code"
echo "End Time: $(date)"
echo "================================================================"

exit $exit_code

