#!/bin/bash

#############################
# ===== Slurm 资源申请 =====
#############################
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J emilia-data-extract
# ——把 Slurm 日志固定到项目根 logs/——
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.err
#SBATCH --signal=TERM@300
#SBATCH --mail-user=haoguangz@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A punim2341

# 环境设置
echo "=== 开始 Emilia 数据提取任务 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

# 设置项目路径
PROJECT_ROOT="/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
cd $PROJECT_ROOT

# 激活conda环境 (如果需要)
# source /data/gpfs/projects/punim2341/haoguangzhou/miniconda3/bin/activate
# conda activate audio_research

# 设置Python路径
export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH

# 创建日志目录
mkdir -p logs

# 获取批次参数 (从环境变量或默认值)
BATCH_ID=${BATCH_ID:-0}
TOTAL_BATCHES=${TOTAL_BATCHES:-10}
TARGET_HOURS=${TARGET_HOURS:-50}
K_VARIANTS=${K_VARIANTS:-1}

echo "=== 参数配置 ==="
echo "Batch ID: $BATCH_ID"
echo "Total Batches: $TOTAL_BATCHES"  
echo "Target Hours per Language: $TARGET_HOURS"
echo "K Variants: $K_VARIANTS"

# 运行数据提取
echo "=== 开始数据提取 ==="
python src/emilia_mel_generator/spartan_batch_job.py \
    --batch-id $BATCH_ID \
    --total-batches $TOTAL_BATCHES \
    --target-hours $TARGET_HOURS \
    --k-variants $K_VARIANTS

exit_code=$?

echo "=== 任务完成 ==="
echo "Exit code: $exit_code"
echo "End time: $(date)"

# 如果是最后一个批次，尝试合并结果
if [ $BATCH_ID -eq $((TOTAL_BATCHES-1)) ]; then
    echo "=== 开始合并结果 ==="
    python src/emilia_mel_generator/spartan_batch_job.py \
        --merge-only \
        --total-batches $TOTAL_BATCHES
fi

exit $exit_code
