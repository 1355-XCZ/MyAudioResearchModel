#!/bin/bash

#############################
# ===== Slurm 资源申请 =====
#############################
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J emilia-vevo-batch
# ——把 Slurm 日志固定到项目根 logs/——
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.err
#SBATCH --signal=TERM@300
#SBATCH --mail-user=haoguangz@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A punim2341

# 作业信息
echo "=== 开始 Emilia + Vevo TTS 批处理任务 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

# 设置项目路径
PROJECT_ROOT="/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
cd $PROJECT_ROOT

# 激活conda环境 (如果需要)
source /data/gpfs/projects/punim2341/haoguangzhou/miniconda3/bin/activate
conda activate marm5

# 设置Python路径
export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH

# 检查GPU
echo "GPU信息:"
nvidia-smi

# 创建日志目录
mkdir -p /data/gpfs/projects/punim2341/haoguangzhou/logs
mkdir -p "${OUTPUT_BASE_PATH}/logs"
mkdir -p "${OUTPUT_BASE_PATH}/checkpoints"

# 设置数据和模型（在服务器上重新获取）
echo "=== 设置数据和模型 ==="
python src/emilia_mel_generator/production_batch/setup_data_and_models.py \
    --base_path "/data/gpfs/projects/punim2341/haoguangzhou" \
    --cache_path "/data/gpfs/projects/punim2341/haoguangzhou/cache" \
    --download_dataset \
    --download_models \
    --check_space

if [ $? -ne 0 ]; then
    echo "❌ 数据和模型设置失败"
    exit 1
fi

# 获取批次参数 (从环境变量或默认值)
BATCH_ID=${BATCH_ID:-0}
TOTAL_BATCHES=${TOTAL_BATCHES:-10}
BATCH_SIZE=${BATCH_SIZE:-100}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-50}

echo "=== 参数配置 ==="
echo "输入路径: ${INPUT_DATA_PATH}"
echo "输出路径: ${OUTPUT_BASE_PATH}"
echo "缓存路径: /data/gpfs/projects/punim2341/haoguangzhou/cache"
echo "Batch ID: $BATCH_ID"
echo "Total Batches: $TOTAL_BATCHES"
echo "Batch Size: $BATCH_SIZE"
echo "Checkpoint Interval: $CHECKPOINT_INTERVAL"

# 运行批处理脚本
echo "=== 开始批处理任务 ==="
python src/emilia_mel_generator/production_batch/emilia_batch_processor.py \
    --input_data_path "${INPUT_DATA_PATH}" \
    --output_base_path "${OUTPUT_BASE_PATH}" \
    --batch_size "$BATCH_SIZE" \
    --device "${DEVICE:-cuda}" \
    --resume \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --log_level "${LOG_LEVEL:-INFO}" \
    --log_file "emilia_vevo_batch_${SLURM_JOB_ID}.log"

exit_code=$?

echo "=== 任务完成 ==="
echo "Exit code: $exit_code"
echo "End time: $(date)"

# 如果是最后一个批次，尝试合并结果
if [ $BATCH_ID -eq $((TOTAL_BATCHES-1)) ]; then
    echo "=== 开始合并结果 ==="
    python src/emilia_mel_generator/production_batch/emilia_batch_processor.py \
        --merge-only \
        --output_base_path "${OUTPUT_BASE_PATH}" \
        --total_batches $TOTAL_BATCHES
fi

# 生成作业统计信息
echo "作业统计信息:" > "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_${SLURM_JOB_ID}.txt"
echo "作业ID: $SLURM_JOB_ID" >> "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_${SLURM_JOB_ID}.txt"
echo "节点: $SLURM_JOB_NODELIST" >> "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_${SLURM_JOB_ID}.txt"
echo "退出代码: $exit_code" >> "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_${SLURM_JOB_ID}.txt"
echo "完成时间: $(date)" >> "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_${SLURM_JOB_ID}.txt"

exit $exit_code
