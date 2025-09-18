#!/bin/bash
#SBATCH --job-name=emilia-batch-processing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/emilia_batch_%j.out
#SBATCH --error=logs/emilia_batch_%j.err

# 作业信息
echo "======================================"
echo "Emilia数据集批处理任务"
echo "======================================"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_JOB_NODELIST"
echo "开始时间: $(date)"
echo "======================================"

# 环境设置
source ~/.bashrc

# 激活conda环境
echo "激活conda环境..."
conda activate marm5

# 检查GPU
echo "GPU信息:"
nvidia-smi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建必要的目录
mkdir -p logs
mkdir -p ${OUTPUT_BASE_PATH}/logs
mkdir -p ${OUTPUT_BASE_PATH}/checkpoints

# 运行批处理脚本
echo "开始批处理任务..."
echo "输入路径: ${INPUT_DATA_PATH}"
echo "输出路径: ${OUTPUT_BASE_PATH}"

python emilia_batch_processor.py \
    --input_data_path "${INPUT_DATA_PATH}" \
    --output_base_path "${OUTPUT_BASE_PATH}" \
    --batch_size "${BATCH_SIZE:-100}" \
    --device "${DEVICE:-cuda}" \
    --resume \
    --checkpoint_interval "${CHECKPOINT_INTERVAL:-50}" \
    --log_level "${LOG_LEVEL:-INFO}" \
    --log_file "emilia_batch_${SLURM_JOB_ID}.log"

# 检查退出状态
EXIT_CODE=$?
echo "======================================"
echo "任务完成时间: $(date)"
echo "退出代码: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 批处理任务成功完成！"
else
    echo "❌ 批处理任务失败，退出代码: $EXIT_CODE"
fi

echo "======================================"

# 生成作业统计信息
echo "作业统计信息:" > "logs/job_stats_${SLURM_JOB_ID}.txt"
echo "作业ID: $SLURM_JOB_ID" >> "logs/job_stats_${SLURM_JOB_ID}.txt"
echo "节点: $SLURM_JOB_NODELIST" >> "logs/job_stats_${SLURM_JOB_ID}.txt"
echo "退出代码: $EXIT_CODE" >> "logs/job_stats_${SLURM_JOB_ID}.txt"
echo "完成时间: $(date)" >> "logs/job_stats_${SLURM_JOB_ID}.txt"

exit $EXIT_CODE
