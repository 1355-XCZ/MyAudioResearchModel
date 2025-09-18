#!/bin/bash

# Emilia数据集批处理任务提交脚本 - Spartan集群专用
# 基于提供的Spartan集群环境配置

set -e  # 遇到错误立即退出

# Spartan集群默认配置
DEFAULT_INPUT_PATH="/data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset"
DEFAULT_OUTPUT_PATH="/data/gpfs/projects/punim2341/haoguangzhou/emilia_output"
DEFAULT_BATCH_SIZE=100
DEFAULT_DEVICE="cuda"
DEFAULT_CHECKPOINT_INTERVAL=50
DEFAULT_LOG_LEVEL="INFO"

# 显示使用说明
show_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "Spartan集群专用 - Emilia数据集批处理任务提交"
    echo ""
    echo "选项:"
    echo "  -i, --input PATH        输入数据路径 (默认: $DEFAULT_INPUT_PATH)"
    echo "  -o, --output PATH       输出路径 (默认: $DEFAULT_OUTPUT_PATH)"
    echo "  -b, --batch-size N      批处理大小 (默认: $DEFAULT_BATCH_SIZE)"
    echo "  -d, --device DEVICE     计算设备 (默认: $DEFAULT_DEVICE)"
    echo "  -c, --checkpoint N      检查点间隔 (默认: $DEFAULT_CHECKPOINT_INTERVAL)"
    echo "  -l, --log-level LEVEL   日志级别 (默认: $DEFAULT_LOG_LEVEL)"
    echo "  -r, --resume            从检查点恢复"
    echo "  -t, --time TIME         作业时间限制 (默认: 04:00:00)"
    echo "  -p, --partition PART    SLURM分区 (默认: gpu-a100-short)"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                              # 使用默认配置"
    echo "  $0 -b 200 -r                                   # 批大小200，从检查点恢复"
    echo "  $0 -t 08:00:00 -p gpu-a100                     # 8小时时限，使用gpu-a100分区"
    echo ""
    echo "Spartan集群信息:"
    echo "  项目账号: punim2341"
    echo "  用户: haoguangz@student.unimelb.edu.au"
    echo "  项目根目录: /data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
}

# 解析命令行参数
INPUT_PATH="$DEFAULT_INPUT_PATH"
OUTPUT_PATH="$DEFAULT_OUTPUT_PATH"
BATCH_SIZE=$DEFAULT_BATCH_SIZE
DEVICE=$DEFAULT_DEVICE
CHECKPOINT_INTERVAL=$DEFAULT_CHECKPOINT_INTERVAL
LOG_LEVEL=$DEFAULT_LOG_LEVEL
RESUME_FLAG=""
TIME_LIMIT="04:00:00"
PARTITION="gpu-a100-short"

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT_INTERVAL="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p "$OUTPUT_PATH"
mkdir -p "/data/gpfs/projects/punim2341/haoguangzhou/logs"

# 显示配置信息
echo "======================================"
echo "🚀 提交Emilia数据集批处理任务 (Spartan)"
echo "======================================"
echo "📁 输入路径: $INPUT_PATH"
echo "📁 输出路径: $OUTPUT_PATH"
echo "📦 批处理大小: $BATCH_SIZE"
echo "🖥️  计算设备: $DEVICE"
echo "💾 检查点间隔: $CHECKPOINT_INTERVAL"
echo "📊 日志级别: $LOG_LEVEL"
echo "⏰ 时间限制: $TIME_LIMIT"
echo "🎯 SLURM分区: $PARTITION"
echo "🔄 恢复模式: $([ -n "$RESUME_FLAG" ] && echo "启用" || echo "禁用")"
echo "👤 账号: punim2341"
echo "======================================"

# 设置环境变量供SLURM脚本使用
export INPUT_DATA_PATH="$INPUT_PATH"
export OUTPUT_BASE_PATH="$OUTPUT_PATH"
export BATCH_SIZE="$BATCH_SIZE"
export DEVICE="$DEVICE"
export CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL"
export LOG_LEVEL="$LOG_LEVEL"

# 创建临时SLURM脚本（基于配置定制）
TEMP_SLURM_SCRIPT=$(mktemp /tmp/emilia_batch_XXXXXX.slurm)

cat > "$TEMP_SLURM_SCRIPT" << EOF
#!/bin/bash

#############################
# ===== Slurm 资源申请 =====
#############################
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t $TIME_LIMIT
#SBATCH -J emilia-vevo-batch
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.err
#SBATCH --signal=TERM@300
#SBATCH --mail-user=haoguangz@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A punim2341

# 作业信息
echo "=== 开始 Emilia + Vevo TTS 批处理任务 ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "GPU: \$CUDA_VISIBLE_DEVICES"
echo "Time: \$(date)"

# 设置项目路径
PROJECT_ROOT="/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
cd \$PROJECT_ROOT

# 激活conda环境
source /data/gpfs/projects/punim2341/haoguangzhou/miniconda3/bin/activate
conda activate marm5

# 设置Python路径
export PYTHONPATH=\$PROJECT_ROOT/src:\$PYTHONPATH

# 检查GPU
echo "GPU信息:"
nvidia-smi

# 创建日志目录
mkdir -p /data/gpfs/projects/punim2341/haoguangzhou/logs
mkdir -p "\${OUTPUT_BASE_PATH}/logs"
mkdir -p "\${OUTPUT_BASE_PATH}/checkpoints"

echo "=== 参数配置 ==="
echo "输入路径: \${INPUT_DATA_PATH}"
echo "输出路径: \${OUTPUT_BASE_PATH}"
echo "Batch Size: \${BATCH_SIZE}"
echo "Checkpoint Interval: \${CHECKPOINT_INTERVAL}"

# 运行批处理脚本
echo "=== 开始批处理任务 ==="
python src/emilia_mel_generator/production_batch/emilia_batch_processor.py \\
    --input_data_path "\${INPUT_DATA_PATH}" \\
    --output_base_path "\${OUTPUT_BASE_PATH}" \\
    --batch_size "\${BATCH_SIZE}" \\
    --device "\${DEVICE}" \\
    $RESUME_FLAG \\
    --checkpoint_interval "\${CHECKPOINT_INTERVAL}" \\
    --log_level "\${LOG_LEVEL}" \\
    --log_file "emilia_vevo_batch_\${SLURM_JOB_ID}.log"

exit_code=\$?

echo "=== 任务完成 ==="
echo "Exit code: \$exit_code"
echo "End time: \$(date)"

# 生成作业统计信息
echo "作业统计信息:" > "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_\${SLURM_JOB_ID}.txt"
echo "作业ID: \$SLURM_JOB_ID" >> "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_\${SLURM_JOB_ID}.txt"
echo "节点: \$SLURM_JOB_NODELIST" >> "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_\${SLURM_JOB_ID}.txt"
echo "退出代码: \$exit_code" >> "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_\${SLURM_JOB_ID}.txt"
echo "完成时间: \$(date)" >> "/data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_\${SLURM_JOB_ID}.txt"

exit \$exit_code
EOF

# 提交SLURM作业
echo "🎯 提交到Spartan集群..."
JOB_ID=$(sbatch "$TEMP_SLURM_SCRIPT" | grep -o '[0-9]*')

echo "✅ 作业已提交到Spartan集群！"
echo "📋 作业ID: $JOB_ID"
echo "📊 监控命令: squeue -j $JOB_ID"
echo "📄 输出日志: /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-vevo-batch-${JOB_ID}.out"
echo "❌ 错误日志: /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-vevo-batch-${JOB_ID}.err"
echo "📊 作业统计: /data/gpfs/projects/punim2341/haoguangzhou/logs/job_stats_${JOB_ID}.txt"

# 清理临时文件
rm "$TEMP_SLURM_SCRIPT"

echo "======================================"
echo "🎉 Spartan集群作业提交完成！"
echo "======================================"
