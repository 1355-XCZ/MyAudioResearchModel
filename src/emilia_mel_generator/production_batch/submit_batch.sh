#!/bin/bash

# Emilia数据集批处理任务提交脚本
# 简化SLURM作业提交流程

set -e  # 遇到错误立即退出

# 默认配置
DEFAULT_INPUT_PATH="/path/to/emilia/dataset"
DEFAULT_OUTPUT_PATH="/path/to/output"
DEFAULT_BATCH_SIZE=100
DEFAULT_DEVICE="cuda"
DEFAULT_CHECKPOINT_INTERVAL=50
DEFAULT_LOG_LEVEL="INFO"

# 显示使用说明
show_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -i, --input PATH        输入数据路径 (必需)"
    echo "  -o, --output PATH       输出路径 (必需)"
    echo "  -b, --batch-size N      批处理大小 (默认: $DEFAULT_BATCH_SIZE)"
    echo "  -d, --device DEVICE     计算设备 (默认: $DEFAULT_DEVICE)"
    echo "  -c, --checkpoint N      检查点间隔 (默认: $DEFAULT_CHECKPOINT_INTERVAL)"
    echo "  -l, --log-level LEVEL   日志级别 (默认: $DEFAULT_LOG_LEVEL)"
    echo "  -r, --resume            从检查点恢复"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -i /data/emilia -o /output/results"
    echo "  $0 -i /data/emilia -o /output/results -b 200 -r"
}

# 解析命令行参数
INPUT_PATH=""
OUTPUT_PATH=""
BATCH_SIZE=$DEFAULT_BATCH_SIZE
DEVICE=$DEFAULT_DEVICE
CHECKPOINT_INTERVAL=$DEFAULT_CHECKPOINT_INTERVAL
LOG_LEVEL=$DEFAULT_LOG_LEVEL
RESUME_FLAG=""

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

# 检查必需参数
if [[ -z "$INPUT_PATH" ]]; then
    echo "❌ 错误: 必须指定输入数据路径 (-i/--input)"
    show_usage
    exit 1
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    echo "❌ 错误: 必须指定输出路径 (-o/--output)"
    show_usage
    exit 1
fi

# 验证输入路径
if [[ ! -d "$INPUT_PATH" ]]; then
    echo "❌ 错误: 输入路径不存在: $INPUT_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_PATH"
mkdir -p "$OUTPUT_PATH/logs"

# 显示配置信息
echo "======================================"
echo "🚀 提交Emilia数据集批处理任务"
echo "======================================"
echo "📁 输入路径: $INPUT_PATH"
echo "📁 输出路径: $OUTPUT_PATH"
echo "📦 批处理大小: $BATCH_SIZE"
echo "🖥️  计算设备: $DEVICE"
echo "💾 检查点间隔: $CHECKPOINT_INTERVAL"
echo "📊 日志级别: $LOG_LEVEL"
echo "🔄 恢复模式: $([ -n "$RESUME_FLAG" ] && echo "启用" || echo "禁用")"
echo "======================================"

# 检查SLURM是否可用
if command -v sbatch &> /dev/null; then
    echo "🎯 使用SLURM提交作业..."
    
    # 设置环境变量供SLURM脚本使用
    export INPUT_DATA_PATH="$INPUT_PATH"
    export OUTPUT_BASE_PATH="$OUTPUT_PATH"
    export BATCH_SIZE="$BATCH_SIZE"
    export DEVICE="$DEVICE"
    export CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL"
    export LOG_LEVEL="$LOG_LEVEL"
    
    # 提交SLURM作业
    JOB_ID=$(sbatch submit_batch_job.slurm | grep -o '[0-9]*')
    
    echo "✅ 作业已提交！"
    echo "📋 作业ID: $JOB_ID"
    echo "📊 监控命令: squeue -j $JOB_ID"
    echo "📄 日志位置: logs/emilia_batch_${JOB_ID}.out"
    echo "❌ 错误日志: logs/emilia_batch_${JOB_ID}.err"
    
else
    echo "⚠️  SLURM不可用，直接运行脚本..."
    
    # 直接运行Python脚本
    python emilia_batch_processor.py \
        --input_data_path "$INPUT_PATH" \
        --output_base_path "$OUTPUT_PATH" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        $RESUME_FLAG \
        --checkpoint_interval "$CHECKPOINT_INTERVAL" \
        --log_level "$LOG_LEVEL" \
        --log_file "emilia_batch_$(date +%Y%m%d_%H%M%S).log"
fi

echo "======================================"
