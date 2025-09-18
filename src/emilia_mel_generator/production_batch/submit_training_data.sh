#!/bin/bash

# Emilia训练数据生成提交脚本 - Spartan集群专用
# 简化版本，专注于训练数据生成

set -e

# Spartan集群配置
DEFAULT_DATASET_PATH="/data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset"
DEFAULT_OUTPUT_PATH="/data/gpfs/projects/punim2341/haoguangzhou/emilia_training_data"
DEFAULT_CACHE_PATH="/data/gpfs/projects/punim2341/haoguangzhou/cache"

show_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "Emilia训练数据生成 - Spartan集群专用"
    echo "🎯 使用 Emo-Emilia 数据集中的中性标签音频作为风格参考"
    echo "🔗 数据源: ASLP-lab/Emo-Emilia (1400个样本, 7种情感)"
    echo ""
    echo "选项:"
    echo "  -d, --dataset PATH      数据集路径 (默认: $DEFAULT_DATASET_PATH)"
    echo "  -o, --output PATH       输出路径 (默认: $DEFAULT_OUTPUT_PATH)"
    echo "  -c, --cache PATH        缓存路径 (默认: $DEFAULT_CACHE_PATH)"
    echo "  -m, --max-samples N     最大样本数（测试用）"
    echo "  -H, --hours-per-lang N  每种语言的目标时长 (默认: 50.0)"
    echo "  -t, --time TIME         作业时间限制 (默认: 04:00:00)"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置"
    echo "  $0 -m 1000                           # 测试模式，只处理1000个样本"
    echo "  $0 -H 25.0                           # 每种语言25小时（共50小时）"
    echo "  $0 -t 08:00:00                       # 8小时时间限制"
}

# 解析参数
DATASET_PATH="$DEFAULT_DATASET_PATH"
OUTPUT_PATH="$DEFAULT_OUTPUT_PATH"
CACHE_PATH="$DEFAULT_CACHE_PATH"
MAX_SAMPLES=""
HOURS_PER_LANG="50.0"
TIME_LIMIT="04:00:00"

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -c|--cache)
            CACHE_PATH="$2"
            shift 2
            ;;
        -m|--max-samples)
            MAX_SAMPLES="--max_samples $2"
            shift 2
            ;;
        -H|--hours-per-lang)
            HOURS_PER_LANG="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
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

# 创建目录
mkdir -p "$OUTPUT_PATH"
mkdir -p "/data/gpfs/projects/punim2341/haoguangzhou/logs"

echo "======================================"
echo "🎯 提交Emilia训练数据生成作业"
echo "======================================"
echo "📁 数据集: $DATASET_PATH"
echo "📁 输出: $OUTPUT_PATH"
echo "📁 缓存: $CACHE_PATH"
echo "🕰 目标时长: 每种语言 $HOURS_PER_LANG 小时"
echo "⏰ 时间限制: $TIME_LIMIT"
echo "======================================"

# 创建SLURM脚本
TEMP_SLURM=$(mktemp /tmp/training_data_XXXXXX.slurm)

cat > "$TEMP_SLURM" << EOF
#!/bin/bash
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t $TIME_LIMIT
#SBATCH -J emilia-training-data
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.err
#SBATCH --mail-user=haoguangz@student.unimelb.edu.au
#SBATCH --mail-type=END,FAIL
#SBATCH -A punim2341

echo "🎯 Emilia训练数据生成作业开始"
echo "作业ID: \$SLURM_JOB_ID"
echo "时间: \$(date)"

# 设置环境
PROJECT_ROOT="/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
cd \$PROJECT_ROOT

source /data/gpfs/projects/punim2341/haoguangzhou/miniconda3/bin/activate
conda activate marm5

export PYTHONPATH=\$PROJECT_ROOT/src:\$PYTHONPATH

# 运行训练数据生成
python src/emilia_mel_generator/production_batch/generate_training_data.py \\
    --dataset_path "$DATASET_PATH" \\
    --output_path "$OUTPUT_PATH" \\
    --cache_path "$CACHE_PATH" \\
    --device cuda \\
    --hours_per_lang "$HOURS_PER_LANG" \\
    $MAX_SAMPLES

echo "✅ 训练数据生成完成 - \$(date)"
EOF

# 提交作业
JOB_ID=$(sbatch "$TEMP_SLURM" | grep -o '[0-9]*')

echo "✅ 作业已提交！"
echo "📋 作业ID: $JOB_ID"
echo "📊 监控: squeue -j $JOB_ID"
echo "📄 日志: /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-training-data-${JOB_ID}.out"

# 清理临时文件
rm "$TEMP_SLURM"

echo "======================================"
