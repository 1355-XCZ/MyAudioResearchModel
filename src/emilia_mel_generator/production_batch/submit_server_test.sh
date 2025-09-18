#!/bin/bash

# 服务器小规模测试提交脚本
# 在Spartan集群上测试中英文各10条样本

set -e

echo "======================================"
echo "🧪 Emilia服务器小规模测试"
echo "======================================"
echo "🎯 目标: 验证完整处理流程"
echo "📊 规模: 中英文各10条样本 (共20条)"
echo "⏰ 预计时间: 30-60分钟"
echo "🔧 测试内容:"
echo "  ✅ Emo-Emilia数据集加载"
echo "  ✅ 语言匹配验证"
echo "  ✅ Vevo兼容性检查"
echo "  ✅ emotion2vec特征提取"
echo "  ✅ CSV元信息生成"
echo "======================================"

# 默认配置
DEFAULT_OUTPUT="/data/gpfs/projects/punim2341/haoguangzhou/emilia_test"
DEFAULT_SAMPLES=20

# 解析参数
OUTPUT_PATH="$DEFAULT_OUTPUT"
SAMPLES="$DEFAULT_SAMPLES"

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -s|--samples)
            SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -o, --output PATH    测试输出路径 (默认: $DEFAULT_OUTPUT)"
            echo "  -s, --samples N      测试样本数 (默认: $DEFAULT_SAMPLES)"
            echo "  -h, --help           显示帮助"
            echo ""
            echo "示例:"
            echo "  $0                           # 使用默认配置"
            echo "  $0 -s 10                     # 只测试10个样本"
            echo "  $0 -o /path/to/test/output   # 自定义输出路径"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 显示配置
echo "📋 测试配置:"
echo "  输出路径: $OUTPUT_PATH"
echo "  样本数: $SAMPLES"
echo ""

# 确认提交
echo "确认提交服务器测试吗？(y/N)"
read -r confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "❌ 测试已取消"
    exit 0
fi

# 创建输出目录
mkdir -p "$OUTPUT_PATH"
mkdir -p "/data/gpfs/projects/punim2341/haoguangzhou/logs"

# 提交测试作业
echo "🚀 提交测试作业..."

# 创建临时SLURM脚本
TEMP_SCRIPT=$(mktemp /tmp/emilia_test_XXXXXX.slurm)

cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J emilia-test-small
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-%j.err
#SBATCH --mail-user=haoguangz@student.unimelb.edu.au
#SBATCH --mail-type=END,FAIL
#SBATCH -A punim2341

echo "🧪 Emilia小规模测试开始"
echo "作业ID: \$SLURM_JOB_ID"
echo "样本数: $SAMPLES"
echo "时间: \$(date)"

# 设置环境
PROJECT_ROOT="/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
cd \$PROJECT_ROOT

source /data/gpfs/projects/punim2341/haoguangzhou/miniconda3/bin/activate
conda activate marm5

export PYTHONPATH=\$PROJECT_ROOT/src:\$PYTHONPATH

# 检查环境
echo "=== 环境检查 ==="
echo "Python: \$(python --version)"
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: \$(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi

# 创建测试输出目录
TEST_OUTPUT="$OUTPUT_PATH/test_small_\$(date +%Y%m%d_%H%M%S)"
mkdir -p "\$TEST_OUTPUT"

echo "📁 测试输出: \$TEST_OUTPUT"

# 设置缓存环境
export HF_HOME="/data/gpfs/projects/punim2341/haoguangzhou/cache/huggingface"
export TRANSFORMERS_CACHE="/data/gpfs/projects/punim2341/haoguangzhou/cache/transformers"
export TORCH_HOME="/data/gpfs/projects/punim2341/haoguangzhou/cache/torch"

mkdir -p "\$HF_HOME" "\$TRANSFORMERS_CACHE" "\$TORCH_HOME"

# 测试Emo-Emilia数据集加载
echo "=== 测试Emo-Emilia数据集 ==="
python -c "
from datasets import load_dataset
print('🧪 测试Emo-Emilia加载...')
try:
    ds = load_dataset('ASLP-lab/Emo-Emilia')
    print(f'✅ Emo-Emilia加载成功: {len(ds[\"train\"])} 个样本')
    
    # 统计neutral样本
    neutral_count = sum(1 for item in ds['train'] if item['emotion'] == 'neutral')
    print(f'📊 Neutral样本: {neutral_count} 个')
    
    # 语言分布
    lang_count = {}
    for item in ds['train']:
        if item['emotion'] == 'neutral':
            lang = item['language']
            lang_count[lang] = lang_count.get(lang, 0) + 1
    print(f'📊 Neutral语言分布: {lang_count}')
    
except Exception as e:
    print(f'❌ Emo-Emilia加载失败: {e}')
    exit(1)
"

if [ \$? -ne 0 ]; then
    echo "❌ Emo-Emilia测试失败"
    exit 1
fi

# 运行训练数据生成测试
echo "=== 运行训练数据生成测试 ==="
python src/emilia_mel_generator/production_batch/generate_training_data.py \\
    --dataset_path "/data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset" \\
    --output_path "\$TEST_OUTPUT" \\
    --cache_path "/data/gpfs/projects/punim2341/haoguangzhou/cache" \\
    --device cuda \\
    --max_samples $SAMPLES \\
    --hours_per_lang 0.1

exit_code=\$?

echo "=== 测试结果验证 ==="
if [ \$exit_code -eq 0 ]; then
    echo "✅ 训练数据生成成功"
    
    # 文件统计
    echo "📊 生成文件统计:"
    echo "  原始mel: \$(find "\$TEST_OUTPUT/original_mels" -name "*.npz" 2>/dev/null | wc -l)"
    echo "  中性mel: \$(find "\$TEST_OUTPUT/neutral_mels" -name "*.npz" 2>/dev/null | wc -l)"
    echo "  情感特征: \$(find "\$TEST_OUTPUT/emotion_features" -name "*.npz" 2>/dev/null | wc -l)"
    
    # 验证语言匹配
    if [ -f "src/emilia_mel_generator/production_batch/verify_language_matching.py" ]; then
        echo "🌍 验证语言匹配..."
        python src/emilia_mel_generator/production_batch/verify_language_matching.py \\
            --output_path "\$TEST_OUTPUT"
    fi
    
    echo "🎉 小规模测试完成！"
    echo "📁 结果位置: \$TEST_OUTPUT"
else
    echo "❌ 训练数据生成失败"
fi

echo "=== 测试完成 ==="
echo "时间: \$(date)"
echo "退出代码: \$exit_code"

exit \$exit_code
EOF

# 提交作业
JOB_ID=$(sbatch "$TEMP_SCRIPT" | grep -o '[0-9]*')

echo "✅ 测试作业已提交到Spartan集群！"
echo "📋 作业ID: $JOB_ID"
echo "📊 监控命令: squeue -j $JOB_ID"
echo "📄 实时日志: tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-${JOB_ID}.out"

# 清理临时文件
rm "$TEMP_SCRIPT"

echo ""
echo "======================================"
echo "🎯 测试提交完成！"
echo ""
echo "📋 后续步骤:"
echo "1. 监控作业状态: squeue -j $JOB_ID"
echo "2. 查看实时日志: tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-${JOB_ID}.out"
echo "3. 测试完成后验证结果"
echo "4. 如果测试成功，可以提交大规模处理任务"
echo "======================================"
