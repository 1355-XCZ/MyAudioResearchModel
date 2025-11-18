#!/bin/bash

# ================================================================
# 完整实验复现管道 - 一键运行所有步骤
# ================================================================
# 功能：
#   1. 检查/创建虚拟环境（如果已存在则跳过）
#   2. 安装依赖（如果已安装则跳过）
#   3. 准备评估数据子集
#   4. 运行完整评估（47个码率点 × 3个数据集）
#   5. 生成所有论文图表
#
# 使用方法：
#   本地运行:    bash reproduce_full_pipeline.sh
#   SLURM集群:   sbatch scripts/run_full_reproduce.slurm
# ================================================================

set -e  # 遇到错误立即退出

PROJECT_ROOT=$(pwd)
VENV_DIR="$PROJECT_ROOT/reproduce_venv"

echo "================================================================"
echo "🚀 情感RVQ实验完整复现管道"
echo "================================================================"
echo "项目目录: $PROJECT_ROOT"
echo "开始时间: $(date)"
echo ""

# ================================================================
# 步骤1: 检查/创建虚拟环境
# ================================================================
echo "================================================================"
echo "步骤1: 检查Python虚拟环境"
echo "================================================================"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
    echo "✅ 发现现有虚拟环境: $VENV_DIR"
    echo "   跳过环境创建步骤"
else
    echo "📦 创建新的虚拟环境..."
    python3 -m venv "$VENV_DIR"
    echo "✅ 虚拟环境已创建: $VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
echo "✅ 虚拟环境已激活"
echo "   Python: $(which python)"
echo "   版本: $(python --version)"
echo ""

# ================================================================
# 步骤2: 检查/安装依赖
# ================================================================
echo "================================================================"
echo "步骤2: 检查项目依赖"
echo "================================================================"

# 检查关键包是否已安装
NEED_INSTALL=false

if ! python -c "import torch" 2>/dev/null; then
    echo "❌ torch 未安装"
    NEED_INSTALL=true
elif ! python -c "import funasr" 2>/dev/null; then
    echo "❌ funasr 未安装"
    NEED_INSTALL=true
elif ! python -c "import vector_quantize_pytorch" 2>/dev/null; then
    echo "❌ vector_quantize_pytorch 未安装"
    NEED_INSTALL=true
else
    echo "✅ 核心依赖已安装"
    echo "   跳过安装步骤"
fi

if [ "$NEED_INSTALL" = true ]; then
    echo ""
    echo "📦 安装项目依赖..."
    echo "   这可能需要10-15分钟，请耐心等待..."
    pip install --upgrade pip setuptools wheel -q
    pip install -r requirements.txt
    echo "✅ 依赖安装完成"
fi

echo ""

# ================================================================
# 步骤3: 验证环境完整性
# ================================================================
echo "================================================================"
echo "步骤3: 验证环境完整性"
echo "================================================================"

python test_dependencies.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 环境验证失败！请检查错误信息"
    exit 1
fi

echo ""
echo "✅ 环境验证通过"
echo ""

# ================================================================
# 步骤4: 准备评估数据子集
# ================================================================
echo "================================================================"
echo "步骤4: 准备评估数据子集"
echo "================================================================"
echo "配置:"
echo "  - 数据集: ESD, IEMOCAP, RAVDESS"
echo "  - 样本数: 100/情感"
echo "  - 随机种子: 42 (保证可复现)"
echo ""

if [ -d "data_subset" ] && [ -f "data_subset/ESD/subset_info.json" ]; then
    echo "✅ 发现现有数据子集: data_subset/"
    echo "   跳过数据准备步骤"
    echo "   (如需重新生成，请删除 data_subset/ 目录)"
else
    echo "📊 准备数据子集..."
    python prepare_evaluation_subset.py \
        --datasets esd iemocap ravdess \
        --samples 100 \
        --seed 42
    echo "✅ 数据子集已准备"
fi

echo ""

# ================================================================
# 步骤5: 运行完整评估
# ================================================================
echo "================================================================"
echo "步骤5: 运行完整评估"
echo "================================================================"
echo "配置:"
echo "  - 码率点: 47个"
echo "    • 10-200 BPF (步长5): 39个点"
echo "    • 220-300 BPF (步长20): 5个点"
echo "  - 数据集: 3个 (ESD, IEMOCAP, RAVDESS)"
echo "  - 总评估: ~75,200次"
echo "  - 预计时间: 2-4小时"
echo ""

# 生成码率点列表
RATE_POINTS=$(python3 -c "print(','.join(map(str, list(range(10, 201, 5)) + list(range(220, 301, 20)))))")

OUTPUT_DIR="evaluation_results"
mkdir -p "$OUTPUT_DIR"

# 评估每个数据集
for DATASET in esd iemocap ravdess; do
    RESULT_FILE="$OUTPUT_DIR/${DATASET^^}_evaluation_results.json"
    
    if [ -f "$RESULT_FILE" ]; then
        echo "⚠️  发现现有评估结果: $RESULT_FILE"
        echo "   跳过 ${DATASET^^} 评估"
        echo "   (如需重新评估，请删除该文件)"
    else
        echo ""
        echo "📊 评估 ${DATASET^^}..."
        python run_evaluation.py \
            --dataset "$DATASET" \
            --data-root data_subset \
            --samples 100 \
            --rates "$RATE_POINTS" \
            --output-dir "$OUTPUT_DIR"
        echo "✅ ${DATASET^^} 评估完成"
    fi
done

echo ""
echo "✅ 所有数据集评估完成"
echo ""

# ================================================================
# 步骤6: 生成论文图表
# ================================================================
echo "================================================================"
echo "步骤6: 生成论文图表"
echo "================================================================"
echo "将生成5张关键图表:"
echo "  1. Overall Weighted F1 Score"
echo "  2. Overall Model Confidence"
echo "  3. Per-class Accuracy (Vertical Layout)"
echo "  4. Model Confidence by Emotion (Vertical Layout)"
echo "  5. Confusion Matrices 3×4 Grid"
echo ""

python generate_paper_figures.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 图表生成完成"
    echo ""
    echo "生成的图表:"
    ls -lh evaluation_results/figures/*.png
else
    echo ""
    echo "❌ 图表生成失败"
    exit 1
fi

# ================================================================
# 完成总结
# ================================================================
echo ""
echo "================================================================"
echo "🎉 实验复现完成！"
echo "================================================================"
echo ""
echo "📁 生成的文件:"
echo "   数据子集:     data_subset/"
echo "   评估结果:     evaluation_results/*_evaluation_results.json"
echo "   论文图表:     evaluation_results/figures/"
echo ""
echo "📊 关键图表:"
for fig in evaluation_results/figures/*.png; do
    if [ -f "$fig" ]; then
        echo "   - $(basename "$fig")"
    fi
done
echo ""
echo "完成时间: $(date)"
echo "================================================================"

