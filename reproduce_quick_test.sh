#!/bin/bash

# ================================================================
# 超快速测试管道 - 10样本/情感
# ================================================================
# 功能：
#   1. 检查/创建虚拟环境（如果已存在则跳过）
#   2. 安装依赖（如果已安装则跳过）
#   3. 准备快速测试数据子集（10样本/情感）
#   4. 运行快速评估（4个码率点 × 3个数据集）
#   5. 生成所有论文图表
#
# 使用方法：
#   本地运行:    bash reproduce_quick_test.sh
#   SLURM集群:   sbatch scripts/quick_test_10samples.slurm
#
# 预计时间: 5-10分钟（首次需加15分钟安装）
# ================================================================

set -e  # 遇到错误立即退出

PROJECT_ROOT=$(pwd)
VENV_DIR="$PROJECT_ROOT/reproduce_venv"

echo "================================================================"
echo "⚡ 超快速测试管道（10样本/情感）"
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
# 步骤4: 准备快速测试数据子集
# ================================================================
echo "================================================================"
echo "步骤4: 准备快速测试数据子集"
echo "================================================================"
echo "⚠️  这是超快速测试版本"
echo ""
echo "配置:"
echo "  - 数据集: ESD, IEMOCAP, RAVDESS"
echo "  - 样本数: 10/情感（快速测试）"
echo "  - 随机种子: 42 (保证可复现)"
echo ""

echo "📊 准备快速测试数据子集..."
python prepare_evaluation_subset.py \
    --datasets esd iemocap ravdess \
    --samples 10 \
    --seed 42 \
    --target-data data_subset_quick

echo "✅ 快速测试数据子集已准备"
echo ""

# ================================================================
# 步骤5: 运行快速测试评估（4个码率点，10样本）
# ================================================================
echo "================================================================"
echo "步骤5: 运行快速测试评估"
echo "================================================================"
echo ""
echo "配置:"
echo "  - 码率点: 4个 (10, 100, 200, 300 BPF)"
echo "  - 数据集: 3个 (ESD, IEMOCAP, RAVDESS)"
echo "  - 样本数: 10/情感"
echo "  - 总评估: ~160次"
echo "  - 预计时间: 5-10分钟 ⚡"
echo ""

# 测试用码率点
TEST_RATES="10,100,200,300"

OUTPUT_DIR="evaluation_results_quick"
mkdir -p "$OUTPUT_DIR"

# 评估每个数据集
for DATASET in esd iemocap ravdess; do
    echo ""
    echo "📊 评估 ${DATASET^^}..."
    python run_evaluation.py \
        --dataset "$DATASET" \
        --data-root data_subset_quick \
        --samples 10 \
        --rates "$TEST_RATES" \
        --output-dir "$OUTPUT_DIR"
    echo "✅ ${DATASET^^} 评估完成"
done

echo ""
echo "✅ 所有数据集快速测试评估完成"
echo ""

# ================================================================
# 步骤6: 生成论文图表
# ================================================================
echo "================================================================"
echo "步骤6: 生成论文图表"
echo "================================================================"
echo "将生成5张关键图表 (基于快速测试数据):"
echo "  1. Overall Weighted F1 Score"
echo "  2. Overall Model Confidence"
echo "  3. Per-class Accuracy (Vertical Layout)"
echo "  4. Model Confidence by Emotion (Vertical Layout)"
echo "  5. Confusion Matrices 3×4 Grid"
echo ""

# 复制快速测试结果到标准位置以便绘图
mkdir -p evaluation_results
cp "$OUTPUT_DIR/rate_sweep_ESD.json" evaluation_results/ESD_evaluation_results.json
cp "$OUTPUT_DIR/rate_sweep_IEMOCAP.json" evaluation_results/IEMOCAP_evaluation_results.json
cp "$OUTPUT_DIR/rate_sweep_RAVDESS.json" evaluation_results/RAVDESS_evaluation_results.json

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
echo "🎉 快速测试完成！"
echo "================================================================"
echo ""
echo "📁 生成的文件:"
echo "   快速测试数据:   data_subset_quick/"
echo "   快速测试结果:   evaluation_results_quick/"
echo "   论文图表:       evaluation_results/figures/"
echo ""
echo "📊 关键图表:"
for fig in evaluation_results/figures/*.png; do
    if [ -f "$fig" ]; then
        echo "   - $(basename "$fig")"
    fi
done
echo ""
echo "⚠️  提醒: 这是10样本的快速测试结果"
echo ""
echo "下一步："
echo "  1. 检查流程和图表是否正常"
echo "  2. 运行100样本测试: bash reproduce_test_pipeline.sh"
echo "  3. 运行完整实验: bash reproduce_full_pipeline.sh"
echo ""
echo "完成时间: $(date)"
echo "================================================================"

