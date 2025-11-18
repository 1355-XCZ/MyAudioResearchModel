#!/bin/bash
# 一键运行论文实验复现
# 
# 此脚本将:
# 1. 准备评估子集数据（每个情感100样本）
# 2. 运行评估实验（3个数据集，47个码率点）
# 3. 生成所有论文图表
#
# 预计运行时间: 2-4小时（取决于GPU性能）

set -e  # 遇到错误立即退出

echo "================================================================"
echo "论文实验一键复现脚本"
echo "================================================================"
echo ""

# 检查Python环境
echo "检查Python环境..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "❌ PyTorch未安装"; exit 1; }
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
echo ""

# 步骤1: 准备评估子集数据
echo "================================================================"
echo "步骤 1/3: 准备评估子集数据"
echo "================================================================"
echo "随机抽取每个情感100个样本（种子=42，可复现）"
echo ""

if [ -d "data_subset" ]; then
    echo "⚠️  data_subset目录已存在"
    read -p "是否重新生成？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf data_subset
        python3 prepare_evaluation_subset.py
    else
        echo "跳过数据准备，使用现有子集"
    fi
else
    python3 prepare_evaluation_subset.py
fi

echo ""
echo "✅ 数据准备完成"
echo ""

# 步骤2 & 3: 运行评估和绘图
echo "================================================================"
echo "步骤 2-3/3: 运行评估实验并生成图表"
echo "================================================================"
echo "数据集: ESD, IEMOCAP, RAVDESS"
echo "码率点: 47个 (10-200 BPF步长5, 200-300 BPF步长20)"
echo "样本数: 100/情感"
echo ""
echo "⏰ 预计运行时间: 2-4小时"
echo ""

python3 reproduce_experiments.py --mode all

echo ""
echo "================================================================"
echo "🎉 实验复现完成！"
echo "================================================================"
echo ""
echo "结果位置:"
echo "  - 评估数据: evaluation_results/"
echo "  - 论文图表: evaluation_results/figures/"
echo ""
echo "查看结果:"
echo "  ls -lh evaluation_results/*.json"
echo "  ls -lh evaluation_results/figures/*.png"
echo ""

