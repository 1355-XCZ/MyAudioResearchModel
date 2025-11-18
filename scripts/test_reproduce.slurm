#!/bin/bash
#SBATCH --job-name=test_reproduce
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/test_reproduce_%j.out
#SBATCH --error=logs/test_reproduce_%j.err

# 快速测试实验复现系统
# 用于验证环境和脚本是否正常工作
# 测试配置：每个情感10样本，3个码率点

echo "================================================================"
echo "实验复现系统测试"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo ""

# 创建日志目录
mkdir -p logs

# 加载环境配置
if [ -f "local_config.sh" ]; then
    echo "加载本地配置..."
    source local_config.sh
else
    echo "⚠️  未找到 local_config.sh，使用默认环境"
fi

# 激活Python环境
if [ -n "$CONDA_ENV" ]; then
    echo "激活Conda环境: $CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV
elif [ -n "$VENV_PATH" ]; then
    echo "激活虚拟环境: $VENV_PATH"
    source $VENV_PATH/bin/activate
fi

echo ""
echo "================================================================"
echo "步骤1: 检查依赖"
echo "================================================================"
python3 test_dependencies.py
if [ $? -ne 0 ]; then
    echo "❌ 依赖检查失败"
    exit 1
fi

echo ""
echo "================================================================"
echo "步骤2: 准备测试子集（10样本/情感）"
echo "================================================================"

# 如果测试子集已存在，先删除
if [ -d "data_subset_test" ]; then
    echo "删除旧的测试子集..."
    rm -rf data_subset_test
fi

python3 prepare_evaluation_subset.py \
    --source-data data \
    --target-data data_subset_test \
    --samples 10 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ 数据准备失败"
    exit 1
fi

echo ""
echo "================================================================"
echo "步骤3: 运行快速评估（3个码率点）"
echo "================================================================"

# 测试ESD数据集
echo "测试 ESD 数据集..."
python3 run_evaluation.py \
    --dataset esd \
    --data-root data_subset_test \
    --samples 10 \
    --rates "10,50,100" \
    --output-dir evaluation_results_test

if [ $? -ne 0 ]; then
    echo "❌ ESD评估失败"
    exit 1
fi

echo ""
echo "================================================================"
echo "步骤4: 测试绘图脚本"
echo "================================================================"

# 测试单个绘图脚本
echo "测试绘图功能..."
python3 plot_overall_accuracy_and_confusion.py

if [ $? -ne 0 ]; then
    echo "⚠️  绘图测试失败（可能缺少评估结果）"
    echo "   继续测试..."
fi

echo ""
echo "================================================================"
echo "测试完成"
echo "================================================================"

# 显示生成的文件
echo "生成的测试文件："
if [ -d "data_subset_test" ]; then
    echo "  ✅ 测试子集："
    du -sh data_subset_test/
    ls -1 data_subset_test/
fi

if [ -d "evaluation_results_test" ]; then
    echo "  ✅ 评估结果："
    ls -lh evaluation_results_test/*.json 2>/dev/null | head -5
fi

echo ""
echo "🎉 快速测试通过！"
echo ""
echo "完整实验运行命令："
echo "  sbatch scripts/run_full_reproduce.slurm"
echo ""
echo "或手动运行："
echo "  python3 prepare_evaluation_subset.py"
echo "  python3 reproduce_experiments.py --mode all"

# 清理测试数据（可选）
read -t 10 -p "是否删除测试数据？(y/n，10秒后自动保留): " cleanup
if [[ $cleanup =~ ^[Yy]$ ]]; then
    echo "清理测试数据..."
    rm -rf data_subset_test
    rm -rf evaluation_results_test
    echo "✅ 清理完成"
else
    echo "保留测试数据供检查"
fi

echo "完成时间: $(date)"

