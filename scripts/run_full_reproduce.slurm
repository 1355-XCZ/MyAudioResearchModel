#!/bin/bash
#SBATCH --job-name=full_reproduce
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/full_reproduce_%j.out
#SBATCH --error=logs/full_reproduce_%j.err

# ================================================================
# 完整实验复现管道 - SLURM版本
# 使用智能脚本 reproduce_full_pipeline.sh
# 47个码率点 × 3个数据集
# ================================================================

set -e

echo "================================================================"
echo "🚀 SLURM完整实验复现管道（47码率点）"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

mkdir -p logs

echo "系统信息："
echo "  Python: $(which python3) ($(python3 --version))"
echo "  Working directory: $(pwd)"
echo ""

echo "GPU信息："
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

echo "================================================================"
echo "运行完整实验管道"
echo "================================================================"
echo ""

# 调用智能完整脚本
bash reproduce_full_pipeline.sh

RESULT=$?

echo ""
echo "================================================================"
echo "SLURM作业完成"
echo "================================================================"

if [ $RESULT -eq 0 ]; then
    echo "✅ 完整实验成功！"
    echo ""
    echo "生成的文件:"
    echo "  评估结果: evaluation_results/"
    ls -lh evaluation_results/*_evaluation_results.json 2>/dev/null || true
    echo ""
    echo "  论文图表: evaluation_results/figures/"
    ls -lh evaluation_results/figures/*.png 2>/dev/null || true
    echo ""
    echo "🎉 所有实验结果已生成，可用于论文！"
else
    echo "❌ 实验失败，请检查日志"
    echo "  输出: logs/full_reproduce_${SLURM_JOB_ID}.out"
    echo "  错误: logs/full_reproduce_${SLURM_JOB_ID}.err"
fi

echo ""
echo "完成时间: $(date)"

exit $RESULT
