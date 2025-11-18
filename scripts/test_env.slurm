#!/bin/bash
#SBATCH --job-name=test_env_install
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/test_env_%j.out
#SBATCH --error=logs/test_env_%j.err

# 严格的环境测试脚本
# 从零开始：创建虚拟环境、安装依赖、验证环境

set -e  # 遇到错误立即退出

echo "================================================================"
echo "环境完整性测试（从零开始）"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

# 创建日志目录
mkdir -p logs

# 定义虚拟环境路径
VENV_PATH="./test_venv_${SLURM_JOB_ID}"

# 清理可能存在的旧测试环境
if [ -d "$VENV_PATH" ]; then
    echo "删除旧的测试环境: $VENV_PATH"
    rm -rf "$VENV_PATH"
fi

# 显示系统信息
echo "系统信息："
echo "  Python: $(which python3) ($(python3 --version))"
echo "  Pip: $(which pip3)"
echo "  Working directory: $(pwd)"
echo ""

# 检查GPU
echo "GPU信息："
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️  nvidia-smi 不可用"
fi

echo ""
echo "================================================================"
echo "步骤1: 创建虚拟环境"
echo "================================================================"

python3 -m venv "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "❌ 虚拟环境创建失败"
    exit 1
fi

echo "✅ 虚拟环境已创建: $VENV_PATH"

# 激活虚拟环境
source "$VENV_PATH/bin/activate"

echo "✅ 虚拟环境已激活"
echo "  Python: $(which python)"
echo "  Pip: $(which pip)"
echo ""

echo "================================================================"
echo "步骤2: 升级pip"
echo "================================================================"

pip install --upgrade pip setuptools wheel
echo "✅ Pip已升级"
echo ""

echo "================================================================"
echo "步骤3: 安装依赖（从requirements.txt）"
echo "================================================================"

pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    deactivate
    rm -rf "$VENV_PATH"
    exit 1
fi

echo "✅ 所有依赖已安装"
echo ""

echo "================================================================"
echo "步骤4: 运行依赖检查脚本"
echo "================================================================"

python test_dependencies.py

TEST_RESULT=$?

echo ""
echo "================================================================"
echo "步骤5: 清理测试环境"
echo "================================================================"

# 停用虚拟环境
deactivate

# 删除测试虚拟环境
echo "删除测试环境: $VENV_PATH"
rm -rf "$VENV_PATH"

echo "✅ 测试环境已清理"
echo ""

echo "================================================================"
echo "测试结果总结"
echo "================================================================"

if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ 环境测试完全通过！"
    echo ""
    echo "所有依赖包可以正确安装和导入。"
    echo ""
    echo "下一步："
    echo "  1. 在您的工作环境中安装依赖："
    echo "     pip install -r requirements.txt"
    echo ""
    echo "  2. 运行快速测试："
    echo "     sbatch scripts/test_reproduce.slurm"
    echo ""
    echo "  3. 运行完整实验："
    echo "     sbatch scripts/run_full_reproduce.slurm"
    exit 0
else
    echo "❌ 依赖检查失败"
    echo ""
    echo "虽然依赖包安装成功，但某些包无法正确导入。"
    echo "请检查日志: logs/test_env_${SLURM_JOB_ID}.out"
    exit 1
fi
