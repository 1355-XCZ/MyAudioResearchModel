#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=test_rvq
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/test_rvq_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/test_rvq_%j.err

# 加载模块
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load GCC/11.3.0
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# 激活虚拟环境
source /data/gpfs/projects/punim2341/haoguangzhou/venvs/vevo-source-fix/bin/activate

# 进入项目目录
cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

# 设置环境变量
export PYTHONPATH=/data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src:$PYTHONPATH

# 测试模式：只运行3个epoch验证流程
export TEST_MODE=true
export TEST_EPOCHS=3

# 运行RVQ训练测试
echo "=========================================="
echo "RVQ训练测试（gpu-l40s，3 epochs）"
echo "=========================================="
echo "开始时间: $(date)"

python train_rvq.py

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "完成时间: $(date)"
echo "退出码: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试成功！可以提交到gpu-h100"
else
    echo "❌ 测试失败，请检查错误日志"
fi

exit $EXIT_CODE

