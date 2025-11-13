#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=compute_norm
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/compute_norm_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/compute_norm_%j.err

# 加载模块
module purge
module load GCCcore/11.3.0
module load Python/3.10.4

# 激活虚拟环境
source /data/gpfs/projects/punim2341/haoguangzhou/venvs/vevo-source-fix/bin/activate

# 进入项目目录
cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

# 设置环境变量
export PYTHONPATH=/data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src:$PYTHONPATH

# 运行归一化参数计算
echo "=========================================="
echo "计算100h中英文数据的归一化参数"
echo "=========================================="
echo "开始时间: $(date)"
echo "数据目录: /data/gpfs/projects/punim2341/haoguangzhou/data/emilia_vevo_training_50h"
echo "输出文件: ev2_mean_std_100h_EN_ZH.npz"
echo ""

python compute_normalization.py \
  --data-dir /data/gpfs/projects/punim2341/haoguangzhou/data/emilia_vevo_training_50h \
  --output ev2_mean_std_100h_EN_ZH.npz

echo ""
echo "=========================================="
echo "完成时间: $(date)"
echo "=========================================="

# 验证输出文件
if [ -f "ev2_mean_std_100h_EN_ZH.npz" ]; then
    echo "✅ 归一化参数文件已生成"
    ls -lh ev2_mean_std_100h_EN_ZH.npz
else
    echo "❌ 归一化参数文件生成失败"
    exit 1
fi

