#!/bin/bash

# Spartan集群部署脚本
# 用于将批处理系统部署到Spartan集群

set -e

echo "======================================"
echo "🚀 部署Emilia批处理系统到Spartan集群"
echo "======================================"

# Spartan集群配置
SPARTAN_USER="haoguangz"
SPARTAN_HOST="spartan.hpc.unimelb.edu.au"
PROJECT_ROOT="/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
LOCAL_PROJECT_ROOT="../.."  # 相对于production_batch的项目根目录

# 检查本地文件
echo "📋 检查本地文件..."
required_files=(
    "emilia_batch_processor.py"
    "spartan_config.yaml"
    "submit_spartan.sh"
    "submit_spartan.slurm"
    "BATCH_PROCESSING_GUIDE.md"
    "README.md"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ 缺少文件: $file"
        exit 1
    fi
done

echo "✅ 所有必需文件存在"

# 创建部署包
echo "📦 创建部署包..."
DEPLOY_PACKAGE="emilia_batch_system_$(date +%Y%m%d_%H%M%S).tar.gz"

tar -czf "$DEPLOY_PACKAGE" \
    emilia_batch_processor.py \
    spartan_config.yaml \
    batch_config.yaml \
    submit_spartan.sh \
    submit_spartan.slurm \
    submit_batch.sh \
    submit_batch_job.slurm \
    test_batch_processor.py \
    BATCH_PROCESSING_GUIDE.md \
    README.md

echo "✅ 部署包创建完成: $DEPLOY_PACKAGE"

# 显示部署信息
echo ""
echo "======================================"
echo "📋 部署信息"
echo "======================================"
echo "🎯 目标服务器: $SPARTAN_HOST"
echo "👤 用户: $SPARTAN_USER"
echo "📁 项目路径: $PROJECT_ROOT"
echo "📦 部署包: $DEPLOY_PACKAGE"
echo ""

# 显示部署命令
echo "======================================"
echo "🚀 部署命令 (请手动执行)"
echo "======================================"
echo ""
echo "1. 上传部署包到Spartan:"
echo "   scp $DEPLOY_PACKAGE $SPARTAN_USER@$SPARTAN_HOST:~/"
echo ""
echo "2. 登录到Spartan并部署:"
echo "   ssh $SPARTAN_USER@$SPARTAN_HOST"
echo "   cd $PROJECT_ROOT"
echo "   tar -xzf ~/$DEPLOY_PACKAGE -C src/emilia_mel_generator/production_batch/"
echo "   chmod +x src/emilia_mel_generator/production_batch/*.sh"
echo ""
echo "3. 测试部署:"
echo "   cd src/emilia_mel_generator/production_batch"
echo "   python test_batch_processor.py"
echo ""
echo "4. 提交作业:"
echo "   ./submit_spartan.sh"
echo ""

# 显示监控命令
echo "======================================"
echo "📊 监控命令"
echo "======================================"
echo ""
echo "查看作业状态:"
echo "  squeue -u $SPARTAN_USER"
echo ""
echo "查看作业详情:"
echo "  scontrol show job JOBID"
echo ""
echo "查看日志:"
echo "  tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-vevo-batch-JOBID.out"
echo ""
echo "取消作业:"
echo "  scancel JOBID"
echo ""

echo "======================================"
echo "✅ 部署准备完成！"
echo "请按照上述命令手动完成部署。"
echo "======================================"
