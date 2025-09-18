#!/bin/bash

# 项目清理和部署准备脚本
# 用于清理本地测试文件并准备部署到服务器

set -e

echo "======================================"
echo "🚀 准备部署到服务器"
echo "======================================"

# 检查当前目录
if [[ ! -f "emilia_batch_processor.py" ]]; then
    echo "❌ 错误: 请在 production_batch 目录下运行此脚本"
    exit 1
fi

# 1. 清理本地测试文件
echo "🧹 步骤 1: 清理本地测试文件..."
python cleanup_local_files.py

# 2. 检查git状态
echo ""
echo "📋 步骤 2: 检查git状态..."
cd ../../../  # 回到项目根目录

echo "当前git状态:"
git status --porcelain

# 3. 显示要提交的更改
echo ""
echo "📝 步骤 3: 准备git提交..."
echo "将要添加的新文件:"
git ls-files --others --exclude-standard | grep -E "(production_batch|\.gitignore)" || echo "  (无新文件)"

echo ""
echo "将要提交的修改:"
git diff --name-only --cached 2>/dev/null || echo "  (无暂存修改)"
git diff --name-only 2>/dev/null || echo "  (无未暂存修改)"

# 4. 创建部署包
echo ""
echo "📦 步骤 4: 创建部署包..."
cd src/emilia_mel_generator/production_batch

DEPLOY_PACKAGE="emilia_batch_system_$(date +%Y%m%d_%H%M%S).tar.gz"

tar -czf "$DEPLOY_PACKAGE" \
    emilia_batch_processor.py \
    spartan_config.yaml \
    batch_config.yaml \
    submit_spartan.sh \
    submit_spartan.slurm \
    submit_batch.sh \
    submit_batch_job.slurm \
    setup_data_and_models.py \
    test_batch_processor.py \
    cleanup_local_files.py \
    BATCH_PROCESSING_GUIDE.md \
    README.md \
    QUICK_START.md

echo "✅ 部署包创建完成: $DEPLOY_PACKAGE"

# 5. 显示部署说明
echo ""
echo "======================================"
echo "🎯 部署到Spartan集群"
echo "======================================"
echo ""
echo "1. 提交代码到git:"
echo "   cd ../../../"
echo "   git add ."
echo "   git commit -m 'Add production batch processing system'"
echo "   git push"
echo ""
echo "2. 上传部署包到Spartan:"
echo "   scp $DEPLOY_PACKAGE haoguangz@spartan.hpc.unimelb.edu.au:~/"
echo ""
echo "3. 在Spartan上部署:"
echo "   ssh haoguangz@spartan.hpc.unimelb.edu.au"
echo "   cd /data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
echo "   git pull  # 获取最新代码"
echo "   cd src/emilia_mel_generator"
echo "   mkdir -p production_batch"
echo "   cd production_batch"
echo "   tar -xzf ~/$DEPLOY_PACKAGE"
echo "   chmod +x *.sh"
echo ""
echo "4. 测试部署:"
echo "   python test_batch_processor.py"
echo ""
echo "5. 设置数据和模型:"
echo "   python setup_data_and_models.py \\"
echo "     --base_path /data/gpfs/projects/punim2341/haoguangzhou \\"
echo "     --cache_path /data/gpfs/projects/punim2341/haoguangzhou/cache \\"
echo "     --download_dataset \\"
echo "     --download_models \\"
echo "     --check_space"
echo ""
echo "6. 提交批处理作业:"
echo "   ./submit_spartan.sh -i /data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset \\"
echo "                       -o /data/gpfs/projects/punim2341/haoguangzhou/emilia_output"
echo ""
echo "======================================"
echo "📊 监控命令:"
echo "  squeue -u haoguangz                    # 查看作业状态"
echo "  tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-vevo-batch-*.out"
echo "======================================"

echo ""
echo "🎉 部署准备完成！"
echo "请按照上述步骤完成部署。"
