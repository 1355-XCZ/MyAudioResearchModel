#!/bin/bash

# Spartan 多批次提交脚本
# 用于分批处理大规模数据提取任务

# 配置参数
TOTAL_BATCHES=10        # 总批次数 (可调整)
TARGET_HOURS=50         # 每语言目标小时数
K_VARIANTS=1            # K值 (默认1，可根据需要调整)

echo "=== Spartan 多批次任务提交 ==="
echo "总批次数: $TOTAL_BATCHES"
echo "每语言目标小时: $TARGET_HOURS"
echo "K值: $K_VARIANTS"
echo "=================================="

# 创建日志目录
mkdir -p /data/gpfs/projects/punim2341/haoguangzhou/logs

# 提交所有批次
for batch_id in $(seq 0 $((TOTAL_BATCHES-1))); do
    echo "提交批次 $((batch_id+1))/$TOTAL_BATCHES..."
    
    # 设置环境变量并提交作业
    sbatch \
        --export=BATCH_ID=$batch_id,TOTAL_BATCHES=$TOTAL_BATCHES,TARGET_HOURS=$TARGET_HOURS,K_VARIANTS=$K_VARIANTS \
        --job-name="emilia-batch-$batch_id" \
        submit_spartan.slurm
    
    # 等待一小段时间避免同时提交太多作业
    sleep 2
done

echo "=== 所有批次已提交 ==="
echo "使用以下命令监控作业状态:"
echo "  squeue -u \$USER"
echo "  squeue -j <job_id>"
echo ""
echo "作业日志位置:"
echo "  /data/gpfs/projects/punim2341/haoguangzhou/logs/"
echo ""
echo "完成后的数据位置:"
echo "  /data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset/"
