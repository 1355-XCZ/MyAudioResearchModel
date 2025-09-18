#!/bin/bash

# 批任务管理器提交脚本 - 支持大规模数据集分批处理
# 解决集群无法一次性处理完整数据集的问题

set -e

# Spartan集群默认配置
DEFAULT_DATASET_PATH="/data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset"
DEFAULT_OUTPUT_PATH="/data/gpfs/projects/punim2341/haoguangzhou/emilia_training_data"
DEFAULT_CACHE_PATH="/data/gpfs/projects/punim2341/haoguangzhou/cache"

show_usage() {
    echo "用法: $0 [选项] [动作]"
    echo ""
    echo "批任务管理器 - 支持大规模数据集分批处理"
    echo "🎯 自动分批、断点续传、并行处理"
    echo ""
    echo "选项:"
    echo "  -d, --dataset PATH      数据集路径 (默认: $DEFAULT_DATASET_PATH)"
    echo "  -o, --output PATH       输出路径 (默认: $DEFAULT_OUTPUT_PATH)"
    echo "  -c, --cache PATH        缓存路径 (默认: $DEFAULT_CACHE_PATH)"
    echo "  -s, --samples-per-batch N  每批样本数 (默认: 500)"
    echo "  -j, --max-jobs N        最大并发作业数 (默认: 3)"
    echo "  -H, --hours-per-lang N  每种语言目标时长 (默认: 50.0)"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "动作:"
    echo "  start                   开始批任务管理 (默认)"
    echo "  monitor                 只监控现有作业"
    echo "  merge                   合并完成的batch结果"
    echo "  status                  显示当前状态"
    echo "  restart                 重启失败的batch"
    echo ""
    echo "示例:"
    echo "  $0                                    # 开始批任务管理"
    echo "  $0 -s 1000 -j 5                     # 每批1000样本，最多5个并发"
    echo "  $0 monitor                           # 只监控现有作业"
    echo "  $0 merge                             # 合并结果"
    echo "  $0 -H 25.0 start                    # 每种语言25小时"
    echo ""
    echo "特性:"
    echo "  ✅ 自动分批处理大数据集"
    echo "  ✅ 断点续传，任务中断可恢复"
    echo "  ✅ 并行处理多个batch"
    echo "  ✅ 自动重试失败的batch"
    echo "  ✅ 使用Emo-Emilia中性参考音频"
}

# 解析参数
DATASET_PATH="$DEFAULT_DATASET_PATH"
OUTPUT_PATH="$DEFAULT_OUTPUT_PATH"
CACHE_PATH="$DEFAULT_CACHE_PATH"
SAMPLES_PER_BATCH=500
MAX_JOBS=3
HOURS_PER_LANG=50.0
ACTION="start"

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -c|--cache)
            CACHE_PATH="$2"
            shift 2
            ;;
        -s|--samples-per-batch)
            SAMPLES_PER_BATCH="$2"
            shift 2
            ;;
        -j|--max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -H|--hours-per-lang)
            HOURS_PER_LANG="$2"
            shift 2
            ;;
        start|monitor|merge|status|restart)
            ACTION="$1"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 创建必要的目录
mkdir -p "$OUTPUT_PATH"
mkdir -p "/data/gpfs/projects/punim2341/haoguangzhou/logs"

echo "======================================"
echo "🎯 批任务管理器 - Spartan集群"
echo "======================================"
echo "📁 数据集: $DATASET_PATH"
echo "📁 输出: $OUTPUT_PATH"
echo "📁 缓存: $CACHE_PATH"
echo "📦 每批样本数: $SAMPLES_PER_BATCH"
echo "⚡ 最大并发: $MAX_JOBS"
echo "🕐 目标时长: 每种语言 $HOURS_PER_LANG 小时"
echo "🎬 执行动作: $ACTION"
echo "======================================"

# 根据动作执行不同操作
case $ACTION in
    start)
        echo "🚀 启动批任务管理..."
        python batch_task_manager.py \
            --dataset_path "$DATASET_PATH" \
            --output_base_path "$OUTPUT_PATH" \
            --cache_path "$CACHE_PATH" \
            --samples_per_batch "$SAMPLES_PER_BATCH" \
            --max_concurrent_jobs "$MAX_JOBS" \
            --hours_per_lang "$HOURS_PER_LANG" \
            --action start
        ;;
    monitor)
        echo "👀 监控现有作业..."
        python batch_task_manager.py \
            --dataset_path "$DATASET_PATH" \
            --output_base_path "$OUTPUT_PATH" \
            --cache_path "$CACHE_PATH" \
            --action monitor
        ;;
    merge)
        echo "🔗 合并batch结果..."
        python batch_task_manager.py \
            --dataset_path "$DATASET_PATH" \
            --output_base_path "$OUTPUT_PATH" \
            --cache_path "$CACHE_PATH" \
            --action merge
        ;;
    status)
        echo "📊 显示任务状态..."
        python batch_task_manager.py \
            --dataset_path "$DATASET_PATH" \
            --output_base_path "$OUTPUT_PATH" \
            --cache_path "$CACHE_PATH" \
            --action status
        ;;
    restart)
        echo "🔄 重启批任务管理..."
        python batch_task_manager.py \
            --dataset_path "$DATASET_PATH" \
            --output_base_path "$OUTPUT_PATH" \
            --cache_path "$CACHE_PATH" \
            --samples_per_batch "$SAMPLES_PER_BATCH" \
            --max_concurrent_jobs "$MAX_JOBS" \
            --hours_per_lang "$HOURS_PER_LANG" \
            --action start
        ;;
esac

echo "======================================"
echo "✅ 批任务管理器执行完成"
echo ""
echo "📋 常用监控命令:"
echo "  squeue -u haoguangz                    # 查看所有作业"
echo "  ./submit_batch_manager.sh monitor     # 监控batch状态"
echo "  ./submit_batch_manager.sh status      # 显示详细状态"
echo "  ./submit_batch_manager.sh merge       # 手动合并结果"
echo "======================================"
