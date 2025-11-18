#!/usr/bin/env python3
"""
复现论文实验的完整流程脚本

用法:
    python reproduce_experiments.py --mode all  # 运行评估+绘图
    python reproduce_experiments.py --mode eval  # 只运行评估
    python reproduce_experiments.py --mode plot  # 只绘图（需要已有评估结果）
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== 实验配置 ==========
# 码率采样点：10-200（步长5），200-300（步长20）
RATE_POINTS = list(range(10, 201, 5)) + list(range(220, 301, 20))

# 数据集配置
DATASETS = ['esd', 'iemocap', 'ravdess']
SAMPLES_PER_EMOTION = 100

# 数据目录（使用子集数据）
DATA_ROOT = 'data_subset'  # 使用准备好的评估子集

# 模型检查点
RVQ_CHECKPOINT = 'checkpoints/grouped_rvq_best.pt'
ENTROPY_CHECKPOINT = 'checkpoints/entropy_model_best.pt'

# 输出目录
OUTPUT_DIR = 'evaluation_results'
PLOT_OUTPUT_DIR = 'evaluation_results/figures'


def check_prerequisites():
    """检查前置条件"""
    logger.info("=" * 80)
    logger.info("检查前置条件")
    logger.info("=" * 80)
    
    # 检查模型文件
    if not Path(RVQ_CHECKPOINT).exists():
        logger.error(f"❌ RVQ模型不存在: {RVQ_CHECKPOINT}")
        return False
    logger.info(f"✅ RVQ模型: {RVQ_CHECKPOINT}")
    
    if not Path(ENTROPY_CHECKPOINT).exists():
        logger.error(f"❌ 熵模型不存在: {ENTROPY_CHECKPOINT}")
        return False
    logger.info(f"✅ 熵模型: {ENTROPY_CHECKPOINT}")
    
    # 检查数据集
    data_root = Path(DATA_ROOT)
    if not data_root.exists():
        logger.error(f"❌ 数据目录不存在: {data_root}")
        logger.error(f"   请先运行: python prepare_evaluation_subset.py")
        return False
    
    for dataset in DATASETS:
        dataset_path = data_root / dataset.upper()
        if not dataset_path.exists():
            logger.error(f"❌ 数据集不存在: {dataset_path}")
            logger.error(f"   请先运行: python prepare_evaluation_subset.py")
            return False
        
        # 检查子集信息文件
        subset_info = data_root / f"{dataset.upper()}_subset_info.json"
        if subset_info.exists():
            with open(subset_info) as f:
                info = json.load(f)
            total_samples = sum(e['count'] for e in info['emotions'].values())
            logger.info(f"✅ 数据集: {dataset.upper()} ({total_samples}样本)")
        else:
            logger.info(f"✅ 数据集: {dataset_path}")
    
    # 检查Python依赖
    try:
        import torch
        import numpy as np
        import matplotlib
        import seaborn
        import sklearn
        import tqdm
        logger.info(f"✅ PyTorch版本: {torch.__version__}")
        logger.info(f"✅ CUDA可用: {torch.cuda.is_available()}")
    except ImportError as e:
        logger.error(f"❌ 缺少依赖: {e}")
        return False
    
    return True


def run_evaluation():
    """运行所有数据集的评估"""
    logger.info("\n" + "=" * 80)
    logger.info("开始评估实验")
    logger.info("=" * 80)
    
    rates_str = ",".join(map(str, RATE_POINTS))
    
    for dataset in DATASETS:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"评估数据集: {dataset.upper()}")
        logger.info(f"{'=' * 60}")
        logger.info(f"  样本数: {SAMPLES_PER_EMOTION}/情感")
        logger.info(f"  码率点: {len(RATE_POINTS)}个 ({RATE_POINTS[0]}-{RATE_POINTS[-1]} BPF)")
        
        cmd = [
            sys.executable, 'run_evaluation.py',
            '--dataset', dataset,
            '--samples', str(SAMPLES_PER_EMOTION),
            '--rates', rates_str,
            '--data-root', DATA_ROOT,  # 使用子集数据
            '--output-dir', OUTPUT_DIR,
            '--rvq-checkpoint', RVQ_CHECKPOINT,
            '--entropy-checkpoint', ENTROPY_CHECKPOINT
        ]
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f"✅ {dataset.upper()}评估完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {dataset.upper()}评估失败: {e}")
            return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 所有数据集评估完成")
    logger.info("=" * 80)
    return True


def run_plotting():
    """运行所有绘图脚本"""
    logger.info("\n" + "=" * 80)
    logger.info("开始生成论文图表")
    logger.info("=" * 80)
    
    # 确保输出目录存在
    Path(PLOT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 定义绘图脚本列表
    plot_scripts = [
        {
            'script': 'plot_overall_accuracy_and_confusion.py',
            'description': '整体准确率和混淆矩阵（单数据集）',
            'datasets': DATASETS
        },
        {
            'script': 'plot_overall_confidence_with_ci.py',
            'description': '整体置信度（带置信区间）',
            'datasets': DATASETS
        },
        {
            'script': 'plot_overall_f1_with_ci.py',
            'description': '整体F1分数（带置信区间）',
            'datasets': DATASETS
        },
        {
            'script': 'plot_emotion_curves_with_ci.py',
            'description': '各情感类别准确率曲线（带置信区间）',
            'datasets': DATASETS
        },
        {
            'script': 'plot_emotion_curves_unified_colors.py',
            'description': '统一颜色方案的情感准确率曲线',
            'datasets': DATASETS
        },
        {
            'script': 'plot_emotion_model_confidence.py',
            'description': '各情感的模型置信度曲线',
            'datasets': DATASETS
        },
        {
            'script': 'plot_all_datasets_emotion_accuracy_confidence_vertical.py',
            'description': '跨数据集情感准确率和置信度对比图（垂直布局）',
            'datasets': None  # 处理所有数据集
        },
        {
            'script': 'plot_all_datasets_confusion_matrices_3x4.py',
            'description': '跨数据集混淆矩阵（3×4网格）',
            'datasets': None  # 处理所有数据集
        }
    ]
    
    for plot_config in plot_scripts:
        script = plot_config['script']
        description = plot_config['description']
        datasets = plot_config['datasets']
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"生成图表: {description}")
        logger.info(f"脚本: {script}")
        
        if datasets:
            # 需要为每个数据集分别运行
            for dataset in datasets:
                logger.info(f"  处理: {dataset.upper()}")
                
                # 检查评估结果是否存在
                result_file = Path(OUTPUT_DIR) / f"{dataset.upper()}_evaluation_results.json"
                if not result_file.exists():
                    logger.warning(f"⚠️  评估结果不存在，跳过: {result_file}")
                    continue
                
                cmd = [sys.executable, script]
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    logger.info(f"  ✅ {dataset.upper()}图表生成完成")
                except subprocess.CalledProcessError as e:
                    logger.error(f"  ❌ 生成失败: {e}")
                    if e.stderr:
                        logger.error(f"  错误信息: {e.stderr}")
        else:
            # 处理所有数据集的综合图表
            cmd = [sys.executable, script]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"  ✅ 综合图表生成完成")
            except subprocess.CalledProcessError as e:
                logger.error(f"  ❌ 生成失败: {e}")
                if e.stderr:
                    logger.error(f"  错误信息: {e.stderr}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 所有图表生成完成")
    logger.info("=" * 80)
    logger.info(f"图表保存位置: {PLOT_OUTPUT_DIR}")
    return True


def summarize_results():
    """汇总实验结果"""
    logger.info("\n" + "=" * 80)
    logger.info("实验结果汇总")
    logger.info("=" * 80)
    
    for dataset in DATASETS:
        result_file = Path(OUTPUT_DIR) / f"{dataset.upper()}_evaluation_results.json"
        
        if not result_file.exists():
            logger.warning(f"⚠️  {dataset.upper()}: 结果文件不存在")
            continue
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"\n{dataset.upper()}:")
            logger.info(f"  样本数: {data.get('samples', 'N/A')}")
            logger.info(f"  码率点: {len(data.get('rate_points', {}))}个")
            
            # 找到准确率最高和最低的点
            rate_points = data.get('rate_points', {})
            if rate_points:
                accuracies = [(rp['target_rate_bpf'], rp['accuracy']) 
                             for rp in rate_points.values()]
                accuracies.sort(key=lambda x: x[1])
                
                min_acc = accuracies[0]
                max_acc = accuracies[-1]
                
                logger.info(f"  准确率范围: {min_acc[1]*100:.2f}% (@{min_acc[0]}BPF) - {max_acc[1]*100:.2f}% (@{max_acc[0]}BPF)")
        
        except Exception as e:
            logger.error(f"❌ 读取{dataset.upper()}结果失败: {e}")
    
    # 列出生成的图表
    logger.info(f"\n生成的图表:")
    if Path(PLOT_OUTPUT_DIR).exists():
        plot_files = list(Path(PLOT_OUTPUT_DIR).glob("*.png"))
        for pf in sorted(plot_files):
            logger.info(f"  - {pf.name}")
    
    logger.info("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='复现论文实验')
    parser.add_argument('--mode', type=str, choices=['all', 'eval', 'plot'], default='all',
                       help='运行模式: all=评估+绘图, eval=只评估, plot=只绘图')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("论文实验复现脚本")
    logger.info("=" * 80)
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"数据集: {', '.join([d.upper() for d in DATASETS])}")
    logger.info(f"码率点: {len(RATE_POINTS)}个 ({RATE_POINTS[0]}-{RATE_POINTS[-1]} BPF)")
    logger.info(f"样本数: {SAMPLES_PER_EMOTION}/情感")
    
    # 检查前置条件
    if not check_prerequisites():
        logger.error("❌ 前置条件检查失败，请先安装依赖并准备数据")
        sys.exit(1)
    
    # 执行相应模式
    success = True
    
    if args.mode in ['all', 'eval']:
        success = run_evaluation()
        if not success:
            logger.error("❌ 评估失败")
            sys.exit(1)
    
    if args.mode in ['all', 'plot']:
        success = run_plotting()
        if not success:
            logger.error("❌ 绘图失败")
            sys.exit(1)
    
    # 汇总结果
    summarize_results()
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 实验复现完成！")
    logger.info("=" * 80)
    logger.info(f"评估结果: {OUTPUT_DIR}/")
    logger.info(f"论文图表: {PLOT_OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

