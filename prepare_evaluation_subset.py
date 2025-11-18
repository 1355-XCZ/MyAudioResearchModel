#!/usr/bin/env python3
"""
准备评估子集数据

从完整数据集中随机抽取指定数量的样本，用于论文实验复现
- 每个情感类别随机抽取100个样本
- 保持数据平衡
- 可设置随机种子以确保可复现性
"""

import os
import sys
import json
import random
import shutil
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_samples_by_emotion(data_root: Path, dataset_name: str):
    """
    收集数据集中按情感分类的所有样本
    
    Returns:
        dict: {emotion: [sample_paths]}
    """
    samples_by_emotion = defaultdict(list)
    
    if dataset_name == 'ESD':
        # ESD结构: data_root/speaker_id/emotion/xxx_ev2_frame.npy
        for feat_file in data_root.glob("**/*_ev2_frame.npy"):
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            # 标准化情感标签
            emotion_normalized = emotion.lower()
            if emotion_normalized == 'surprise':
                emotion_normalized = 'surprised'
            
            samples_by_emotion[emotion_normalized].append({
                'feature_file': feat_file,
                'label_file': label_file,
                'emotion': emotion
            })
    
    elif dataset_name == 'IEMOCAP':
        # IEMOCAP结构: data_root/SessionX/xxx_ev2_frame.npy
        for feat_file in data_root.glob("**/*_ev2_frame.npy"):
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            emotion_normalized = emotion.lower()
            samples_by_emotion[emotion_normalized].append({
                'feature_file': feat_file,
                'label_file': label_file,
                'emotion': emotion
            })
    
    elif dataset_name == 'RAVDESS':
        # RAVDESS结构: data_root/xxx_ev2_frame.npy
        for feat_file in data_root.glob("*_ev2_frame.npy"):
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            emotion_normalized = emotion.lower()
            samples_by_emotion[emotion_normalized].append({
                'feature_file': feat_file,
                'label_file': label_file,
                'emotion': emotion
            })
    
    return samples_by_emotion


def sample_subset(samples_by_emotion, samples_per_emotion: int, seed: int = 42):
    """
    从每个情感类别中随机抽取指定数量的样本
    
    Args:
        samples_by_emotion: 按情感分类的样本字典
        samples_per_emotion: 每个情感抽取的样本数
        seed: 随机种子
    
    Returns:
        dict: {emotion: [selected_samples]}
    """
    random.seed(seed)
    subset = {}
    
    for emotion, samples in samples_by_emotion.items():
        available = len(samples)
        
        if available < samples_per_emotion:
            logger.warning(f"⚠️  {emotion}: 可用样本不足 ({available} < {samples_per_emotion})，使用全部样本")
            subset[emotion] = samples
        else:
            # 随机抽取
            subset[emotion] = random.sample(samples, samples_per_emotion)
            logger.info(f"✅ {emotion}: 从{available}个样本中随机抽取{samples_per_emotion}个")
    
    return subset


def copy_subset(subset, source_root: Path, target_root: Path, dataset_name: str):
    """
    复制子集文件到目标目录
    
    Args:
        subset: 选中的样本子集
        source_root: 源数据根目录
        target_root: 目标数据根目录
        dataset_name: 数据集名称
    """
    target_root.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    for emotion, samples in subset.items():
        for sample in tqdm(samples, desc=f"复制 {emotion}"):
            feat_file = sample['feature_file']
            label_file = sample['label_file']
            
            # 计算相对路径
            rel_path = feat_file.relative_to(source_root)
            
            # 目标路径
            target_feat = target_root / rel_path
            target_label = target_feat.parent / label_file.name
            
            # 创建目录
            target_feat.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            shutil.copy2(feat_file, target_feat)
            shutil.copy2(label_file, target_label)
            
            copied_count += 2
    
    logger.info(f"✅ 共复制{copied_count}个文件")


def save_subset_info(subset, output_file: Path, dataset_name: str, samples_per_emotion: int, seed: int):
    """
    保存子集信息到JSON文件
    
    Args:
        subset: 选中的样本子集
        output_file: 输出JSON文件路径
        dataset_name: 数据集名称
        samples_per_emotion: 每个情感的样本数
        seed: 随机种子
    """
    info = {
        'dataset': dataset_name,
        'samples_per_emotion': samples_per_emotion,
        'random_seed': seed,
        'emotions': {}
    }
    
    for emotion, samples in subset.items():
        info['emotions'][emotion] = {
            'count': len(samples),
            'files': [str(s['feature_file'].name) for s in samples]
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 子集信息保存到: {output_file}")


def prepare_dataset_subset(dataset_name: str, source_root: str, target_root: str, 
                          samples_per_emotion: int = 100, seed: int = 42):
    """
    准备单个数据集的评估子集
    
    Args:
        dataset_name: 数据集名称 (ESD/IEMOCAP/RAVDESS)
        source_root: 源数据根目录
        target_root: 目标数据根目录
        samples_per_emotion: 每个情感的样本数
        seed: 随机种子
    """
    logger.info("=" * 80)
    logger.info(f"准备 {dataset_name} 评估子集")
    logger.info("=" * 80)
    
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    if not source_path.exists():
        logger.error(f"❌ 源数据不存在: {source_path}")
        return False
    
    # 1. 收集所有样本
    logger.info(f"📂 扫描源数据目录: {source_path}")
    samples_by_emotion = collect_samples_by_emotion(source_path, dataset_name)
    
    if not samples_by_emotion:
        logger.error(f"❌ 未找到任何样本")
        return False
    
    logger.info(f"✅ 找到{len(samples_by_emotion)}个情感类别:")
    for emotion, samples in sorted(samples_by_emotion.items()):
        logger.info(f"  - {emotion}: {len(samples)}个样本")
    
    # 2. 随机抽取子集
    logger.info(f"\n🎲 随机抽取子集 (种子={seed}, 每类{samples_per_emotion}个)")
    subset = sample_subset(samples_by_emotion, samples_per_emotion, seed)
    
    total_samples = sum(len(samples) for samples in subset.values())
    logger.info(f"✅ 子集总样本数: {total_samples}")
    
    # 3. 复制文件
    logger.info(f"\n📋 复制文件到: {target_path}")
    copy_subset(subset, source_path, target_path, dataset_name)
    
    # 4. 保存子集信息
    info_file = target_path.parent / f"{dataset_name}_subset_info.json"
    save_subset_info(subset, info_file, dataset_name, samples_per_emotion, seed)
    
    logger.info(f"\n✅ {dataset_name} 子集准备完成")
    return True


def main():
    parser = argparse.ArgumentParser(description='准备评估子集数据')
    parser.add_argument('--source-data', type=str, default='data',
                       help='源数据根目录 (默认: data)')
    parser.add_argument('--target-data', type=str, default='data_subset',
                       help='目标数据根目录 (默认: data_subset)')
    parser.add_argument('--samples', type=int, default=100,
                       help='每个情感的样本数 (默认: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['ESD', 'IEMOCAP', 'RAVDESS'],
                       help='要处理的数据集 (默认: ESD IEMOCAP RAVDESS)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("评估子集数据准备工具")
    logger.info("=" * 80)
    logger.info(f"源数据目录: {args.source_data}")
    logger.info(f"目标目录: {args.target_data}")
    logger.info(f"每个情感样本数: {args.samples}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"数据集: {', '.join(args.datasets)}")
    
    success_count = 0
    
    for dataset in args.datasets:
        # 统一转换为大写（数据目录是大写的）
        dataset_upper = dataset.upper()
        source_root = Path(args.source_data) / dataset_upper
        target_root = Path(args.target_data) / dataset_upper
        
        success = prepare_dataset_subset(
            dataset_name=dataset_upper,
            source_root=str(source_root),
            target_root=str(target_root),
            samples_per_emotion=args.samples,
            seed=args.seed
        )
        
        if success:
            success_count += 1
        
        print()  # 空行分隔
    
    # 最终汇总
    logger.info("=" * 80)
    logger.info("汇总")
    logger.info("=" * 80)
    logger.info(f"成功: {success_count}/{len(args.datasets)}")
    logger.info(f"子集数据保存到: {args.target_data}/")
    logger.info(f"子集信息文件: {args.target_data}/*_subset_info.json")
    
    if success_count == len(args.datasets):
        logger.info("\n🎉 所有子集准备完成！")
        logger.info("\n下一步:")
        logger.info("  1. 检查子集信息: cat data_subset/*_subset_info.json")
        logger.info("  2. 运行实验: python reproduce_experiments.py --mode all")
    else:
        logger.error("\n❌ 部分数据集处理失败")
        sys.exit(1)


if __name__ == '__main__':
    main()

