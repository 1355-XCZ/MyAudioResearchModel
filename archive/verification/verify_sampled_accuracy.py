"""
验证采样数据的分类准确率
评估100样本/情感的采样是否保持代表性
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from funasr import AutoModel
import numpy as np
from tqdm import tqdm
from collections import Counter
import logging

# 本地dataset导入
dataset_module_path = Path(__file__).parent / 'datasets'
if str(dataset_module_path) not in sys.path:
    sys.path.insert(0, str(dataset_module_path))

from iemocap_dataset import IEMOCAPDataset
from ravdess_dataset import RAVDESSDataset
from esd_dataset import ESDDataset

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_dataset_accuracy(dataset, dataset_name, model):
    """测试数据集的分类准确率"""
    # 加载样本（包含采样）
    dataset.samples = dataset.load_samples()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"测试{dataset_name}（采样后）")
    logger.info(f"{'='*60}")
    logger.info(f"总样本数: {len(dataset.samples)}")
    
    # 统计情感分布
    emotions = [s['emotion'] for s in dataset.samples]
    emotion_counts = Counter(emotions)
    logger.info(f"情感分布: {dict(emotion_counts)}")
    
    # 分类测试
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}
    
    for sample in tqdm(dataset.samples, desc=f"分类测试"):
        audio_path = sample['audio_path']
        features_path = audio_path.replace('.wav', '_ev2_frame.npy')
        
        if not Path(features_path).exists():
            continue
        
        # 加载特征
        features = np.load(features_path)  # (T, 768)
        
        # emotion2vec分类
        result = model.generate(features, granularity="frame", extract_embedding=False)
        pred = result[0]['labels'][0][0]  # 第一帧的预测
        
        # 映射
        ground_truth = dataset.map_emotion_to_ev2(sample['emotion'])
        
        # 处理RAVDESS的disgust映射
        if dataset_name == 'RAVDESS' and pred == 'disgusted':
            pred = 'disgust'
        
        # 统计
        if ground_truth not in per_class_total:
            per_class_total[ground_truth] = 0
            per_class_correct[ground_truth] = 0
        
        per_class_total[ground_truth] += 1
        total += 1
        
        if pred == ground_truth:
            per_class_correct[ground_truth] += 1
            correct += 1
    
    # 输出结果
    accuracy = correct / total * 100 if total > 0 else 0
    logger.info(f"\n总体准确率: {accuracy:.2f}% ({correct}/{total})")
    
    logger.info(f"\n每类准确率:")
    for emo in sorted(per_class_total.keys()):
        class_acc = per_class_correct[emo] / per_class_total[emo] * 100
        logger.info(f"  {emo}: {class_acc:.2f}% ({per_class_correct[emo]}/{per_class_total[emo]})")
    
    return accuracy


def main():
    logger.info("="*60)
    logger.info("验证采样数据的分类准确率（100样本/情感）")
    logger.info("="*60)
    
    # 加载emotion2vec模型
    logger.info("\n加载emotion2vec模型...")
    model = AutoModel(model="iic/emotion2vec_plus_base", hub="ms")
    if hasattr(model, 'model'):
        model.model = model.model.cuda()
    logger.info("✅ 模型已加载")
    
    # 测试3个数据集
    features_root = "/data/gpfs/projects/punim2341/haoguangzhou/data/evaluation_features"
    
    # IEMOCAP
    iemocap_dataset = IEMOCAPDataset(f"{features_root}/IEMOCAP")
    acc_iemocap = test_dataset_accuracy(iemocap_dataset, "IEMOCAP", model)
    
    # RAVDESS
    ravdess_dataset = RAVDESSDataset(f"{features_root}/RAVDESS")
    acc_ravdess = test_dataset_accuracy(ravdess_dataset, "RAVDESS", model)
    
    # ESD
    esd_dataset = ESDDataset(f"{features_root}/ESD", languages=['english', 'chinese'])
    acc_esd = test_dataset_accuracy(esd_dataset, "ESD", model)
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("总结")
    logger.info("="*60)
    logger.info(f"IEMOCAP: {acc_iemocap:.2f}%")
    logger.info(f"RAVDESS: {acc_ravdess:.2f}%")
    logger.info(f"ESD: {acc_esd:.2f}%")
    logger.info("\n✅ 验证完成")


if __name__ == "__main__":
    main()

