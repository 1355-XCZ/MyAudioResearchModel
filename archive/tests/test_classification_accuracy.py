"""测试3个数据集的分类准确率，对比emotion_bottleneck_rate结果"""

import torch
import numpy as np
from pathlib import Path
from collections import Counter
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# emotion_bottleneck_rate的结果（EMOTION2VEC_CLASSIFICATION_ANALYSIS_REPORT.md）
EXPECTED_ACC = {
    'ESD': 0.9584,
    'RAVDESS': 0.9319,
    'IEMOCAP': 0.7416
}

def test_dataset_accuracy(dataset_name, model):
    """测试单个数据集的分类准确率（按类别统计）"""
    features_root = Path(f"/data/gpfs/projects/punim2341/haoguangzhou/data/evaluation_features/{dataset_name}")
    feat_files = list(features_root.glob("**/*_ev2_frame.npy"))
    
    if len(feat_files) == 0:
        return None
    
    # 采样测试（ESD太大，采样3000个）
    sample_size = min(3000 if dataset_name == 'ESD' else len(feat_files), len(feat_files))
    samples = random.sample(feat_files, sample_size)
    
    # 按类别统计
    per_class_stats = {}  # {ground_truth: {'correct': x, 'total': y}}
    
    for feat_file in samples:
        label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
        if not label_file.exists():
            continue
        
        with open(label_file) as f:
            ground_truth = f.read().strip()
        
        # 加载特征并预测
        features = np.load(feat_file)
        feat_mean = torch.from_numpy(features.mean(axis=0)).unsqueeze(0).float().cuda()
        
        with torch.no_grad():
            logits = model.model.proj(feat_mean)
            pred_idx = logits.argmax(dim=-1).item()
        
        # emotion2vec的9类
        ev2_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']
        pred = ev2_labels[pred_idx] if pred_idx < len(ev2_labels) else 'unknown'
        
        # 映射到数据集标签
        if dataset_name == 'ESD':
            # ESD 5类映射
            mapping = {
                'angry': 'angry', 'disgusted': 'angry',
                'happy': 'happy',
                'neutral': 'neutral',
                'sad': 'sad', 'fearful': 'sad',
                'surprised': 'surprise'
            }
            pred = mapping.get(pred, pred)
        elif dataset_name == 'RAVDESS':
            # RAVDESS 8类映射（处理拼写差异）
            # emotion2vec的disgusted -> RAVDESS的disgust
            if pred == 'disgusted':
                pred = 'disgust'
            # RAVDESS的calm在emotion2vec中可能被识别为neutral
            # 这里不做映射，看实际预测
        elif dataset_name == 'IEMOCAP':
            # IEMOCAP需要标签文件或从annotation解析
            pass
        
        # 统计每个类别
        if ground_truth not in per_class_stats:
            per_class_stats[ground_truth] = {'correct': 0, 'total': 0}
        
        # 排除unknown（IEMOCAP的xxx）
        if ground_truth == 'unknown':
            continue
        
        per_class_stats[ground_truth]['total'] += 1
        if pred.lower() == ground_truth.lower():
            per_class_stats[ground_truth]['correct'] += 1
    
    # 计算总体和每类准确率
    total_correct = sum(s['correct'] for s in per_class_stats.values())
    total_samples = sum(s['total'] for s in per_class_stats.values())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    
    per_class_acc = {cls: stats['correct'] / stats['total'] if stats['total'] > 0 else 0 
                     for cls, stats in per_class_stats.items()}
    
    return overall_acc, total_samples, per_class_stats, per_class_acc

def main():
    logger.info("="*60)
    logger.info("3个数据集分类准确率测试")
    logger.info("="*60)
    
    # 加载模型
    from funasr import AutoModel
    logger.info("加载emotion2vec...")
    model = AutoModel(model="iic/emotion2vec_plus_base", hub="ms")
    model.model = model.model.cuda()
    logger.info("✅ 模型加载成功")
    
    # 测试3个数据集
    results = {}
    for dataset in ['ESD', 'RAVDESS', 'IEMOCAP']:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试{dataset}")
        logger.info(f"{'='*60}")
        
        result = test_dataset_accuracy(dataset, model)
        if result is not None:
            acc, total, per_class_stats, per_class_acc = result
            results[dataset] = acc
            expected = EXPECTED_ACC[dataset]
            diff = acc - expected
            
            logger.info(f"测试样本: {total}")
            logger.info(f"总体准确率: {acc:.2%}")
            logger.info(f"期望准确率: {expected:.2%}（emotion_bottleneck_rate）")
            logger.info(f"差异: {diff:+.2%}")
            
            # 每类准确率
            logger.info("\n每类准确率:")
            for cls, cls_acc in sorted(per_class_acc.items()):
                stats = per_class_stats[cls]
                logger.info(f"  {cls}: {cls_acc:.2%} ({stats['correct']}/{stats['total']})")
            
            if abs(diff) < 0.05:
                logger.info(f"\n✅ 总体准确率与期望一致（差异<5%）")
            else:
                logger.warning(f"\n⚠️ 总体准确率与期望有差异")
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("总结")
    logger.info("="*60)
    for ds, acc in results.items():
        status = "✅" if abs(acc - EXPECTED_ACC[ds]) < 0.05 else "⚠️"
        logger.info(f"{ds}: {acc:.2%} (期望{EXPECTED_ACC[ds]:.2%}) {status}")
    
    logger.info("\n✅ 验证完成")

if __name__ == "__main__":
    main()

