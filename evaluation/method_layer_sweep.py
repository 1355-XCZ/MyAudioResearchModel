"""
方法2: 层数扫描评估
通过控制使用的量化层数，评估emotion2vec分类性能
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Optional
import sys

logger = logging.getLogger(__name__)


def layer_sweep_evaluation(
    rvq_model,
    dataset,  # EmotionDataset实例
    num_layers_list: List[int],
    classifier,  # EmotionClassifierV2实例
    output_dir: str,
    device: str = 'cuda'
) -> Dict:
    """
    层数扫描评估主函数
    
    与emotion_information_bottleneck类似，但使用分组RVQ
    
    Args:
        rvq_model: 训练好的GroupedRVQ模型
        dataset: EmotionDataset实例
        num_layers_list: 使用的层数列表（总层数=12组×3层=36）
        classifier: emotion2vec分类器
        output_dir: 输出目录
        device: 设备
    
    Returns:
        results: 评估结果字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rvq_model.eval()
    
    # 获取情感映射
    emotion_mapping = dataset.get_emotion_mapping()
    
    logger.info(f"开始层数扫描评估: {dataset.name}")
    logger.info(f"  层数列表: {num_layers_list}")
    logger.info(f"  样本数: {len(dataset)}")
    
    # 结果存储
    results = {
        'dataset': dataset.name,
        'num_layers_list': num_layers_list,
        'emotion_mapping': emotion_mapping,
        'layer_points': {}
    }
    
    # 加载样本
    if not dataset.samples:
        dataset.samples = dataset.load_samples()
    
    # 总层数
    total_layers = rvq_model.num_groups * rvq_model.config.num_fine_layers
    
    logger.info(f"样本总数: {len(dataset.samples)}")
    logger.info(f"总层数: {total_layers}")
    
    # 对每个层数进行评估
    for num_layers in tqdm(num_layers_list, desc="层数点"):
        if num_layers > total_layers:
            logger.warning(f"层数 {num_layers} 超过总层数 {total_layers}，跳过")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"使用层数: {num_layers}/{total_layers}")
        
        # 初始化这个层数点的结果
        layer_results = {
            'num_layers': num_layers,
            'predictions': [],
            'ground_truths': [],
            'confidences': [],
        }
        
        # 对每个样本进行量化和分类
        for sample in tqdm(dataset.samples, desc=f"样本 @ {num_layers}层"):
            audio_path = sample['audio_path']
            emotion_original = sample['emotion']
            
            # 映射情感标签
            emotion_ev2 = dataset.map_emotion_to_ev2(emotion_original)
            
            # 加载特征
            features_path = audio_path.replace('.wav', '_ev2_frame.npy')
            
            if not Path(features_path).exists():
                logger.warning(f"特征文件不存在，跳过: {features_path}")
                continue
            
            try:
                features = np.load(features_path)  # (T, 768)
                features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, T, 768)
                
                # 使用指定层数进行量化（不使用ECVQ，直接量化）
                with torch.no_grad():
                    # 禁用ECVQ，直接重建
                    quantized = quantize_with_num_layers(
                        rvq_model,
                        features_tensor,
                        num_layers,
                        device
                    )
                
                # 使用emotion2vec分类量化后的特征
                quantized_np = quantized[0].cpu().numpy()  # (T, 768)
                
                # 调用分类器
                result = classifier.classify_from_features(
                    quantized_np,
                    esd_ground_truth=emotion_ev2
                )
                
                # 记录结果
                layer_results['predictions'].append(result['ev2_predicted'])
                layer_results['ground_truths'].append(emotion_ev2)
                layer_results['confidences'].append(result['ev2_confidence'])
                
            except Exception as e:
                logger.error(f"处理样本失败: {audio_path}, 错误: {e}")
                continue
        
        # 计算这个层数点的统计指标
        if len(layer_results['predictions']) > 0:
            predictions = np.array(layer_results['predictions'])
            ground_truths = np.array(layer_results['ground_truths'])
            confidences = np.array(layer_results['confidences'])
            
            accuracy = (predictions == ground_truths).mean()
            avg_confidence = confidences.mean()
            
            layer_results['accuracy'] = float(accuracy)
            layer_results['avg_confidence'] = float(avg_confidence)
            
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"  平均置信度: {avg_confidence:.4f}")
        
        # 保存这个层数点的结果
        results['layer_points'][f'{num_layers}_layers'] = layer_results
    
    # 保存完整结果
    output_file = output_dir / f'layer_sweep_{dataset.name}.json'
    os.makedirs(output_file.parent, exist_ok=True)  # 确保目录存在
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ 层数扫描评估完成: {dataset.name}")
    logger.info(f"  结果已保存: {output_file}")
    
    return results


def quantize_with_num_layers(rvq_model, features, num_layers, device):
    """
    使用指定层数进行量化（辅助函数）
    
    实现方式：
    1. 使用RVQ正常量化
    2. 只使用前num_layers层的码字进行重建
    
    Args:
        rvq_model: GroupedRVQ模型
        features: (B, T, 768) 特征
        num_layers: 使用的层数
        device: 设备
    
    Returns:
        quantized: (B, T, 768) 量化后的特征
    """
    B, T, D = features.shape
    
    # 重塑为分组形式
    features_grouped = features.view(B, T, rvq_model.num_groups, rvq_model.group_dim)
    
    # 逐组量化（只使用前num_layers_per_group层）
    total_layers = rvq_model.num_groups * rvq_model.config.num_fine_layers
    layers_per_group = num_layers // rvq_model.num_groups
    remaining_layers = num_layers % rvq_model.num_groups
    
    reconstructed_grouped = torch.zeros_like(features_grouped)
    
    for g in range(rvq_model.num_groups):
        # 确定这个组使用多少层
        if g < remaining_layers:
            group_layers = layers_per_group + 1
        else:
            group_layers = layers_per_group
        
        group_layers = min(group_layers, rvq_model.config.num_fine_layers)
        
        # 残差量化
        residual = features_grouped[:, :, g, :]  # (B, T, group_dim)
        
        for m in range(group_layers):
            # 量化
            vq = rvq_model.fine_vqs[g][m]
            quantized_layer, indices, commit_loss = vq(residual)
            
            # 累加
            reconstructed_grouped[:, :, g, :] += quantized_layer
            
            # 更新残差
            residual = residual - quantized_layer
    
    # 重塑回原始形状
    quantized = reconstructed_grouped.view(B, T, D)
    
    return quantized

