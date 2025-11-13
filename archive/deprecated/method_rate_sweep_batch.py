"""
批处理版本的码率扫描（保留原版作为备份）
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Optional
import sys
from torch.utils.data import DataLoader

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rate_controller import RateController
from eval_dataset import EvalFeatureDataset, collate_fn_eval  # 修复：不用相对导入

logger = logging.getLogger(__name__)


def rate_sweep_evaluation_batch(
    rvq_model,
    entropy_model,
    dataset,
    target_rates_bpf: List[float],
    classifier,
    output_dir: str,
    device: str = 'cuda',
    frame_rate_hz: float = 50.0,
    batch_size: int = 16,  # 批处理大小
    num_workers: int = 4
) -> Dict:
    """
    批处理版本的码率扫描评估
    
    保留真ECVQ，但用批处理加速I/O和推理
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rvq_model.eval()
    entropy_model.eval()
    
    emotion_mapping = dataset.get_emotion_mapping()
    
    logger.info(f"开始批处理码率扫描评估: {dataset.name}")
    logger.info(f"  目标码率: {target_rates_bpf} bpf")
    logger.info(f"  批处理: batch_size={batch_size}, num_workers={num_workers}")
    
    # 加载样本
    if not dataset.samples:
        dataset.samples = dataset.load_samples()
    
    logger.info(f"  样本总数: {len(dataset.samples)}")
    
    # 创建DataLoader
    eval_dataset = EvalFeatureDataset(dataset.samples, dataset.name)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_eval,
        pin_memory=True if device == 'cuda' else False
    )
    
    # 结果存储
    results = {
        'dataset': dataset.name,
        'target_rates_bpf': target_rates_bpf,
        'emotion_mapping': emotion_mapping,
        'samples': {},
        'rate_points': {}
    }
    
    # 码率控制器
    from config import RateControlConfig
    rate_config = RateControlConfig()
    rate_controller = RateController(rate_config)
    
    # 对每个目标码率进行评估
    for target_rate_bpf in tqdm(target_rates_bpf, desc="码率点"):
        logger.info(f"\n{'='*60}")
        logger.info(f"目标码率: {target_rate_bpf} bpf ({target_rate_bpf * frame_rate_hz:.1f} bps)")
        
        rate_results = {
            'target_rate_bpf': target_rate_bpf,
            'achieved_rates': [],
            'lambda_values': [],
            'predictions': [],
            'ground_truths': [],
            'confidences': [],
        }
        
        # 在第一个batch上进行二分搜索找optimal_lambda
        optimal_lambda = 0.5  # 默认值
        lambda_found = False
        
        # 批处理循环
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"样本 @ {target_rate_bpf} bpf", leave=False)):
            features = batch['features'].to(device, non_blocking=True)  # (B, T_max, 768)
            valid_mask = batch['valid_mask'].to(device, non_blocking=True)  # (B, T_max)
            emotions = batch['emotions']
            audio_paths = batch['audio_paths']
            
            # 只在第一个batch的第一个样本上搜索λ
            if batch_idx == 0 and not lambda_found:
                # 使用第一个样本搜索
                first_sample = features[0:1]  # (1, T, 768)
                first_mask = valid_mask[0:1]  # (1, T)
                
                def encoder_fn(lambda_val):
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                        _, indices, _, _ = rvq_model(
                            first_sample,
                            lambda_rate=torch.tensor(lambda_val, device=device),
                            entropy_model=entropy_model,
                            valid_mask=first_mask
                        )
                        bits = entropy_model.compute_bits(indices, valid_mask=first_mask).item()
                        num_frames = first_mask.sum().item()
                        actual_rate_bpf = bits / num_frames if num_frames > 0 else 0.0
                        return indices, actual_rate_bpf
                
                optimal_lambda, achieved_rate_bpf = rate_controller.binary_search(
                    encoder_fn,
                    target_rate_bpf,
                    tolerance_bps=rate_config.rate_tolerance_bps
                )
                lambda_found = True
                logger.info(f"  最优λ: {optimal_lambda:.4f}, 实际码率: {achieved_rate_bpf:.2f} bpf")
            
            # 批量编码和分类
            try:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    quantized, indices, _, _ = rvq_model(
                        features,
                        lambda_rate=torch.tensor(optimal_lambda, device=device),
                        entropy_model=entropy_model,
                        valid_mask=valid_mask
                    )
                    
                    # 计算每个样本的码率
                    for i in range(features.size(0)):
                        sample_mask = valid_mask[i:i+1]
                        sample_indices = indices[i:i+1]
                        
                        bits = entropy_model.compute_bits(sample_indices, valid_mask=sample_mask).item()
                        num_frames = sample_mask.sum().item()
                        actual_rate_bpf = bits / num_frames if num_frames > 0 else 0.0
                        
                        # 分类
                        quantized_np = quantized[i, :num_frames].cpu().float().numpy()  # (T, 768)
                        emotion_original = emotions[i]
                        emotion_ev2 = dataset.map_emotion_to_ev2(emotion_original)
                        
                        result = classifier.classify_from_features(
                            quantized_np,
                            esd_ground_truth=emotion_ev2
                        )
                        
                        # 记录结果
                        rate_results['achieved_rates'].append(actual_rate_bpf)
                        rate_results['lambda_values'].append(optimal_lambda)
                        rate_results['predictions'].append(result['ev2_predicted'])
                        rate_results['ground_truths'].append(emotion_ev2)
                        rate_results['confidences'].append(result['ev2_confidence'])
                
            except Exception as e:
                logger.error(f"批处理失败: {e}")
                continue
        
        # 计算准确率
        if len(rate_results['predictions']) > 0:
            correct = sum(p == g for p, g in zip(rate_results['predictions'], rate_results['ground_truths']))
            accuracy = correct / len(rate_results['predictions']) * 100
            avg_rate = np.mean(rate_results['achieved_rates'])
            avg_confidence = np.mean(rate_results['confidences'])
            
            logger.info(f"  准确率: {accuracy:.2f}% ({correct}/{len(rate_results['predictions'])})")
            logger.info(f"  平均码率: {avg_rate:.2f} bpf")
            logger.info(f"  平均置信度: {avg_confidence:.4f}")
            
            results['rate_points'][f'{target_rate_bpf}_bpf'] = rate_results
        else:
            logger.warning(f"  无有效结果")
    
    # 保存结果
    output_file = output_dir / f"rate_sweep_{dataset.name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ 码率扫描评估完成: {dataset.name}")
    logger.info(f"  结果已保存: {output_file}")
    
    return results

