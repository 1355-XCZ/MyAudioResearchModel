"""
方法1: 码率扫描评估
通过λ参数控制码率，评估emotion2vec分类性能
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Optional
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rate_controller import RateController

logger = logging.getLogger(__name__)


def rate_sweep_evaluation(
    rvq_model,
    entropy_model,
    dataset,  # EmotionDataset实例
    target_rates_bpf: List[float],
    classifier,  # EmotionClassifierV2实例
    output_dir: str,
    device: str = 'cuda',
    frame_rate_hz: float = 50.0,
    batch_size: int = 8,  # ChatGPT建议：批处理
    num_workers: int = 4  # ChatGPT建议：多进程I/O
) -> Dict:
    """
    码率扫描评估主函数
    
    Args:
        rvq_model: 训练好的GroupedRVQ模型
        entropy_model: 训练好的无条件熵模型 q(z)
        dataset: EmotionDataset实例（IEMOCAP/RAVDESS/ESD）
        target_rates_bpf: 目标码率列表（bits per frame）
        classifier: emotion2vec分类器
        output_dir: 输出目录
        device: 设备
        frame_rate_hz: 帧率
    
    Returns:
        results: 评估结果字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rvq_model.eval()
    entropy_model.eval()
    
    # 获取情感映射
    emotion_mapping = dataset.get_emotion_mapping()
    
    logger.info(f"开始码率扫描评估: {dataset.name}")
    logger.info(f"  目标码率: {target_rates_bpf} bpf")
    logger.info(f"  样本数: {len(dataset)}")
    logger.info(f"  情感映射: {emotion_mapping}")
    
    # 结果存储
    results = {
        'dataset': dataset.name,
        'target_rates_bpf': target_rates_bpf,
        'emotion_mapping': emotion_mapping,
        'samples': {},
        'rate_points': {}
    }
    
    # 码率控制器配置
    from config import RateControlConfig
    rate_config = RateControlConfig()
    rate_controller = RateController(rate_config)
    
    # 加载样本
    if not dataset.samples:
        dataset.samples = dataset.load_samples()
    
    logger.info(f"样本总数: {len(dataset.samples)}")
    
    # 对每个目标码率进行评估
    for target_rate_bpf in tqdm(target_rates_bpf, desc="码率点"):
        logger.info(f"\n{'='*60}")
        logger.info(f"目标码率: {target_rate_bpf} bpf ({target_rate_bpf * frame_rate_hz:.1f} bps)")
        
        # 初始化这个码率点的结果
        rate_results = {
            'target_rate_bpf': target_rate_bpf,
            'achieved_rates': [],
            'lambda_values': [],
            'predictions': [],
            'ground_truths': [],
            'confidences': [],
        }
        
        # 记录第一个样本的λ，用于加速后续样本的搜索
        first_lambda = None
        
        # 对每个样本进行编码和分类
        for idx, sample in enumerate(tqdm(dataset.samples, desc=f"样本 @ {target_rate_bpf} bpf")):
            audio_path = sample['audio_path']
            emotion_original = sample['emotion']
            
            # 映射情感标签到emotion2vec
            emotion_ev2 = dataset.map_emotion_to_ev2(emotion_original)
            
            # 加载或提取emotion2vec特征
            # TODO: 这里假设有预提取的特征，实际需要处理音频提取
            features_path = audio_path.replace('.wav', '_ev2_frame.npy')
            
            if not Path(features_path).exists():
                # 如果没有预提取特征，从音频提取
                logger.warning(f"特征文件不存在，跳过: {features_path}")
                continue
            
            try:
                features = np.load(features_path)  # (T, 768)
                features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, T, 768)
                valid_mask = torch.ones(1, features_tensor.size(1), dtype=torch.bool, device=device)
                
                # 每个样本单独搜索λ（确保达到目标码率）
                def encoder_fn(lambda_val):
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                        _, indices, _, _ = rvq_model(
                            features_tensor,
                            lambda_rate=torch.tensor(lambda_val, device=device),
                            entropy_model=entropy_model,
                            valid_mask=valid_mask
                        )
                        bits = entropy_model.compute_bits(indices, valid_mask=valid_mask).item()
                        num_frames = valid_mask.sum().item()
                        actual_rate_bpf = bits / num_frames if num_frames > 0 else 0.0
                        return indices, actual_rate_bpf
                
                # 利用第一个样本的λ作为先验，加速收敛
                sample_lambda, sample_rate_bpf = rate_controller.binary_search(
                    encoder_fn,
                    target_rate_bpf,
                    tolerance_bpf=rate_config.rate_tolerance_bpf,
                    lambda_hint=first_lambda  # 传入先验λ
                )
                
                # 记录第一个样本的λ
                if idx == 0:
                    first_lambda = sample_lambda
                    logger.info(f"  第一个样本: λ={sample_lambda:.4f}, R={sample_rate_bpf:.2f} bpf")
                
                # 使用找到的λ进行量化（实际上binary_search已经编码过了，但为了一致性重新编码）
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    quantized, indices, _, stats = rvq_model(
                        features_tensor,
                        lambda_rate=torch.tensor(sample_lambda, device=device, dtype=features_tensor.dtype),
                        entropy_model=entropy_model,
                        valid_mask=valid_mask
                    )
                    
                    # 计算实际码率
                    bits = entropy_model.compute_bits(indices, valid_mask=valid_mask).item()
                    num_frames = valid_mask.sum().item()
                    actual_rate_bpf = bits / num_frames
                
                # 使用emotion2vec分类量化后的特征
                quantized_np = quantized[0].cpu().numpy()  # (T, 768)
                
                # 调用分类器
                result = classifier.classify_from_features(
                    quantized_np,
                    esd_ground_truth=emotion_ev2
                )
                
                # 记录结果
                rate_results['achieved_rates'].append(actual_rate_bpf)
                rate_results['lambda_values'].append(sample_lambda)  # 每个样本的λ
                rate_results['predictions'].append(result['ev2_predicted'])
                rate_results['ground_truths'].append(emotion_ev2)
                rate_results['confidences'].append(result['ev2_confidence'])
                
            except Exception as e:
                logger.error(f"处理样本失败: {audio_path}, 错误: {e}")
                continue
        
        # 计算这个码率点的统计指标
        if len(rate_results['predictions']) > 0:
            predictions = np.array(rate_results['predictions'])
            ground_truths = np.array(rate_results['ground_truths'])
            confidences = np.array(rate_results['confidences'])
            
            accuracy = (predictions == ground_truths).mean()
            avg_confidence = confidences.mean()
            avg_rate_bpf = np.mean(rate_results['achieved_rates'])
            
            rate_results['accuracy'] = float(accuracy)
            rate_results['avg_confidence'] = float(avg_confidence)
            rate_results['avg_rate_bpf'] = float(avg_rate_bpf)
            
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"  平均置信度: {avg_confidence:.4f}")
            logger.info(f"  平均码率: {avg_rate_bpf:.2f} bpf")
        
        # 保存这个码率点的结果
        rate_key = "original" if target_rate_bpf == float('inf') else f'{target_rate_bpf}_bpf'
        results['rate_points'][rate_key] = rate_results
    
    # 保存完整结果
    output_file = output_dir / f'rate_sweep_{dataset.name}.json'
    os.makedirs(output_file.parent, exist_ok=True)  # 确保目录存在
    
    # 处理inf值：在JSON中用"original"替代
    results_serializable = results.copy()
    results_serializable['target_rates_bpf'] = [
        "original" if rate == float('inf') else rate 
        for rate in target_rates_bpf
    ]
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"\n✅ 码率扫描评估完成: {dataset.name}")
    logger.info(f"  结果已保存: {output_file}")
    
    return results

