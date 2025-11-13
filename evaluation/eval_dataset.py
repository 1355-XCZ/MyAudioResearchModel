"""
评估数据集 - 批处理优化
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class EvalFeatureDataset(Dataset):
    """
    评估特征数据集（用于批处理）
    
    从预提取的emotion2vec特征加载
    """
    
    def __init__(self, samples: List[Dict], dataset_name: str):
        """
        Args:
            samples: 数据集的sample列表（每个dict包含audio_path, emotion等）
            dataset_name: 数据集名称（用于日志）
        """
        self.samples = samples
        self.dataset_name = dataset_name
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载特征
        audio_path = sample['audio_path']
        features_path = audio_path.replace('.wav', '_ev2_frame.npy')
        
        try:
            features = np.load(features_path)  # (T, 768)
            
            return {
                'features': torch.from_numpy(features).float(),  # (T, 768)
                'emotion': sample['emotion'],
                'audio_path': audio_path,
                'length': features.shape[0],
            }
        except Exception as e:
            logger.error(f"加载失败: {features_path}, 错误: {e}")
            # 返回一个dummy样本
            return {
                'features': torch.zeros(1, 768),
                'emotion': 'unknown',
                'audio_path': audio_path,
                'length': 1,
            }


def collate_fn_eval(batch):
    """
    批处理collate函数（处理变长序列）
    
    Args:
        batch: list of dicts from __getitem__
    
    Returns:
        dict with batched tensors
    """
    # 过滤掉加载失败的样本
    batch = [b for b in batch if b['length'] > 1]
    
    if len(batch) == 0:
        # 返回空batch
        return {
            'features': torch.zeros(1, 1, 768),
            'lengths': torch.tensor([1]),
            'emotions': ['unknown'],
            'audio_paths': [''],
            'valid_mask': torch.zeros(1, 1, dtype=torch.bool),
        }
    
    # 获取最大长度
    max_len = max(b['length'] for b in batch)
    batch_size = len(batch)
    
    # 初始化batch tensors
    features_padded = torch.zeros(batch_size, max_len, 768)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    emotions = []
    audio_paths = []
    
    # 填充
    for i, item in enumerate(batch):
        length = item['length']
        features_padded[i, :length] = item['features']
        lengths[i] = length
        emotions.append(item['emotion'])
        audio_paths.append(item['audio_path'])
    
    # 创建valid_mask
    valid_mask = torch.arange(max_len)[None, :] < lengths[:, None]  # (B, T)
    
    return {
        'features': features_padded,      # (B, T_max, 768)
        'lengths': lengths,                # (B,)
        'emotions': emotions,              # list of str
        'audio_paths': audio_paths,        # list of str
        'valid_mask': valid_mask,          # (B, T_max)
    }

