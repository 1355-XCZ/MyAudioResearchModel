"""
数据集抽象层 - 面向对象设计
支持灵活的情感标签映射和过滤
"""

from .base_dataset import EmotionDataset
from .iemocap_dataset import IEMOCAPDataset
from .ravdess_dataset import RAVDESSDataset
from .esd_dataset import ESDDataset

__all__ = [
    'EmotionDataset',
    'IEMOCAPDataset',
    'RAVDESSDataset',
    'ESDDataset',
]

