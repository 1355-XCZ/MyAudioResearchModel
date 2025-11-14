"""
数据集基类 - 面向对象设计
定义统一的接口，支持灵活的情感标签映射和过滤
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
import random


class EmotionDataset(ABC):
    """
    情感数据集抽象基类
    
    每个具体数据集需要实现：
    1. load_samples() - 加载样本列表
    2. get_emotion_mapping() - 情感标签映射到emotion2vec 9类
    3. get_emotion_filter() - 启用的情感类别（用户后续配置）
    """
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: 数据集根目录
        """
        self.data_root = Path(data_root)
        self.samples = []
        
    @abstractmethod
    def load_samples(self) -> List[Dict]:
        """
        加载数据样本
        
        Returns:
            List of dicts, each containing:
                - 'audio_path': str, 音频文件路径
                - 'features_path': str (可选), ev2特征文件路径
                - 'emotion': str, 原始情感标签
                - 'speaker_id': str (可选), 说话人ID
                - ... 其他元数据
        """
        pass
    
    @abstractmethod
    def get_emotion_mapping(self) -> Dict[str, str]:
        """
        返回情感标签映射：数据集标签 -> emotion2vec 9类
        
        emotion2vec 9类：
        ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 
         'other', 'sad', 'surprised', 'unknown']
        
        Returns:
            Dict[原始标签, emotion2vec标签]
        
        注意：
            - TODO: 用户后续确认映射策略
            - 映射应该基于语义相似性
        """
        pass
    
    @abstractmethod
    def get_emotion_filter(self) -> Optional[List[str]]:
        """
        返回启用的情感类别（数据集原始标签）
        
        Returns:
            List of str: 启用的情感类别，None表示使用全部
        
        注意：
            - TODO: 用户后续配置（如只使用某些情感类别）
            - 默认返回None（使用全部）
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据集名称"""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """原始情感类别数"""
        pass
    
    def filter_samples_by_emotion(self, samples: List[Dict]) -> List[Dict]:
        """
        根据emotion_filter过滤样本
        
        Args:
            samples: 样本列表
        
        Returns:
            过滤后的样本列表
        """
        emotion_filter = self.get_emotion_filter()
        
        if emotion_filter is None:
            return samples
        
        filtered = [s for s in samples if s['emotion'] in emotion_filter]
        
        return filtered
    
    def sample_balanced(self, samples: List[Dict], samples_per_emotion: int = 500, seed: int = 1344871) -> List[Dict]:
        """
        每个情感类别采样固定数量的样本
        
        Args:
            samples: 样本列表
            samples_per_emotion: 每个情感采样的样本数
            seed: 随机种子
        
        Returns:
            采样后的样本列表
        """
        # 按情感分组
        emotion_groups = {}
        for sample in samples:
            emotion = sample['emotion']
            if emotion not in emotion_groups:
                emotion_groups[emotion] = []
            emotion_groups[emotion].append(sample)
        
        # 每个情感采样
        random.seed(seed)
        sampled = []
        for emotion, group in emotion_groups.items():
            if len(group) <= samples_per_emotion:
                # 样本数不足，全部保留
                sampled.extend(group)
            else:
                # 随机采样
                sampled.extend(random.sample(group, samples_per_emotion))
        
        # 打乱样本顺序，避免按情感分组排列
        # 这样可以防止某些情感集中在特定区间，导致评估时出现系统性偏差
        random.shuffle(sampled)
        
        return sampled
    
    def map_emotion_to_ev2(self, emotion: str) -> str:
        """
        将数据集标签映射到emotion2vec标签
        
        Args:
            emotion: 数据集原始标签
        
        Returns:
            emotion2vec标签
        """
        mapping = self.get_emotion_mapping()
        
        # 尝试直接映射
        if emotion in mapping:
            return mapping[emotion]
        
        # 尝试小写映射
        emotion_lower = emotion.lower()
        if emotion_lower in mapping:
            return mapping[emotion_lower]
        
        # 未找到映射，返回'other'
        return 'other'
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        if not self.samples:
            self.samples = self.load_samples()
        
        from collections import Counter
        
        emotion_counts = Counter([s['emotion'] for s in self.samples])
        
        stats = {
            'name': self.name,
            'num_samples': len(self.samples),
            'num_classes': self.num_classes,
            'emotion_distribution': dict(emotion_counts),
            'emotion_mapping': self.get_emotion_mapping(),
            'emotion_filter': self.get_emotion_filter(),
        }
        
        return stats
    
    def __len__(self) -> int:
        """返回样本数量"""
        if not self.samples:
            self.samples = self.load_samples()
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        if not self.samples:
            self.samples = self.load_samples()
        return self.samples[idx]

