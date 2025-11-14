"""
RAVDESS数据集实现
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    from .base_dataset import EmotionDataset
except ImportError:
    from base_dataset import EmotionDataset

logger = logging.getLogger(__name__)


class RAVDESSDataset(EmotionDataset):
    """
    RAVDESS数据集
    
    特点：
    - 8类情感
    - 完全平衡（每类192个speech样本）
    - 文件名编码包含所有信息
    """
    
    def __init__(self, data_root: str, samples_per_emotion: int = 100):
        super().__init__(data_root)
        self._num_classes = 8
        self.samples_per_emotion = samples_per_emotion
        
        # RAVDESS情感编码
        self.emotion_codes = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        # 加载样本
        self.samples = self.load_samples()
    
    @property
    def name(self) -> str:
        return "RAVDESS"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    def get_emotion_mapping(self) -> Dict[str, str]:
        """
        RAVDESS标签 -> emotion2vec 9类映射
        
        映射策略：
        - neutral → neutral ✓
        - calm → neutral     # TODO: 用户确认（当前calm→neutral，分析显示正确）
        - happy → happy ✓
        - sad → sad ✓
        - angry → angry ✓
        - fearful → fearful ✓
        - disgust → disgusted ✓
        - surprised → surprised ✓
        """
        return {
            'neutral': 'neutral',
            'calm': 'neutral',        # TODO: 用户确认映射策略
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fearful',
            'disgust': 'disgusted',
            'disgusted': 'disgusted',  # 已映射的标签
            'surprised': 'surprised',
        }
    
    def get_emotion_filter(self) -> Optional[List[str]]:
        """
        返回启用的情感类别
        
        默认：None（使用全部）
        TODO: 用户后续可能只使用部分情感
        """
        return None  # TODO: 用户后续配置
    
    def load_samples(self) -> List[Dict]:
        """
        加载RAVDESS样本（从已提取的特征文件）
        
        evaluation_features/RAVDESS/目录结构：
            03-01-06-01-02-01-12_ev2_frame.npy
            03-01-06-01-02-01-12_emotion.txt
        """
        samples = []
        
        # 直接查找所有*_ev2_frame.npy文件
        feature_files = list(self.data_root.glob("*_ev2_frame.npy"))
        
        for feat_file in feature_files:
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            # 解析文件名获取元信息
            parts = feat_file.stem.replace('_ev2_frame', '').split('-')
            if len(parts) == 7:
                actor = parts[6]
                intensity = parts[3]
            else:
                actor = 'unknown'
                intensity = '01'
            
            # 兼容method_rate_sweep.py
            fake_audio_path = str(feat_file).replace('_ev2_frame.npy', '.wav')
            
            sample = {
                'audio_path': fake_audio_path,
                'emotion': emotion,
                'speaker_id': f'Actor_{actor}',
                'intensity': intensity,
            }
            samples.append(sample)
        
        # 应用emotion_filter
        samples = self.filter_samples_by_emotion(samples)
        
        logger.info(f"✅ RAVDESS: 加载了 {len(samples)} 个样本")
        
        # 采样（随机种子1344871保证可复现）
        samples = self.sample_balanced(samples, samples_per_emotion=self.samples_per_emotion, seed=1344871)
        logger.info(f"   采样后: {len(samples)} 个样本（每类{self.samples_per_emotion}）")
        
        return samples

