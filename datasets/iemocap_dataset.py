"""
IEMOCAP数据集实现
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    from .base_dataset import EmotionDataset
except ImportError:
    from base_dataset import EmotionDataset

logger = logging.getLogger(__name__)


class IEMOCAPDataset(EmotionDataset):
    """
    IEMOCAP数据集
    
    特点：
    - 10类情感（排除xxx后）
    - 有frustrated, excited等特殊标签
    - 样本数：~7,266（排除xxx后）
    """
    
    def __init__(self, data_root: str, samples_per_emotion: int = 100):
        super().__init__(data_root)
        self._num_classes = 10  # 有效情感类别数（排除xxx）
        self.samples_per_emotion = samples_per_emotion
        
        # 加载样本
        self.samples = self.load_samples()
    
    @property
    def name(self) -> str:
        return "IEMOCAP"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    def get_emotion_mapping(self) -> Dict[str, str]:
        """
        IEMOCAP标签 -> emotion2vec 9类映射
        
        映射策略：
        - ang → angry ✓
        - hap → happy ✓
        - sad → sad ✓
        - neu → neutral ✓
        - fru → angry  # TODO: 用户确认（当前frustrated→angry，但分析显示应映射到neutral）
        - exc → happy  # TODO: 用户确认（当前excited→happy，分析显示正确）
        - sur → surprised
        - fea → fearful
        - dis → disgusted
        - oth → other
        """
        return {
            'ang': 'angry',
            'hap': 'happy',
            'sad': 'sad',
            'neu': 'neutral',
            'fru': 'angry',       # TODO: 用户确认映射策略
            'exc': 'happy',       # TODO: 用户确认映射策略
            'sur': 'surprised',
            'fea': 'fearful',
            'dis': 'disgusted',
            'oth': 'other',
        }
    
    def get_emotion_filter(self) -> Optional[List[str]]:
        """
        返回启用的情感类别
        
        默认：None（使用全部，但排除'xxx'）
        TODO: 用户后续可能只使用部分情感（如只用4类核心情感）
        """
        # 默认使用全部，排除'xxx'（未知标签）
        return None  # TODO: 用户后续配置
    
    def load_samples(self) -> List[Dict]:
        """
        加载IEMOCAP样本（从已提取的特征文件）
        
        evaluation_features/IEMOCAP/目录结构：
            Session1/
                Ses01F_impro01/
                    Ses01F_impro01_F000_ev2_frame.npy
                    Ses01F_impro01_F000_emotion.txt
        """
        samples = []
        
        # 直接查找所有*_ev2_frame.npy文件
        feature_files = list(self.data_root.glob("**/*_ev2_frame.npy"))
        
        for feat_file in feature_files:
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            # 从文件名解析说话人
            utterance_id = feat_file.stem.replace('_ev2_frame', '')
            speaker_id = utterance_id.split('_')[0]  # 如Ses01F
            
            # 兼容method_rate_sweep.py
            fake_audio_path = str(feat_file).replace('_ev2_frame.npy', '.wav')
            
            sample = {
                'audio_path': fake_audio_path,
                'emotion': emotion,
                'speaker_id': speaker_id,
            }
            samples.append(sample)
        
        # 应用emotion_filter
        samples = self.filter_samples_by_emotion(samples)
        
        logger.info(f"✅ IEMOCAP: 加载了 {len(samples)} 个样本")
        
        # 采样（随机种子1344871保证可复现）
        samples = self.sample_balanced(samples, samples_per_emotion=self.samples_per_emotion, seed=1344871)
        logger.info(f"   采样后: {len(samples)} 个样本（每类{self.samples_per_emotion}）")
        
        return samples
    
    def _parse_emotion_from_filename(self, filename: str) -> str:
        """
        从文件名解析情感标签
        
        IEMOCAP文件名格式通常包含情感信息
        这里提供简化实现，实际需要根据数据集具体格式调整
        """
        # 简化实现：返回占位符
        # TODO: 实际实现需要解析IEMOCAP的标注文件
        return 'neu'  # 占位符

