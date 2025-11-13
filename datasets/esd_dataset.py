"""
ESD (Emotional Speech Dataset) 数据集实现
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    from .base_dataset import EmotionDataset
except ImportError:
    from base_dataset import EmotionDataset

logger = logging.getLogger(__name__)


class ESDDataset(EmotionDataset):
    """
    ESD数据集
    
    特点：
    - 5类情感
    - 完全平衡（每类7,000个样本，中英文各3,500）
    - 目录结构清晰
    """
    
    def __init__(self, data_root: str, languages: Optional[List[str]] = None, samples_per_emotion: int = 100):
        """
        Args:
            data_root: ESD数据集根目录
            languages: 使用的语言列表，None表示使用全部（['english', 'chinese']）
            samples_per_emotion: 每个情感采样的样本数
        """
        super().__init__(data_root)
        self._num_classes = 5
        self.languages = languages or ['english', 'chinese']
        self.samples_per_emotion = samples_per_emotion
        
    @property
    def name(self) -> str:
        return "ESD"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    def get_emotion_mapping(self) -> Dict[str, str]:
        """
        ESD标签 -> emotion2vec 9类映射
        
        ESD标签（中英文）：
        - English: Angry, Happy, Neutral, Sad, Surprise
        - Chinese: 生气, 快乐, 中立, 伤心, 惊喜
        
        映射策略：
        - 直接对应，只需处理拼写差异（surprise → surprised）
        """
        return {
            # 英文标签
            'Angry': 'angry',
            'angry': 'angry',
            'Happy': 'happy',
            'happy': 'happy',
            'Neutral': 'neutral',
            'neutral': 'neutral',
            'Sad': 'sad',
            'sad': 'sad',
            'Surprise': 'surprised',  # 注意拼写转换
            'surprise': 'surprised',
            # 中文标签
            '生气': 'angry',
            '快乐': 'happy',
            '中立': 'neutral',
            '伤心': 'sad',
            '惊喜': 'surprised',
        }
    
    def get_emotion_filter(self) -> Optional[List[str]]:
        """
        返回启用的情感类别
        
        默认：None（使用全部5类）
        TODO: 用户后续可能只使用部分情感
        """
        return None  # TODO: 用户后续配置
    
    def load_samples(self) -> List[Dict]:
        """
        加载ESD样本（从已提取的特征文件）
        
        evaluation_features/ESD/目录结构：
            0001/
                Angry/
                    0001_000351_ev2_frame.npy
                    0001_000351_emotion.txt
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
            
            # 从路径推断说话人ID
            speaker_id = feat_file.parent.parent.name  # 如0001
            
            # 判断语言（0001-0010中文，0011-0020英文）
            if speaker_id.isdigit():
                sid_num = int(speaker_id)
                lang = 'chinese' if sid_num <= 10 else 'english'
            else:
                lang = 'unknown'
            
            # 过滤语言
            if lang not in self.languages:
                continue
            
            # 为了兼容method_rate_sweep.py，把features_path当作audio_path
            # method_rate_sweep.py会用.replace('.wav', '_ev2_frame.npy')查找特征
            # 我们直接给它正确的特征路径
            fake_audio_path = str(feat_file).replace('_ev2_frame.npy', '.wav')
            
            sample = {
                'audio_path': fake_audio_path,  # 兼容method_rate_sweep
                'emotion': emotion,
                'speaker_id': f'{lang}_{speaker_id}',
                'language': lang,
            }
            samples.append(sample)
        
        # 应用emotion_filter
        samples = self.filter_samples_by_emotion(samples)
        
        logger.info(f"✅ ESD: 加载了 {len(samples)} 个样本 (语言: {self.languages})")
        
        # 采样（随机种子1344871保证可复现）
        samples = self.sample_balanced(samples, samples_per_emotion=self.samples_per_emotion, seed=1344871)
        logger.info(f"   采样后: {len(samples)} 个样本（每类{self.samples_per_emotion}）")
        
        return samples

