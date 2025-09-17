"""
Emotion2Vec 特征提取器
基于 emotion2vec 模型提取情感语音表征
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
import librosa
import logging

# 注意: 这里假设 emotion2vec 已安装
# 实际使用时需要根据 emotion2vec 的具体 API 进行调整
try:
    # 假设的 emotion2vec 导入，实际需要根据官方文档调整
    from emotion2vec import Emotion2Vec
    EMOTION2VEC_AVAILABLE = True
except ImportError:
    EMOTION2VEC_AVAILABLE = False
    logging.warning("emotion2vec not available, using dummy implementation")


class Emotion2VecExtractor(nn.Module):
    """
    Emotion2Vec 特征提取器
    
    提取音频的情感表征，用于后续的向量量化
    """
    
    def __init__(
        self,
        model_name: str = "emotion2vec_base",
        sample_rate: int = 16000,
        feature_dim: int = 1024,  # emotion2vec 特征维度
        layer_idx: int = -1,  # 使用哪一层的特征
        normalize: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.layer_idx = layer_idx
        self.normalize = normalize
        self.device = device
        
        if EMOTION2VEC_AVAILABLE:
            # 加载 emotion2vec 模型
            self.model = Emotion2Vec.from_pretrained(model_name)
            self.model.eval()
            self.model.to(device)
        else:
            # Dummy implementation for testing
            self.model = None
            logging.warning("Using dummy emotion2vec implementation")
        
        # 特征归一化统计量 (需要从数据中计算或预先提供)
        self.register_buffer("feat_mean", torch.zeros(feature_dim))
        self.register_buffer("feat_std", torch.ones(feature_dim))
        
    def load_normalization_stats(self, stats_path: str):
        """
        加载特征归一化统计量
        
        Args:
            stats_path: 包含 mean 和 std 的 .npz 文件路径
        """
        stats = np.load(stats_path)
        self.feat_mean.data.copy_(torch.from_numpy(stats["mean"]))
        self.feat_std.data.copy_(torch.from_numpy(stats["std"]))
        
    @torch.no_grad()
    def extract_features(
        self, 
        audio: Union[torch.Tensor, np.ndarray, str],
        audio_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        从音频中提取 emotion2vec 特征
        
        Args:
            audio: 音频数据 (可以是tensor、numpy数组或文件路径)
            audio_length: 音频长度 (用于batch处理)
            
        Returns:
            features: emotion2vec 特征 [B, T, D]
        """
        
        # 处理输入格式
        if isinstance(audio, str):
            # 从文件加载音频
            audio, _ = librosa.load(audio, sr=self.sample_rate)
            audio = torch.from_numpy(audio).float()
        elif isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
            
        # 确保是正确的维度
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [B, T]
            
        audio = audio.to(self.device)
        
        if EMOTION2VEC_AVAILABLE and self.model is not None:
            # 使用真实的 emotion2vec 模型
            features = self.model.extract_features(
                audio, 
                layer=self.layer_idx,
                return_attention=False
            )
        else:
            # Dummy implementation - 生成随机特征用于测试
            batch_size, seq_len = audio.shape
            # 模拟 emotion2vec 的时间下采样 (通常是 50Hz -> 25Hz)
            feature_len = seq_len // (self.sample_rate // 25)  # 假设25fps
            features = torch.randn(
                batch_size, feature_len, self.feature_dim, 
                device=self.device
            )
            
        # 特征归一化
        if self.normalize:
            features = (features - self.feat_mean.to(features)) / self.feat_std.to(features)
            
        return features
        
    def forward(
        self, 
        audio: torch.Tensor, 
        audio_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            audio: 音频波形 [B, T]
            audio_length: 音频长度 [B]
            
        Returns:
            features: emotion2vec 特征 [B, T, D]
        """
        return self.extract_features(audio, audio_length)


class DummyEmotion2VecExtractor(Emotion2VecExtractor):
    """
    用于测试的 Dummy Emotion2Vec 提取器
    生成随机特征，保持与真实模型相同的接口
    """
    
    def __init__(self, **kwargs):
        # 强制设置为 dummy 模式
        super().__init__(**kwargs)
        self.model = None
        
    @torch.no_grad()
    def extract_features(
        self, 
        audio: Union[torch.Tensor, np.ndarray, str],
        audio_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """生成随机特征用于测试"""
        
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=self.sample_rate)
            audio = torch.from_numpy(audio).float()
        elif isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
            
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        batch_size, seq_len = audio.shape
        # 模拟特征提取的时间下采样
        feature_len = seq_len // (self.sample_rate // 25)  # 25fps
        
        # 生成具有一定结构的随机特征 (而非完全随机)
        features = torch.randn(batch_size, feature_len, self.feature_dim)
        
        # 添加一些时间相关性，使特征更真实
        for i in range(1, feature_len):
            features[:, i] = 0.8 * features[:, i-1] + 0.2 * features[:, i]
            
        features = features.to(self.device)
        
        if self.normalize:
            # 简单的归一化
            features = torch.nn.functional.layer_norm(features, [self.feature_dim])
            
        return features


def create_emotion2vec_extractor(
    model_name: str = "emotion2vec_base",
    use_dummy: bool = False,
    **kwargs
) -> Emotion2VecExtractor:
    """
    创建 emotion2vec 特征提取器的工厂函数
    
    Args:
        model_name: 模型名称
        use_dummy: 是否使用 dummy 实现 (用于测试)
        **kwargs: 其他参数
        
    Returns:
        Emotion2VecExtractor 实例
    """
    if use_dummy or not EMOTION2VEC_AVAILABLE:
        return DummyEmotion2VecExtractor(model_name=model_name, **kwargs)
    else:
        return Emotion2VecExtractor(model_name=model_name, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("Testing Emotion2Vec Extractor...")
    
    # 创建提取器 (使用 dummy 模式)
    extractor = create_emotion2vec_extractor(use_dummy=True)
    
    # 测试随机音频
    dummy_audio = torch.randn(2, 16000)  # 1秒音频，batch_size=2
    features = extractor(dummy_audio)
    
    print(f"Input audio shape: {dummy_audio.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Feature dim: {features.shape[-1]}")
    
    print("Emotion2Vec Extractor test completed!")

