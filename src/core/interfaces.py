"""
核心接口定义，支持模块化实验设计
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioData:
    """音频数据结构"""
    waveform: np.ndarray  # 音频波形
    sample_rate: int
    duration: float
    file_path: Optional[str] = None


@dataclass
class ProcessedData:
    """处理后的数据结构"""
    phonemes: List[str]  # 音素序列
    emotion_features: np.ndarray  # emotion2vec特征
    source_audio: AudioData  # 原始音频数据引用
    text: Optional[str] = None  # 原始文本（如果有）


@dataclass
class ModelOutput:
    """模型输出数据结构"""
    mel_spectrogram: torch.Tensor  # Mel频谱
    attention_weights: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class DataProcessor(ABC):
    """数据处理器抽象基类"""
    
    @abstractmethod
    def process_audio(self, audio_data: AudioData) -> ProcessedData:
        """
        处理音频数据
        输入: 音频数据
        输出: (内容音素, emotion2vec表征)
        """
        pass
    
    @abstractmethod
    def extract_phonemes(self, audio_data: AudioData, text: Optional[str] = None) -> List[str]:
        """提取音素"""
        pass
    
    @abstractmethod
    def extract_emotion_features(self, audio_data: AudioData) -> np.ndarray:
        """提取情感特征"""
        pass


class StageAModel(ABC):
    """阶段A模型抽象基类: 内容音素 -> M0"""
    
    @abstractmethod
    def forward(self, phonemes: List[str]) -> ModelOutput:
        """
        前向传播
        输入: 内容音素
        输出: M0 (中性Mel频谱)
        """
        pass
    
    @abstractmethod
    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """训练步骤"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        pass


class EmotionQuantizer(ABC):
    """情感量化器抽象基类: VQ-VAE量化emotion2vec表征"""
    
    @abstractmethod
    def quantize(self, emotion_features: np.ndarray, codebook_size: int) -> Tuple[np.ndarray, float]:
        """
        量化情感特征
        输入: emotion2vec表征, 码本大小
        输出: (量化后的特征, 重建损失)
        """
        pass
    
    @abstractmethod
    def train_codebook(self, emotion_dataset: List[np.ndarray]) -> None:
        """
        训练码本（具体方法由实现决定）
        输入: 情感特征数据集
        """
        pass
    
    @abstractmethod
    def experiment_codebook_sizes(self, emotion_features: np.ndarray) -> Dict[int, Tuple[np.ndarray, float]]:
        """
        实验不同码本大小的效果
        输入: emotion2vec表征
        输出: {码本大小: (量化特征, 损失)}
        """
        pass
    
    @abstractmethod
    def get_available_codebook_sizes(self) -> List[int]:
        """获取可用的码本大小列表"""
        pass


class StageBModel(ABC):
    """阶段B模型抽象基类: 包含B1和B2两个子阶段"""
    
    @abstractmethod
    def forward_b1(self, mel_input: torch.Tensor, emotion_features: np.ndarray) -> ModelOutput:
        """
        B1阶段前向传播
        输入: (M0, emotion2vec表征)
        输出: M1 (带情感的Mel频谱)
        """
        pass
    
    @abstractmethod
    def forward_b2(self, mel_input: torch.Tensor, emotion_features: np.ndarray) -> ModelOutput:
        """
        B2阶段前向传播 (可选，扩散模型细化)
        输入: (M1, emotion2vec表征)
        输出: M2 (细化的Mel频谱)
        """
        pass
    
    @abstractmethod
    def forward(self, mel_input: torch.Tensor, emotion_features: np.ndarray, use_b2: bool = False) -> ModelOutput:
        """
        完整B阶段前向传播
        输入: (M0, emotion2vec表征)
        输出: M1 或 M2 (根据use_b2参数)
        """
        pass
    
    @abstractmethod
    def train_step_b1(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """B1训练步骤"""
        pass
    
    @abstractmethod
    def train_step_b2(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """B2训练步骤"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        pass


class Vocoder(ABC):
    """声码器抽象基类: Mel频谱 -> 音频"""
    
    @abstractmethod
    def synthesize(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """
        合成音频
        输入: Mel频谱
        输出: 音频波形
        """
        pass
    
    @abstractmethod
    def load_pretrained(self, model_path: str) -> None:
        """加载预训练模型"""
        pass


class Pipeline(ABC):
    """完整流水线抽象基类"""
    
    def __init__(self, 
                 data_processor: DataProcessor,
                 stage_a_model: StageAModel,
                 stage_b_model: StageBModel,
                 vocoder: Vocoder,
                 emotion_quantizer: Optional[EmotionQuantizer] = None):
        self.data_processor = data_processor
        self.stage_a_model = stage_a_model
        self.stage_b_model = stage_b_model
        self.vocoder = vocoder
        self.emotion_quantizer = emotion_quantizer
    
    @abstractmethod
    def train(self, config: Dict[str, Any]) -> None:
        """训练完整流水线"""
        pass
    
    @abstractmethod
    def train_quantizer(self, emotion_dataset: List[np.ndarray]) -> None:
        """训练量化器（如果启用的话）"""
        pass
    
    @abstractmethod
    def inference(self, input_data, use_quantizer: bool = False, 
                 codebook_size: Optional[int] = None, use_b2: bool = False) -> np.ndarray:
        """
        推理生成音频
        Args:
            input_data: AudioData 或 ProcessedData
            use_quantizer: 是否使用VQ-VAE量化器
            codebook_size: VQ-VAE码本大小
            use_b2: 是否使用B2阶段
        Returns:
            重建的音频波形
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: List[AudioData]) -> Dict[str, float]:
        """评估模型性能"""
        pass


class ModelFactory(ABC):
    """模型工厂抽象基类，支持动态切换不同模型实现"""
    
    @abstractmethod
    def create_data_processor(self, config: Dict[str, Any]) -> DataProcessor:
        """创建数据处理器"""
        pass
    
    @abstractmethod
    def create_stage_a_model(self, config: Dict[str, Any]) -> StageAModel:
        """创建阶段A模型"""
        pass
    
    @abstractmethod
    def create_stage_b_model(self, config: Dict[str, Any]) -> StageBModel:
        """创建阶段B模型"""
        pass
    
    @abstractmethod
    def create_vocoder(self, config: Dict[str, Any]) -> Vocoder:
        """创建声码器"""
        pass
    
    @abstractmethod
    def create_emotion_quantizer(self, config: Dict[str, Any]) -> Optional[EmotionQuantizer]:
        """创建情感量化器（可选）"""
        pass
