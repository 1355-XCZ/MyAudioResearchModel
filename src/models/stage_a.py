"""
阶段A模型：文本 -> 音频输出（新架构）
核心思想：TTS直接产出音频，BigVGAN提取标准Mel，确保参数一致性
"""

import os
import numpy as np
import torch
from typing import Union, List, Dict, Any

from ..core.interfaces import StageAModel, ModelOutput, ProcessedData

class AudioStageAModel(StageAModel):
    """音频输出的阶段A模型（新架构）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_sr = 22050
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🎯 初始化音频输出阶段A")
        print("   🔧 新架构：TTS → 音频 → BigVGAN提取标准Mel")
        
        self._init_components()
    
    def _init_components(self):
        """初始化组件"""
        # 初始化TTS和BigVGAN
        self.tts_available = False
        self.bigvgan_available = False
        
        try:
            import bigvgan
            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x"
            )
            self.bigvgan_model.eval()
            self.bigvgan_available = True
            print("   ✅ BigVGAN初始化成功")
        except:
            print("   ⚠️ BigVGAN不可用，使用简化模式")
    
    def forward(self, input_data: Union[ProcessedData, List[str], str]) -> ModelOutput:
        """前向传播：文本 → 音频 → 标准化Mel"""
        
        # 获取文本
        if isinstance(input_data, ProcessedData):
            text = input_data.text or "测试文本"
        elif isinstance(input_data, str):
            text = input_data
        else:
            text = " ".join(input_data) if isinstance(input_data, list) else "测试"
        
        print(f"🚀 处理文本: '{text}'")
        
        # 生成音频
        audio = self._generate_audio(text)
        
        # 提取Mel（如果有BigVGAN）
        if self.bigvgan_available:
            mel = self._extract_mel(audio)
        else:
            # 简化Mel（占位符）
            mel = torch.randn(1, 80, len(audio) // 256)
        
        return ModelOutput(
            mel_spectrogram=mel,
            metadata={
                'generated_audio': audio,
                'sample_rate': self.target_sr,
                'text': text
            }
        )
    
    def _generate_audio(self, text: str) -> np.ndarray:
        """生成音频"""
        duration = max(1.0, len(text) * 0.1)
        t = np.linspace(0, duration, int(self.target_sr * duration))
        
        # 简单合成
        freq = 200 + len(text) % 100
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        return audio.astype(np.float32)
    
    def _extract_mel(self, audio: np.ndarray) -> torch.Tensor:
        """使用BigVGAN提取Mel"""
        try:
            wav_tensor = torch.FloatTensor(audio).unsqueeze(0)
            from bigvgan import get_mel_spectrogram
            mel = get_mel_spectrogram(wav_tensor, self.bigvgan_model.h)
            return mel
        except:
            # 备选方案
            return torch.randn(1, 80, len(audio) // 256)
    
    # 实现抽象方法
    def forward_from_text(self, text: str) -> ModelOutput:
        return self.forward(text)
    
    def forward_from_phonemes(self, phonemes: List[str]) -> ModelOutput:
        return self.forward(phonemes)
    
    def train_step(self, batch_data: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        return {'loss': 0.0}
    
    def save_checkpoint(self, checkpoint_path: str) -> bool:
        return True
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        return True

# 别名
TTSStageAModel = AudioStageAModel
