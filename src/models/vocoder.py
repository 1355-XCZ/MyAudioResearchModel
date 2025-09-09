"""
声码器实现: Mel频谱 -> 音频
支持HiFi-GAN, MelGAN等不同声码器的接口
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
import os

from ..core.interfaces import Vocoder


class HiFiGANVocoder(Vocoder):
    """
    HiFi-GAN声码器实现
    输入: Mel频谱
    输出: 音频波形
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model_config', {})
        
        # 音频参数
        self.sampling_rate = self.model_config.get('sampling_rate', 22050)
        self.hop_length = self.model_config.get('hop_length', 256)
        self.win_length = self.model_config.get('win_length', 1024)
        self.n_mel_channels = self.model_config.get('n_mel_channels', 80)
        
        # 构建模型
        self.generator = self._build_generator()
        self.discriminator = None  # 推理时不需要判别器
        
    def _build_generator(self) -> nn.Module:
        """构建HiFi-GAN生成器"""
        return HiFiGANGenerator(
            n_mel_channels=self.n_mel_channels,
            hop_length=self.hop_length
        )
    
    def synthesize(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """
        合成音频
        输入: Mel频谱 (batch, mel_dim, time) 或 (mel_dim, time)
        输出: 音频波形
        """
        # 确保输入维度正确
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)  # (1, mel_dim, time)
        elif mel_spectrogram.dim() == 3 and mel_spectrogram.shape[2] == self.n_mel_channels:
            # 如果输入是 (batch, time, mel_dim)，转换为 (batch, mel_dim, time)
            mel_spectrogram = mel_spectrogram.transpose(1, 2)
        
        # 生成音频
        self.generator.eval()
        with torch.no_grad():
            audio = self.generator(mel_spectrogram)
        
        # 转换为numpy数组
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # 移除通道维度
        if audio.dim() == 2:
            audio = audio.squeeze(0)  # 如果batch_size=1，移除batch维度
        
        return audio.cpu().numpy()
    
    def load_pretrained(self, model_path: str) -> None:
        """加载预训练模型"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 尝试不同的状态字典键名
            if 'generator' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator'])
            elif 'model_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.generator.load_state_dict(checkpoint)
            
            print(f"已加载预训练声码器: {model_path}")
        else:
            print(f"警告: 预训练模型不存在 {model_path}，使用随机初始化的模型")


class MelGANVocoder(Vocoder):
    """
    MelGAN声码器实现（简化版）
    作为HiFi-GAN的替代选择
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model_config', {})
        
        self.sampling_rate = self.model_config.get('sampling_rate', 22050)
        self.hop_length = self.model_config.get('hop_length', 256)
        self.n_mel_channels = self.model_config.get('n_mel_channels', 80)
        
        self.generator = self._build_generator()
    
    def _build_generator(self) -> nn.Module:
        """构建MelGAN生成器"""
        return MelGANGenerator(
            n_mel_channels=self.n_mel_channels,
            hop_length=self.hop_length
        )
    
    def synthesize(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """合成音频"""
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
        elif mel_spectrogram.dim() == 3 and mel_spectrogram.shape[2] == self.n_mel_channels:
            mel_spectrogram = mel_spectrogram.transpose(1, 2)
        
        self.generator.eval()
        with torch.no_grad():
            audio = self.generator(mel_spectrogram)
        
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        
        return audio.cpu().numpy()
    
    def load_pretrained(self, model_path: str) -> None:
        """加载预训练模型"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'generator' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator'])
            else:
                self.generator.load_state_dict(checkpoint)
            print(f"已加载预训练MelGAN声码器: {model_path}")
        else:
            print(f"警告: 预训练模型不存在 {model_path}，使用随机初始化的模型")


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN生成器（简化实现）
    """
    
    def __init__(self, n_mel_channels: int = 80, hop_length: int = 256):
        super().__init__()
        
        self.n_mel_channels = n_mel_channels
        self.hop_length = hop_length
        
        # 上采样比率
        self.upsample_rates = [8, 8, 2, 2]  # 总共256倍上采样
        self.upsample_kernel_sizes = [16, 16, 4, 4]
        
        # 初始卷积
        self.conv_pre = nn.Conv1d(n_mel_channels, 512, 7, 1, padding=3)
        
        # 上采样层
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(
                512 // (2**i), 
                512 // (2**(i+1)),
                k, u, padding=(k-u)//2
            ))
        
        # 残差块
        self.resblocks = nn.ModuleList()
        ch = 512
        for i in range(len(self.ups)):
            ch //= 2
            for j in range(3):  # 每个上采样层后3个残差块
                self.resblocks.append(ResBlock(ch))
        
        # 输出卷积
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        
        # 激活函数
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (batch, n_mel_channels, time)
        Returns:
            audio: (batch, 1, time * hop_length)
        """
        x = self.conv_pre(x)
        
        for i in range(len(self.ups)):
            x = self.activation(x)
            x = self.ups[i](x)
            
            # 应用残差块
            xs = None
            for j in range(3):
                if xs is None:
                    xs = self.resblocks[i*3+j](x)
                else:
                    xs += self.resblocks[i*3+j](x)
            x = xs / 3  # 平均
        
        x = self.activation(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class MelGANGenerator(nn.Module):
    """
    MelGAN生成器（简化实现）
    """
    
    def __init__(self, n_mel_channels: int = 80, hop_length: int = 256):
        super().__init__()
        
        self.n_mel_channels = n_mel_channels
        
        # 上采样层
        self.layers = nn.Sequential(
            nn.Conv1d(n_mel_channels, 512, 7, padding=3),
            nn.LeakyReLU(0.2),
            
            # 第一个上采样块 (x8)
            nn.ConvTranspose1d(512, 256, 16, 8, padding=4),
            nn.LeakyReLU(0.2),
            
            # 第二个上采样块 (x8)
            nn.ConvTranspose1d(256, 128, 16, 8, padding=4),
            nn.LeakyReLU(0.2),
            
            # 第三个上采样块 (x2)
            nn.ConvTranspose1d(128, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 第四个上采样块 (x2)
            nn.ConvTranspose1d(64, 32, 4, 2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 输出层
            nn.Conv1d(32, 1, 7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.layers(x)


class ResBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3),
            nn.Conv1d(channels, channels, 3, 1, dilation=5, padding=5),
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3),
            nn.Conv1d(channels, channels, 3, 1, dilation=5, padding=5),
        ])
        
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.activation(x)
            xt = c1(xt)
            xt = self.activation(xt)
            xt = c2(xt)
            x = xt + x
        return x


class VocoderFactory:
    """声码器工厂类"""
    
    @staticmethod
    def create_vocoder(vocoder_type: str, config: Dict[str, Any]) -> Vocoder:
        """
        创建声码器实例
        Args:
            vocoder_type: 声码器类型 ('hifigan', 'melgan')
            config: 配置字典
        Returns:
            声码器实例
        """
        if vocoder_type.lower() == 'hifigan':
            return HiFiGANVocoder(config)
        elif vocoder_type.lower() == 'melgan':
            return MelGANVocoder(config)
        else:
            raise ValueError(f"不支持的声码器类型: {vocoder_type}")
    
    @staticmethod
    def get_available_vocoders() -> list:
        """获取可用的声码器类型"""
        return ['hifigan', 'melgan']
