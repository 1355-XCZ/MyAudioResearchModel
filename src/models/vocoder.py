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

# BigVGAN相关导入
try:
    import bigvgan
    from transformers import AutoModel
    BIGVGAN_AVAILABLE = True
except ImportError:
    BIGVGAN_AVAILABLE = False
    print("警告: BigVGAN库未安装，BigVGAN声码器将不可用")

# NVIDIA NeMo HiFi-GAN相关导入
try:
    from nemo.collections.tts.models import HifiGanModel
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("警告: NeMo库未安装，NVIDIA预训练HiFi-GAN将不可用")


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


class NVIDIAHiFiGANVocoder(Vocoder):
    """
    NVIDIA预训练HiFi-GAN声码器
    基于NeMo toolkit的预训练模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model_config', {})
        
        # 模型参数
        self.model_name = self.model_config.get('model_name', 'nvidia/tts_hifigan')
        self.sampling_rate = 22050  # NVIDIA HiFi-GAN固定采样率
        self.hop_length = 256
        self.n_mel_channels = 80
        
        # 初始化模型
        self.model = None
        self._load_nvidia_hifigan()
    
    def _load_nvidia_hifigan(self):
        """加载NVIDIA预训练HiFi-GAN模型"""
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo库未安装，请安装: pip install nemo_toolkit[all]")
        
        try:
            print(f"正在加载NVIDIA HiFi-GAN模型: {self.model_name}")
            
            # 使用NeMo加载预训练HiFi-GAN
            self.model = HifiGanModel.from_pretrained(model_name=self.model_name)
            self.model.eval()
            
            print(f"✅ 成功加载NVIDIA HiFi-GAN预训练模型")
            
        except Exception as e:
            print(f"❌ NVIDIA HiFi-GAN模型加载失败: {e}")
            print("请检查:")
            print("1. 网络连接是否正常")
            print("2. NeMo toolkit是否正确安装")
            print("3. 模型名称是否正确")
            self.model = None
            raise
    
    def synthesize(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """
        使用NVIDIA HiFi-GAN合成音频
        输入: Mel频谱 (batch, mel_dim, time) 或 (mel_dim, time)
        输出: 音频波形
        """
        if self.model is None:
            raise RuntimeError("NVIDIA HiFi-GAN模型未正确加载")
        
        # 确保输入维度正确
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)  # (1, mel_dim, time)
        elif mel_spectrogram.dim() == 3 and mel_spectrogram.shape[2] == self.n_mel_channels:
            # 如果输入是 (batch, time, mel_dim)，转换为 (batch, mel_dim, time)
            mel_spectrogram = mel_spectrogram.transpose(1, 2)
        
        # 检查Mel频谱维度
        if mel_spectrogram.shape[1] != self.n_mel_channels:
            print(f"警告: 输入Mel频谱维度 {mel_spectrogram.shape[1]} 与NVIDIA HiFi-GAN期望维度 {self.n_mel_channels} 不匹配")
            # 调整维度
            if mel_spectrogram.shape[1] < self.n_mel_channels:
                # 填充
                padding = torch.zeros(mel_spectrogram.shape[0], 
                                    self.n_mel_channels - mel_spectrogram.shape[1], 
                                    mel_spectrogram.shape[2])
                mel_spectrogram = torch.cat([mel_spectrogram, padding], dim=1)
            else:
                # 截断
                mel_spectrogram = mel_spectrogram[:, :self.n_mel_channels, :]
        
        # 生成音频
        try:
            with torch.no_grad():
                # NeMo HiFi-GAN的标准接口
                audio = self.model.convert_spectrogram_to_audio(spec=mel_spectrogram)
            
            # 处理输出格式
            if audio.dim() == 3:
                audio = audio.squeeze(1)  # 移除通道维度
            if audio.dim() == 2:
                audio = audio.squeeze(0)  # 如果batch_size=1，移除batch维度
            
            return audio.cpu().numpy()
            
        except Exception as e:
            print(f"NVIDIA HiFi-GAN音频生成失败: {e}")
            # 返回静音作为备选
            return np.zeros(mel_spectrogram.shape[2] * self.hop_length, dtype=np.float32)
    
    def load_pretrained(self, model_path: str) -> None:
        """
        NVIDIA HiFi-GAN使用NeMo自动管理预训练模型
        但保留接口兼容性
        """
        if os.path.exists(model_path):
            print(f"警告: NVIDIA HiFi-GAN使用NeMo自动下载，忽略本地路径: {model_path}")
        else:
            print(f"NVIDIA HiFi-GAN将通过NeMo自动下载: {self.model_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取NVIDIA HiFi-GAN模型信息"""
        return {
            'model_name': self.model_name,
            'sampling_rate': self.sampling_rate,
            'hop_length': self.hop_length,
            'n_mel_channels': self.n_mel_channels,
            'model_loaded': self.model is not None,
            'model_type': 'NVIDIA_PreTrained_HiFiGAN'
        }


class BigVGANVocoder(Vocoder):
    """
    NVIDIA BigVGAN声码器实现
    支持44kHz, 24kHz, 22kHz三个版本
    """
    
    # 预定义的BigVGAN模型配置
    BIGVGAN_MODELS = {
        '44khz': {
            'model_name': 'nvidia/bigvgan_v2_44khz_128band_512x',
            'sampling_rate': 44100,
            'hop_length': 512,
            'n_mel_channels': 128
        },
        '24khz': {
            'model_name': 'nvidia/bigvgan_v2_24khz_100band_256x', 
            'sampling_rate': 24000,
            'hop_length': 256,
            'n_mel_channels': 100
        },
        '22khz': {
            'model_name': 'nvidia/bigvgan_v2_22khz_80band_fmax8k_256x',
            'sampling_rate': 22050,
            'hop_length': 256,
            'n_mel_channels': 80
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model_config', {})
        
        # 获取BigVGAN版本 (44khz, 24khz, 22khz)
        self.version = self.model_config.get('version', '22khz')
        
        if self.version not in self.BIGVGAN_MODELS:
            raise ValueError(f"不支持的BigVGAN版本: {self.version}. 支持的版本: {list(self.BIGVGAN_MODELS.keys())}")
        
        # 获取模型信息
        model_info = self.BIGVGAN_MODELS[self.version]
        self.model_name = model_info['model_name']
        self.sampling_rate = model_info['sampling_rate']
        self.hop_length = model_info['hop_length'] 
        self.n_mel_channels = model_info['n_mel_channels']
        
        # 初始化模型
        self.model = None
        self._load_bigvgan_model()
        
    def _load_bigvgan_model(self):
        """加载BigVGAN预训练模型"""
        if not BIGVGAN_AVAILABLE:
            raise ImportError("BigVGAN库未安装，请安装: pip install bigvgan")
        
        try:
            print(f"正在加载BigVGAN模型: {self.model_name}")
            
            # 方法1: 尝试使用bigvgan库直接加载
            try:
                self.model = bigvgan.BigVGAN.from_pretrained(self.model_name)
                print(f"✅ 成功加载BigVGAN模型 ({self.version})")
                
            except Exception as e:
                print(f"尝试bigvgan库加载失败: {e}")
                
                # 方法2: 尝试使用transformers库加载
                try:
                    from transformers import AutoModel
                    self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                    print(f"✅ 通过transformers成功加载BigVGAN模型 ({self.version})")
                    
                except Exception as e2:
                    print(f"transformers加载也失败: {e2}")
                    
                    # 方法3: 使用torch.hub加载
                    try:
                        import torch
                        self.model = torch.hub.load('NVIDIA/BigVGAN', 'bigvgan', 
                                                  model_name=self.model_name.split('/')[-1])
                        print(f"✅ 通过torch.hub成功加载BigVGAN模型 ({self.version})")
                        
                    except Exception as e3:
                        print(f"所有加载方法都失败:")
                        print(f"  - bigvgan: {e}")
                        print(f"  - transformers: {e2}")
                        print(f"  - torch.hub: {e3}")
                        raise RuntimeError("无法加载BigVGAN模型，请检查网络连接和依赖安装")
            
            # 设置为评估模式
            if self.model is not None:
                self.model.eval()
                
        except Exception as e:
            print(f"❌ BigVGAN模型加载失败: {e}")
            self.model = None
            raise
    
    def synthesize(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """
        使用BigVGAN合成音频
        输入: Mel频谱 (batch, mel_dim, time) 或 (mel_dim, time)
        输出: 音频波形
        """
        if self.model is None:
            raise RuntimeError("BigVGAN模型未正确加载")
        
        # 确保输入维度正确
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)  # (1, mel_dim, time)
        elif mel_spectrogram.dim() == 3 and mel_spectrogram.shape[2] == self.n_mel_channels:
            # 如果输入是 (batch, time, mel_dim)，转换为 (batch, mel_dim, time)
            mel_spectrogram = mel_spectrogram.transpose(1, 2)
        
        # 检查Mel频谱维度是否匹配
        if mel_spectrogram.shape[1] != self.n_mel_channels:
            print(f"警告: 输入Mel频谱维度 {mel_spectrogram.shape[1]} 与模型期望维度 {self.n_mel_channels} 不匹配")
            # 尝试调整维度
            if mel_spectrogram.shape[1] < self.n_mel_channels:
                # 如果输入维度小，用零填充
                padding = torch.zeros(mel_spectrogram.shape[0], 
                                    self.n_mel_channels - mel_spectrogram.shape[1], 
                                    mel_spectrogram.shape[2])
                mel_spectrogram = torch.cat([mel_spectrogram, padding], dim=1)
            else:
                # 如果输入维度大，截断
                mel_spectrogram = mel_spectrogram[:, :self.n_mel_channels, :]
        
        # 生成音频
        try:
            with torch.no_grad():
                # 不同的BigVGAN实现可能有不同的接口
                if hasattr(self.model, 'forward'):
                    audio = self.model.forward(mel_spectrogram)
                elif hasattr(self.model, 'generate'):
                    audio = self.model.generate(mel_spectrogram)
                elif callable(self.model):
                    audio = self.model(mel_spectrogram)
                else:
                    raise AttributeError("BigVGAN模型没有可调用的推理方法")
            
            # 处理输出格式
            if isinstance(audio, tuple):
                audio = audio[0]  # 如果返回tuple，取第一个元素
            
            # 转换为numpy数组
            if audio.dim() == 3:
                audio = audio.squeeze(1)  # 移除通道维度
            if audio.dim() == 2:
                audio = audio.squeeze(0)  # 如果batch_size=1，移除batch维度
            
            return audio.cpu().numpy()
            
        except Exception as e:
            print(f"BigVGAN音频生成失败: {e}")
            # 返回静音作为备选
            return np.zeros(mel_spectrogram.shape[2] * self.hop_length, dtype=np.float32)
    
    def load_pretrained(self, model_path: str) -> None:
        """
        BigVGAN使用HuggingFace Hub，通常不需要手动加载本地权重
        但保留接口兼容性
        """
        if os.path.exists(model_path):
            print(f"警告: BigVGAN通常使用HuggingFace Hub自动下载，忽略本地路径: {model_path}")
        else:
            print(f"BigVGAN模型将从HuggingFace Hub自动下载: {self.model_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前BigVGAN模型信息"""
        return {
            'version': self.version,
            'model_name': self.model_name,
            'sampling_rate': self.sampling_rate,
            'hop_length': self.hop_length,
            'n_mel_channels': self.n_mel_channels,
            'model_loaded': self.model is not None
        }


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
            vocoder_type: 声码器类型 ('hifigan', 'hifigan_custom', 'hifigan_nvidia', 'melgan', 'bigvgan_44khz', 'bigvgan_24khz', 'bigvgan_22khz')
            config: 配置字典
        Returns:
            声码器实例
        """
        if vocoder_type.lower() == 'hifigan':
            # 默认使用自定义HiFi-GAN (保持向后兼容)
            return HiFiGANVocoder(config)
        elif vocoder_type.lower() == 'hifigan_custom':
            # 明确使用自定义HiFi-GAN
            return HiFiGANVocoder(config)
        elif vocoder_type.lower() == 'hifigan_nvidia':
            # 使用NVIDIA预训练HiFi-GAN
            return NVIDIAHiFiGANVocoder(config)
        elif vocoder_type.lower() == 'melgan':
            return MelGANVocoder(config)
        elif vocoder_type.lower() == 'bigvgan_44khz':
            # 44kHz BigVGAN
            config_copy = config.copy()
            config_copy['model_config'] = config_copy.get('model_config', {})
            config_copy['model_config']['version'] = '44khz'
            return BigVGANVocoder(config_copy)
        elif vocoder_type.lower() == 'bigvgan_24khz':
            # 24kHz BigVGAN
            config_copy = config.copy()
            config_copy['model_config'] = config_copy.get('model_config', {})
            config_copy['model_config']['version'] = '24khz'
            return BigVGANVocoder(config_copy)
        elif vocoder_type.lower() == 'bigvgan_22khz':
            # 22kHz BigVGAN
            config_copy = config.copy()
            config_copy['model_config'] = config_copy.get('model_config', {})
            config_copy['model_config']['version'] = '22khz'
            return BigVGANVocoder(config_copy)
        elif vocoder_type.lower() == 'bigvgan':
            # 默认使用22kHz版本
            config_copy = config.copy()
            config_copy['model_config'] = config_copy.get('model_config', {})
            config_copy['model_config']['version'] = config_copy['model_config'].get('version', '22khz')
            return BigVGANVocoder(config_copy)
        else:
            raise ValueError(f"不支持的声码器类型: {vocoder_type}. 支持的类型: {VocoderFactory.get_available_vocoders()}")
    
    @staticmethod
    def get_available_vocoders() -> list:
        """获取可用的声码器类型"""
        base_vocoders = ['hifigan', 'hifigan_custom', 'melgan']
        bigvgan_vocoders = ['bigvgan', 'bigvgan_44khz', 'bigvgan_24khz', 'bigvgan_22khz']
        nvidia_vocoders = ['hifigan_nvidia']
        
        available = base_vocoders.copy()
        
        if NEMO_AVAILABLE:
            available.extend(nvidia_vocoders)
            
        if BIGVGAN_AVAILABLE:
            available.extend(bigvgan_vocoders)
            
        return available
    
    @staticmethod
    def get_bigvgan_models() -> Dict[str, Dict[str, Any]]:
        """获取BigVGAN模型信息"""
        if BIGVGAN_AVAILABLE:
            return BigVGANVocoder.BIGVGAN_MODELS
        else:
            return {}
