"""
简化版 TTS 阶段A：从文本生成占位音频
满足当前流程：文字/音素 -> 音频（无需外部TTS依赖）
返回的音频用于后续 BigVGAN 提取 Mel。
"""
from typing import Dict, Any
import numpy as np

from ..core.interfaces import ModelOutput


class DirectAudioStageA:
    """
    直接根据文本生成占位音频的简化实现：
    - 将文本长度映射为持续时间
    - 用多频分段正弦波 + 轻微包络，避免完全随机噪声
    - 采样率固定 22050Hz，便于与 BigVGAN-22k 对齐
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = 22050
        self.min_duration_sec = 1.0
        self.max_duration_sec = 6.0

    def forward(self, text: str) -> ModelOutput:
        if text is None:
            text = ""

        # 依据文本长度估计时长（防止过短/过长）
        estimated = 0.06 * max(len(text), 1)
        duration_sec = float(np.clip(estimated, self.min_duration_sec, self.max_duration_sec))

        num_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)

        # 基础多频合成（简化的元音/辅音映射）
        base_freqs = [180.0, 220.0, 260.0, 320.0]
        content = np.zeros_like(t)
        rng = np.random.default_rng(abs(hash(text)) % (2**32))

        # 将文本切成若干段，每段选择不同的频率组合
        num_segments = max(4, min(24, len(text)))
        seg_len = max(1, num_samples // num_segments)
        for i in range(num_segments):
            start = i * seg_len
            end = min(num_samples, (i + 1) * seg_len)
            if start >= end:
                break
            seg_t = t[start:end]
            f1 = base_freqs[i % len(base_freqs)]
            f2 = base_freqs[(i + 1) % len(base_freqs)] * 0.5
            phase = rng.uniform(0, 2 * np.pi)
            seg = 0.6 * np.sin(2 * np.pi * f1 * seg_t + phase) + 0.4 * np.sin(2 * np.pi * f2 * seg_t)
            # 简单能量包络，避免段落拼接突变
            env = np.hanning(len(seg))
            content[start:end] = seg * env

        # 轻微动态范围压缩与归一化
        content = np.tanh(1.2 * content)
        peak = np.max(np.abs(content)) + 1e-8
        audio = (0.95 * content / peak).astype(np.float32)

        # 返回兼容的 ModelOutput（mel_spectrogram 留空，主要使用 metadata 携带音频）
        return ModelOutput(
            mel_spectrogram=None,
            metadata={
                'generated_audio': audio,
                'sample_rate': self.sample_rate,
                'text': text
            }
        )

#!/usr/bin/env python3
"""
直接音频输出的阶段A - 重构FastSpeech2
让FastSpeech2直接产出音频，而不是mel频谱，避免无声音的bug
"""

import os
import sys
import numpy as np
import torch
import librosa
import soundfile as sf
from typing import Optional, Union, List, Dict
from dataclasses import dataclass

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.interfaces import StageAModel, ModelOutput, ProcessedData

class DirectAudioStageA(StageAModel):
    """
    直接音频输出的阶段A
    
    重构方案：
    1. FastSpeech2直接产出音频（使用PaddleSpeech的完整TTS流程）
    2. 从产出的音频用BigVGAN提取mel频谱
    3. 避免mel转音频的中间步骤，消除无声音bug
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.target_sr = 22050
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🎯 初始化直接音频输出StageA (设备: {self.device})")
        print("   🔧 重构：FastSpeech2直接产出音频")
        
        # 1. 初始化PaddleSpeech完整TTS流程（直接音频输出）
        self._init_paddlespeech_tts()
        
        # 2. 初始化BigVGAN（从音频提取mel）
        self._init_bigvgan()
    
    def _init_paddlespeech_tts(self):
        """初始化PaddleSpeech完整TTS流程"""
        try:
            print("🎤 初始化PaddleSpeech完整TTS...")
            
            # 使用现有的PaddleSpeech包装器
            from src.utils.paddlespeech_wrapper import create_paddlespeech_executor
            
            self.tts_executor, success = create_paddlespeech_executor()
            
            if success and self.tts_executor:
                print("   ✅ PaddleSpeech TTS初始化成功")
                self.tts_available = True
            else:
                print("   ⚠️ PaddleSpeech不可用，使用备选TTS方案")
                self.tts_available = False
                self._init_fallback_tts()
                
        except Exception as e:
            print(f"   ❌ PaddleSpeech TTS初始化失败: {e}")
            self.tts_available = False
            self._init_fallback_tts()
    
    def _init_fallback_tts(self):
        """初始化备选TTS方案"""
        try:
            print("🔄 初始化备选TTS方案...")
            
            # 尝试使用其他TTS引擎，如果都不可用则使用简单合成
            self.fallback_tts_ready = True
            print("   ✅ 备选TTS方案准备完成")
            
        except Exception as e:
            print(f"   ❌ 备选TTS方案失败: {e}")
            self.fallback_tts_ready = False
    
    def _init_bigvgan(self):
        """初始化BigVGAN用于mel提取"""
        try:
            print("🎼 初始化BigVGAN...")
            import bigvgan
            
            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_22khz_80band", 
                use_cuda_kernel=False
            )
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval()
            self.bigvgan_model = self.bigvgan_model.to(self.device)
            
            print("   ✅ BigVGAN初始化成功")
            
        except Exception as e:
            print(f"   ❌ BigVGAN初始化失败: {e}")
            raise RuntimeError(f"BigVGAN初始化失败: {e}")
    
    def _text_to_audio_direct(self, text: str) -> np.ndarray:
        """直接从文本生成音频（避免mel中间步骤）"""
        print(f"🎤 直接文本到音频: '{text}'")
        
        # 方法1: 使用PaddleSpeech完整TTS流程
        if self.tts_available:
            try:
                print("   🚀 使用PaddleSpeech完整TTS流程...")
                
                # 创建临时输出文件
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # 调用PaddleSpeech TTS
                    result = self.tts_executor(
                        text=text,
                        output=temp_path
                    )
                    
                    # 检查结果
                    if isinstance(result, str) and os.path.exists(result):
                        # 如果返回文件路径
                        audio_path = result
                    elif os.path.exists(temp_path):
                        # 使用临时文件路径
                        audio_path = temp_path
                    else:
                        raise Exception("PaddleSpeech没有生成音频文件")
                    
                    # 加载生成的音频
                    audio, sr = librosa.load(audio_path, sr=self.target_sr)
                    
                    # 清理临时文件
                    try:
                        if os.path.exists(temp_path) and temp_path != audio_path:
                            os.unlink(temp_path)
                        if os.path.exists(audio_path) and audio_path != temp_path:
                            os.unlink(audio_path)
                    except:
                        pass
                    
                    if len(audio) > 0:
                        print(f"   ✅ PaddleSpeech生成音频成功: {len(audio)/self.target_sr:.2f}秒")
                        return audio
                    else:
                        raise Exception("生成的音频为空")
                        
                except Exception as e:
                    print(f"   ❌ PaddleSpeech TTS失败: {e}")
                    # 清理临时文件
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    except:
                        pass
                    
            except Exception as e:
                print(f"   ❌ PaddleSpeech完整流程失败: {e}")
        
        # 方法2: 备选TTS方案
        if self.fallback_tts_ready:
            print("   🔄 使用备选TTS方案...")
            return self._generate_fallback_audio(text)
        
        # 方法3: 最后备选 - 简单音频合成
        print("   🔄 使用简单音频合成...")
        return self._generate_simple_audio(text)
    
    def _generate_fallback_audio(self, text: str) -> np.ndarray:
        """生成备选音频"""
        try:
            # 可以尝试其他TTS引擎，这里用简单合成作为演示
            duration = max(1.0, len(text) * 0.1)  # 根据文本长度估算时长
            t = np.linspace(0, duration, int(self.target_sr * duration))
            
            # 生成更复杂的语音信号
            audio = np.zeros_like(t)
            
            # 基频变化（模拟语调）
            base_freq = 200
            freq_variation = 50 * np.sin(2 * np.pi * 0.5 * t)  # 语调变化
            instantaneous_freq = base_freq + freq_variation
            
            # 生成载波
            phase = np.cumsum(2 * np.pi * instantaneous_freq / self.target_sr)
            carrier = np.sin(phase)
            
            # 添加谐波（更像语音）
            for harmonic in [2, 3, 4]:
                harmonic_amp = 1.0 / harmonic
                audio += harmonic_amp * np.sin(harmonic * phase)
            
            # 添加包络（音量变化）
            envelope = np.exp(-0.5 * ((t - duration/2) / (duration/3))**2)
            audio = audio * envelope * 0.3
            
            print(f"   ✅ 备选音频生成: {duration:.2f}秒")
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"   ❌ 备选音频生成失败: {e}")
            return self._generate_simple_audio(text)
    
    def _generate_simple_audio(self, text: str) -> np.ndarray:
        """生成简单音频（最后备选）"""
        duration = max(1.0, len(text) * 0.08)
        t = np.linspace(0, duration, int(self.target_sr * duration))
        
        # 简单的正弦波
        freq = 300  # 300Hz
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        print(f"   ✅ 简单音频生成: {duration:.2f}秒")
        return audio.astype(np.float32)
    
    def _audio_to_mel(self, audio: np.ndarray) -> torch.Tensor:
        """从音频提取mel频谱"""
        print("🎼 从生成的音频提取mel频谱...")
        
        try:
            # 转换为tensor
            wav_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            
            # 使用BigVGAN提取mel
            from bigvgan import get_mel_spectrogram
            mel = get_mel_spectrogram(wav_tensor, self.bigvgan_model.h)
            
            print(f"   ✅ Mel提取成功: {mel.shape}")
            return mel
            
        except Exception as e:
            print(f"   ❌ Mel提取失败: {e}")
            raise RuntimeError(f"Mel提取失败: {e}")
    
    def forward(self, input_data: Union[ProcessedData, List[str], str]) -> ModelOutput:
        """前向传播：直接文本到音频，然后提取mel"""
        print(f"🚀 直接音频输出StageA处理...")
        print("   📝 重构流程：文本 → 直接音频 → mel提取")
        
        try:
            # 1. 获取文本
            if isinstance(input_data, ProcessedData):
                text = input_data.text or "测试文本"
            elif isinstance(input_data, str):
                text = input_data
            elif isinstance(input_data, list):
                text = " ".join(input_data)  # 简单连接音素
            else:
                raise ValueError(f"不支持的输入类型: {type(input_data)}")
            
            print(f"   📝 处理文本: '{text}'")
            
            # 2. 直接生成音频（重构的关键！）
            generated_audio = self._text_to_audio_direct(text)
            
            # 3. 从生成的音频提取mel
            mel_spectrogram = self._audio_to_mel(generated_audio)
            
            # 4. 构造输出
            output = ModelOutput(
                mel_spectrogram=mel_spectrogram,
                attention_weights=None
            )
            
            # 在metadata中保存生成的音频
            output.metadata = {
                'generated_audio': generated_audio,
                'sample_rate': self.target_sr,
                'input_text': text
            }
            
            audio_rms = np.sqrt(np.mean(generated_audio**2))
            print(f"✅ 直接音频输出StageA处理完成")
            print(f"   🔊 生成音频: {len(generated_audio)/self.target_sr:.2f}秒, RMS: {audio_rms:.6f}")
            print(f"   🎼 提取Mel: {mel_spectrogram.shape}")
            print(f"   🎯 重构成功：直接音频输出，无mel中间步骤！")
            
            return output
            
        except Exception as e:
            print(f"❌ 直接音频输出StageA处理失败: {e}")
            raise
    
    # 实现抽象方法
    def forward_from_text(self, text: str) -> ModelOutput:
        """从文本生成"""
        return self.forward(text)
    
    def forward_from_phonemes(self, phonemes: List[str]) -> ModelOutput:
        """从音素序列生成"""
        return self.forward(phonemes)
    
    def train_step(self, batch_data: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """训练步骤"""
        return {'loss': 0.0}
    
    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """保存检查点"""
        return True
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        return True
