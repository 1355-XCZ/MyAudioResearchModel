"""
音频数据处理实现
支持Whisper文本提取和Emotion2Vec情感特征提取
"""
import os
import librosa
import numpy as np
from typing import List, Optional, Dict, Any
import torch
import whisper
from transformers import AutoModel, AutoProcessor

from ..core.interfaces import DataProcessor, AudioData, ProcessedData


class WhisperEmotionProcessor(DataProcessor):
    """基于Whisper和Emotion2Vec的数据处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.whisper_model = None
        self.emotion_model = None
        self.emotion_processor = None
        self._load_models()
    
    def _load_models(self):
        """加载模型"""
        # 加载Whisper模型
        whisper_config = self.config.get('whisper', {})
        model_size = whisper_config.get('model_size', 'base')
        self.whisper_model = whisper.load_model(model_size)
        
        # 加载Emotion2Vec模型
        emotion_config = self.config.get('emotion2vec', {})
        model_path = emotion_config.get('model_path', 'emotion2vec_base')
        
        # 注意：这里需要根据实际的emotion2vec模型进行调整
        # 假设使用HuggingFace格式的模型
        try:
            self.emotion_model = AutoModel.from_pretrained(model_path)
            self.emotion_processor = AutoProcessor.from_pretrained(model_path)
        except:
            print(f"警告：无法加载emotion2vec模型 {model_path}，将使用模拟特征")
            self.emotion_model = None
            self.emotion_processor = None
    
    def process_audio(self, audio_data: AudioData) -> ProcessedData:
        """
        处理音频数据
        输入: 音频数据
        输出: (内容音素, emotion2vec表征)
        """
        # 提取音素
        phonemes = self.extract_phonemes(audio_data)
        
        # 提取情感特征
        emotion_features = self.extract_emotion_features(audio_data)
        
        return ProcessedData(
            phonemes=phonemes,
            emotion_features=emotion_features,
            audio_path=audio_data.file_path
        )
    
    def extract_phonemes(self, audio_data: AudioData, text: Optional[str] = None) -> List[str]:
        """
        提取音素
        如果提供了文本，直接使用；否则使用Whisper进行语音识别
        """
        if text is None:
            # 使用Whisper进行语音识别
            result = self.whisper_model.transcribe(
                audio_data.waveform,
                language=self.config.get('whisper', {}).get('language', 'zh')
            )
            text = result["text"]
        
        # 转换为音素
        phonemes = self._text_to_phonemes(text)
        return phonemes
    
    def _text_to_phonemes(self, text: str) -> List[str]:
        """
        文本转音素
        这里使用简单的拼音转换，实际应用中可能需要更复杂的音素转换
        """
        phoneme_config = self.config.get('phoneme', {})
        g2p_model = phoneme_config.get('g2p_model', 'pypinyin')
        
        if g2p_model == 'pypinyin':
            try:
                from pypinyin import lazy_pinyin, Style
                phonemes = lazy_pinyin(text, style=Style.TONE3)
                return phonemes
            except ImportError:
                print("警告：pypinyin未安装，使用字符级别的音素")
                return list(text.replace(' ', ''))
        else:
            # 其他G2P模型的接口可以在这里扩展
            return list(text.replace(' ', ''))
    
    def extract_emotion_features(self, audio_data: AudioData) -> np.ndarray:
        """
        提取情感特征
        使用Emotion2Vec模型提取情感表征
        """
        if self.emotion_model is None:
            # 如果模型未加载，返回模拟特征
            feature_dim = self.config.get('emotion2vec', {}).get('feature_dim', 768)
            return np.random.randn(feature_dim).astype(np.float32)
        
        try:
            # 预处理音频
            if self.emotion_processor:
                inputs = self.emotion_processor(
                    audio_data.waveform, 
                    sampling_rate=audio_data.sample_rate, 
                    return_tensors="pt"
                )
            else:
                # 简单的预处理
                inputs = {"input_values": torch.tensor(audio_data.waveform).unsqueeze(0)}
            
            # 提取特征
            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
                # 假设模型输出的最后一层隐藏状态作为情感特征
                emotion_features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            return emotion_features.astype(np.float32)
            
        except Exception as e:
            print(f"情感特征提取失败: {e}")
            # 返回模拟特征
            feature_dim = self.config.get('emotion2vec', {}).get('feature_dim', 768)
            return np.random.randn(feature_dim).astype(np.float32)


class AudioLoader:
    """音频加载工具类"""
    
    @staticmethod
    def load_audio(file_path: str, target_sr: int = 22050) -> AudioData:
        """
        加载音频文件
        """
        waveform, sr = librosa.load(file_path, sr=target_sr)
        duration = len(waveform) / sr
        
        return AudioData(
            waveform=waveform,
            sample_rate=sr,
            duration=duration,
            file_path=file_path
        )
    
    @staticmethod
    def save_audio(audio_data: AudioData, output_path: str):
        """
        保存音频文件
        """
        import soundfile as sf
        sf.write(output_path, audio_data.waveform, audio_data.sample_rate)


class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audio_loader = AudioLoader()
    
    def load_train_data(self) -> List[AudioData]:
        """加载训练数据"""
        train_config = self.config.get('train_data', {})
        audio_dir = train_config.get('audio_dir')
        
        audio_files = []
        if audio_dir and os.path.exists(audio_dir):
            for file_name in os.listdir(audio_dir):
                if file_name.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(audio_dir, file_name)
                    audio_files.append(self.audio_loader.load_audio(file_path))
        
        return audio_files
    
    def load_test_data(self) -> List[AudioData]:
        """加载测试数据（ESD数据集）"""
        test_config = self.config.get('test_data', {})
        audio_dir = test_config.get('audio_dir')
        
        audio_files = []
        if audio_dir and os.path.exists(audio_dir):
            for file_name in os.listdir(audio_dir):
                if file_name.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(audio_dir, file_name)
                    audio_files.append(self.audio_loader.load_audio(file_path))
        
        return audio_files
