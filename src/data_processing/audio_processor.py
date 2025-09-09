"""
音频数据处理实现
支持Whisper文本提取和Emotion2Vec情感特征提取
"""
import os
import librosa
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
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
        model_path = emotion_config.get('model_path', 'iic/emotion2vec_base')
        
        # 使用官方的ModelScope API加载Emotion2Vec
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            self.emotion_pipeline = pipeline(
                task=Tasks.emotion_recognition,
                model=model_path
            )
            print(f"✅ 成功加载emotion2vec模型: {model_path}")
            
        except ImportError:
            print("警告：modelscope未安装，尝试使用FunASR...")
            try:
                from funasr import AutoModel
                self.emotion_model = AutoModel(model=model_path)
                self.emotion_pipeline = None
                print(f"✅ 成功加载emotion2vec模型(FunASR): {model_path}")
            except:
                print(f"警告：无法加载emotion2vec模型 {model_path}，将使用模拟特征")
                self.emotion_pipeline = None
                self.emotion_model = None
    
    def process_audio(self, audio_data: AudioData) -> ProcessedData:
        """
        处理音频数据
        输入: 音频数据
        输出: (内容音素, emotion2vec表征)
        """
        # 提取音素和文本
        phonemes, recognized_text = self.extract_phonemes(audio_data)
        
        # 提取情感特征
        emotion_features = self.extract_emotion_features(audio_data)
        
        return ProcessedData(
            phonemes=phonemes,
            emotion_features=emotion_features,
            source_audio=audio_data,
            text=recognized_text
        )
    
    def extract_phonemes(self, audio_data: AudioData, text: Optional[str] = None) -> Tuple[List[str], str]:
        """
        提取音素
        如果提供了文本，直接使用；否则使用Whisper进行语音识别
        返回: (音素列表, 识别的文本)
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
        return phonemes, text
    
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
        elif g2p_model == 'g2pm':
            # G2pM支持（未来扩展）
            print(f"警告：{g2p_model} 暂未实现，使用pypinyin替代")
            try:
                from pypinyin import lazy_pinyin, Style
                return lazy_pinyin(text, style=Style.TONE3)
            except ImportError:
                return list(text.replace(' ', ''))
                
        elif g2p_model == 'mfa':
            # MFA支持（未来扩展）
            print(f"警告：{g2p_model} 暂未实现，使用pypinyin替代")
            try:
                from pypinyin import lazy_pinyin, Style
                return lazy_pinyin(text, style=Style.TONE3)
            except ImportError:
                return list(text.replace(' ', ''))
        
        else:
            # 未知G2P模型，提示并使用备选方案
            print(f"警告：未知的G2P模型 '{g2p_model}'，使用pypinyin替代")
            try:
                from pypinyin import lazy_pinyin, Style
                return lazy_pinyin(text, style=Style.TONE3)
            except ImportError:
                return list(text.replace(' ', ''))
    
    def extract_emotion_features(self, audio_data: AudioData) -> np.ndarray:
        """
        提取情感特征
        使用官方Emotion2Vec模型提取情感表征
        """
        # 检查模型是否加载成功
        if self.emotion_pipeline is None and self.emotion_model is None:
            # 如果模型未加载，返回模拟特征
            feature_dim = self.config.get('emotion2vec', {}).get('feature_dim', 768)
            print("⚠️ 使用模拟情感特征")
            return np.random.randn(feature_dim).astype(np.float32)
        
        try:
            # 准备音频数据
            # Emotion2Vec要求16kHz采样率
            if audio_data.sample_rate != 16000:
                # 重采样到16kHz
                import librosa
                resampled_audio = librosa.resample(
                    audio_data.waveform, 
                    orig_sr=audio_data.sample_rate, 
                    target_sr=16000
                )
            else:
                resampled_audio = audio_data.waveform
            
            # 获取配置参数
            emotion_config = self.config.get('emotion2vec', {})
            granularity = emotion_config.get('granularity', 'utterance')
            extract_embedding = emotion_config.get('extract_embedding', True)
            
            # 使用ModelScope pipeline
            if self.emotion_pipeline is not None:
                # 创建临时音频文件（ModelScope需要文件路径）
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    sf.write(tmp_file.name, resampled_audio, 16000)
                    
                    result = self.emotion_pipeline(
                        tmp_file.name,
                        output_dir="./temp_outputs",
                        granularity=granularity,
                        extract_embedding=extract_embedding
                    )
                    
                    # 清理临时文件
                    os.unlink(tmp_file.name)
                
                # 提取嵌入特征
                if 'feats' in result:
                    emotion_features = result['feats']
                elif 'embedding' in result:
                    emotion_features = result['embedding']
                else:
                    # 如果格式不符预期，返回模拟特征
                    print(f"⚠️ Emotion2Vec输出格式未知: {result.keys()}")
                    return np.random.randn(768).astype(np.float32)
                
            # 使用FunASR模型
            elif self.emotion_model is not None:
                result = self.emotion_model(
                    input=resampled_audio,
                    granularity=granularity,
                    extract_embedding=extract_embedding
                )
                
                # 提取特征
                if isinstance(result, dict) and 'feats' in result:
                    emotion_features = result['feats']
                else:
                    emotion_features = result
            
            # 确保返回正确的格式
            if isinstance(emotion_features, np.ndarray):
                return emotion_features.astype(np.float32)
            elif isinstance(emotion_features, torch.Tensor):
                return emotion_features.detach().cpu().numpy().astype(np.float32)
            else:
                # 转换为numpy数组
                return np.array(emotion_features, dtype=np.float32)
            
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
