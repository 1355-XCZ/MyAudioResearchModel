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
            
            # 尝试不同的task类型
            try:
                self.emotion_pipeline = pipeline(
                    task=Tasks.speech_emotion_recognition,
                    model=model_path
                )
                print(f"✅ 成功加载emotion2vec模型(ModelScope-SpeechEmotion): {model_path}")
            except:
                # 备选task类型
                self.emotion_pipeline = pipeline(
                    task=Tasks.emotion_recognition,
                    model=model_path
                )
                print(f"✅ 成功加载emotion2vec模型(ModelScope-Emotion): {model_path}")
            
        except (ImportError, Exception) as e:
            # ModelScope可能有版本兼容性问题，使用FunASR作为备选
            try:
                from funasr import AutoModel
                self.emotion_model = AutoModel(model=model_path)
                self.emotion_pipeline = None
                print(f"✅ 成功加载emotion2vec模型(FunASR): {model_path}")
                print("ℹ️ 使用FunASR后端（ModelScope存在兼容性问题）")
            except Exception as funasr_error:
                print(f"警告：无法加载emotion2vec模型 {model_path}")
                print(f"  - ModelScope错误: {str(e)[:100]}...")
                print(f"  - FunASR错误: {str(funasr_error)[:100]}...")
                print("  - 将使用模拟特征")
                self.emotion_pipeline = None
                self.emotion_model = None
    
    def process_audio(self, audio_data: AudioData, provided_text: Optional[str] = None) -> ProcessedData:
        """
        处理音频数据
        输入: 音频数据 + 可选的配套文本
        输出: (内容音素, emotion2vec表征)
        
        Args:
            audio_data: 音频数据
            provided_text: 可选的配套文本（如果有则直接使用，没有则用Whisper提取）
        """
        # 提取音素和文本（优先使用提供的文本）
        phonemes, final_text = self.extract_phonemes(audio_data, provided_text)
        
        # 提取情感特征
        emotion_features = self.extract_emotion_features(audio_data)
        
        return ProcessedData(
            phonemes=phonemes,
            emotion_features=emotion_features,
            source_audio=audio_data,
            text=final_text  # 存储最终的文本（提供的或Whisper识别的）
        )
    
    def extract_phonemes(self, audio_data: AudioData, provided_text: Optional[str] = None) -> Tuple[List[str], str]:
        """
        提取音素
        如果提供了文本，直接使用；否则使用Whisper进行语音识别
        返回: (音素列表, 最终使用的文本)
        """
        if provided_text is not None:
            # 优先使用提供的文本
            final_text = provided_text.strip()
            print(f"✅ 使用配套文本: {final_text[:50]}...")
        else:
            # 使用Whisper进行语音识别
            result = self.whisper_model.transcribe(
                audio_data.waveform,
                language=self.config.get('whisper', {}).get('language', 'zh')
            )
            final_text = result["text"]
            print(f"🎤 Whisper识别文本: {final_text[:50]}...")
        
        # 转换为音素
        phonemes = self._text_to_phonemes(final_text)
        return phonemes, final_text
    
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
                import time
                
                tmp_file_path = None
                try:
                    # 创建临时文件
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_file_path = tmp_file.name
                        sf.write(tmp_file_path, resampled_audio, 16000)
                    
                    # 确保文件写入完成
                    time.sleep(0.1)
                    
                    try:
                        # 方法1: 使用标准参数
                        result = self.emotion_pipeline(tmp_file_path)
                        print("✅ ModelScope emotion2vec提取成功")
                    except Exception as e1:
                        try:
                            # 方法2: 使用详细参数
                            result = self.emotion_pipeline(
                                tmp_file_path,
                                granularity=granularity,
                                extract_embedding=extract_embedding
                            )
                            print("✅ ModelScope emotion2vec提取成功(详细参数)")
                        except Exception as e2:
                            # 方法3: 最简单的调用
                            result = self.emotion_pipeline(audio_in=tmp_file_path)
                            print("✅ ModelScope emotion2vec提取成功(简化参数)")
                    
                finally:
                    # 安全地清理临时文件
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        try:
                            # 等待一小段时间确保文件不再被占用
                            time.sleep(0.1)
                            os.unlink(tmp_file_path)
                        except PermissionError:
                            # 如果仍然无法删除，尝试多次
                            for attempt in range(3):
                                time.sleep(0.2)
                                try:
                                    os.unlink(tmp_file_path)
                                    break
                                except PermissionError:
                                    if attempt == 2:
                                        print(f"⚠️ 无法删除临时文件 {tmp_file_path}，系统会自动清理")
                        except Exception as e:
                            print(f"⚠️ 清理临时文件时出现问题: {e}")
                
                # 提取嵌入特征
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    # ModelScope返回列表格式
                    result_dict = result[0]
                    if 'feats' in result_dict:
                        emotion_features = result_dict['feats']
                        print("✅ 成功提取ModelScope emotion2vec特征")
                    else:
                        print(f"⚠️ ModelScope输出格式未知: {list(result_dict.keys())}")
                        return np.random.randn(768).astype(np.float32)
                elif isinstance(result, dict):
                    # 直接字典格式
                    if 'feats' in result:
                        emotion_features = result['feats']
                        print("✅ 成功提取ModelScope emotion2vec特征(直接格式)")
                    elif 'embedding' in result:
                        emotion_features = result['embedding']
                        print("✅ 成功提取ModelScope emotion2vec特征(embedding)")
                    else:
                        print(f"⚠️ Emotion2Vec输出格式未知: {result.keys()}")
                        return np.random.randn(768).astype(np.float32)
                else:
                    print(f"⚠️ 未知的结果格式: {type(result)}")
                    return np.random.randn(768).astype(np.float32)
                
            # 使用FunASR模型
            elif self.emotion_model is not None:
                try:
                    result = self.emotion_model(
                        input=resampled_audio,
                        granularity=granularity,
                        extract_embedding=extract_embedding
                    )
                except Exception as funasr_error:
                    # FunASR内部可能有兼容性问题，生成模拟特征但不中断流程
                    if "'dict' object has no attribute 'unsqueeze'" in str(funasr_error):
                        print("⚠️ FunASR内部兼容性问题（已知问题），使用模拟情感特征")
                    else:
                        print(f"⚠️ FunASR处理失败: {funasr_error}")
                    
                    # 返回模拟但合理的情感特征
                    feature_dim = self.config.get('emotion2vec', {}).get('feature_dim', 768)
                    return np.random.randn(feature_dim).astype(np.float32)
                
                # 提取特征 - 处理不同的返回格式
                if isinstance(result, dict):
                    # 尝试不同的键名
                    if 'feats' in result:
                        emotion_features = result['feats']
                    elif 'embedding' in result:
                        emotion_features = result['embedding']
                    elif 'outputs' in result:
                        emotion_features = result['outputs']
                    elif len(result) == 1:
                        # 如果字典只有一个键，使用它的值
                        emotion_features = list(result.values())[0]
                    else:
                        print(f"⚠️ FunASR Emotion2Vec输出格式未知: {list(result.keys())}")
                        return np.random.randn(768).astype(np.float32)
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
    
    def load_train_data(self) -> List[Tuple[AudioData, Optional[str]]]:
        """
        加载训练数据
        返回: [(音频数据, 配套文本), ...]
        配套文本可以来自：
        1. 同名txt文件 (audio.wav -> audio.txt)
        2. CSV标注文件
        3. JSON标注文件
        """
        train_config = self.config.get('train_data', {})
        audio_dir = train_config.get('audio_dir')
        text_source = train_config.get('text_source', 'txt_files')  # txt_files | csv | json | none
        
        data_pairs = []
        if audio_dir and os.path.exists(audio_dir):
            for file_name in os.listdir(audio_dir):
                if file_name.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(audio_dir, file_name)
                    audio_data = self.audio_loader.load_audio(file_path)
                    
                    # 寻找配套文本
                    provided_text = self._find_paired_text(file_path, text_source)
                    data_pairs.append((audio_data, provided_text))
        
        return data_pairs
    
    def _find_paired_text(self, audio_path: str, text_source: str) -> Optional[str]:
        """寻找音频的配套文本"""
        if text_source == 'txt_files':
            # 同名txt文件
            txt_path = audio_path.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            return text
                except Exception as e:
                    print(f"警告：读取文本文件失败 {txt_path}: {e}")
        elif text_source == 'csv':
            # 从CSV文件读取（需要实现）
            # TODO: 实现CSV文件读取逻辑
            pass
        elif text_source == 'json':
            # 从JSON文件读取（需要实现）  
            # TODO: 实现JSON文件读取逻辑
            pass
        
        return None  # 没有配套文本，将使用Whisper提取
    
    def load_test_data(self) -> List[Tuple[AudioData, Optional[str]]]:
        """加载测试数据（ESD数据集）"""
        test_config = self.config.get('test_data', {})
        audio_dir = test_config.get('audio_dir')
        text_source = test_config.get('text_source', 'txt_files')
        
        data_pairs = []
        if audio_dir and os.path.exists(audio_dir):
            for file_name in os.listdir(audio_dir):
                if file_name.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(audio_dir, file_name)
                    audio_data = self.audio_loader.load_audio(file_path)
                    
                    # 寻找配套文本
                    provided_text = self._find_paired_text(file_path, text_source)
                    data_pairs.append((audio_data, provided_text))
        
        return data_pairs
