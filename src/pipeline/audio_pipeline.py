"""
音频处理流水线
核心架构：TTS→音频→BigVGAN提取Mel→阶段B→BigVGAN合成
确保参数一致性，实现模块解耦
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging

from ..core.interfaces import Pipeline, DataProcessor, StageAModel, StageBModel, Vocoder, EmotionQuantizer, AudioData, ProcessedData
from ..data_processing.audio_processor import WhisperEmotionProcessor
from ..models.stage_a import AudioStageAModel
from ..models.stage_b import TwoStageEmotionModel
from ..models.vocoder import BigVGANVocoder
from ..models.emotion_quantizer import create_emotion_quantizer

class AudioEmotionPipeline(Pipeline):
    """
    音频情感处理流水线
    
    架构特点：
    1. 阶段A直接输出音频而非Mel频谱
    2. 使用BigVGAN提取标准化Mel频谱
    3. 阶段B处理标准化Mel
    4. 最终使用相同BigVGAN合成音频
    5. 确保整个流程的参数一致性
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        print("🚀 初始化音频情感处理流水线")
        print("   架构：TTS→音频→BigVGAN提取Mel→阶段B→BigVGAN合成")
        
        # 初始化组件
        self.data_processor = self._create_data_processor()
        self.stage_a_model = self._create_stage_a_model()
        self.bigvgan_extractor = self._create_bigvgan_extractor()
        self.stage_b_model = self._create_stage_b_model()
        self.bigvgan_synthesizer = self._create_bigvgan_synthesizer()
        self.emotion_quantizer = self._create_emotion_quantizer()
        
        # 验证参数一致性
        self._verify_parameter_consistency()
        
        # 输出目录
        self.output_dir = self.config.get('experiment', {}).get('output_dir', 'outputs/audio_pipeline')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("✅ 音频情感处理流水线初始化完成")
    
    def _create_stage_a_model(self) -> StageAModel:
        """创建阶段A模型（音频输出版本）"""
        print("🎤 创建阶段A模型（音频输出）...")
        stage_a_config = self.config.get('stage_a', {})
        return AudioStageAModel(stage_a_config)
    
    def _create_bigvgan_extractor(self) -> BigVGANVocoder:
        """创建BigVGAN Mel提取器"""
        print("🎼 创建BigVGAN Mel提取器...")
        vocoder_config = self.config.get('vocoder', {})
        return BigVGANVocoder(vocoder_config)
    
    def _create_bigvgan_synthesizer(self) -> BigVGANVocoder:
        """创建BigVGAN音频合成器（与提取器相同配置）"""
        print("🔊 创建BigVGAN音频合成器...")
        vocoder_config = self.config.get('vocoder', {})
        return BigVGANVocoder(vocoder_config)
    
    def _verify_parameter_consistency(self):
        """验证参数一致性"""
        print("🔍 验证参数一致性...")
        
        try:
            extractor_config = self.bigvgan_extractor.get_mel_config()
            synthesizer_config = self.bigvgan_synthesizer.get_mel_config()
            
            key_params = ['sampling_rate', 'hop_length', 'n_mel_channels', 'version']
            
            consistent = True
            for param in key_params:
                if extractor_config.get(param) != synthesizer_config.get(param):
                    print(f"   ❌ 参数不一致: {param}")
                    consistent = False
                else:
                    print(f"   ✅ 参数一致: {param} = {extractor_config.get(param)}")
            
            if consistent:
                print("🎯 参数一致性验证成功！")
            else:
                raise RuntimeError("参数一致性验证失败")
                
        except Exception as e:
            print(f"❌ 参数一致性验证失败: {e}")
            raise
    
    def inference(self, input_data: AudioData, 
                 use_quantizer: bool = False, 
                 codebook_size: Optional[int] = None, 
                 use_b2: bool = False) -> np.ndarray:
        """
        完整推理流程（只支持音频输入）
        
        流程：
        1. 音频输入→预处理（Whisper+Emotion2Vec）
        2. 阶段A：文本→音频输出
        3. BigVGAN提取：音频→标准化Mel(M0)
        4. [可选]情感量化
        5. 阶段B：M0+情感→情感化Mel
        6. BigVGAN合成：情感化Mel→最终音频
        """
        self.logger.info("🚀 开始音频情感处理...")
        
        if not isinstance(input_data, AudioData):
            raise ValueError("只支持AudioData输入，请提供音频文件")
        
        try:
            # 步骤1: 预处理 - 从音频提取文本和情感特征
            self.logger.info("🔧 预处理阶段：音频→文本+情感特征")
            processed = self.data_processor.process_audio(input_data)
            input_text = processed.text
            emotion_features = processed.emotion_features
            self.logger.info(f"Whisper提取文本: {input_text}")
            self.logger.info(f"Emotion2Vec特征: {emotion_features.shape}")
            
            # 步骤2: 阶段A - 文本→音频输出
            self.logger.info("🎤 阶段A: 文本→音频输出")
            stage_a_output = self.stage_a_model.forward(input_text)
            
            generated_audio = stage_a_output.metadata.get('generated_audio')
            if generated_audio is None:
                raise RuntimeError("阶段A没有产出音频")
            
            self.logger.info(f"阶段A产出音频: {len(generated_audio)/22050:.2f}秒")
            
            # 步骤3: BigVGAN提取标准化Mel频谱
            self.logger.info("🎼 BigVGAN提取标准化Mel频谱")
            m0_mel = self.bigvgan_extractor.extract_mel_from_audio(generated_audio)
            self.logger.info(f"提取M0 Mel: {m0_mel.shape}")
            
            # 步骤4: [可选]情感量化
            final_emotion_features = emotion_features
            if use_quantizer and self.emotion_quantizer is not None:
                if codebook_size is None:
                    codebook_size = 256
                
                quantized_emotion, vq_loss = self.emotion_quantizer.quantize(
                    emotion_features, codebook_size
                )
                final_emotion_features = quantized_emotion
                self.logger.info(f"情感量化: 码本大小{codebook_size}, VQ损失{vq_loss:.4f}")
            
            # 步骤5: 阶段B - M0+情感→情感化Mel
            self.logger.info("🎭 阶段B: 情感处理")
            stage_b_output = self.stage_b_model.forward(
                m0_mel, final_emotion_features, use_b2=use_b2
            )
            
            emotional_mel = stage_b_output.mel_spectrogram
            stage_info = "B2" if use_b2 else "B1"
            self.logger.info(f"阶段{stage_info}产出情感化Mel: {emotional_mel.shape}")
            
            # 步骤6: BigVGAN合成最终音频
            self.logger.info("🔊 BigVGAN合成最终音频")
            final_audio = self.bigvgan_synthesizer.synthesize(emotional_mel)
            
            self.logger.info(f"最终音频合成: {len(final_audio)/22050:.2f}秒")
            self.logger.info("✅ 音频情感处理完成")
            
            return final_audio
            
        except Exception as e:
            self.logger.error(f"❌ 音频情感处理失败: {e}")
            raise
    
    
    def save_inference_results(self, audio: np.ndarray, 
                             input_text: str, 
                             output_path: str = None,
                             mode: str = "complete") -> str:
        """保存推理结果"""
        if output_path is None:
            import time
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"{mode}_inference_{timestamp}.wav")
        
        # 保存音频
        import soundfile as sf
        sf.write(output_path, audio, 22050)
        
        # 保存元数据
        metadata = {
            'input_text': input_text,
            'output_path': output_path,
            'sample_rate': 22050,
            'duration': len(audio) / 22050,
            'mode': mode,
            'architecture': 'audio_emotion_pipeline'
        }
        
        metadata_path = output_path.replace('.wav', '_metadata.json')
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存: {output_path}")
        return output_path
    
    # 实现抽象方法
    def train(self, config: Dict[str, Any]) -> None:
        """训练（主要训练阶段B和VQ-VAE）"""
        self.logger.info("开始训练...")
        pass
    
    def train_quantizer(self, emotion_dataset: List[np.ndarray]) -> None:
        """训练量化器"""
        if self.emotion_quantizer is not None:
            self.emotion_quantizer.train_codebook(emotion_dataset)
    
    def evaluate(self, test_data: List[AudioData]) -> Dict[str, float]:
        """评估模型性能"""
        return {'accuracy': 0.0}
    
    # 基础设施方法
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('AudioEmotionPipeline')
        logger.setLevel(logging.INFO)
        return logger
    
    def _create_data_processor(self) -> DataProcessor:
        """创建数据处理器"""
        return WhisperEmotionProcessor(self.config.get('data_processing', {}))
    
    def _create_stage_b_model(self) -> StageBModel:
        """创建阶段B模型"""
        return TwoStageEmotionModel(self.config.get('stage_b', {}))
    
    def _create_emotion_quantizer(self) -> Optional[EmotionQuantizer]:
        """创建情感量化器"""
        quantizer_config = self.config.get('emotion_quantizer', {})
        if quantizer_config.get('enabled', False):
            return create_emotion_quantizer(quantizer_config)
        return None
