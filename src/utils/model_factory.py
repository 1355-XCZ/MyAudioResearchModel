"""
模型工厂实现
支持动态切换不同模型实现的工厂模式
"""
from typing import Dict, Any, Optional
from ..core.interfaces import (
    ModelFactory, DataProcessor, StageAModel, StageBModel, 
    Vocoder, EmotionQuantizer
)
from ..data_processing.audio_processor import WhisperEmotionProcessor
from ..models.stage_a import TTSStageAModel
from ..models.stage_b import TwoStageEmotionModel
from ..models.vocoder import VocoderFactory
from ..models.emotion_quantizer import create_emotion_quantizer


class DefaultModelFactory(ModelFactory):
    """
    默认模型工厂实现
    根据配置创建相应的模型实例
    """
    
    def create_data_processor(self, config: Dict[str, Any]) -> DataProcessor:
        """创建数据处理器"""
        processor_type = config.get('processor_type', 'whisper_emotion')
        
        if processor_type == 'whisper_emotion':
            return WhisperEmotionProcessor(config)
        else:
            raise ValueError(f"不支持的数据处理器类型: {processor_type}")
    
    def create_stage_a_model(self, config: Dict[str, Any]) -> StageAModel:
        """创建阶段A模型"""
        model_type = config.get('model_type', 'tts_model')
        
        if model_type.lower() == 'tts_model':
            return TTSStageAModel(config)
        # 保持向后兼容
        elif model_type.lower() == 'fastspeech2':
            # 如果配置文件还在使用旧的model_type，自动转换
            config_copy = config.copy()
            config_copy['tts_architecture'] = 'fastspeech2'
            return TTSStageAModel(config_copy)
        else:
            raise ValueError(f"不支持的阶段A模型类型: {model_type}")
    
    def create_stage_b_model(self, config: Dict[str, Any]) -> StageBModel:
        """创建阶段B模型"""
        model_type = config.get('model_type', 'two_stage_emotion')
        
        if model_type.lower() == 'two_stage_emotion':
            return TwoStageEmotionModel(config)
        # 可以在这里添加其他阶段B模型
        # elif model_type.lower() == 'single_stage_emotion':
        #     return SingleStageEmotionModel(config)
        else:
            raise ValueError(f"不支持的阶段B模型类型: {model_type}")
    
    def create_vocoder(self, config: Dict[str, Any]) -> Vocoder:
        """创建声码器"""
        model_type = config.get('model_type', 'hifigan')
        return VocoderFactory.create_vocoder(model_type, config)
    
    def create_emotion_quantizer(self, config: Dict[str, Any]) -> Optional[EmotionQuantizer]:
        """创建情感量化器（可选）"""
        if not config.get('enabled', False):
            return None
        
        quantizer_type = config.get('type', 'vqvae')
        return create_emotion_quantizer(quantizer_type, config)


class ExperimentalModelFactory(ModelFactory):
    """
    实验性模型工厂
    用于测试新的模型实现
    """
    
    def __init__(self, base_factory: ModelFactory):
        self.base_factory = base_factory
    
    def create_data_processor(self, config: Dict[str, Any]) -> DataProcessor:
        """创建数据处理器（可以扩展实验性处理器）"""
        processor_type = config.get('processor_type', 'whisper_emotion')
        
        # 这里可以添加实验性的数据处理器
        if processor_type == 'experimental_processor':
            # return ExperimentalProcessor(config)
            pass
        
        # 回退到基础工厂
        return self.base_factory.create_data_processor(config)
    
    def create_stage_a_model(self, config: Dict[str, Any]) -> StageAModel:
        """创建阶段A模型（可以扩展实验性模型）"""
        model_type = config.get('model_type', 'fastspeech2')
        
        # 这里可以添加实验性的阶段A模型
        if model_type.lower() == 'experimental_tts':
            # return ExperimentalTTSModel(config)
            pass
        
        return self.base_factory.create_stage_a_model(config)
    
    def create_stage_b_model(self, config: Dict[str, Any]) -> StageBModel:
        """创建阶段B模型（可以扩展实验性模型）"""
        model_type = config.get('model_type', 'two_stage_emotion')
        
        # 这里可以添加实验性的阶段B模型
        if model_type.lower() == 'experimental_emotion':
            # return ExperimentalEmotionModel(config)
            pass
        
        return self.base_factory.create_stage_b_model(config)
    
    def create_vocoder(self, config: Dict[str, Any]) -> Vocoder:
        """创建声码器"""
        return self.base_factory.create_vocoder(config)
    
    def create_emotion_quantizer(self, config: Dict[str, Any]) -> Optional[EmotionQuantizer]:
        """创建情感量化器"""
        return self.base_factory.create_emotion_quantizer(config)