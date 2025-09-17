"""
Emilia Mel Generator Package
增强数据生成工具 - 基于用户需求的S1/S2增强策略
"""

from .corrected_data_generator import CorrectedDataGenerator, VevoTTSEmulator, Emotion2VecExtractor
from .config import load_config, get_default_vevo_config
from .utils import validate_mel_config, test_mel_consistency

__version__ = "2.0.0"
__all__ = [
    "CorrectedDataGenerator",
    "VevoTTSEmulator", 
    "Emotion2VecExtractor",
    "load_config",
    "get_default_vevo_config",
    "validate_mel_config",
    "test_mel_consistency"
]
