"""
评估模块
包含emotion2vec分类器、码率扫描、层数扫描和结果分析
"""

try:
    from .emotion_classifier import EmotionClassifierV2
    from .method_rate_sweep import rate_sweep_evaluation
    from .method_layer_sweep import layer_sweep_evaluation
    from .analyzer import ResultAnalyzer
except ImportError as e:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    parent = Path(__file__).parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    
    from emotion_classifier import EmotionClassifierV2
    from method_rate_sweep import rate_sweep_evaluation
    from method_layer_sweep import layer_sweep_evaluation
    from analyzer import ResultAnalyzer

__all__ = [
    'EmotionClassifierV2',
    'rate_sweep_evaluation',
    'layer_sweep_evaluation',
    'ResultAnalyzer',
]

