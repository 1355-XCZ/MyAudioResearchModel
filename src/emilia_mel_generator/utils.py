"""
工具函数模块
包含配置验证、测试和辅助功能
"""

import torch
import torchaudio
import numpy as np
import json
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import logging

from .config import get_default_vevo_config, get_flowse_compatible_config

logger = logging.getLogger(__name__)


def validate_mel_config(config: Dict) -> bool:
    """
    验证 mel 频谱图配置的有效性
    
    Args:
        config: mel 配置字典
        
    Returns:
        配置是否有效
    """
    required_keys = [
        "sample_rate", "n_fft", "hop_size", "win_size", 
        "num_mels", "fmin", "fmax"
    ]
    
    # 检查必需的键
    for key in required_keys:
        if key not in config:
            logger.error(f"缺少必需的配置项: {key}")
            return False
    
    # 检查数值范围
    checks = [
        config["sample_rate"] > 0,
        config["n_fft"] > 0,
        config["hop_size"] > 0,
        config["win_size"] > 0,
        config["num_mels"] > 0,
        config["fmin"] >= 0,
        config["fmax"] > config["fmin"],
        config["fmax"] <= config["sample_rate"] / 2,  # Nyquist frequency
        config["hop_size"] <= config["n_fft"],
        config["win_size"] <= config["n_fft"]
    ]
    
    if not all(checks):
        logger.error("配置参数超出有效范围")
        return False
    
    return True


def test_mel_consistency() -> bool:
    """
    测试 Vevo 和 FlowSE 配置的一致性
    
    Returns:
        配置是否一致
    """
    print("🔍 测试 Mel 频谱图配置一致性...")
    
    vevo_config = get_default_vevo_config()
    flowse_config = get_flowse_compatible_config()
    
    # 打印配置对比
    print("\n📊 配置对比:")
    comparisons = [
        ("Sample Rate", vevo_config['sample_rate'], flowse_config['target_sample_rate']),
        ("N FFT", vevo_config['n_fft'], flowse_config['n_fft']),
        ("Hop Length", vevo_config['hop_size'], flowse_config['hop_length']),
        ("Win Length", vevo_config['win_size'], flowse_config['win_length']),
        ("Mel Channels", vevo_config['num_mels'], flowse_config['n_mel_channels']),
        ("F Min", vevo_config['fmin'], flowse_config['f_min']),
        ("F Max", vevo_config['fmax'], flowse_config['f_max'])
    ]
    
    all_match = True
    for name, vevo_val, flowse_val in comparisons:
        match = "✅" if vevo_val == flowse_val else "❌"
        print(f"{match} {name}: Vevo={vevo_val}, FlowSE={flowse_val}")
        if vevo_val != flowse_val:
            all_match = False
    
    if all_match:
        print("\n✅ 所有配置参数一致！")
    else:
        print("\n❌ 存在配置不一致，请检查参数！")
    
    return all_match


def create_test_mel_spectrograms(config: Dict, duration: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建测试用的 mel 频谱图
    
    Args:
        config: mel 配置
        duration: 测试音频时长（秒）
        
    Returns:
        (测试音频, mel 频谱图)
    """
    sample_rate = config["sample_rate"]
    samples = int(sample_rate * duration)
    
    # 生成测试信号 (440Hz 正弦波 + 白噪声)
    t = torch.linspace(0, duration, samples)
    test_audio = (
        0.8 * torch.sin(2 * np.pi * 440 * t) +  # 主音调
        0.2 * torch.randn(samples)               # 白噪声
    )
    
    # 创建 mel 变换器
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config["sample_rate"],
        n_fft=config["n_fft"],
        win_length=config["win_size"],
        hop_length=config["hop_size"],
        n_mels=config["num_mels"],
        f_min=config["fmin"],
        f_max=config["fmax"],
        power=config.get("power", 2.0),
        normalized=config.get("normalized", False)
    )
    
    # 提取 mel 频谱图
    mel = mel_transform(test_audio)
    
    return test_audio, mel


def analyze_mel_properties(mel: torch.Tensor, config: Dict) -> Dict:
    """
    分析 mel 频谱图的属性
    
    Args:
        mel: mel 频谱图 [n_mels, frames]
        config: mel 配置
        
    Returns:
        分析结果字典
    """
    n_mels, n_frames = mel.shape
    duration = n_frames * config["hop_size"] / config["sample_rate"]
    
    return {
        "shape": list(mel.shape),
        "n_mels": n_mels,
        "n_frames": n_frames,
        "duration": duration,
        "frame_rate": config["sample_rate"] / config["hop_size"],
        "time_resolution": config["hop_size"] / config["sample_rate"],
        "frequency_resolution": config["sample_rate"] / config["n_fft"],
        "mel_statistics": {
            "min": float(mel.min()),
            "max": float(mel.max()),
            "mean": float(mel.mean()),
            "std": float(mel.std())
        }
    }


def save_mel_analysis(output_dir: str, analysis_results: List[Dict]) -> None:
    """
    保存 mel 分析结果
    
    Args:
        output_dir: 输出目录
        analysis_results: 分析结果列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存详细分析结果
    with open(output_path / "mel_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # 创建摘要报告
    if analysis_results:
        summary = {
            "total_samples": len(analysis_results),
            "average_duration": np.mean([r["duration"] for r in analysis_results]),
            "average_frames": np.mean([r["n_frames"] for r in analysis_results]),
            "mel_shape_consistency": len(set(tuple(r["shape"]) for r in analysis_results)) == 1,
            "configuration": analysis_results[0] if analysis_results else {}
        }
        
        with open(output_path / "mel_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"分析结果已保存到: {output_path}")


def check_audio_quality(audio: torch.Tensor, sample_rate: int) -> Dict:
    """
    检查音频质量
    
    Args:
        audio: 音频张量
        sample_rate: 采样率
        
    Returns:
        质量检查结果
    """
    # 基本统计
    duration = len(audio) / sample_rate
    rms = torch.sqrt(torch.mean(audio ** 2))
    peak = torch.max(torch.abs(audio))
    
    # 检查是否有裁剪
    clipping_threshold = 0.99
    is_clipped = peak > clipping_threshold
    
    # 检查是否过于安静
    silence_threshold = 1e-4
    is_silent = rms < silence_threshold
    
    # 检查动态范围
    dynamic_range = 20 * torch.log10(peak / (rms + 1e-8))
    
    return {
        "duration": float(duration),
        "rms": float(rms),
        "peak": float(peak),
        "dynamic_range_db": float(dynamic_range),
        "is_clipped": bool(is_clipped),
        "is_silent": bool(is_silent),
        "quality_score": float(1.0 - is_clipped - is_silent)  # 简单质量评分
    }


def create_file_structure_summary(base_dir: str) -> Dict:
    """
    创建文件结构摘要
    
    Args:
        base_dir: 基础目录
        
    Returns:
        文件结构摘要
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return {"error": f"目录不存在: {base_dir}"}
    
    summary = {
        "base_directory": str(base_path),
        "languages": {},
        "total_files": 0,
        "total_size_mb": 0
    }
    
    for lang_dir in base_path.iterdir():
        if lang_dir.is_dir() and lang_dir.name in ["EN", "ZH"]:
            mel_files = list(lang_dir.glob("*.npy"))
            json_files = list(lang_dir.glob("*.json"))
            
            total_size = sum(f.stat().st_size for f in mel_files + json_files)
            
            summary["languages"][lang_dir.name] = {
                "mel_files": len(mel_files),
                "metadata_files": len(json_files),
                "total_size_mb": total_size / (1024 * 1024)
            }
            
            summary["total_files"] += len(mel_files) + len(json_files)
            summary["total_size_mb"] += total_size / (1024 * 1024)
    
    return summary


def run_full_validation(config: Dict) -> bool:
    """
    运行完整的验证流程
    
    Args:
        config: 配置字典
        
    Returns:
        验证是否通过
    """
    print("🚀 开始完整验证流程...")
    
    # 1. 验证配置
    if not validate_mel_config(config["vevo_config"]):
        print("❌ 配置验证失败")
        return False
    print("✅ 配置验证通过")
    
    # 2. 测试一致性
    if not test_mel_consistency():
        print("❌ 一致性测试失败")
        return False
    print("✅ 一致性测试通过")
    
    # 3. 创建测试频谱图
    test_audio, test_mel = create_test_mel_spectrograms(config["vevo_config"])
    print(f"✅ 测试频谱图创建成功: {test_mel.shape}")
    
    # 4. 分析频谱图属性
    analysis = analyze_mel_properties(test_mel, config["vevo_config"])
    print(f"✅ 频谱图分析完成: {analysis['duration']:.2f}s, {analysis['n_frames']} frames")
    
    # 5. 检查音频质量
    quality = check_audio_quality(test_audio, config["vevo_config"]["sample_rate"])
    print(f"✅ 音频质量检查: 质量评分 {quality['quality_score']:.2f}")
    
    print("🎉 所有验证通过！")
    return True
