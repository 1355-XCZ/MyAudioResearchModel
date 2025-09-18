"""
配置管理模块
处理 Vevo, FlowSE, Vocoder 的配置一致性
"""

import yaml
from typing import Dict, Optional
from pathlib import Path


def get_default_vevo_config() -> Dict:
    """
    获取默认的 Vevo 配置
    基于 Vevo 的 Vocoder.json 和 Vq8192ToMels.json
    
    Returns:
        默认的 Vevo 配置字典
    """
    return {
        # 音频参数
        "sample_rate": 24000,      # Vevo 使用 24kHz
        "max_length": 36000,       # 最大音频长度 (1.5s at 24kHz)
        "min_length": 2400,        # 最小音频长度 (0.1s at 24kHz)
        
        # Mel 频谱图参数 - 与 Vevo 完全一致
        "hop_size": 480,           # Vevo 的 hop_size (20ms)
        "n_fft": 1920,            # Vevo 的 n_fft  
        "win_size": 1920,          # Vevo 的 win_size
        "num_mels": 128,           # Vevo 使用 128 mel channels
        "fmin": 0,                 # 最小频率
        "fmax": 12000,             # 最大频率 (24000/2)
        
        # 标准化参数 - Vevo 的统计值
        "mel_mean": -4.92,         # Vevo 的 mel 均值
        "mel_var": 8.14,           # Vevo 的 mel 方差
        
        # 处理参数
        "power": 2.0,              # 功率谱
        "normalized": False,       # 不使用 torchaudio 的标准化
    }


def get_bigvgan_24khz_config() -> Dict:
    """
    获取BigVGAN 24kHz配置
    用于与Vevo兼容的音频重建
    
    Returns:
        BigVGAN 24kHz配置字典
    """
    return {
        # 音频参数
        "sample_rate": 24000,      # 24kHz采样率
        "max_length": 240000,      # 最大音频长度 (10s at 24kHz)
        "min_length": 2400,        # 最小音频长度 (0.1s at 24kHz)
        
        # Mel 频谱图参数 - BigVGAN 24kHz标准
        "hop_size": 256,           # BigVGAN 24kHz hop_size
        "n_fft": 1024,            # BigVGAN 24kHz n_fft  
        "win_size": 1024,          # BigVGAN 24kHz win_size
        "num_mels": 100,           # BigVGAN 24kHz使用100 mel channels
        "fmin": 0,                 # 最小频率
        "fmax": 12000,             # 最大频率 (24000/2)
        
        # BigVGAN特定参数
        "model_name": "nvidia/bigvgan_v2_24khz_100band_256x",
        "version": "24khz",
        
        # 处理参数
        "power": 2.0,              # 功率谱
        "normalized": False,       # 不使用 torchaudio 的标准化
    }


def get_flowse_compatible_config() -> Dict:
    """
    获取与 FlowSE 兼容的配置
    
    Returns:
        FlowSE 兼容的配置字典
    """
    vevo_config = get_default_vevo_config()
    
    return {
        "target_sample_rate": vevo_config["sample_rate"],
        "n_mel_channels": vevo_config["num_mels"],
        "hop_length": vevo_config["hop_size"],
        "win_length": vevo_config["win_size"],
        "n_fft": vevo_config["n_fft"],
        "f_min": vevo_config["fmin"],
        "f_max": vevo_config["fmax"],
        "mel_spec_type": "vocos"
    }


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认配置
        
    Returns:
        配置字典
    """
    if config_path is None:
        # 返回默认配置
        return {
            "output": {
                "base_dir": "data/emilia_neutral_mels",
                "create_file_lists": True
            },
            "dataset": {
                "name": "amphion/Emilia-Dataset",
                "languages": ["EN", "ZH"],
                "streaming": True,
                "balance_languages": True,
                "target_hours_per_lang": 50,        # 每种语言50小时
                "max_samples_per_lang": 60000,      # 足够大的数量确保达到50小时
                "avg_segment_length": 3.0,          # 预期平均片段长度
                "k_neutral_variants": 1             # 默认K=1，可根据需要调整
            },
            "vevo_config": get_default_vevo_config(),
            "split": {
                "train_ratio": 0.9,
                "val_ratio": 0.1,
                "shuffle": True
            },
            "processing": {
                "batch_size": 1,
                "num_workers": 4,
                "save_metadata": True,
                "log_interval": 100
            }
        }
    
    # 加载 YAML 配置文件
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 合并默认配置
    default_config = load_config()
    
    def merge_config(default: Dict, custom: Dict) -> Dict:
        """递归合并配置"""
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    return merge_config(default_config, config)


def validate_config_consistency() -> bool:
    """
    验证 Vevo 和 FlowSE 配置的一致性
    
    Returns:
        配置是否一致
    """
    vevo_config = get_default_vevo_config()
    flowse_config = get_flowse_compatible_config()
    
    # 检查关键参数一致性
    checks = [
        vevo_config["sample_rate"] == flowse_config["target_sample_rate"],
        vevo_config["n_fft"] == flowse_config["n_fft"],
        vevo_config["hop_size"] == flowse_config["hop_length"],
        vevo_config["win_size"] == flowse_config["win_length"],
        vevo_config["num_mels"] == flowse_config["n_mel_channels"],
        vevo_config["fmin"] == flowse_config["f_min"],
        vevo_config["fmax"] == flowse_config["f_max"]
    ]
    
    return all(checks)


def save_config_template(output_path: str) -> None:
    """
    保存配置模板文件
    
    Args:
        output_path: 输出路径
    """
    template_config = load_config()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(template_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"配置模板已保存到: {output_path}")


def get_config_summary() -> Dict:
    """
    获取配置摘要信息
    
    Returns:
        配置摘要字典
    """
    vevo_config = get_default_vevo_config()
    
    return {
        "unified_mel_config": {
            "sample_rate": vevo_config["sample_rate"],
            "n_fft": vevo_config["n_fft"],
            "hop_length": vevo_config["hop_size"],
            "win_length": vevo_config["win_size"],
            "n_mel_channels": vevo_config["num_mels"],
            "f_min": vevo_config["fmin"],
            "f_max": vevo_config["fmax"],
            "power": vevo_config["power"],
            "normalized": vevo_config["normalized"]
        },
        "compatibility": {
            "vevo_compatible": True,
            "flowse_compatible": True,
            "vocoder_compatible": True
        },
        "derived_properties": {
            "frame_rate": vevo_config["sample_rate"] / vevo_config["hop_size"],  # 50 Hz
            "frequency_resolution": vevo_config["sample_rate"] / vevo_config["n_fft"],  # ~12.5 Hz
            "time_resolution": vevo_config["hop_size"] / vevo_config["sample_rate"],  # 0.02 s (20ms)
            "nyquist_frequency": vevo_config["sample_rate"] / 2,  # 12000 Hz
            "mel_frequency_range": f"{vevo_config['fmin']}-{vevo_config['fmax']} Hz"
        }
    }
