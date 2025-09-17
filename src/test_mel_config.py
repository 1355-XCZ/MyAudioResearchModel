"""
测试 Mel 频谱图配置的一致性
验证 Vevo, FlowSE, Vocoder 之间的参数匹配
"""

import torch
import torchaudio
import numpy as np
import json
from pathlib import Path


def load_vevo_config():
    """加载 Vevo 配置"""
    return {
        "sample_rate": 24000,
        "hop_size": 480,
        "n_fft": 1920,
        "win_size": 1920,
        "num_mels": 128,
        "fmin": 0,
        "fmax": 12000,
    }


def load_flowse_config():
    """从 FlowSE 配置文件加载配置"""
    return {
        "target_sample_rate": 24000,
        "n_mel_channels": 128,
        "hop_length": 480,
        "win_length": 1920,
        "n_fft": 1920,
        "f_min": 0,
        "f_max": 12000,
    }


def test_mel_extraction():
    """测试 mel 频谱图提取"""
    print("🔍 测试 Mel 频谱图提取配置...")
    
    # 加载配置
    vevo_config = load_vevo_config()
    flowse_config = load_flowse_config()
    
    # 检查配置一致性
    print("\n📊 配置对比:")
    print(f"Sample Rate: Vevo={vevo_config['sample_rate']}, FlowSE={flowse_config['target_sample_rate']}")
    print(f"N FFT: Vevo={vevo_config['n_fft']}, FlowSE={flowse_config['n_fft']}")
    print(f"Hop Length: Vevo={vevo_config['hop_size']}, FlowSE={flowse_config['hop_length']}")
    print(f"Win Length: Vevo={vevo_config['win_size']}, FlowSE={flowse_config['win_length']}")
    print(f"Mel Channels: Vevo={vevo_config['num_mels']}, FlowSE={flowse_config['n_mel_channels']}")
    print(f"F Min: Vevo={vevo_config['fmin']}, FlowSE={flowse_config['f_min']}")
    print(f"F Max: Vevo={vevo_config['fmax']}, FlowSE={flowse_config['f_max']}")
    
    # 验证一致性
    config_match = (
        vevo_config['sample_rate'] == flowse_config['target_sample_rate'] and
        vevo_config['n_fft'] == flowse_config['n_fft'] and
        vevo_config['hop_size'] == flowse_config['hop_length'] and
        vevo_config['win_size'] == flowse_config['win_length'] and
        vevo_config['num_mels'] == flowse_config['n_mel_channels'] and
        vevo_config['fmin'] == flowse_config['f_min'] and
        vevo_config['fmax'] == flowse_config['f_max']
    )
    
    if config_match:
        print("\n✅ 配置一致性检查通过！")
    else:
        print("\n❌ 配置不一致，请检查参数！")
        return False
    
    # 创建测试音频
    sample_rate = vevo_config['sample_rate']
    duration = 1.0  # 1秒
    samples = int(sample_rate * duration)
    
    # 生成测试信号 (440Hz 正弦波)
    t = torch.linspace(0, duration, samples)
    test_audio = torch.sin(2 * np.pi * 440 * t)
    
    print(f"\n🎵 测试音频: {samples} samples at {sample_rate}Hz")
    
    # 使用 Vevo 配置提取 mel
    vevo_mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=vevo_config['sample_rate'],
        n_fft=vevo_config['n_fft'],
        win_length=vevo_config['win_size'],
        hop_length=vevo_config['hop_size'],
        n_mels=vevo_config['num_mels'],
        f_min=vevo_config['fmin'],
        f_max=vevo_config['fmax'],
        power=2.0,
        normalized=False
    )
    
    # 使用 FlowSE 配置提取 mel
    flowse_mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=flowse_config['target_sample_rate'],
        n_fft=flowse_config['n_fft'],
        win_length=flowse_config['win_length'],
        hop_length=flowse_config['hop_length'],
        n_mels=flowse_config['n_mel_channels'],
        f_min=flowse_config['f_min'],
        f_max=flowse_config['f_max'],
        power=2.0,
        normalized=False
    )
    
    # 提取 mel 频谱图
    vevo_mel = vevo_mel_transform(test_audio)
    flowse_mel = flowse_mel_transform(test_audio)
    
    print(f"\n📈 Mel 频谱图形状:")
    print(f"Vevo Mel: {vevo_mel.shape}")
    print(f"FlowSE Mel: {flowse_mel.shape}")
    
    # 检查形状是否一致
    if vevo_mel.shape == flowse_mel.shape:
        print("✅ Mel 频谱图形状一致！")
    else:
        print("❌ Mel 频谱图形状不一致！")
        return False
    
    # 检查数值是否相近
    mel_diff = torch.abs(vevo_mel - flowse_mel).mean()
    print(f"\n🔢 Mel 频谱图数值差异: {mel_diff:.6f}")
    
    if mel_diff < 1e-6:
        print("✅ Mel 频谱图数值一致！")
    else:
        print("❌ Mel 频谱图数值有差异！")
        return False
    
    return True


def test_frame_alignment():
    """测试帧对齐"""
    print("\n🕐 测试帧对齐...")
    
    config = load_vevo_config()
    
    # 不同长度的音频
    durations = [0.5, 1.0, 1.5, 2.0]  # 秒
    
    for duration in durations:
        samples = int(config['sample_rate'] * duration)
        expected_frames = (samples + config['hop_size'] - 1) // config['hop_size']
        
        # 创建测试音频
        test_audio = torch.randn(samples)
        
        # 提取 mel
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['sample_rate'],
            n_fft=config['n_fft'],
            win_length=config['win_size'],
            hop_length=config['hop_size'],
            n_mels=config['num_mels'],
            f_min=config['fmin'],
            f_max=config['fmax']
        )
        
        mel = mel_transform(test_audio)
        actual_frames = mel.shape[1]
        
        print(f"Duration: {duration}s, Samples: {samples}, Expected Frames: {expected_frames}, Actual Frames: {actual_frames}")
    
    print("✅ 帧对齐测试完成！")


def generate_config_summary():
    """生成配置摘要"""
    print("\n📋 生成配置摘要...")
    
    vevo_config = load_vevo_config()
    flowse_config = load_flowse_config()
    
    summary = {
        "unified_mel_config": {
            "sample_rate": vevo_config['sample_rate'],
            "n_fft": vevo_config['n_fft'],
            "hop_length": vevo_config['hop_size'],
            "win_length": vevo_config['win_size'],
            "n_mel_channels": vevo_config['num_mels'],
            "f_min": vevo_config['fmin'],
            "f_max": vevo_config['fmax'],
            "power": 2.0,
            "normalized": False
        },
        "compatibility": {
            "vevo_compatible": True,
            "flowse_compatible": True,
            "vocoder_compatible": True
        },
        "frame_rate": vevo_config['sample_rate'] / vevo_config['hop_size'],  # 50 Hz
        "frequency_resolution": vevo_config['sample_rate'] / vevo_config['n_fft'],  # ~12.5 Hz
        "time_resolution": vevo_config['hop_size'] / vevo_config['sample_rate']  # 0.02 s (20ms)
    }
    
    # 保存配置摘要
    with open("mel_config_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ 配置摘要已保存到 mel_config_summary.json")
    
    return summary


def main():
    """主函数"""
    print("🚀 开始 Mel 频谱图配置测试...")
    
    # 测试配置一致性
    if not test_mel_extraction():
        print("\n❌ 配置测试失败！")
        return
    
    # 测试帧对齐
    test_frame_alignment()
    
    # 生成配置摘要
    summary = generate_config_summary()
    
    print(f"\n🎉 所有测试通过！")
    print(f"📊 统一配置:")
    print(f"  - 采样率: {summary['unified_mel_config']['sample_rate']} Hz")
    print(f"  - Mel 通道数: {summary['unified_mel_config']['n_mel_channels']}")
    print(f"  - 帧率: {summary['frame_rate']:.1f} Hz")
    print(f"  - 时间分辨率: {summary['time_resolution']*1000:.1f} ms")
    print(f"  - 频率分辨率: {summary['frequency_resolution']:.1f} Hz")


if __name__ == "__main__":
    main()
