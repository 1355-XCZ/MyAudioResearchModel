#!/usr/bin/env python3
"""
NVIDIA预训练HiFi-GAN声码器使用示例
展示如何使用NVIDIA NeMo toolkit的预训练HiFi-GAN模型
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from models.vocoder import VocoderFactory, NVIDIAHiFiGANVocoder
from models.stage_a import TTSStageAModel

def test_nvidia_hifigan():
    """测试NVIDIA预训练HiFi-GAN声码器"""
    print("=" * 60)
    print("NVIDIA预训练HiFi-GAN声码器测试")
    print("=" * 60)
    
    # 检查可用的声码器类型
    available_vocoders = VocoderFactory.get_available_vocoders()
    print(f"可用声码器类型: {available_vocoders}")
    
    if 'hifigan_nvidia' not in available_vocoders:
        print("❌ NVIDIA HiFi-GAN不可用")
        print("请安装NeMo toolkit: pip install nemo_toolkit[all]")
        return False
    
    print("✅ 发现NVIDIA HiFi-GAN声码器")
    
    # 创建阶段A模型生成测试Mel频谱
    print("\n1. 准备测试数据...")
    stage_a_config = {
        'tts_architecture': 'fastspeech2',
        'model_config': {
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 8,
            'dropout': 0.1
        },
        'phoneme_vocab': {'source': 'pypinyin_auto'}
    }
    
    try:
        tts_model = TTSStageAModel(stage_a_config)
        test_text = "NVIDIA HiFi-GAN提供高质量的语音合成"
        mel_result = tts_model.forward_from_text(test_text)
        test_mel = mel_result.mel_spectrogram
        print(f"✅ 生成测试Mel频谱，形状: {test_mel.shape}")
    except Exception as e:
        print(f"⚠ 阶段A模型生成失败，使用随机Mel频谱: {e}")
        # 生成随机Mel频谱用于测试
        test_mel = torch.randn(1, 80, 100)  # (batch, mel_dim, time)
    
    # 测试NVIDIA HiFi-GAN
    print(f"\n2. 测试NVIDIA预训练HiFi-GAN...")
    
    try:
        # 创建NVIDIA HiFi-GAN声码器
        nvidia_config = {
            'model_config': {
                'model_name': 'nvidia/tts_hifigan'
            }
        }
        
        print("   正在创建NVIDIA HiFi-GAN声码器...")
        nvidia_vocoder = VocoderFactory.create_vocoder('hifigan_nvidia', nvidia_config)
        
        # 获取模型信息
        if hasattr(nvidia_vocoder, 'get_model_info'):
            model_info = nvidia_vocoder.get_model_info()
            print(f"   ✅ 模型信息:")
            print(f"     - 模型名称: {model_info['model_name']}")
            print(f"     - 采样率: {model_info['sampling_rate']} Hz")
            print(f"     - 帧移: {model_info['hop_length']}")
            print(f"     - Mel维度: {model_info['n_mel_channels']}")
            print(f"     - 模型类型: {model_info['model_type']}")
            print(f"     - 加载状态: {model_info['model_loaded']}")
        
        # 合成音频
        print(f"   正在使用NVIDIA HiFi-GAN合成音频...")
        audio = nvidia_vocoder.synthesize(test_mel)
        
        print(f"   ✅ 音频合成成功!")
        print(f"     - 音频长度: {len(audio)} samples")
        print(f"     - 音频时长: {len(audio) / nvidia_vocoder.sampling_rate:.2f} 秒")
        print(f"     - 采样率: {nvidia_vocoder.sampling_rate} Hz")
        
        # 保存音频
        try:
            import soundfile as sf
            output_path = "nvidia_hifigan_output.wav"
            sf.write(output_path, audio, nvidia_vocoder.sampling_rate)
            print(f"     - 已保存到: {output_path}")
        except ImportError:
            print(f"     - soundfile未安装，无法保存音频文件")
        
        return True
        
    except Exception as e:
        print(f"   ❌ NVIDIA HiFi-GAN测试失败: {e}")
        print("   请检查:")
        print("   1. 网络连接是否正常")
        print("   2. NeMo toolkit是否正确安装")
        print("   3. NVIDIA模型是否可访问")
        return False

def compare_hifigan_versions():
    """对比不同版本的HiFi-GAN"""
    print(f"\n" + "=" * 60)
    print("HiFi-GAN版本对比")
    print("=" * 60)
    
    # 测试用Mel频谱
    test_mel = torch.randn(1, 80, 100)
    
    # 不同版本的HiFi-GAN配置
    hifigan_configs = {
        'hifigan_custom': {
            'description': '自定义HiFi-GAN (轻量级)',
            'config': {'model_config': {}}
        },
        'hifigan_nvidia': {
            'description': 'NVIDIA预训练HiFi-GAN (高质量)',
            'config': {'model_config': {'model_name': 'nvidia/tts_hifigan'}}
        }
    }
    
    results = {}
    available_vocoders = VocoderFactory.get_available_vocoders()
    
    for version, info in hifigan_configs.items():
        print(f"\n测试 {version} - {info['description']}:")
        
        if version not in available_vocoders:
            print(f"   ⚠ {version} 不可用")
            continue
        
        try:
            # 创建声码器
            vocoder = VocoderFactory.create_vocoder(version, info['config'])
            
            # 合成音频
            audio = vocoder.synthesize(test_mel)
            
            print(f"   ✅ 合成成功")
            print(f"     - 音频长度: {len(audio)} samples")
            print(f"     - 采样率: {vocoder.sampling_rate} Hz")
            
            # 保存对比音频
            try:
                import soundfile as sf
                output_path = f"{version}_comparison.wav"
                sf.write(output_path, audio, vocoder.sampling_rate)
                print(f"     - 对比文件: {output_path}")
            except ImportError:
                pass
            
            results[version] = {
                'success': True,
                'audio_length': len(audio),
                'sampling_rate': vocoder.sampling_rate
            }
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            results[version] = {'success': False, 'error': str(e)}
    
    # 对比总结
    print(f"\n对比总结:")
    success_count = sum(1 for r in results.values() if r['success'])
    
    for version, result in results.items():
        info = hifigan_configs[version]
        if result['success']:
            print(f"✅ {version}: {info['description']}")
        else:
            print(f"❌ {version}: 失败")
    
    print(f"\n成功率: {success_count}/{len(results)} ")
    
    if success_count > 1:
        print(f"\n💡 使用建议:")
        print(f"   - hifigan_custom: 快速开发，资源占用少")
        print(f"   - hifigan_nvidia: 生产环境，音质更好")

def show_usage_examples():
    """显示使用示例"""
    print(f"\n" + "=" * 60)
    print("NVIDIA HiFi-GAN使用示例")
    print("=" * 60)
    
    print(f"\n1. 配置文件方式 (推荐):")
    print("""
# 在 config/base_config.yaml 中设置
vocoder:
  model_type: "hifigan_nvidia"
  model_config:
    model_name: "nvidia/tts_hifigan"
""")
    
    print(f"\n2. 代码中直接使用:")
    print("""
from models.vocoder import VocoderFactory

# 创建NVIDIA HiFi-GAN声码器
config = {
    'model_config': {
        'model_name': 'nvidia/tts_hifigan'
    }
}
vocoder = VocoderFactory.create_vocoder('hifigan_nvidia', config)

# 合成音频
audio = vocoder.synthesize(mel_spectrogram)
""")
    
    print(f"\n3. 在流水线中使用:")
    print("""
# 修改配置文件后直接使用
pipeline = EmotionAudioPipeline('config/base_config.yaml')
# 配置文件中设置 model_type: "hifigan_nvidia" 即可自动使用
""")
    
    print(f"\n4. 与其他声码器对比:")
    print("""
# 对比不同声码器
vocoders = ['hifigan_custom', 'hifigan_nvidia', 'bigvgan_22khz']
for vocoder_type in vocoders:
    vocoder = VocoderFactory.create_vocoder(vocoder_type, config)
    audio = vocoder.synthesize(mel_spectrogram)
    save_audio(f'comparison_{vocoder_type}.wav', audio)
""")

def main():
    """主函数"""
    print("NVIDIA预训练HiFi-GAN声码器集成测试")
    
    # 测试NVIDIA HiFi-GAN
    success = test_nvidia_hifigan()
    
    if success:
        # 对比不同版本
        compare_hifigan_versions()
        
        # 显示使用示例
        show_usage_examples()
        
        print(f"\n" + "🎉" * 20)
        print("NVIDIA HiFi-GAN集成成功!")
        print("现在您可以选择使用:")
        print("• 自定义HiFi-GAN (轻量级)")
        print("• NVIDIA预训练HiFi-GAN (高质量)")
        print("• BigVGAN (世界级质量)")
        print("🎉" * 20)
    else:
        print(f"\n" + "⚠️" * 20)
        print("NVIDIA HiFi-GAN集成失败")
        print("但您仍然可以使用:")
        print("• 自定义HiFi-GAN (hifigan_custom)")
        print("• BigVGAN (如果已安装)")
        print("⚠️" * 20)

if __name__ == "__main__":
    main()
