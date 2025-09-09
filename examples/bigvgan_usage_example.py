#!/usr/bin/env python3
"""
BigVGAN声码器使用示例
展示如何使用NVIDIA BigVGAN的三个版本进行高质量语音合成
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from models.vocoder import VocoderFactory, BigVGANVocoder
from models.stage_a import TTSStageAModel

def test_bigvgan_models():
    """测试所有BigVGAN模型版本"""
    print("=" * 60)
    print("BigVGAN声码器测试")
    print("=" * 60)
    
    # 获取可用的声码器类型
    available_vocoders = VocoderFactory.get_available_vocoders()
    print(f"可用声码器类型: {available_vocoders}")
    
    # BigVGAN模型配置
    bigvgan_configs = {
        'bigvgan_22khz': {
            'description': '22kHz版本 - 与现有系统兼容',
            'model_type': 'bigvgan_22khz',
            'model_config': {'version': '22khz'}
        },
        'bigvgan_24khz': {
            'description': '24kHz版本 - 更高质量',
            'model_type': 'bigvgan_24khz', 
            'model_config': {'version': '24khz'}
        },
        'bigvgan_44khz': {
            'description': '44kHz版本 - 最高质量',
            'model_type': 'bigvgan_44khz',
            'model_config': {'version': '44khz'}
        }
    }
    
    # 创建测试用的Mel频谱
    print("\n1. 准备测试数据...")
    
    # 使用阶段A模型生成Mel频谱
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
        test_text = "欢迎使用BigVGAN高质量声码器"
        mel_result = tts_model.forward_from_text(test_text)
        test_mel = mel_result.mel_spectrogram
        print(f"✅ 生成测试Mel频谱，形状: {test_mel.shape}")
    except Exception as e:
        print(f"⚠ 阶段A模型生成失败，使用随机Mel频谱: {e}")
        # 生成随机Mel频谱用于测试
        test_mel = torch.randn(1, 80, 100)  # (batch, mel_dim, time)
    
    # 测试每个BigVGAN版本
    results = {}
    
    for model_name, config in bigvgan_configs.items():
        print(f"\n2. 测试 {model_name} - {config['description']}")
        
        try:
            # 创建声码器
            vocoder = VocoderFactory.create_vocoder(config['model_type'], config)
            
            # 获取模型信息
            if hasattr(vocoder, 'get_model_info'):
                model_info = vocoder.get_model_info()
                print(f"  模型信息:")
                print(f"    - HuggingFace模型: {model_info['model_name']}")
                print(f"    - 采样率: {model_info['sampling_rate']} Hz")
                print(f"    - 帧移: {model_info['hop_length']}")
                print(f"    - Mel维度: {model_info['n_mel_channels']}")
                print(f"    - 模型已加载: {model_info['model_loaded']}")
            
            # 调整Mel频谱维度以匹配模型要求
            current_mel = test_mel.clone()
            expected_mel_dim = vocoder.n_mel_channels
            
            if current_mel.shape[1] != expected_mel_dim:
                print(f"  调整Mel频谱维度: {current_mel.shape[1]} → {expected_mel_dim}")
                if current_mel.shape[1] < expected_mel_dim:
                    # 填充
                    padding = torch.zeros(current_mel.shape[0], 
                                        expected_mel_dim - current_mel.shape[1], 
                                        current_mel.shape[2])
                    current_mel = torch.cat([current_mel, padding], dim=1)
                else:
                    # 截断
                    current_mel = current_mel[:, :expected_mel_dim, :]
            
            # 合成音频
            print(f"  正在合成音频...")
            audio = vocoder.synthesize(current_mel)
            
            print(f"  ✅ 合成成功!")
            print(f"    - 音频长度: {len(audio)} samples")
            print(f"    - 音频时长: {len(audio) / vocoder.sampling_rate:.2f} 秒")
            print(f"    - 采样率: {vocoder.sampling_rate} Hz")
            
            # 保存音频
            try:
                import soundfile as sf
                output_path = f"bigvgan_{config['model_config']['version']}_output.wav"
                sf.write(output_path, audio, vocoder.sampling_rate)
                print(f"    - 已保存到: {output_path}")
                results[model_name] = {
                    'success': True,
                    'audio': audio,
                    'sampling_rate': vocoder.sampling_rate,
                    'output_path': output_path
                }
            except ImportError:
                print(f"    - soundfile未安装，无法保存音频文件")
                results[model_name] = {
                    'success': True,
                    'audio': audio,
                    'sampling_rate': vocoder.sampling_rate
                }
                
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            results[model_name] = {'success': False, 'error': str(e)}
    
    # 总结结果
    print(f"\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    success_count = 0
    for model_name, result in results.items():
        config = bigvgan_configs[model_name]
        if result['success']:
            success_count += 1
            print(f"✅ {model_name}: {config['description']}")
            if 'output_path' in result:
                print(f"   📁 输出文件: {result['output_path']}")
            print(f"   🎵 采样率: {result['sampling_rate']} Hz")
        else:
            print(f"❌ {model_name}: 失败 - {result['error']}")
    
    print(f"\n成功率: {success_count}/{len(bigvgan_configs)} ({success_count/len(bigvgan_configs)*100:.1f}%)")
    
    if success_count > 0:
        print(f"\n🎉 BigVGAN集成成功!")
        print("您现在可以使用以下配置切换到BigVGAN:")
        print("\n在 config/base_config.yaml 中设置:")
        for model_name, config in bigvgan_configs.items():
            if results[model_name]['success']:
                print(f"\n# {config['description']}")
                print(f"vocoder:")
                print(f"  model_type: \"{config['model_type']}\"")
    else:
        print(f"\n⚠ 所有BigVGAN模型都加载失败")
        print("请检查:")
        print("1. 网络连接是否正常")
        print("2. 是否安装了BigVGAN依赖: pip install bigvgan")
        print("3. HuggingFace Hub是否可访问")

def demonstrate_quality_comparison():
    """演示不同质量级别的BigVGAN对比"""
    print("\n" + "=" * 60)
    print("BigVGAN质量对比演示")
    print("=" * 60)
    
    # 质量级别说明
    quality_levels = {
        'bigvgan_22khz': {
            'quality': '标准',
            'compatibility': '与现有系统兼容',
            'use_case': '一般应用，兼容性优先'
        },
        'bigvgan_24khz': {
            'quality': '高',
            'compatibility': '需要调整采样率',
            'use_case': '高质量应用'
        },
        'bigvgan_44khz': {
            'quality': '最高',
            'compatibility': '需要高端硬件',
            'use_case': '专业音频制作'
        }
    }
    
    print("BigVGAN版本对比:")
    for model, info in quality_levels.items():
        print(f"\n{model}:")
        print(f"  质量等级: {info['quality']}")
        print(f"  兼容性: {info['compatibility']}")
        print(f"  适用场景: {info['use_case']}")
    
    print(f"\n选择建议:")
    print("• 开发测试阶段: 使用 bigvgan_22khz (快速，兼容)")
    print("• 一般应用: 使用 bigvgan_24khz (质量与速度平衡)")
    print("• 专业制作: 使用 bigvgan_44khz (最高质量)")

def show_usage_examples():
    """显示使用示例"""
    print("\n" + "=" * 60)
    print("BigVGAN使用示例代码")
    print("=" * 60)
    
    print("\n1. 配置文件方式 (推荐):")
    print("""
# 在 config/base_config.yaml 中设置
vocoder:
  model_type: "bigvgan_22khz"  # 或 bigvgan_24khz, bigvgan_44khz
  model_config:
    version: "22khz"
""")
    
    print("\n2. 代码中直接使用:")
    print("""
from models.vocoder import VocoderFactory

# 创建BigVGAN声码器
config = {'model_config': {'version': '22khz'}}
vocoder = VocoderFactory.create_vocoder('bigvgan_22khz', config)

# 合成音频
audio = vocoder.synthesize(mel_spectrogram)
""")
    
    print("\n3. 在流水线中使用:")
    print("""
from pipeline.training_pipeline import EmotionAudioPipeline

# 修改配置使用BigVGAN
pipeline = EmotionAudioPipeline('config/base_config.yaml')
# 配置文件中设置 model_type: "bigvgan_22khz" 即可自动使用
""")

if __name__ == "__main__":
    print("BigVGAN声码器集成测试")
    
    # 测试BigVGAN模型
    test_bigvgan_models()
    
    # 演示质量对比
    demonstrate_quality_comparison()
    
    # 显示使用示例
    show_usage_examples()
    
    print(f"\n" + "🎯" * 20)
    print("BigVGAN集成完成!")
    print("现在您可以使用世界级的NVIDIA BigVGAN声码器了!")
    print("🎯" * 20)
