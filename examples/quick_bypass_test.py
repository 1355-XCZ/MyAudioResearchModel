#!/usr/bin/env python3
"""
快速绕过B阶段测试
最小化测试，快速验证核心功能是否正常
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np

def quick_stage_a_test():
    """快速测试阶段A功能"""
    print("🚀 快速阶段A测试")
    
    try:
        from models.stage_a import TTSStageAModel
        
        # 最简配置
        config = {
            'tts_architecture': 'fastspeech2',
            'model_config': {'d_model': 128, 'n_layers': 2, 'n_heads': 4, 'dropout': 0.1},
            'phoneme_vocab': {'source': 'pypinyin_auto'}
        }
        
        print("  创建阶段A模型...")
        model = TTSStageAModel(config)
        print("  ✅ 模型创建成功")
        
        # 测试文本输入
        test_text = "你好世界"
        print(f"  测试文本: {test_text}")
        
        result = model.forward_from_text(test_text)
        print(f"  ✅ 生成Mel频谱: {result.mel_spectrogram.shape}")
        
        return True, result.mel_spectrogram
        
    except Exception as e:
        print(f"  ❌ 阶段A测试失败: {e}")
        return False, None

def quick_vocoder_test(mel_spectrogram):
    """快速测试声码器功能"""
    print("\n🔊 快速声码器测试")
    
    try:
        from models.vocoder import VocoderFactory
        
        # 获取可用声码器
        available = VocoderFactory.get_available_vocoders()
        print(f"  可用声码器: {available}")
        
        # 选择第一个可用的声码器
        if not available:
            print("  ❌ 没有可用的声码器")
            return False
        
        vocoder_type = available[0]
        print(f"  使用声码器: {vocoder_type}")
        
        # 创建声码器
        vocoder = VocoderFactory.create_vocoder(vocoder_type, {'model_config': {}})
        print("  ✅ 声码器创建成功")
        
        # 合成音频
        audio = vocoder.synthesize(mel_spectrogram)
        print(f"  ✅ 音频合成成功: {len(audio)} samples")
        
        # 保存音频
        try:
            import soundfile as sf
            sf.write("quick_test_output.wav", audio, vocoder.sampling_rate)
            print("  📁 已保存: quick_test_output.wav")
        except ImportError:
            print("  ⚠ soundfile未安装，无法保存音频")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 声码器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("⚡" * 30)
    print("快速绕过B阶段测试")
    print("验证核心TTS功能是否正常")
    print("⚡" * 30)
    
    # 测试阶段A
    stage_a_success, mel_output = quick_stage_a_test()
    
    if not stage_a_success:
        print("\n❌ 阶段A测试失败，无法继续")
        return
    
    # 测试声码器
    vocoder_success = quick_vocoder_test(mel_output)
    
    # 总结
    print("\n" + "📊" * 20)
    print("测试结果总结")
    print("📊" * 20)
    
    if stage_a_success and vocoder_success:
        print("🎉 所有核心组件工作正常！")
        print("✅ 文本 → Mel频谱 → 音频 流程完整")
        print("💡 您可以专注于开发阶段B，其他部分已就绪")
        print("📁 生成了 quick_test_output.wav 供测试")
    elif stage_a_success:
        print("⚠ 阶段A工作正常，但声码器有问题")
        print("💡 建议检查声码器依赖和配置")
    else:
        print("❌ 阶段A有问题，需要先修复基础组件")
    
    print("\n🎯 可用功能:")
    if stage_a_success:
        print("✅ 中文文本转Mel频谱")
        print("✅ 音素处理和TTS模型")
    if vocoder_success:
        print("✅ Mel频谱转音频")
        print("✅ 多种声码器选择")
    
    if stage_a_success and vocoder_success:
        print("\n🚀 建议下一步:")
        print("1. 运行 python examples/bypass_stage_b_test.py 进行完整测试")
        print("2. 开始开发和训练阶段B模型")
        print("3. 实现VQ-VAE情感量化器的训练逻辑")

if __name__ == "__main__":
    main()
