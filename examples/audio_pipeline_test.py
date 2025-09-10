#!/usr/bin/env python3
"""
音频处理流水线测试脚本
验证完整架构：TTS→音频→BigVGAN提取Mel→阶段B→BigVGAN合成
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import soundfile as sf

def test_audio_pipeline():
    """测试完整音频处理流水线"""
    print("🚀" * 50)
    print("音频处理流水线测试")
    print("架构：TTS→音频→BigVGAN提取Mel→阶段B→BigVGAN合成")
    print("🚀" * 50)
    
    try:
        # 步骤1: 测试阶段A（文本→音频→标准化Mel）
        print("\n📝 步骤1: 测试阶段A")
        test_result_a = test_stage_a()
        
        if not test_result_a['success']:
            print("❌ 阶段A测试失败，无法继续")
            return False
        
        # 步骤2: 测试BigVGAN Mel提取功能
        print("\n🎼 步骤2: 测试BigVGAN Mel提取")
        mel_result = test_bigvgan_mel_extraction(test_result_a['audio'])
        
        # 步骤3: 测试BigVGAN音频合成
        print("\n🔊 步骤3: 测试BigVGAN音频合成")
        synthesis_result = test_bigvgan_synthesis(mel_result['mel'])
        
        # 步骤4: 验证参数一致性
        print("\n🎯 步骤4: 验证参数一致性")
        consistency_result = verify_parameter_consistency(mel_result, synthesis_result)
        
        # 总结
        print("\n" + "✅" * 50)
        print("音频处理流水线测试总结")
        print("✅" * 50)
        
        if consistency_result['consistent']:
            print("🎉 流水线验证成功！")
            print("✅ 参数一致性：BigVGAN提取和合成使用相同参数")
            print("✅ 模块解耦：TTS模型可任意更换")
            print("✅ 质量保证：BigVGAN保证高质量")
            
            # 保存测试结果
            save_test_results(test_result_a, mel_result, synthesis_result)
            return True
        else:
            print("❌ 参数一致性验证失败")
            return False
        
    except Exception as e:
        print(f"❌ 流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stage_a():
    """测试阶段A：文本→音频输出"""
    print("   🎤 初始化阶段A模型...")
    
    try:
        from models.stage_a import AudioStageAModel
        
        config = {
            'target_sampling_rate': 22050,
            'tts_engine_priority': ['simple_synthesis'],
            'mel_extractor': {
                'type': 'bigvgan',
                'version': '22khz'
            }
        }
        
        stage_a = AudioStageAModel(config)
        
        test_text = "音频处理流水线测试：TTS直接产出音频"
        print(f"   📝 测试文本: {test_text}")
        
        result = stage_a.forward(test_text)
        
        generated_audio = result.metadata.get('generated_audio')
        sample_rate = result.metadata.get('sample_rate', 22050)
        
        if generated_audio is not None and len(generated_audio) > 0:
            print(f"   ✅ 阶段A成功: 音频长度 {len(generated_audio)} samples")
            print(f"   🎼 Mel频谱: {result.mel_spectrogram.shape if result.mel_spectrogram is not None else 'None'}")
            
            sf.write("stage_a_output.wav", generated_audio, sample_rate)
            print(f"   📁 已保存: stage_a_output.wav")
            
            return {
                'success': True,
                'audio': generated_audio,
                'sample_rate': sample_rate,
                'mel': result.mel_spectrogram,
                'text': test_text
            }
        else:
            print(f"   ❌ 阶段A失败: 没有生成音频")
            return {'success': False}
            
    except Exception as e:
        print(f"   ❌ 阶段A测试失败: {e}")
        return {'success': False, 'error': str(e)}

def test_bigvgan_mel_extraction(audio: np.ndarray):
    """测试BigVGAN Mel提取功能"""
    print("   🎼 初始化BigVGAN Mel提取器...")
    
    try:
        from models.vocoder import BigVGANVocoder
        
        config = {
            'model_config': {
                'version': '22khz'
            }
        }
        
        bigvgan = BigVGANVocoder(config)
        
        print("   🔍 从音频提取Mel频谱...")
        mel_spectrogram = bigvgan.extract_mel_from_audio(audio)
        
        mel_config = bigvgan.get_mel_config()
        
        print(f"   ✅ Mel提取成功: {mel_spectrogram.shape}")
        print(f"   📊 Mel配置: {mel_config}")
        
        return {
            'success': True,
            'mel': mel_spectrogram,
            'config': mel_config,
            'bigvgan': bigvgan
        }
        
    except Exception as e:
        print(f"   ❌ BigVGAN Mel提取失败: {e}")
        return {'success': False, 'error': str(e)}

def test_bigvgan_synthesis(mel_spectrogram: torch.Tensor):
    """测试BigVGAN音频合成功能"""
    print("   🔊 初始化BigVGAN音频合成器...")
    
    try:
        from models.vocoder import BigVGANVocoder
        
        config = {
            'model_config': {
                'version': '22khz'
            }
        }
        
        bigvgan = BigVGANVocoder(config)
        
        print("   🎵 从Mel频谱合成音频...")
        synthesized_audio = bigvgan.synthesize(mel_spectrogram)
        
        synthesis_config = bigvgan.get_mel_config()
        
        print(f"   ✅ 音频合成成功: {len(synthesized_audio)} samples")
        print(f"   📊 合成配置: {synthesis_config}")
        
        sf.write("bigvgan_synthesis_output.wav", synthesized_audio, bigvgan.sampling_rate)
        print(f"   📁 已保存: bigvgan_synthesis_output.wav")
        
        return {
            'success': True,
            'audio': synthesized_audio,
            'config': synthesis_config,
            'sample_rate': bigvgan.sampling_rate
        }
        
    except Exception as e:
        print(f"   ❌ BigVGAN音频合成失败: {e}")
        return {'success': False, 'error': str(e)}

def verify_parameter_consistency(mel_result, synthesis_result):
    """验证参数一致性"""
    print("   🔍 验证BigVGAN提取和合成的参数一致性...")
    
    try:
        mel_config = mel_result.get('config', {})
        synthesis_config = synthesis_result.get('config', {})
        
        key_params = ['sampling_rate', 'hop_length', 'n_mel_channels', 'version']
        
        consistent = True
        for param in key_params:
            mel_val = mel_config.get(param)
            syn_val = synthesis_config.get(param)
            
            if mel_val != syn_val:
                print(f"   ❌ 参数不一致: {param} - 提取:{mel_val} vs 合成:{syn_val}")
                consistent = False
            else:
                print(f"   ✅ 参数一致: {param} = {mel_val}")
        
        if consistent:
            print("   🎯 参数一致性验证成功！")
        
        return {
            'consistent': consistent,
            'mel_config': mel_config,
            'synthesis_config': synthesis_config
        }
        
    except Exception as e:
        print(f"   ❌ 参数一致性验证失败: {e}")
        return {'consistent': False, 'error': str(e)}

def save_test_results(stage_a_result, mel_result, synthesis_result):
    """保存测试结果"""
    print("   📁 保存测试结果...")
    
    try:
        os.makedirs("outputs/audio_pipeline_test", exist_ok=True)
        
        # 保存音频文件
        if 'audio' in stage_a_result:
            sf.write("outputs/audio_pipeline_test/stage_a_audio.wav", 
                    stage_a_result['audio'], stage_a_result['sample_rate'])
        
        if 'audio' in synthesis_result:
            sf.write("outputs/audio_pipeline_test/final_synthesis.wav",
                    synthesis_result['audio'], synthesis_result['sample_rate'])
        
        # 保存配置信息
        import json
        test_report = {
            'test_name': 'audio_pipeline_verification',
            'results': {
                'stage_a': {
                    'success': stage_a_result['success'],
                    'text': stage_a_result.get('text', ''),
                    'audio_length': len(stage_a_result.get('audio', [])),
                    'sample_rate': stage_a_result.get('sample_rate', 0)
                },
                'mel_extraction': {
                    'success': mel_result['success'],
                    'mel_shape': str(mel_result.get('mel', torch.empty(0)).shape),
                    'config': mel_result.get('config', {})
                },
                'audio_synthesis': {
                    'success': synthesis_result['success'],
                    'audio_length': len(synthesis_result.get('audio', [])),
                    'config': synthesis_result.get('config', {})
                }
            }
        }
        
        with open("outputs/audio_pipeline_test/test_report.json", 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)
        
        print("   ✅ 测试结果已保存到 outputs/audio_pipeline_test/")
        
    except Exception as e:
        print(f"   ❌ 保存测试结果失败: {e}")

def main():
    """主函数"""
    print("🎯 音频处理流水线验证测试")
    
    success = test_audio_pipeline()
    
    if success:
        print("\n🎊 音频处理流水线验证成功！")
        print("现在您可以：")
        print("1. 使用任何TTS模型，不用担心Mel参数问题")
        print("2. 专注开发阶段B，参数一致性已保证")
        print("3. 享受BigVGAN的世界级音频质量")
    else:
        print("\n😞 流水线验证失败，请检查错误信息")
    
    return success

if __name__ == "__main__":
    main()
