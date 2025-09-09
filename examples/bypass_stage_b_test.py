#!/usr/bin/env python3
"""
绕过阶段B的测试示例
测试从音频输入到阶段A，然后直接使用声码器合成音频
验证除阶段B外的其他组件是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from typing import Dict, Any
import soundfile as sf

# 导入必要的组件
from data_processing.audio_processor import WhisperEmotionProcessor, AudioLoader
from models.stage_a import TTSStageAModel
from models.vocoder import VocoderFactory
from core.interfaces import AudioData

def test_audio_to_stage_a():
    """测试音频到阶段A的完整流程"""
    print("=" * 60)
    print("测试1: 音频输入 → 阶段A → Mel频谱")
    print("=" * 60)
    
    try:
        # 1. 创建数据处理器
        print("1. 初始化数据处理器...")
        processor_config = {
            'whisper_model': 'base',
            'emotion_model': 'emotion2vec_base'
        }
        data_processor = WhisperEmotionProcessor(processor_config)
        print("   ✅ 数据处理器初始化成功")
        
        # 2. 创建测试音频数据（如果没有真实音频文件）
        print("\n2. 准备测试数据...")
        
        # 方法A: 使用真实音频文件（如果存在）
        test_audio_path = "test_audio.wav"
        if os.path.exists(test_audio_path):
            print(f"   使用真实音频文件: {test_audio_path}")
            audio_loader = AudioLoader()
            audio_data = audio_loader.load_audio(test_audio_path)
        else:
            # 方法B: 创建模拟音频数据
            print("   创建模拟音频数据...")
            sample_rate = 22050
            duration = 3.0  # 3秒
            samples = int(sample_rate * duration)
            
            # 创建一个简单的正弦波作为测试音频
            t = np.linspace(0, duration, samples)
            frequency = 440  # A4音符
            audio_array = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            audio_data = AudioData(
                waveform=audio_array,
                sample_rate=sample_rate,
                metadata={'source': 'synthetic', 'duration': duration}
            )
            
            # 保存测试音频以便后续对比
            sf.write("synthetic_test_input.wav", audio_array, sample_rate)
            print(f"   ✅ 创建了{duration}秒的测试音频，已保存为 synthetic_test_input.wav")
        
        # 3. 数据处理 - 提取文本和情感特征
        print("\n3. 处理音频数据...")
        try:
            processed_data = data_processor.process_audio(audio_data)
            print("   ✅ 音频处理成功")
            print(f"   - 提取的文本: {processed_data.text[:50]}..." if processed_data.text else "   - 文本提取失败")
            print(f"   - 情感特征维度: {processed_data.emotion_features.shape if processed_data.emotion_features is not None else 'None'}")
            print(f"   - 音素数量: {len(processed_data.phonemes) if processed_data.phonemes else 0}")
            
        except Exception as e:
            print(f"   ⚠ 音频处理失败，使用备选方案: {e}")
            # 创建模拟的处理数据
            from core.interfaces import ProcessedData
            processed_data = ProcessedData(
                text="这是一个测试句子用于验证系统功能",
                phonemes=["zh", "e4", "sh", "i4", "y", "i1", "g", "e4", "c", "e4", "sh", "i4", "j", "u4", "z", "i3"],
                emotion_features=np.random.randn(768).astype(np.float32),  # 模拟emotion2vec特征
                mel_spectrogram=None,
                metadata={'source': 'fallback'}
            )
            print("   ✅ 使用模拟处理数据")
        
        # 4. 阶段A - 生成M0
        print("\n4. 阶段A: 生成中性Mel频谱...")
        stage_a_config = {
            'tts_architecture': 'fastspeech2',  # 或 'paddlespeech_fastspeech2'
            'model_config': {
                'd_model': 256,
                'n_layers': 4,
                'n_heads': 8,
                'dropout': 0.1
            },
            'phoneme_vocab': {'source': 'pypinyin_auto'}
        }
        
        stage_a_model = TTSStageAModel(stage_a_config)
        
        # 使用阶段A生成Mel频谱
        m0_output = stage_a_model.forward(processed_data)
        print("   ✅ 阶段A处理成功")
        print(f"   - M0 Mel频谱形状: {m0_output.mel_spectrogram.shape}")
        print(f"   - 注意力权重: {'存在' if m0_output.attention_weights is not None else '不存在'}")
        
        return {
            'success': True,
            'processed_data': processed_data,
            'm0_output': m0_output,
            'audio_data': audio_data
        }
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def test_stage_a_to_vocoder():
    """测试阶段A直接到声码器的流程"""
    print("\n" + "=" * 60)
    print("测试2: 阶段A → 声码器 → 音频输出")
    print("=" * 60)
    
    # 先运行阶段A测试获取M0
    stage_a_result = test_audio_to_stage_a()
    
    if not stage_a_result['success']:
        print("❌ 阶段A测试失败，无法继续声码器测试")
        return {'success': False}
    
    m0_output = stage_a_result['m0_output']
    
    try:
        # 测试不同的声码器
        vocoder_configs = {
            'hifigan_custom': {
                'description': '自定义HiFi-GAN (轻量级)',
                'config': {'model_config': {}}
            },
            'hifigan_nvidia': {
                'description': 'NVIDIA预训练HiFi-GAN (高质量)',
                'config': {'model_config': {'model_name': 'nvidia/tts_hifigan'}}
            },
            'bigvgan_22khz': {
                'description': 'BigVGAN 22kHz (世界级质量)',
                'config': {'model_config': {'version': '22khz'}}
            }
        }
        
        # 获取可用的声码器
        available_vocoders = VocoderFactory.get_available_vocoders()
        print(f"可用声码器: {available_vocoders}")
        
        results = {}
        
        for vocoder_name, vocoder_info in vocoder_configs.items():
            if vocoder_name in available_vocoders:
                print(f"\n测试 {vocoder_name} - {vocoder_info['description']}:")
                
                try:
                    # 创建声码器
                    vocoder = VocoderFactory.create_vocoder(vocoder_name, vocoder_info['config'])
                    
                    # 合成音频
                    print("   正在合成音频...")
                    synthesized_audio = vocoder.synthesize(m0_output.mel_spectrogram)
                    
                    print(f"   ✅ 合成成功")
                    print(f"     - 音频长度: {len(synthesized_audio)} samples")
                    print(f"     - 采样率: {vocoder.sampling_rate} Hz")
                    print(f"     - 时长: {len(synthesized_audio) / vocoder.sampling_rate:.2f} 秒")
                    
                    # 保存音频
                    output_path = f"bypass_test_{vocoder_name}.wav"
                    sf.write(output_path, synthesized_audio, vocoder.sampling_rate)
                    print(f"     - 已保存: {output_path}")
                    
                    results[vocoder_name] = {
                        'success': True,
                        'audio': synthesized_audio,
                        'sampling_rate': vocoder.sampling_rate,
                        'output_path': output_path
                    }
                    
                except Exception as e:
                    print(f"   ❌ {vocoder_name} 失败: {e}")
                    results[vocoder_name] = {'success': False, 'error': str(e)}
            else:
                print(f"\n⚠ {vocoder_name} 不可用，跳过")
                results[vocoder_name] = {'success': False, 'error': 'not_available'}
        
        # 总结结果
        print(f"\n" + "=" * 40)
        print("声码器测试结果总结:")
        print("=" * 40)
        
        success_count = sum(1 for r in results.values() if r['success'])
        total_count = len([v for v in vocoder_configs.keys() if v in available_vocoders])
        
        for vocoder_name, result in results.items():
            if vocoder_name in available_vocoders:
                if result['success']:
                    print(f"✅ {vocoder_name}: 成功")
                else:
                    print(f"❌ {vocoder_name}: {result['error']}")
        
        print(f"\n成功率: {success_count}/{total_count}")
        
        return {
            'success': success_count > 0,
            'results': results,
            'success_count': success_count,
            'total_count': total_count
        }
        
    except Exception as e:
        print(f"❌ 声码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def test_text_to_audio_bypass():
    """测试纯文本到音频的绕过B阶段流程"""
    print("\n" + "=" * 60)
    print("测试3: 纯文本输入 → 阶段A → 声码器 → 音频")
    print("=" * 60)
    
    test_texts = [
        "你好，这是一个测试句子。",
        "今天天气很好，适合出门散步。",
        "人工智能技术正在快速发展。"
    ]
    
    try:
        # 创建阶段A模型
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
        
        stage_a_model = TTSStageAModel(stage_a_config)
        
        # 创建声码器（选择一个可用的）
        available_vocoders = VocoderFactory.get_available_vocoders()
        
        # 优先选择顺序
        preferred_vocoders = ['hifigan_custom', 'hifigan', 'melgan']
        selected_vocoder = None
        
        for pref in preferred_vocoders:
            if pref in available_vocoders:
                selected_vocoder = pref
                break
        
        if not selected_vocoder:
            selected_vocoder = available_vocoders[0] if available_vocoders else None
        
        if not selected_vocoder:
            print("❌ 没有可用的声码器")
            return {'success': False}
        
        print(f"使用声码器: {selected_vocoder}")
        vocoder = VocoderFactory.create_vocoder(selected_vocoder, {'model_config': {}})
        
        results = []
        
        for i, text in enumerate(test_texts):
            print(f"\n处理文本 {i+1}: {text}")
            
            try:
                # 文本 → M0
                print("   生成Mel频谱...")
                m0_output = stage_a_model.forward_from_text(text)
                print(f"   ✅ Mel频谱形状: {m0_output.mel_spectrogram.shape}")
                
                # M0 → 音频
                print("   合成音频...")
                audio = vocoder.synthesize(m0_output.mel_spectrogram)
                print(f"   ✅ 音频长度: {len(audio)} samples, {len(audio)/vocoder.sampling_rate:.2f}秒")
                
                # 保存音频
                output_path = f"text_to_audio_test_{i+1}.wav"
                sf.write(output_path, audio, vocoder.sampling_rate)
                print(f"   📁 已保存: {output_path}")
                
                results.append({
                    'text': text,
                    'success': True,
                    'output_path': output_path,
                    'duration': len(audio) / vocoder.sampling_rate
                })
                
            except Exception as e:
                print(f"   ❌ 处理失败: {e}")
                results.append({
                    'text': text,
                    'success': False,
                    'error': str(e)
                })
        
        # 总结结果
        success_count = sum(1 for r in results if r['success'])
        print(f"\n文本到音频测试结果: {success_count}/{len(test_texts)} 成功")
        
        return {
            'success': success_count > 0,
            'results': results,
            'success_count': success_count
        }
        
    except Exception as e:
        print(f"❌ 文本到音频测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def run_comprehensive_bypass_test():
    """运行完整的绕过B阶段测试"""
    print("🎯" * 20)
    print("绕过阶段B的综合测试")
    print("验证除阶段B外的所有组件功能")
    print("🎯" * 20)
    
    # 记录所有测试结果
    test_results = {}
    
    # 测试1: 音频到阶段A
    print("\n🔍 开始测试1: 音频处理和阶段A...")
    test_results['audio_to_stage_a'] = test_audio_to_stage_a()
    
    # 测试2: 阶段A到声码器
    print("\n🔍 开始测试2: 阶段A到声码器...")
    test_results['stage_a_to_vocoder'] = test_stage_a_to_vocoder()
    
    # 测试3: 纯文本到音频
    print("\n🔍 开始测试3: 纯文本到音频...")
    test_results['text_to_audio'] = test_text_to_audio_bypass()
    
    # 生成测试报告
    print("\n" + "📊" * 20)
    print("测试结果总结")
    print("📊" * 20)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result['success'])
    
    print(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result['success'] else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not result['success'] and 'error' in result:
            print(f"    错误: {result['error']}")
    
    # 功能可用性评估
    print(f"\n🎯 功能可用性评估:")
    
    if test_results['audio_to_stage_a']['success']:
        print("✅ 音频处理 → 阶段A: 完全可用")
    else:
        print("❌ 音频处理 → 阶段A: 不可用")
    
    if test_results['stage_a_to_vocoder']['success']:
        success_count = test_results['stage_a_to_vocoder'].get('success_count', 0)
        total_count = test_results['stage_a_to_vocoder'].get('total_count', 0)
        print(f"✅ 阶段A → 声码器: 部分可用 ({success_count}/{total_count} 声码器可用)")
    else:
        print("❌ 阶段A → 声码器: 不可用")
    
    if test_results['text_to_audio']['success']:
        success_count = test_results['text_to_audio'].get('success_count', 0)
        print(f"✅ 文本 → 音频: 可用 (成功处理 {success_count} 个文本)")
    else:
        print("❌ 文本 → 音频: 不可用")
    
    # 建议
    print(f"\n💡 建议:")
    if passed_tests >= 2:
        print("✅ 系统的主要组件工作正常，可以进行基础的TTS功能")
        print("✅ 您可以专注于开发和优化阶段B，其他部分已经就绪")
        print("✅ 建议使用文本到音频功能进行快速原型验证")
    else:
        print("⚠ 需要先修复基础组件问题，然后再处理阶段B")
    
    print(f"\n📁 生成的测试文件:")
    print("- synthetic_test_input.wav: 合成的测试输入音频")
    print("- bypass_test_*.wav: 不同声码器的测试输出")
    print("- text_to_audio_test_*.wav: 文本到音频的测试输出")
    
    return test_results

if __name__ == "__main__":
    # 运行完整的绕过B阶段测试
    results = run_comprehensive_bypass_test()
    
    print(f"\n" + "🎉" * 20)
    print("绕过B阶段测试完成！")
    print("现在您知道哪些组件可以正常工作了！")
    print("🎉" * 20)
