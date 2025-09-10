#!/usr/bin/env python3
"""
音频到音频的完整流程演示
音频输入 → 文字提取 → Mel频谱生成 → 音频输出
绕过阶段B，展示基础的语音重建能力
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import soundfile as sf
from typing import Optional

def audio_to_audio_pipeline(input_audio_path: str, output_audio_path: str = "reconstructed_audio.wav"):
    """
    完整的音频到音频流水线
    
    Args:
        input_audio_path: 输入音频文件路径
        output_audio_path: 输出音频文件路径
    
    Returns:
        bool: 是否成功完成流程
    """
    print("🎵" * 50)
    print("音频到音频重建流水线")
    print(f"输入: {input_audio_path}")
    print(f"输出: {output_audio_path}")
    print("🎵" * 50)
    
    try:
        # 步骤1: 加载音频文件
        print("\n📂 步骤1: 加载音频文件...")
        
        from data_processing.audio_processor import AudioLoader, WhisperEmotionProcessor
        from core.interfaces import AudioData
        
        # 加载音频
        audio_loader = AudioLoader()
        
        if os.path.exists(input_audio_path):
            audio_data = audio_loader.load_audio(input_audio_path)
            print(f"   ✅ 成功加载音频文件")
            print(f"   - 采样率: {audio_data.sample_rate} Hz")
            print(f"   - 时长: {len(audio_data.waveform) / audio_data.sample_rate:.2f} 秒")
            print(f"   - 音频形状: {audio_data.waveform.shape}")
        else:
            print(f"   ❌ 音频文件不存在: {input_audio_path}")
            return False
        
        # 步骤2: 提取文字和情感特征
        print("\n🎤 步骤2: 提取文字内容...")
        
        processor_config = {
            'whisper_model': 'base',
            'emotion_model': 'emotion2vec_base'
        }
        
        try:
            data_processor = WhisperEmotionProcessor(processor_config)
            processed_data = data_processor.process_audio(audio_data)
            
            print(f"   ✅ 文字提取成功")
            print(f"   - 提取的文字: {processed_data.text}")
            print(f"   - 音素数量: {len(processed_data.phonemes) if processed_data.phonemes else 0}")
            print(f"   - 情感特征维度: {processed_data.emotion_features.shape if processed_data.emotion_features is not None else 'None'}")
            
        except Exception as e:
            print(f"   ⚠ 自动文字提取失败: {e}")
            print(f"   使用备选方案: 手动输入文字")
            
            # 备选方案：手动输入或使用默认文字
            manual_text = input("请输入这段音频的文字内容（回车使用默认）: ").strip()
            if not manual_text:
                manual_text = "这是一段测试音频，用于验证音频重建功能"
            
            from core.interfaces import ProcessedData
            processed_data = ProcessedData(
                text=manual_text,
                phonemes=None,  # 将由阶段A自动处理
                emotion_features=np.random.randn(768).astype(np.float32),  # 模拟情感特征
                mel_spectrogram=None,
                metadata={'source': 'manual_input'}
            )
            print(f"   ✅ 使用文字: {manual_text}")
        
        # 步骤3: 生成Mel频谱 (阶段A)
        print("\n📊 步骤3: 生成Mel频谱...")
        
        from models.stage_a import TTSStageAModel
        
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
        
        # 如果有文字，直接从文字生成；否则使用processed_data
        if processed_data.text:
            m0_output = stage_a_model.forward_from_text(processed_data.text)
        else:
            m0_output = stage_a_model.forward(processed_data)
        
        print(f"   ✅ Mel频谱生成成功")
        print(f"   - M0频谱形状: {m0_output.mel_spectrogram.shape}")
        print(f"   - 频谱时长: {m0_output.mel_spectrogram.shape[-1]} 帧")
        
        # 步骤4: 音频合成 (声码器)
        print("\n🔊 步骤4: 合成音频...")
        
        from models.vocoder import VocoderFactory
        
        # 获取可用声码器并选择最佳的
        available_vocoders = VocoderFactory.get_available_vocoders()
        print(f"   可用声码器: {available_vocoders}")
        
        # 优先选择顺序
        preferred_vocoders = ['hifigan_nvidia', 'bigvgan_22khz', 'hifigan_custom', 'hifigan', 'melgan']
        selected_vocoder = None
        
        for pref in preferred_vocoders:
            if pref in available_vocoders:
                selected_vocoder = pref
                break
        
        if not selected_vocoder:
            selected_vocoder = available_vocoders[0] if available_vocoders else None
        
        if not selected_vocoder:
            print("   ❌ 没有可用的声码器")
            return False
        
        print(f"   使用声码器: {selected_vocoder}")
        
        # 创建声码器
        if selected_vocoder == 'hifigan_nvidia':
            vocoder_config = {'model_config': {'model_name': 'nvidia/tts_hifigan'}}
        elif 'bigvgan' in selected_vocoder:
            version = selected_vocoder.split('_')[1] if '_' in selected_vocoder else '22khz'
            vocoder_config = {'model_config': {'version': version}}
        else:
            vocoder_config = {'model_config': {}}
        
        vocoder = VocoderFactory.create_vocoder(selected_vocoder, vocoder_config)
        
        # 合成音频
        synthesized_audio = vocoder.synthesize(m0_output.mel_spectrogram)
        
        print(f"   ✅ 音频合成成功")
        print(f"   - 输出音频长度: {len(synthesized_audio)} samples")
        print(f"   - 输出采样率: {vocoder.sampling_rate} Hz")
        print(f"   - 输出时长: {len(synthesized_audio) / vocoder.sampling_rate:.2f} 秒")
        
        # 步骤5: 保存音频文件
        print("\n💾 步骤5: 保存音频文件...")
        
        sf.write(output_audio_path, synthesized_audio, vocoder.sampling_rate)
        print(f"   ✅ 音频已保存: {output_audio_path}")
        
        # 流程总结
        print("\n" + "🎉" * 50)
        print("音频重建流程完成！")
        print("🎉" * 50)
        
        print(f"\n📊 流程总结:")
        print(f"   原始音频: {input_audio_path}")
        print(f"   提取文字: {processed_data.text[:50]}..." if processed_data.text else "   提取文字: 无")
        print(f"   生成频谱: {m0_output.mel_spectrogram.shape}")
        print(f"   使用声码器: {selected_vocoder}")
        print(f"   输出音频: {output_audio_path}")
        
        print(f"\n💡 说明:")
        print(f"   - 这是基础的语音重建，跳过了情感处理")
        print(f"   - 输出音频是标准音色，不保留原说话人特征")
        print(f"   - 文字内容保持一致，但韵律会标准化")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 流程执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_audio_processing(input_dir: str, output_dir: str):
    """批量处理音频文件"""
    print(f"\n🔄 批量音频处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的音频格式
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    
    # 查找音频文件
    audio_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(file)
    
    if not audio_files:
        print(f"❌ 在 {input_dir} 中没有找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    success_count = 0
    for i, audio_file in enumerate(audio_files):
        print(f"\n处理 {i+1}/{len(audio_files)}: {audio_file}")
        
        input_path = os.path.join(input_dir, audio_file)
        output_file = f"reconstructed_{os.path.splitext(audio_file)[0]}.wav"
        output_path = os.path.join(output_dir, output_file)
        
        if audio_to_audio_pipeline(input_path, output_path):
            success_count += 1
            print(f"   ✅ 成功处理: {output_file}")
        else:
            print(f"   ❌ 处理失败: {audio_file}")
    
    print(f"\n批量处理完成: {success_count}/{len(audio_files)} 成功")

def create_demo_audio():
    """创建演示用的音频文件"""
    print("🎵 创建演示音频...")
    
    # 创建一个简单的测试音频
    sample_rate = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 创建一个包含多个频率的复合音调
    frequencies = [440, 523, 659, 784]  # A4, C5, E5, G5
    audio = np.zeros_like(t)
    
    for i, freq in enumerate(frequencies):
        start_time = i * duration / len(frequencies)
        end_time = (i + 1) * duration / len(frequencies)
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        audio[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx]) * 0.5
    
    # 保存演示音频
    demo_path = "demo_input_audio.wav"
    sf.write(demo_path, audio, sample_rate)
    print(f"✅ 演示音频已创建: {demo_path}")
    
    return demo_path

def main():
    """主函数"""
    print("🎯" * 60)
    print("音频到音频重建演示")
    print("完整流程: 音频 → 文字 → Mel频谱 → 音频")
    print("🎯" * 60)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        input_audio = sys.argv[1]
        output_audio = sys.argv[2] if len(sys.argv) > 2 else "reconstructed_audio.wav"
    else:
        print("\n选择测试模式:")
        print("1. 使用现有音频文件")
        print("2. 创建演示音频文件")
        print("3. 批量处理音频文件")
        
        choice = input("请选择 (1/2/3): ").strip()
        
        if choice == "1":
            input_audio = input("请输入音频文件路径: ").strip()
            output_audio = input("请输入输出文件路径 (回车使用默认): ").strip()
            if not output_audio:
                output_audio = "reconstructed_audio.wav"
                
        elif choice == "2":
            input_audio = create_demo_audio()
            output_audio = "reconstructed_demo_audio.wav"
            
        elif choice == "3":
            input_dir = input("请输入输入目录路径: ").strip()
            output_dir = input("请输入输出目录路径: ").strip()
            batch_audio_processing(input_dir, output_dir)
            return
        else:
            print("无效选择")
            return
    
    # 运行音频到音频流水线
    success = audio_to_audio_pipeline(input_audio, output_audio)
    
    if success:
        print(f"\n🎊 成功完成音频重建！")
        print(f"📁 原始音频: {input_audio}")
        print(f"📁 重建音频: {output_audio}")
        print(f"🎵 现在可以播放对比两个音频文件了！")
    else:
        print(f"\n😞 音频重建失败，请检查错误信息")

if __name__ == "__main__":
    main()
