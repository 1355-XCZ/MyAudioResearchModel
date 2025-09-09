"""
基础使用示例
展示如何使用更新后的情感音频建模系统进行训练和推理
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.training_pipeline import EmotionAudioPipeline
from src.data_processing.audio_processor import AudioLoader


def basic_inference_example():
    """基础推理示例"""
    print("=== 基础推理示例 ===")
    
    # 创建流水线
    config_path = 'config/base_config.yaml'
    pipeline = EmotionAudioPipeline(config_path)
    
    # 加载测试音频（需要提供实际的音频文件路径）
    audio_loader = AudioLoader()
    
    # 示例：如果有测试音频文件
    test_audio_path = "data/test/sample.wav"  # 需要替换为实际路径
    
    if os.path.exists(test_audio_path):
        # 加载音频
        test_audio = audio_loader.load_audio(test_audio_path)
        
        # 运行不同模式的推理
        print(f"处理音频: {test_audio_path}")
        print(f"音频长度: {test_audio.duration:.2f}秒")
        
        # 1. 基线推理（不量化，仅B1）
        print("运行基线推理（B1阶段）...")
        baseline_b1_audio = pipeline.inference(test_audio, use_quantizer=False, use_b2=False)
        output_path = os.path.join(pipeline.output_dir, 'baseline_b1_audio.wav')
        pipeline._save_audio(baseline_b1_audio, output_path, test_audio.sample_rate)
        print(f"基线B1音频已保存到: {output_path}")
        
        # 2. 基线推理（不量化，B1+B2）
        if pipeline.stage_b_model.enable_b2:
            print("运行基线推理（B1+B2阶段）...")
            baseline_b2_audio = pipeline.inference(test_audio, use_quantizer=False, use_b2=True)
            output_path = os.path.join(pipeline.output_dir, 'baseline_b2_audio.wav')
            pipeline._save_audio(baseline_b2_audio, output_path, test_audio.sample_rate)
            print(f"基线B2音频已保存到: {output_path}")
        
        # 3. VQ-VAE量化推理（B1）
        if pipeline.emotion_quantizer is not None:
            print("运行VQ-VAE量化推理（B1阶段）...")
            quantized_b1_audio = pipeline.inference(test_audio, use_quantizer=True, codebook_size=256, use_b2=False)
            output_path = os.path.join(pipeline.output_dir, 'quantized_b1_audio.wav')
            pipeline._save_audio(quantized_b1_audio, output_path, test_audio.sample_rate)
            print(f"量化B1音频已保存到: {output_path}")
        
    else:
        print(f"测试音频文件不存在: {test_audio_path}")
        print("请将音频文件放置在正确的路径，或修改示例中的路径")


def vq_vae_experiment_example():
    """VQ-VAE码本实验示例"""
    print("\n=== VQ-VAE码本实验示例 ===")
    
    # 创建流水线
    config_path = 'config/base_config.yaml'
    pipeline = EmotionAudioPipeline(config_path)
    
    # 检查VQ-VAE是否启用
    if pipeline.emotion_quantizer is None:
        print("VQ-VAE实验未启用，请在配置文件中设置 emotion_quantizer.enabled: true")
        return
    
    # 加载测试音频
    test_audio_path = "data/test/sample.wav"
    
    if os.path.exists(test_audio_path):
        audio_loader = AudioLoader()
        test_audio = audio_loader.load_audio(test_audio_path)
        
        # 运行VQ-VAE码本大小实验
        print("开始VQ-VAE码本实验...")
        vq_results = pipeline.run_vq_vae_experiment(test_audio)
        
        print("\n实验完成！生成的音频文件：")
        
        # 显示基线结果
        print("基线结果（无量化）：")
        print("  baseline_b1.wav - B1阶段结果")
        if pipeline.stage_b_model.enable_b2:
            print("  baseline_b2.wav - B2阶段结果")
        
        # 显示量化结果
        codebook_sizes = pipeline.emotion_quantizer.get_available_codebook_sizes()
        print(f"\n量化结果（码本大小: {codebook_sizes}）：")
        for k in codebook_sizes:
            print(f"  vq_k{k}_b1.wav - 码本{k}的B1结果")
            if pipeline.stage_b_model.enable_b2:
                print(f"  vq_k{k}_b2.wav - 码本{k}的B2结果")
        
        print("\n实验假设验证：")
        print("- 大码本 (K=1024): 应保留细腻情感细节")
        print("- 中等码本 (K=256, K=512): 保留主要情感，丢失部分细节")
        print("- 小码本 (K=64, K=128): 只保留基础情感特征")
        print("- 基线（无量化）: 保留完整的情感信息")
        
        # 显示量化损失信息
        print("\n量化损失信息：")
        for key, result in vq_results.items():
            if key != 'baseline':
                print(f"  {key}: VQ损失 = {result['vq_loss']:.4f}")
        
    else:
        print(f"测试音频文件不存在: {test_audio_path}")


def stage_comparison_example():
    """阶段对比实验示例"""
    print("\n=== 阶段对比实验示例 ===")
    
    # 创建流水线
    config_path = 'config/base_config.yaml'
    pipeline = EmotionAudioPipeline(config_path)
    
    test_audio_path = "data/test/sample.wav"
    
    if os.path.exists(test_audio_path):
        audio_loader = AudioLoader()
        test_audio = audio_loader.load_audio(test_audio_path)
        
        print("对比不同阶段的效果...")
        
        # 1. 仅阶段A（中性声音）
        print("生成阶段A结果（中性声音M0）...")
        processed = pipeline.data_processor.process_audio(test_audio)
        m0_output = pipeline.stage_a_model.forward(processed.phonemes)
        m0_audio = pipeline.vocoder.synthesize(m0_output.mel_spectrogram)
        output_path = os.path.join(pipeline.output_dir, 'stage_a_only.wav')
        pipeline._save_audio(m0_audio, output_path, processed.source_audio.sample_rate)
        print(f"阶段A音频已保存到: {output_path}")
        
        # 2. 阶段A + B1（带情感）
        print("生成阶段A+B1结果（带情感M1）...")
        ab1_audio = pipeline.inference(test_audio, use_quantizer=False, use_b2=False)
        output_path = os.path.join(pipeline.output_dir, 'stage_a_b1.wav')
        pipeline._save_audio(ab1_audio, output_path, test_audio.sample_rate)
        print(f"阶段A+B1音频已保存到: {output_path}")
        
        # 3. 阶段A + B1 + B2（细化情感）
        if pipeline.stage_b_model.enable_b2:
            print("生成阶段A+B1+B2结果（细化情感M2）...")
            ab2_audio = pipeline.inference(test_audio, use_quantizer=False, use_b2=True)
            output_path = os.path.join(pipeline.output_dir, 'stage_a_b1_b2.wav')
            pipeline._save_audio(ab2_audio, output_path, test_audio.sample_rate)
            print(f"阶段A+B1+B2音频已保存到: {output_path}")
        
        print("\n对比说明：")
        print("- stage_a_only.wav: 中性声音，无情感信息")
        print("- stage_a_b1.wav: 添加了情感信息的声音")
        if pipeline.stage_b_model.enable_b2:
            print("- stage_a_b1_b2.wav: 经过扩散模型细化的声音")
    
    else:
        print(f"测试音频文件不存在: {test_audio_path}")


def training_example():
    """训练示例"""
    print("\n=== 训练示例 ===")
    
    # 创建流水线
    config_path = 'config/base_config.yaml'
    pipeline = EmotionAudioPipeline(config_path)
    
    # 检查训练数据
    train_data = pipeline.dataset_loader.load_train_data()
    
    if len(train_data) > 0:
        print(f"找到 {len(train_data)} 个训练样本")
        
        # 开始训练
        print("开始训练...")
        pipeline.train(pipeline.config)
        
        print("训练完成！")
    else:
        print("没有找到训练数据")
        print("请将音频文件放置在配置文件指定的训练数据目录中")
        
        # 显示配置的数据路径
        train_config = pipeline.config.get('dataset', {}).get('train_data', {})
        audio_dir = train_config.get('audio_dir', 'data/train/audio')
        print(f"训练音频目录: {audio_dir}")


def evaluation_example():
    """评估示例"""
    print("\n=== 评估示例 ===")
    
    # 创建流水线
    config_path = 'config/base_config.yaml'
    pipeline = EmotionAudioPipeline(config_path)
    
    # 加载测试数据
    test_data = pipeline.dataset_loader.load_test_data()
    
    if len(test_data) > 0:
        print(f"找到 {len(test_data)} 个测试样本")
        
        # 运行评估
        print("开始评估...")
        metrics = pipeline.evaluate(test_data)
        
        print("评估结果:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    else:
        print("没有找到测试数据")
        print("请将测试音频文件放置在配置文件指定的测试数据目录中")
        
        # 显示配置的数据路径
        test_config = pipeline.config.get('dataset', {}).get('test_data', {})
        audio_dir = test_config.get('audio_dir', 'data/test/esd/audio')
        print(f"测试音频目录: {audio_dir}")


def paddlespeech_tts_example():
    """PaddleSpeech TTS集成示例"""
    print("\n=== PaddleSpeech TTS集成示例 ===")
    
    try:
        from src.models.stage_a import TTSStageAModel
        
        # 配置PaddleSpeech FastSpeech2
        config = {
            'tts_architecture': 'paddlespeech_fastspeech2',  # 明确指定使用PaddleSpeech
            'model_config': {
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 8,
                'dropout': 0.1
            },
            'phoneme_vocab': {
                'source': 'pypinyin_auto'
            }
        }
        
        print("1. 初始化PaddleSpeech FastSpeech2模型...")
        tts_model = TTSStageAModel(config)
        print(f"✓ 模型初始化完成，使用架构: {tts_model.tts_architecture}")
        
        # 检查PaddleSpeech状态
        if hasattr(tts_model.model, 'model_initialized'):
            if tts_model.model.model_initialized:
                print("✓ PaddleSpeech模型初始化成功，使用预训练模型")
            else:
                print("⚠ PaddleSpeech模型初始化失败，使用备选实现")
        
        # 测试文本到语音转换
        test_texts = [
            "你好，世界！",
            "这是PaddleSpeech集成测试。",
            "中文语音合成效果如何？",
            "欢迎使用情感音频建模系统。"
        ]
        
        print("\n2. 测试文本到语音转换...")
        for i, text in enumerate(test_texts, 1):
            print(f"  测试文本 {i}: {text}")
            try:
                # 使用新的PaddleSpeech集成
                result = tts_model.forward_from_text(text)
                print(f"    ✓ 生成成功，Mel频谱形状: {result.mel_spectrogram.shape}")
                print(f"    ✓ 处理方法: {result.metadata.get('method', 'phoneme_based')}")
            except Exception as e:
                print(f"    ❌ 生成失败: {e}")
        
        # 测试音素输入（对比）
        print("\n3. 测试音素输入（传统方法）...")
        test_phonemes = ['ni3', 'hao3', 'shi4', 'jie4']
        try:
            result = tts_model.forward_from_phonemes(test_phonemes)
            print(f"  ✓ 音素输入成功，Mel频谱形状: {result.mel_spectrogram.shape}")
            print(f"  ✓ 输入音素: {test_phonemes}")
        except Exception as e:
            print(f"  ❌ 音素输入失败: {e}")
        
        # 显示模型信息
        print(f"\n4. 模型详细信息:")
        print(f"  - 模型类型: {type(tts_model.model).__name__}")
        print(f"  - TTS架构: {tts_model.tts_architecture}")
        print(f"  - 音素词汇表大小: {len(tts_model.phoneme_to_idx)}")
        print(f"  - 支持直接文本输入: {tts_model._supports_text_input()}")
        
        # PaddleSpeech特有功能测试
        if hasattr(tts_model.model, 'generate_mel_from_text') and tts_model.model.model_initialized:
            print("\n5. 测试PaddleSpeech直接文本生成...")
            try:
                mel_output = tts_model.model.generate_mel_from_text("测试直接生成")
                print(f"  ✓ 直接生成成功，输出形状: {mel_output.shape}")
            except Exception as e:
                print(f"  ❌ 直接生成失败: {e}")
        
        print("\n✓ PaddleSpeech集成测试完成！")
        
        # 使用建议
        print("\n📝 使用建议:")
        print("1. 首次运行会自动下载PaddleSpeech预训练模型，可能需要一些时间")
        print("2. 如果网络问题导致下载失败，系统会自动使用备选实现")
        print("3. PaddleSpeech模型支持直接文本输入，无需手动转换音素")
        print("4. 预训练模型在中文语音合成上有更好的效果")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保已安装PaddleSpeech: pip install paddlespeech paddlepaddle")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def nvidia_hifigan_example():
    """NVIDIA预训练HiFi-GAN声码器示例"""
    print("\n=== NVIDIA预训练HiFi-GAN声码器示例 ===")
    
    try:
        from src.models.vocoder import VocoderFactory, NVIDIAHiFiGANVocoder
        from src.models.stage_a import TTSStageAModel
        
        # 检查NVIDIA HiFi-GAN是否可用
        available_vocoders = VocoderFactory.get_available_vocoders()
        
        if 'hifigan_nvidia' not in available_vocoders:
            print("⚠ NVIDIA HiFi-GAN不可用，请安装: pip install nemo_toolkit[all]")
            return
        
        print("✅ 发现NVIDIA预训练HiFi-GAN声码器")
        
        # 创建阶段A模型生成Mel频谱
        stage_a_config = {
            'tts_architecture': 'fastspeech2',
            'model_config': {'d_model': 256, 'n_layers': 4, 'n_heads': 8, 'dropout': 0.1},
            'phoneme_vocab': {'source': 'pypinyin_auto'}
        }
        
        tts_model = TTSStageAModel(stage_a_config)
        test_text = "NVIDIA预训练HiFi-GAN提供专业级语音合成质量"
        
        print(f"\n1. 生成测试Mel频谱...")
        mel_result = tts_model.forward_from_text(test_text)
        print(f"   ✅ Mel频谱形状: {mel_result.mel_spectrogram.shape}")
        
        # 对比自定义和NVIDIA HiFi-GAN
        hifigan_versions = [
            ('hifigan_custom', '自定义HiFi-GAN (轻量级)'),
            ('hifigan_nvidia', 'NVIDIA预训练HiFi-GAN (高质量)')
        ]
        
        print(f"\n2. 对比不同HiFi-GAN版本...")
        
        for version, description in hifigan_versions:
            if version in available_vocoders:
                print(f"\n   测试 {version} - {description}:")
                try:
                    # 创建声码器
                    if version == 'hifigan_nvidia':
                        config = {'model_config': {'model_name': 'nvidia/tts_hifigan'}}
                    else:
                        config = {'model_config': {}}
                    
                    vocoder = VocoderFactory.create_vocoder(version, config)
                    
                    # 获取模型信息
                    if hasattr(vocoder, 'get_model_info'):
                        info = vocoder.get_model_info()
                        print(f"     - 模型类型: {info.get('model_type', 'Custom')}")
                        print(f"     - 采样率: {info['sampling_rate']} Hz")
                    
                    # 合成音频
                    audio = vocoder.synthesize(mel_result.mel_spectrogram)
                    print(f"     ✅ 合成成功: {len(audio)} samples, {len(audio)/vocoder.sampling_rate:.2f}秒")
                    
                    # 保存音频
                    try:
                        import soundfile as sf
                        output_path = f"output_{version}.wav"
                        sf.write(output_path, audio, vocoder.sampling_rate)
                        print(f"     📁 已保存: {output_path}")
                    except ImportError:
                        print(f"     ⚠ soundfile未安装，无法保存音频")
                        
                except Exception as e:
                    print(f"     ❌ 失败: {e}")
            else:
                print(f"   ⚠ {version} 不可用")
        
        print(f"\n✅ NVIDIA HiFi-GAN测试完成!")
        print(f"\n💡 质量对比:")
        print(f"   - hifigan_custom: ⭐⭐⭐ (轻量级，快速)")
        print(f"   - hifigan_nvidia: ⭐⭐⭐⭐ (预训练，高质量)")
        print(f"   - bigvgan: ⭐⭐⭐⭐⭐ (世界级质量)")
        
        print(f"\n🔧 配置方式:")
        print(f"   vocoder:")
        print(f"     model_type: \"hifigan_nvidia\"  # 使用NVIDIA预训练版本")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请安装NeMo toolkit: pip install nemo_toolkit[all]")
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def bigvgan_vocoder_example():
    """BigVGAN声码器集成示例"""
    print("\n=== BigVGAN高质量声码器示例 ===")
    
    try:
        from src.models.vocoder import VocoderFactory, BigVGANVocoder
        from src.models.stage_a import TTSStageAModel
        
        # 检查BigVGAN是否可用
        available_vocoders = VocoderFactory.get_available_vocoders()
        bigvgan_vocoders = [v for v in available_vocoders if 'bigvgan' in v.lower()]
        
        if not bigvgan_vocoders:
            print("⚠ BigVGAN声码器不可用，请安装: pip install bigvgan")
            return
        
        print(f"✅ 发现BigVGAN声码器: {bigvgan_vocoders}")
        
        # 创建阶段A模型生成Mel频谱
        stage_a_config = {
            'tts_architecture': 'fastspeech2',
            'model_config': {'d_model': 256, 'n_layers': 4, 'n_heads': 8, 'dropout': 0.1},
            'phoneme_vocab': {'source': 'pypinyin_auto'}
        }
        
        tts_model = TTSStageAModel(stage_a_config)
        test_text = "BigVGAN提供世界级的语音合成质量"
        
        print(f"\n1. 生成测试Mel频谱...")
        mel_result = tts_model.forward_from_text(test_text)
        print(f"   ✅ Mel频谱形状: {mel_result.mel_spectrogram.shape}")
        
        # 测试不同版本的BigVGAN
        bigvgan_versions = ['bigvgan_22khz', 'bigvgan_24khz', 'bigvgan_44khz']
        
        print(f"\n2. 测试BigVGAN不同版本...")
        
        for version in bigvgan_versions:
            if version in available_vocoders:
                print(f"\n   测试 {version}:")
                try:
                    # 创建BigVGAN声码器
                    config = {'model_config': {'version': version.split('_')[1]}}
                    vocoder = VocoderFactory.create_vocoder(version, config)
                    
                    # 获取模型信息
                    if hasattr(vocoder, 'get_model_info'):
                        info = vocoder.get_model_info()
                        print(f"     - 模型: {info['model_name']}")
                        print(f"     - 采样率: {info['sampling_rate']} Hz")
                        print(f"     - Mel维度: {info['n_mel_channels']}")
                    
                    # 合成音频
                    audio = vocoder.synthesize(mel_result.mel_spectrogram)
                    print(f"     ✅ 合成成功: {len(audio)} samples, {len(audio)/vocoder.sampling_rate:.2f}秒")
                    
                    # 保存音频
                    try:
                        import soundfile as sf
                        output_path = f"output_{version}.wav"
                        sf.write(output_path, audio, vocoder.sampling_rate)
                        print(f"     📁 已保存: {output_path}")
                    except ImportError:
                        print(f"     ⚠ soundfile未安装，无法保存音频")
                        
                except Exception as e:
                    print(f"     ❌ 失败: {e}")
            else:
                print(f"   ⚠ {version} 不可用")
        
        print(f"\n✅ BigVGAN测试完成!")
        print(f"\n💡 使用建议:")
        print(f"   - bigvgan_22khz: 与现有系统兼容，适合开发测试")
        print(f"   - bigvgan_24khz: 更高质量，适合一般应用")
        print(f"   - bigvgan_44khz: 最高质量，适合专业制作")
        
        print(f"\n🔧 配置方式:")
        print(f"   在 config/base_config.yaml 中设置:")
        print(f"   vocoder:")
        print(f"     model_type: \"bigvgan_22khz\"  # 选择版本")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请安装BigVGAN: pip install bigvgan")
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def bypass_stage_b_example():
    """绕过阶段B的测试示例"""
    print("\n=== 绕过阶段B测试 (验证其他组件) ===")
    
    try:
        from src.models.stage_a import TTSStageAModel
        from src.models.vocoder import VocoderFactory
        
        print("1. 测试文本到Mel频谱 (阶段A)...")
        
        # 创建阶段A模型
        stage_a_config = {
            'tts_architecture': 'fastspeech2',
            'model_config': {'d_model': 128, 'n_layers': 2, 'n_heads': 4, 'dropout': 0.1},
            'phoneme_vocab': {'source': 'pypinyin_auto'}
        }
        
        stage_a_model = TTSStageAModel(stage_a_config)
        
        # 测试文本
        test_text = "绕过阶段B进行快速功能验证"
        print(f"   测试文本: {test_text}")
        
        # 生成Mel频谱
        m0_result = stage_a_model.forward_from_text(test_text)
        print(f"   ✅ 生成M0 Mel频谱: {m0_result.mel_spectrogram.shape}")
        
        print("\n2. 测试Mel频谱到音频 (声码器)...")
        
        # 获取可用声码器
        available_vocoders = VocoderFactory.get_available_vocoders()
        print(f"   可用声码器: {available_vocoders[:3]}...")  # 只显示前3个
        
        # 选择一个可用的声码器进行测试
        test_vocoders = ['hifigan_custom', 'hifigan', 'melgan']
        selected_vocoder = None
        
        for vocoder_type in test_vocoders:
            if vocoder_type in available_vocoders:
                selected_vocoder = vocoder_type
                break
        
        if not selected_vocoder:
            selected_vocoder = available_vocoders[0] if available_vocoders else None
        
        if selected_vocoder:
            print(f"   使用声码器: {selected_vocoder}")
            vocoder = VocoderFactory.create_vocoder(selected_vocoder, {'model_config': {}})
            
            # 直接从M0合成音频 (绕过阶段B)
            audio = vocoder.synthesize(m0_result.mel_spectrogram)
            print(f"   ✅ 直接合成音频: {len(audio)} samples, {len(audio)/vocoder.sampling_rate:.2f}秒")
            
            # 保存测试音频
            try:
                import soundfile as sf
                output_path = "bypass_stage_b_test.wav"
                sf.write(output_path, audio, vocoder.sampling_rate)
                print(f"   📁 已保存: {output_path}")
            except ImportError:
                print("   ⚠ soundfile未安装，无法保存音频")
            
            print(f"\n✅ 绕过阶段B测试成功!")
            print(f"💡 核心TTS功能正常:")
            print(f"   - 文本 → M0 (中性Mel频谱)")
            print(f"   - M0 → 音频 (声码器合成)")
            print(f"🎯 您可以专注于开发阶段B，其他组件已就绪!")
            
        else:
            print("   ❌ 没有可用的声码器")
            
    except Exception as e:
        print(f"❌ 绕过阶段B测试失败: {e}")
        import traceback
        traceback.print_exc()


def create_sample_config():
    """创建示例配置文件"""
    print("\n=== 创建示例配置 ===")
    
    # 创建数据目录结构
    dirs_to_create = [
        'data/train/audio',
        'data/train/text',
        'data/test/esd/audio',
        'data/test/esd/text',
        'outputs'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    print("\n目录结构已创建！")
    print("请将音频文件放置在相应目录中：")
    print("- 训练数据: data/train/audio/")
    print("- 测试数据: data/test/esd/audio/")
    print("- 输出结果: outputs/")


def main():
    """主函数"""
    print("情感音频建模系统 - 更新版基础使用示例")
    print("=" * 60)
    
    # 创建必要的目录结构
    create_sample_config()
    
    # 运行不同的示例
    try:
        # PaddleSpeech TTS集成示例
        paddlespeech_tts_example()
        
        # NVIDIA预训练HiFi-GAN示例
        nvidia_hifigan_example()
        
        # BigVGAN高质量声码器示例
        bigvgan_vocoder_example()
        
        # 绕过阶段B的测试示例（新增）
        bypass_stage_b_example()
        
        # 基础推理示例
        basic_inference_example()
        
        # VQ-VAE实验示例
        vq_vae_experiment_example()
        
        # 阶段对比实验
        stage_comparison_example()
        
        # 训练示例（注释掉，避免在没有数据时运行）
        # training_example()
        
        # 评估示例（注释掉，避免在没有数据时运行）
        # evaluation_example()
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("这可能是因为缺少必要的依赖或数据文件")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        print("PaddleSpeech安装: pip install paddlespeech paddlepaddle")
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("\n新架构特点：")
    print("1. 🆕 集成PaddleSpeech预训练FastSpeech2模型")
    print("2. 🆕 支持NVIDIA预训练HiFi-GAN声码器")
    print("3. 🆕 支持NVIDIA BigVGAN高质量声码器 (22kHz/24kHz/44kHz)")
    print("4. VQ-VAE量化作为B阶段的前置处理")
    print("5. B阶段分为B1（情感适配）和B2（扩散细化）两个子阶段")
    print("6. 支持灵活的开关控制各个组件")
    print("7. 完整的码本大小对比实验")
    print("8. 支持直接文本输入和传统音素输入")
    print("\n多层次声码器支持：")
    print("- 自定义HiFi-GAN: 轻量级，快速开发")
    print("- NVIDIA预训练HiFi-GAN: 专业级质量")
    print("- BigVGAN: 世界级音频质量，支持多采样率")
    print("- 模块化设计: 可自由切换不同声码器")
    print("- 配置驱动: 修改配置文件即可切换模型")
    print("\n有关更多信息，请参阅 README.md 文件")


if __name__ == "__main__":
    main()