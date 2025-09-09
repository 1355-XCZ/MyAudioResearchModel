"""
模型模块测试
验证各个模型组件的基本功能
"""
import os
import sys
import yaml
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.stage_a import FastSpeech2StageA
from src.models.stage_b import TwoStageEmotionModel
from src.models.emotion_quantizer import VQVAEEmotionQuantizer, IdentityQuantizer
from src.models.vocoder import HiFiGANVocoder, MelGANVocoder
from src.data_processing.audio_processor import WhisperEmotionProcessor, AudioLoader


def test_stage_a_model():
    """测试阶段A模型"""
    print("=== 测试阶段A模型 ===")
    
    try:
        # 加载配置
        with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建阶段A模型
        stage_a_model = FastSpeech2StageA(config['stage_a'])
        
        # 准备测试音素
        test_phonemes = ["ni3", "hao3", "zhe4", "shi4", "yi1", "ge4", "ce4", "shi4"]
        
        print(f"测试音素: {test_phonemes}")
        
        # 测试前向传播
        print("开始阶段A推理...")
        output = stage_a_model.forward(test_phonemes)
        
        print(f"✅ 阶段A推理成功:")
        print(f"   输出Mel频谱形状: {output.mel_spectrogram.shape}")
        print(f"   元数据: {output.metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ 阶段A测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage_b_model():
    """测试阶段B模型"""
    print("\n=== 测试阶段B模型 ===")
    
    try:
        # 加载配置
        with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建阶段B模型
        stage_b_model = TwoStageEmotionModel(config['stage_b'])
        
        # 准备测试数据
        # 模拟M0 (来自阶段A的输出)
        test_m0 = torch.randn(1, 100, 80)  # (batch, time, mel_dim)
        
        # 模拟情感特征
        test_emotion = np.random.randn(768).astype(np.float32)
        
        print(f"测试M0形状: {test_m0.shape}")
        print(f"测试情感特征形状: {test_emotion.shape}")
        
        # 测试B1阶段
        print("测试B1阶段...")
        b1_output = stage_b_model.forward_b1(test_m0, test_emotion)
        print(f"✅ B1推理成功: {b1_output.mel_spectrogram.shape}")
        
        # 测试B2阶段（如果启用）
        if stage_b_model.enable_b2:
            print("测试B2阶段...")
            b2_output = stage_b_model.forward_b2(b1_output.mel_spectrogram, test_emotion)
            print(f"✅ B2推理成功: {b2_output.mel_spectrogram.shape}")
        else:
            print("ℹ️ B2阶段未启用")
        
        # 测试完整B阶段
        print("测试完整B阶段...")
        full_output = stage_b_model.forward(test_m0, test_emotion, use_b2=True)
        print(f"✅ 完整B阶段推理成功: {full_output.mel_spectrogram.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 阶段B测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emotion_quantizer():
    """测试VQ-VAE情感量化器"""
    print("\n=== 测试VQ-VAE量化器 ===")
    
    try:
        # 加载配置
        with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建量化器
        quantizer = VQVAEEmotionQuantizer(config['emotion_quantizer'])
        
        # 准备测试情感特征
        test_emotion = np.random.randn(768).astype(np.float32)
        print(f"测试情感特征形状: {test_emotion.shape}")
        
        # 测试不同码本大小的量化
        codebook_sizes = quantizer.get_available_codebook_sizes()
        print(f"支持的码本大小: {codebook_sizes}")
        
        for k in [64, 256, 1024]:  # 测试几个代表性大小
            if k in codebook_sizes:
                print(f"测试码本大小 {k}...")
                quantized_features, vq_loss = quantizer.quantize(test_emotion, k)
                
                print(f"✅ 码本{k}量化成功:")
                print(f"   量化后形状: {quantized_features.shape}")
                print(f"   VQ损失: {vq_loss:.4f}")
        
        # 测试批量实验
        print("测试批量码本实验...")
        experiment_results = quantizer.experiment_codebook_sizes(test_emotion)
        
        print(f"✅ 批量实验成功:")
        for k, (features, loss) in experiment_results.items():
            print(f"   码本{k}: 损失={loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 量化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vocoder():
    """测试声码器"""
    print("\n=== 测试声码器 ===")
    
    try:
        # 加载配置
        with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建声码器
        from src.models.vocoder import VocoderFactory
        vocoder = VocoderFactory.create_vocoder(
            config['vocoder']['model_type'], 
            config['vocoder']
        )
        
        # 准备测试Mel频谱
        test_mel = torch.randn(1, 80, 100)  # (batch, mel_dim, time)
        print(f"测试Mel频谱形状: {test_mel.shape}")
        
        # 测试音频合成
        print("开始音频合成...")
        synthesized_audio = vocoder.synthesize(test_mel)
        
        print(f"✅ 音频合成成功:")
        print(f"   合成音频形状: {synthesized_audio.shape}")
        print(f"   音频时长: {len(synthesized_audio) / 22050:.2f}秒")
        print(f"   音频范围: [{synthesized_audio.min():.3f}, {synthesized_audio.max():.3f}]")
        
        # 保存测试音频
        output_path = "tests/outputs/test_vocoder_output.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        import soundfile as sf
        sf.write(output_path, synthesized_audio, 22050)
        print(f"✅ 测试音频已保存: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 声码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_model_tests():
    """运行所有模型测试"""
    print("🧪 开始模型模块完整测试")
    print("=" * 50)
    
    tests = [
        ("阶段A模型", test_stage_a_model),
        ("阶段B模型", test_stage_b_model),
        ("VQ-VAE量化器", test_emotion_quantizer),
        ("声码器", test_vocoder)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results[test_name] = False
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("🏆 模型测试结果汇总:")
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\n总体通过率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")


if __name__ == "__main__":
    run_all_model_tests()
