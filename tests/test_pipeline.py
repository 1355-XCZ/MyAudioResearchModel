"""
完整流水线测试
验证端到端的音频处理流程
"""
import os
import sys
import yaml
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.training_pipeline import EmotionAudioPipeline
from src.data_processing.audio_processor import AudioLoader
from src.core.interfaces import AudioData


def test_pipeline_creation():
    """测试流水线创建"""
    print("=== 测试流水线创建 ===")
    
    try:
        # 创建流水线
        pipeline = EmotionAudioPipeline('config/base_config.yaml')
        
        print("✅ 流水线创建成功:")
        print(f"   数据处理器: {type(pipeline.data_processor).__name__}")
        print(f"   阶段A模型: {type(pipeline.stage_a_model).__name__}")
        print(f"   阶段B模型: {type(pipeline.stage_b_model).__name__}")
        print(f"   声码器: {type(pipeline.vocoder).__name__}")
        print(f"   量化器: {type(pipeline.emotion_quantizer).__name__ if pipeline.emotion_quantizer else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 流水线创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_inference():
    """测试端到端推理"""
    print("\n=== 测试端到端推理 ===")
    
    test_audio_path = "data/test/sample.wav"
    if not os.path.exists(test_audio_path):
        print(f"❌ 测试音频不存在: {test_audio_path}")
        print("请放置一个测试音频文件到该路径")
        return False
    
    try:
        # 创建流水线
        pipeline = EmotionAudioPipeline('config/base_config.yaml')
        
        # 加载测试音频
        audio_data = AudioLoader.load_audio(test_audio_path)
        print(f"✅ 加载测试音频: {audio_data.duration:.2f}秒")
        
        # 测试基线推理（无量化，仅B1）
        print("测试基线推理（B1）...")
        baseline_b1_audio = pipeline.inference(
            audio_data, 
            use_quantizer=False, 
            use_b2=False
        )
        print(f"✅ 基线B1推理成功: {baseline_b1_audio.shape}")
        
        # 测试基线推理（无量化，B1+B2）
        print("测试基线推理（B1+B2）...")
        baseline_b2_audio = pipeline.inference(
            audio_data, 
            use_quantizer=False, 
            use_b2=True
        )
        print(f"✅ 基线B2推理成功: {baseline_b2_audio.shape}")
        
        # 测试VQ-VAE量化推理
        if pipeline.emotion_quantizer is not None:
            print("测试VQ-VAE量化推理...")
            quantized_audio = pipeline.inference(
                audio_data,
                use_quantizer=True,
                codebook_size=256,
                use_b2=False
            )
            print(f"✅ VQ-VAE推理成功: {quantized_audio.shape}")
        else:
            print("ℹ️ VQ-VAE量化器未启用")
        
        # 保存测试结果
        output_dir = "tests/outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        import soundfile as sf
        sf.write(f"{output_dir}/test_baseline_b1.wav", baseline_b1_audio, 22050)
        sf.write(f"{output_dir}/test_baseline_b2.wav", baseline_b2_audio, 22050)
        
        if pipeline.emotion_quantizer is not None:
            sf.write(f"{output_dir}/test_quantized.wav", quantized_audio, 22050)
        
        print(f"✅ 测试音频已保存到: {output_dir}/")
        
        return True
        
    except Exception as e:
        print(f"❌ 端到端推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vq_vae_experiment():
    """测试VQ-VAE码本实验"""
    print("\n=== 测试VQ-VAE码本实验 ===")
    
    test_audio_path = "data/test/sample.wav"
    if not os.path.exists(test_audio_path):
        print(f"❌ 测试音频不存在: {test_audio_path}")
        return False
    
    try:
        # 创建流水线
        pipeline = EmotionAudioPipeline('config/base_config.yaml')
        
        if pipeline.emotion_quantizer is None:
            print("❌ VQ-VAE量化器未启用")
            return False
        
        # 加载测试音频
        audio_data = AudioLoader.load_audio(test_audio_path)
        
        # 运行VQ-VAE实验
        print("开始VQ-VAE码本实验...")
        vq_results = pipeline.run_vq_vae_experiment(audio_data)
        
        print(f"✅ VQ-VAE实验成功:")
        for key, result in vq_results.items():
            if key == 'baseline':
                print(f"   {key}: 基线结果")
            else:
                print(f"   {key}: VQ损失={result.get('vq_loss', 'N/A'):.4f}")
        
        print(f"✅ 实验音频已保存到: {pipeline.output_dir}/")
        
        return True
        
    except Exception as e:
        print(f"❌ VQ-VAE实验测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_flow():
    """测试数据流的完整性"""
    print("\n=== 测试数据流完整性 ===")
    
    test_audio_path = "data/test/sample.wav"
    if not os.path.exists(test_audio_path):
        print(f"❌ 测试音频不存在: {test_audio_path}")
        return False
    
    try:
        # 创建流水线
        pipeline = EmotionAudioPipeline('config/base_config.yaml')
        
        # 加载音频
        audio_data = AudioLoader.load_audio(test_audio_path)
        
        print("跟踪完整数据流...")
        
        # 步骤1: 数据预处理
        processed = pipeline.data_processor.process_audio(audio_data)
        print(f"✅ 预处理: 音素{len(processed.phonemes)}个, 情感特征{processed.emotion_features.shape}")
        
        # 步骤2: 阶段A
        m0_output = pipeline.stage_a_model.forward(processed.phonemes)
        print(f"✅ 阶段A: M0形状={m0_output.mel_spectrogram.shape}")
        
        # 步骤3: 可选量化
        emotion_features = processed.emotion_features
        if pipeline.emotion_quantizer is not None:
            quantized_emotion, vq_loss = pipeline.emotion_quantizer.quantize(emotion_features, 256)
            print(f"✅ 量化: VQ损失={vq_loss:.4f}, 形状={quantized_emotion.shape}")
            emotion_features = quantized_emotion
        
        # 步骤4: 阶段B
        m1_output = pipeline.stage_b_model.forward(
            m0_output.mel_spectrogram, 
            emotion_features, 
            use_b2=False
        )
        print(f"✅ 阶段B: M1形状={m1_output.mel_spectrogram.shape}")
        
        # 步骤5: 声码器
        final_audio = pipeline.vocoder.synthesize(m1_output.mel_spectrogram)
        print(f"✅ 声码器: 音频形状={final_audio.shape}")
        
        # 验证数据一致性
        original_duration = audio_data.duration
        reconstructed_duration = len(final_audio) / 22050
        duration_ratio = reconstructed_duration / original_duration
        
        print(f"✅ 数据流完整性验证:")
        print(f"   原始时长: {original_duration:.2f}秒")
        print(f"   重建时长: {reconstructed_duration:.2f}秒")
        print(f"   时长比例: {duration_ratio:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_pipeline_tests():
    """运行所有流水线测试"""
    print("🧪 开始流水线完整测试")
    print("=" * 50)
    
    tests = [
        ("流水线创建", test_pipeline_creation),
        ("端到端推理", test_end_to_end_inference),
        ("VQ-VAE实验", test_vq_vae_experiment),
        ("数据流完整性", test_data_flow)
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
    print("🏆 流水线测试结果汇总:")
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\n总体通过率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")


if __name__ == "__main__":
    run_all_pipeline_tests()
