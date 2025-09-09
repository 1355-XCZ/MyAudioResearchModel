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
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("\n新架构特点：")
    print("1. VQ-VAE量化作为B阶段的前置处理")
    print("2. B阶段分为B1（情感适配）和B2（扩散细化）两个子阶段")
    print("3. 支持灵活的开关控制各个组件")
    print("4. 完整的码本大小对比实验")
    print("\n有关更多信息，请参阅 README.md 文件")


if __name__ == "__main__":
    main()