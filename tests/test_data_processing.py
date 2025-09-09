"""
数据预处理模块测试
验证Whisper、Emotion2Vec和音素转换功能
"""
import os
import sys
import yaml
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.audio_processor import WhisperEmotionProcessor, AudioLoader, DatasetLoader
from src.core.interfaces import AudioData


def test_audio_loading():
    """测试音频加载功能"""
    print("=== 测试音频加载 ===")
    
    # 测试音频文件路径（需要您提供一个测试音频）
    test_audio_path = "data/test/sample.wav"  # 您需要放置一个测试音频
    
    if not os.path.exists(test_audio_path):
        print(f"❌ 测试音频不存在: {test_audio_path}")
        print("请在该路径放置一个测试音频文件")
        return False
    
    try:
        # 测试音频加载
        audio_data = AudioLoader.load_audio(test_audio_path)
        
        print(f"✅ 音频加载成功:")
        print(f"   文件路径: {audio_data.file_path}")
        print(f"   采样率: {audio_data.sample_rate}Hz")
        print(f"   时长: {audio_data.duration:.2f}秒")
        print(f"   波形形状: {audio_data.waveform.shape}")
        print(f"   波形范围: [{audio_data.waveform.min():.3f}, {audio_data.waveform.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 音频加载失败: {e}")
        return False


def test_whisper_recognition():
    """测试Whisper语音识别"""
    print("\n=== 测试Whisper语音识别 ===")
    
    test_audio_path = "data/test/sample.wav"
    if not os.path.exists(test_audio_path):
        print(f"❌ 测试音频不存在: {test_audio_path}")
        return False
    
    try:
        # 加载配置
        with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建处理器
        processor = WhisperEmotionProcessor(config['data_processing'])
        
        # 加载音频
        audio_data = AudioLoader.load_audio(test_audio_path)
        
        # 测试语音识别
        print("开始语音识别...")
        phonemes, recognized_text = processor.extract_phonemes(audio_data)
        
        print(f"✅ 语音识别成功:")
        print(f"   识别文本: {recognized_text}")
        print(f"   音素序列: {phonemes}")
        print(f"   音素数量: {len(phonemes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper识别失败: {e}")
        print("可能原因:")
        print("1. openai-whisper未安装: pip install openai-whisper")
        print("2. 首次运行需要下载模型，请等待")
        print("3. 音频格式不兼容")
        return False


def test_emotion_features():
    """测试Emotion2Vec情感特征提取"""
    print("\n=== 测试Emotion2Vec特征提取 ===")
    
    test_audio_path = "data/test/sample.wav"
    if not os.path.exists(test_audio_path):
        print(f"❌ 测试音频不存在: {test_audio_path}")
        return False
    
    try:
        # 加载配置
        with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建处理器
        processor = WhisperEmotionProcessor(config['data_processing'])
        
        # 加载音频
        audio_data = AudioLoader.load_audio(test_audio_path)
        
        # 测试情感特征提取
        print("开始情感特征提取...")
        emotion_features = processor.extract_emotion_features(audio_data)
        
        print(f"✅ 情感特征提取完成:")
        print(f"   特征形状: {emotion_features.shape}")
        print(f"   特征类型: {type(emotion_features)}")
        print(f"   特征范围: [{emotion_features.min():.3f}, {emotion_features.max():.3f}]")
        print(f"   特征均值: {emotion_features.mean():.3f}")
        print(f"   特征标准差: {emotion_features.std():.3f}")
        
        # 检查是否是模拟特征
        if abs(emotion_features.std() - 1.0) < 0.1:
            print("⚠️ 可能是模拟特征（标准差接近1.0）")
        else:
            print("✅ 可能是真实特征")
        
        return True
        
    except Exception as e:
        print(f"❌ 情感特征提取失败: {e}")
        print("可能原因:")
        print("1. modelscope未安装: pip install modelscope")
        print("2. funasr未安装: pip install funasr")
        print("3. 首次运行需要下载模型，请等待")
        return False


def test_complete_preprocessing():
    """测试完整的预处理流程"""
    print("\n=== 测试完整预处理流程 ===")
    
    test_audio_path = "data/test/sample.wav"
    if not os.path.exists(test_audio_path):
        print(f"❌ 测试音频不存在: {test_audio_path}")
        return False
    
    try:
        # 加载配置
        with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建处理器
        processor = WhisperEmotionProcessor(config['data_processing'])
        
        # 加载音频
        audio_data = AudioLoader.load_audio(test_audio_path)
        
        # 完整预处理
        print("开始完整预处理...")
        processed_data = processor.process_audio(audio_data)
        
        print(f"✅ 完整预处理成功:")
        print(f"   原始音频: {processed_data.source_audio.file_path}")
        print(f"   识别文本: {processed_data.text}")
        print(f"   音素序列: {processed_data.phonemes[:10]}...")  # 显示前10个
        print(f"   音素总数: {len(processed_data.phonemes)}")
        print(f"   情感特征: {processed_data.emotion_features.shape}")
        
        # 验证数据完整性
        assert isinstance(processed_data.phonemes, list), "音素应该是列表"
        assert len(processed_data.phonemes) > 0, "音素列表不应为空"
        assert processed_data.emotion_features.shape == (768,), "情感特征应该是768维"
        assert processed_data.source_audio.waveform is not None, "原始音频应该保留"
        
        print("✅ 数据完整性验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 完整预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """测试数据集批量加载"""
    print("\n=== 测试数据集加载 ===")
    
    try:
        # 加载配置
        with open('config/base_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建数据集加载器
        dataset_loader = DatasetLoader(config['dataset'])
        
        # 测试训练数据加载
        train_data = dataset_loader.load_train_data()
        print(f"✅ 训练数据加载: 找到 {len(train_data)} 个音频文件")
        
        # 测试测试数据加载
        test_data = dataset_loader.load_test_data()
        print(f"✅ 测试数据加载: 找到 {len(test_data)} 个音频文件")
        
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"   样本示例: {sample.file_path}, {sample.duration:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False


def run_all_preprocessing_tests():
    """运行所有预处理测试"""
    print("🧪 开始预处理模块完整测试")
    print("=" * 50)
    
    # 创建必要的目录
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("data/train/audio", exist_ok=True)
    os.makedirs("data/test/esd/audio", exist_ok=True)
    
    tests = [
        ("音频加载", test_audio_loading),
        ("Whisper识别", test_whisper_recognition),
        ("情感特征提取", test_emotion_features),
        ("完整预处理", test_complete_preprocessing),
        ("数据集加载", test_dataset_loading)
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
    print("🏆 测试结果汇总:")
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\n总体通过率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("🎉 预处理模块测试全部通过！")
        print("可以继续测试其他模块了。")
    else:
        print("⚠️ 部分测试失败，请检查错误信息并修复。")


if __name__ == "__main__":
    run_all_preprocessing_tests()
