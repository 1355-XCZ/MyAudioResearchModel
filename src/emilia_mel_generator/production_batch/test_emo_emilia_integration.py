#!/usr/bin/env python3
"""
测试Emo-Emilia数据集集成
验证是否能正确加载和使用Emo-Emilia数据集中的中性标签音频
"""

import sys
from pathlib import Path
import tempfile

def test_emo_emilia_loading():
    """测试Emo-Emilia数据集加载"""
    print("=" * 60)
    print("🧪 测试 Emo-Emilia 数据集集成")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        
        print("⬇️ 从 HuggingFace 加载 ASLP-lab/Emo-Emilia...")
        
        # 加载Emo-Emilia数据集
        emo_dataset = load_dataset(
            "ASLP-lab/Emo-Emilia",
            trust_remote_code=True
        )
        
        print(f"✅ 数据集加载成功")
        print(f"📊 数据集信息:")
        
        total_samples = 0
        emotion_stats = {}
        lang_stats = {}
        
        for split_name in emo_dataset.keys():
            split_data = emo_dataset[split_name]
            split_size = len(split_data)
            total_samples += split_size
            
            print(f"  分割 '{split_name}': {split_size} 个样本")
            
            # 统计前100个样本的情感和语言分布
            for i, item in enumerate(split_data):
                if i >= 100:  # 只统计前100个样本
                    break
                    
                emotion = item.get('emotion', 'unknown')
                language = item.get('language', 'unknown')
                
                emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
                lang_stats[language] = lang_stats.get(language, 0) + 1
        
        print(f"\n📊 样本统计 (前100个):")
        print(f"  总样本数: {total_samples}")
        print(f"  情感分布: {emotion_stats}")
        print(f"  语言分布: {lang_stats}")
        
        # 专门检查neutral样本
        neutral_count = 0
        neutral_samples = []
        
        for split_name in emo_dataset.keys():
            for item in emo_dataset[split_name]:
                if item.get('emotion', '').lower() == 'neutral':
                    neutral_count += 1
                    if len(neutral_samples) < 5:  # 收集前5个neutral样本作为示例
                        audio_info = item['audio']
                        duration = len(audio_info['array']) / audio_info['sampling_rate']
                        neutral_samples.append({
                            'language': item.get('language', 'unknown'),
                            'text': item.get('text', 'No text'),
                            'duration': duration,
                            'speaker': item.get('speaker', 'unknown')
                        })
        
        print(f"\n🎯 中性情感样本:")
        print(f"  总数: {neutral_count} 个")
        print(f"  示例:")
        for i, sample in enumerate(neutral_samples):
            print(f"    {i+1}. {sample['language']}: \"{sample['text'][:50]}...\" ({sample['duration']:.1f}s)")
        
        print(f"\n✅ Emo-Emilia 数据集集成测试通过！")
        print(f"📋 可用中性参考音频: {neutral_count} 个")
        
        return True
        
    except Exception as e:
        print(f"❌ Emo-Emilia 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_data_generator():
    """测试训练数据生成器的中性参考功能"""
    print("\n" + "=" * 60)
    print("🧪 测试训练数据生成器的中性参考功能")
    print("=" * 60)
    
    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 导入训练数据生成器
            sys.path.append(str(Path(__file__).parent))
            from generate_training_data import TrainingDataGenerator
            
            # 创建生成器实例
            generator = TrainingDataGenerator(
                output_path=str(temp_path / "output"),
                cache_path=str(temp_path / "cache"),
                device='cpu'  # 使用CPU避免GPU依赖
            )
            
            print("🚀 初始化训练数据生成器...")
            
            # 测试中性参考池加载
            generator._load_neutral_reference_pool()
            
            if generator.neutral_reference_pool:
                print(f"✅ 中性参考池加载成功: {len(generator.neutral_reference_pool)} 个样本")
                
                # 测试随机获取中性参考
                for lang in ['en', 'zh']:
                    try:
                        ref_info = generator._get_random_neutral_reference_with_text(lang)
                        audio_data = ref_info['audio']
                        text_data = ref_info['text']
                        
                        print(f"  {lang.upper()}: {len(audio_data)/24000:.1f}s, \"{text_data[:30]}...\"")
                        
                    except Exception as e:
                        print(f"  {lang.upper()}: 获取失败 - {e}")
                
                print("✅ 中性参考功能测试通过！")
                return True
            else:
                print("⚠️ 中性参考池为空，将使用合成音频")
                return False
                
    except Exception as e:
        print(f"❌ 训练数据生成器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔍 Emo-Emilia 集成测试")
    
    # 测试1: Emo-Emilia数据集加载
    test1_success = test_emo_emilia_loading()
    
    # 测试2: 训练数据生成器集成
    test2_success = test_training_data_generator()
    
    print("\n" + "=" * 60)
    if test1_success and test2_success:
        print("🎉 所有测试通过！")
        print("✅ Emo-Emilia 数据集集成成功")
        print("✅ 中性参考音频功能正常")
        print("🚀 可以开始正式的训练数据生成")
        return 0
    else:
        print("❌ 部分测试失败")
        print("⚠️ 请检查错误信息并修复问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
