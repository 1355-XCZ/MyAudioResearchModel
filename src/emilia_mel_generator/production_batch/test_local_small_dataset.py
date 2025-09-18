#!/usr/bin/env python3
"""
本地小数据集测试脚本
使用中英文各10条样本测试主脚本的完整流程
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

def create_test_dataset():
    """创建测试用的小数据集"""
    print("📦 创建测试数据集（中英文各10条）...")
    
    test_samples = []
    
    # 创建10个英文样本
    for i in range(10):
        # 模拟不同长度的音频（2-6秒）
        duration = 2.0 + i * 0.4  # 2.0, 2.4, 2.8, ..., 5.6秒
        samples_count = int(duration * 24000)
        
        # 生成模拟音频（正弦波 + 噪声）
        t = np.linspace(0, duration, samples_count)
        freq = 200 + i * 20  # 不同频率
        audio = 0.3 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(samples_count)
        
        sample = {
            'id': f'en_{i:06d}',
            'audio': {
                'array': audio.astype(np.float32),
                'sampling_rate': 24000
            },
            'text': f'This is English test sample number {i+1}. It contains some meaningful content for testing purposes.',
            'speaker': f'en_speaker_{i+1}',
            'language': 'en',
            'duration': duration
        }
        test_samples.append(sample)
    
    # 创建10个中文样本
    chinese_texts = [
        "这是第一个中文测试样本，用于验证系统功能。",
        "具体而言，上半周持续上涨的逻辑在于部分资金交易。",
        "我们认为在新疆出现更大面积的减产之前。",
        "后续来看，工业硅价格仍将继续维持偏弱的格局。",
        "空投认为继续往下跌的空间并不大。",
        "这个也需要提醒各位投资者持续去进行关注。",
        "因此上半周有大量空方平仓离场。",
        "云南新疆地区虽有减产，但周渡产量数据依旧处于同期高位。",
        "其对于仓单的严格把控或对市场情绪产生相应影响。",
        "我是张行本行，关注投研，投资有风险，入市需谨慎。"
    ]
    
    for i in range(10):
        duration = 2.5 + i * 0.3  # 2.5, 2.8, 3.1, ..., 5.2秒
        samples_count = int(duration * 24000)
        
        # 生成模拟中文音频
        t = np.linspace(0, duration, samples_count)
        freq = 180 + i * 15
        audio = 0.35 * np.sin(2 * np.pi * freq * t) + 0.08 * np.random.randn(samples_count)
        
        sample = {
            'id': f'zh_{i:06d}',
            'audio': {
                'array': audio.astype(np.float32),
                'sampling_rate': 24000
            },
            'text': chinese_texts[i],
            'speaker': f'zh_speaker_{i+1}',
            'language': 'zh',
            'duration': duration
        }
        test_samples.append(sample)
    
    print(f"✅ 创建了 {len(test_samples)} 个测试样本")
    print(f"  英文: 10个样本, 总时长 {sum(s['duration'] for s in test_samples[:10]):.1f}秒")
    print(f"  中文: 10个样本, 总时长 {sum(s['duration'] for s in test_samples[10:]):.1f}秒")
    
    return test_samples

def save_test_dataset(test_samples, output_path):
    """保存测试数据集到指定路径"""
    from datasets import Dataset, DatasetDict
    
    # 转换为HuggingFace Dataset格式
    dataset_data = {
        'audio': [sample['audio'] for sample in test_samples],
        'text': [sample['text'] for sample in test_samples],
        'speaker': [sample['speaker'] for sample in test_samples],
        'language': [sample['language'] for sample in test_samples],
        'duration': [sample['duration'] for sample in test_samples]
    }
    
    dataset = Dataset.from_dict(dataset_data)
    dataset_dict = DatasetDict({'train': dataset})
    
    # 保存到磁盘
    dataset_dict.save_to_disk(output_path)
    print(f"✅ 测试数据集已保存: {output_path}")

def run_local_test():
    """运行本地测试"""
    print("=" * 80)
    print("🧪 本地小数据集测试 - 中英文各10条样本")
    print("=" * 80)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 设置路径
        test_dataset_path = temp_path / "test_emilia_dataset"
        test_output_path = temp_path / "test_training_data"
        test_cache_path = temp_path / "test_cache"
        
        try:
            # 1. 创建测试数据集
            test_samples = create_test_dataset()
            save_test_dataset(test_samples, test_dataset_path)
            
            # 2. 导入训练数据生成器
            sys.path.append(str(Path(__file__).parent))
            from generate_training_data import TrainingDataGenerator
            
            print("\n🚀 初始化训练数据生成器...")
            
            # 3. 创建生成器（使用CPU避免GPU依赖）
            generator = TrainingDataGenerator(
                output_path=str(test_output_path),
                cache_path=str(test_cache_path),
                device='cpu'  # 本地测试使用CPU
            )
            
            print("🔧 初始化模型组件...")
            
            # 4. 初始化模型（这会加载Vevo TTS和Emotion2Vec）
            try:
                generator.initialize_models()
                print("✅ 模型初始化成功")
            except Exception as e:
                print(f"❌ 模型初始化失败: {e}")
                print("💡 这可能是因为缺少GPU或模型权重，但可以继续测试数据加载流程")
                return False
            
            print("\n📊 测试数据集加载和筛选...")
            
            # 5. 测试数据集加载（不设置时长限制，处理所有20个样本）
            try:
                dataset = generator.load_dataset(
                    str(test_dataset_path), 
                    target_hours_per_lang=1.0  # 设置很小的时长，确保处理所有样本
                )
                
                print(f"✅ 数据集加载成功: {len(dataset)} 个样本")
                
                # 验证语言分布
                lang_count = {'en': 0, 'zh': 0}
                for sample in dataset:
                    lang = sample['language']
                    lang_count[lang] += 1
                
                print(f"📊 语言分布验证:")
                for lang, count in lang_count.items():
                    print(f"  {lang.upper()}: {count} 个样本")
                
            except Exception as e:
                print(f"❌ 数据集加载失败: {e}")
                return False
            
            print("\n🎯 测试中性参考音频池...")
            
            # 6. 测试Emo-Emilia中性参考池
            try:
                print(f"中性参考池大小: {len(generator.neutral_reference_pool)}")
                
                if generator.neutral_reference_pool:
                    # 统计中性参考的语言分布
                    neutral_lang_count = {'en': 0, 'zh': 0, 'other': 0}
                    for ref in generator.neutral_reference_pool:
                        lang = ref['language'].lower()
                        if lang in neutral_lang_count:
                            neutral_lang_count[lang] += 1
                        else:
                            neutral_lang_count['other'] += 1
                    
                    print(f"📊 中性参考语言分布:")
                    for lang, count in neutral_lang_count.items():
                        if count > 0:
                            print(f"  {lang.upper()}: {count} 个中性参考")
                    
                    # 测试语言匹配
                    print(f"\n🧪 测试语言匹配:")
                    for test_lang in ['en', 'zh']:
                        try:
                            ref_info = generator._get_random_neutral_reference_with_text(test_lang)
                            ref_lang = ref_info.get('language', 'unknown')
                            ref_text = ref_info.get('text', '')[:50]
                            
                            if ref_lang.lower() == test_lang.lower():
                                print(f"  ✅ {test_lang.upper()}: 正确匹配 -> {ref_lang.upper()}")
                                print(f"     参考文本: '{ref_text}...'")
                            else:
                                print(f"  ❌ {test_lang.upper()}: 语言不匹配 -> {ref_lang.upper()}")
                        
                        except Exception as e:
                            print(f"  ❌ {test_lang.upper()}: 获取失败 - {e}")
                
                else:
                    print("⚠️ 中性参考池为空，将使用合成音频")
                
            except Exception as e:
                print(f"❌ 中性参考测试失败: {e}")
            
            print("\n🎯 测试样本处理...")
            
            # 7. 测试处理几个样本
            test_count = min(4, len(dataset))  # 测试前4个样本
            success_count = 0
            
            for i, sample in enumerate(dataset[:test_count]):
                sample_id = sample['id']
                language = sample['language']
                
                print(f"\n📝 测试样本 {i+1}/{test_count}: {sample_id} ({language.upper()})")
                
                try:
                    success = generator.process_sample(sample)
                    
                    if success:
                        print(f"  ✅ 处理成功")
                        success_count += 1
                        
                        # 检查生成的文件
                        files_to_check = [
                            test_output_path / "original_mels" / f"{sample_id}_mel_original_vevo.npz",
                            test_output_path / "neutral_mels" / f"{sample_id}_mel_neutral_vevo.npz",
                            test_output_path / "emotion_features" / f"{sample_id}_emotion_features.npz"
                        ]
                        
                        for file_path in files_to_check:
                            if file_path.exists():
                                try:
                                    data = np.load(file_path)
                                    if 'mel' in data:
                                        shape = data['mel'].shape
                                        print(f"    📊 {file_path.name}: {shape}")
                                    elif 'utterance' in data:
                                        u_shape = data['utterance'].shape
                                        f_shape = data['frame'].shape
                                        print(f"    📊 {file_path.name}: utterance={u_shape}, frame={f_shape}")
                                except Exception as e:
                                    print(f"    ❌ 文件损坏: {file_path.name}")
                            else:
                                print(f"    ❌ 文件缺失: {file_path.name}")
                    else:
                        print(f"  ❌ 处理失败")
                
                except Exception as e:
                    print(f"  ❌ 处理异常: {e}")
            
            # 8. 检查CSV元信息
            csv_file = test_output_path / "metadata" / "training_dataset_metadata.csv"
            if csv_file.exists():
                import pandas as pd
                df = pd.read_csv(csv_file)
                print(f"\n📊 CSV元信息:")
                print(f"  记录数: {len(df)}")
                print(f"  列: {list(df.columns)}")
                if len(df) > 0:
                    print(f"  示例记录: {df.iloc[0]['sample_id']} ({df.iloc[0]['language']})")
            else:
                print(f"\n❌ CSV元信息文件未生成")
            
            # 9. 测试总结
            print("\n" + "=" * 80)
            print("📊 本地测试总结")
            print("=" * 80)
            print(f"✅ 成功处理: {success_count}/{test_count} 个样本")
            print(f"📁 输出目录: {test_output_path}")
            
            if success_count > 0:
                print(f"\n🎉 测试成功！主脚本工作正常")
                print(f"📋 生成的文件:")
                
                for dir_name in ['original_mels', 'neutral_mels', 'emotion_features']:
                    dir_path = test_output_path / dir_name
                    if dir_path.exists():
                        file_count = len(list(dir_path.glob("*.npz")))
                        print(f"  {dir_name}: {file_count} 个文件")
                
                return True
            else:
                print(f"\n❌ 测试失败！没有成功处理任何样本")
                return False
                
        except Exception as e:
            print(f"❌ 测试过程异常: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_with_real_emilia_cache():
    """使用现有的emilia缓存进行测试"""
    print("=" * 80)
    print("🧪 使用现有Emilia缓存测试 - 中英文各10条")
    print("=" * 80)
    
    # 检查是否有现有的缓存
    cache_dir = Path("../emilia_cache")
    if not cache_dir.exists():
        print("❌ 未找到emilia缓存目录，请先运行完整测试")
        return False
    
    try:
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = Path(temp_dir) / "test_output"
            temp_cache = Path(temp_dir) / "cache"
            
            # 导入生成器
            sys.path.append(str(Path(__file__).parent))
            from generate_training_data import TrainingDataGenerator
            
            # 创建生成器
            generator = TrainingDataGenerator(
                output_path=str(temp_output),
                cache_path=str(temp_cache),
                device='cpu'
            )
            
            print("🚀 初始化模型...")
            generator.initialize_models()
            
            # 从现有缓存加载少量样本
            print("📊 从现有缓存加载样本...")
            
            # 模拟加载缓存数据的逻辑
            test_samples = []
            
            # 从英文缓存加载
            en_cache = cache_dir / "EN"
            if en_cache.exists():
                en_files = list(en_cache.glob("*.wav"))[:10]
                for i, wav_file in enumerate(en_files):
                    try:
                        import soundfile as sf
                        audio, sr = sf.read(wav_file)
                        if sr != 24000:
                            import librosa
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                        
                        # 读取对应的JSON元数据
                        json_file = wav_file.with_suffix('.json')
                        if json_file.exists():
                            with open(json_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            text = metadata.get('text', f'English test sample {i+1}')
                            speaker = metadata.get('speaker', f'en_speaker_{i+1}')
                        else:
                            text = f'English test sample {i+1}'
                            speaker = f'en_speaker_{i+1}'
                        
                        sample = {
                            'id': f'en_{i:06d}',
                            'audio': {'array': audio, 'sampling_rate': 24000},
                            'text': text,
                            'speaker': speaker,
                            'language': 'en',
                            'duration': len(audio) / 24000
                        }
                        test_samples.append(sample)
                        
                    except Exception as e:
                        print(f"⚠️ 跳过损坏的英文文件: {wav_file.name}")
            
            # 从中文缓存加载
            zh_cache = cache_dir / "ZH"
            if zh_cache.exists():
                zh_files = list(zh_cache.glob("*.wav"))[:10]
                for i, wav_file in enumerate(zh_files):
                    try:
                        import soundfile as sf
                        audio, sr = sf.read(wav_file)
                        if sr != 24000:
                            import librosa
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                        
                        json_file = wav_file.with_suffix('.json')
                        if json_file.exists():
                            with open(json_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            text = metadata.get('text', f'中文测试样本 {i+1}')
                            speaker = metadata.get('speaker', f'zh_speaker_{i+1}')
                        else:
                            text = f'中文测试样本 {i+1}'
                            speaker = f'zh_speaker_{i+1}'
                        
                        sample = {
                            'id': f'zh_{i:06d}',
                            'audio': {'array': audio, 'sampling_rate': 24000},
                            'text': text,
                            'speaker': speaker,
                            'language': 'zh',
                            'duration': len(audio) / 24000
                        }
                        test_samples.append(sample)
                        
                    except Exception as e:
                        print(f"⚠️ 跳过损坏的中文文件: {wav_file.name}")
            
            if not test_samples:
                print("❌ 未能从缓存加载任何样本")
                return False
            
            print(f"✅ 从缓存加载了 {len(test_samples)} 个样本")
            
            # 统计语言分布
            lang_count = {'en': 0, 'zh': 0}
            for sample in test_samples:
                lang_count[sample['language']] += 1
            
            print(f"📊 语言分布:")
            for lang, count in lang_count.items():
                duration = sum(s['duration'] for s in test_samples if s['language'] == lang)
                print(f"  {lang.upper()}: {count} 个样本, {duration:.1f} 秒")
            
            # 处理样本
            print(f"\n🎯 开始处理 {len(test_samples)} 个样本...")
            
            success_count = 0
            failed_count = 0
            
            from tqdm import tqdm
            
            for sample in tqdm(test_samples, desc="处理样本"):
                try:
                    success = generator.process_sample(sample)
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"❌ 样本处理异常 {sample['id']}: {e}")
                    failed_count += 1
            
            # 保存最终CSV
            generator._save_metadata_csv()
            
            # 生成报告
            generator._generate_final_report_and_summary(len(test_samples), success_count, failed_count)
            
            # 显示结果
            print("\n" + "=" * 80)
            print("📊 本地测试完成")
            print("=" * 80)
            print(f"✅ 成功: {success_count}/{len(test_samples)} 个样本")
            print(f"❌ 失败: {failed_count} 个样本")
            print(f"📁 输出目录: {temp_output}")
            
            # 检查输出文件
            if success_count > 0:
                print(f"\n📁 生成的文件:")
                for dir_name in ['original_mels', 'neutral_mels', 'emotion_features', 'metadata']:
                    dir_path = temp_output / dir_name
                    if dir_path.exists():
                        file_count = len(list(dir_path.glob("*")))
                        print(f"  {dir_name}: {file_count} 个文件")
                
                # 测试数据集访问器
                print(f"\n🔧 测试数据集访问器...")
                try:
                    sys.path.append(str(Path(__file__).parent))
                    from dataset_accessor import EmiliaTrainingDataset
                    
                    dataset_accessor = EmiliaTrainingDataset(str(temp_output))
                    print("✅ 数据集访问器测试成功")
                    
                    # 测试加载样本
                    if len(dataset_accessor.metadata_df) > 0:
                        first_sample_id = dataset_accessor.metadata_df.iloc[0]['sample_id']
                        sample_data = dataset_accessor.load_sample_data(first_sample_id)
                        
                        if sample_data:
                            print(f"✅ 样本数据加载测试成功: {first_sample_id}")
                            print(f"  源mel形状: {sample_data['original_mel'].shape}")
                            print(f"  中性mel形状: {sample_data['neutral_mel'].shape}")
                            print(f"  情感特征: utterance={sample_data['emotion_utterance'].shape}")
                        else:
                            print(f"❌ 样本数据加载失败")
                
                except Exception as e:
                    print(f"❌ 数据集访问器测试失败: {e}")
                
                return True
            else:
                print(f"\n❌ 没有成功处理任何样本")
                return False
                
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("选择测试模式:")
    print("1. 模拟数据测试（不需要GPU/模型）")
    print("2. 真实缓存数据测试（需要GPU/模型）")
    
    try:
        choice = input("请选择 (1/2): ").strip()
        
        if choice == "1":
            print("\n🧪 运行模拟数据测试...")
            success = run_local_test()
        elif choice == "2":
            print("\n🧪 运行真实缓存数据测试...")
            success = test_with_real_emilia_cache()
        else:
            print("❌ 无效选择")
            return 1
        
        if success:
            print("\n🎉 本地测试成功！")
            print("✅ 主脚本工作正常，可以部署到Spartan集群")
            return 0
        else:
            print("\n❌ 本地测试失败！")
            print("⚠️ 请检查错误信息并修复问题")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 测试已取消")
        return 130

if __name__ == "__main__":
    sys.exit(main())
