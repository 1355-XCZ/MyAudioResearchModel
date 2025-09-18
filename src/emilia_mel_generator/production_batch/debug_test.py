#!/usr/bin/env python3
"""
调试测试脚本 - 逐步检查导入和初始化问题
"""

import sys
import os
from pathlib import Path

def test_step_by_step():
    """逐步测试每个组件"""
    print("🔍 开始逐步调试测试...")
    
    # Step 1: 检查基本导入
    print("\n📋 Step 1: 检查基本导入...")
    try:
        import torch
        import numpy as np
        import soundfile as sf
        print("✅ 基本库导入成功")
        print(f"  Torch版本: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ 基本库导入失败: {e}")
        return False
    
    # Step 2: 检查项目路径
    print("\n📋 Step 2: 检查项目路径...")
    try:
        current_dir = Path.cwd()
        project_root = current_dir.parent.parent.parent
        emilia_dir = current_dir.parent
        
        print(f"  当前目录: {current_dir}")
        print(f"  项目根目录: {project_root}")
        print(f"  Emilia目录: {emilia_dir}")
        
        # 检查关键文件
        corrected_generator = emilia_dir / "corrected_data_generator.py"
        amphion_dir = project_root / "src" / "Amphion"
        
        print(f"  corrected_data_generator.py存在: {corrected_generator.exists()}")
        print(f"  Amphion目录存在: {amphion_dir.exists()}")
        
        if not corrected_generator.exists():
            print("❌ 缺少corrected_data_generator.py")
            return False
            
    except Exception as e:
        print(f"❌ 路径检查失败: {e}")
        return False
    
    # Step 3: 测试corrected_data_generator导入
    print("\n📋 Step 3: 测试corrected_data_generator导入...")
    try:
        # 添加正确的路径
        emilia_generator_path = project_root / "src" / "emilia_mel_generator"
        sys.path.append(str(emilia_generator_path))
        
        print(f"  添加路径: {emilia_generator_path}")
        
        from corrected_data_generator import Emotion2VecExtractor
        print("✅ Emotion2VecExtractor导入成功")
        
        # 尝试初始化（使用CPU）
        extractor = Emotion2VecExtractor(device='cpu')
        print("✅ Emotion2VecExtractor初始化成功")
        
    except Exception as e:
        print(f"❌ Emotion2VecExtractor失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: 测试Vevo TTS导入
    print("\n📋 Step 4: 测试Vevo TTS导入...")
    try:
        amphion_path = project_root / "src" / "Amphion"
        sys.path.insert(0, str(amphion_path))
        
        # 清理模块缓存
        to_delete = [k for k in list(sys.modules.keys()) 
                    if k in ['models', 'utils'] or k.startswith(('models.', 'utils.'))]
        for k in to_delete:
            if k in sys.modules:
                del sys.modules[k]
        
        # 切换到Amphion目录并调整sys.path
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        os.chdir(str(amphion_path))
        sys.path.clear()
        sys.path.extend([str(amphion_path)] + original_path)
        
        # 确保utils包可用
        utils_path = amphion_path / "utils"
        if not (utils_path / "__init__.py").exists():
            with open(utils_path / "__init__.py", 'w') as f:
                f.write("# Amphion utils package\n")
        
        # 先测试utils导入
        import utils
        from utils.util import load_config
        print("  ✅ utils.util导入成功")
        
        # 再测试Vevo TTS导入
        from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio, g2p_
        print("✅ Vevo TTS导入成功")
        
        # 恢复原始环境
        os.chdir(original_cwd)
        sys.path.clear()
        sys.path.extend(original_path)
        
    except Exception as e:
        print(f"❌ Vevo TTS导入失败: {e}")
        import traceback
        traceback.print_exc()
        # 恢复目录
        try:
            os.chdir(original_cwd)
        except:
            pass
        return False
    
    # Step 5: 测试Emo-Emilia数据集加载
    print("\n📋 Step 5: 测试Emo-Emilia数据集加载...")
    try:
        from datasets import load_dataset
        
        print("⬇️ 尝试加载ASLP-lab/Emo-Emilia...")
        
        # 按照官方示例加载，不使用trust_remote_code
        emo_dataset = load_dataset("ASLP-lab/Emo-Emilia")
        
        print("✅ Emo-Emilia数据集加载成功")
        print(f"  数据集结构: {list(emo_dataset.keys())}")
        
        # 检查neutral样本
        neutral_count = 0
        lang_count = {'en': 0, 'zh': 0, 'other': 0}
        
        for item in emo_dataset['train'][:100]:  # 只检查前100个
            emotion = item.get('emotion', '').lower()
            language = item.get('language', 'unknown').lower()
            
            if emotion == 'neutral':
                neutral_count += 1
                if language in lang_count:
                    lang_count[language] += 1
                else:
                    lang_count['other'] += 1
        
        print(f"  前100个样本中的neutral: {neutral_count} 个")
        print(f"  语言分布: {dict(lang_count)}")
        
        if neutral_count == 0:
            print("⚠️ 警告: 前100个样本中没有neutral标签")
        
    except Exception as e:
        print(f"❌ Emo-Emilia加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: 测试现有缓存数据
    print("\n📋 Step 6: 测试现有缓存数据...")
    try:
        cache_dir = Path("../emilia_cache")
        
        if cache_dir.exists():
            en_cache = cache_dir / "EN"
            zh_cache = cache_dir / "ZH"
            
            en_files = list(en_cache.glob("*.wav")) if en_cache.exists() else []
            zh_files = list(zh_cache.glob("*.wav")) if zh_cache.exists() else []
            
            print(f"  英文缓存文件: {len(en_files)} 个")
            print(f"  中文缓存文件: {len(zh_files)} 个")
            
            if len(en_files) >= 10 and len(zh_files) >= 10:
                print("✅ 缓存数据足够进行测试")
                
                # 测试加载一个文件
                test_file = en_files[0]
                audio, sr = sf.read(test_file)
                print(f"  测试文件: {test_file.name}, 长度: {len(audio)/sr:.1f}s, 采样率: {sr}")
                
            else:
                print("⚠️ 缓存数据不足，需要先运行完整测试生成缓存")
                return False
        else:
            print("❌ 缓存目录不存在")
            return False
            
    except Exception as e:
        print(f"❌ 缓存数据检查失败: {e}")
        return False
    
    print("\n🎉 所有组件检查通过！")
    return True

def run_minimal_test():
    """运行最小化测试"""
    print("\n" + "=" * 60)
    print("🧪 运行最小化测试")
    print("=" * 60)
    
    try:
        # 检查是否通过了前置检查
        if not test_step_by_step():
            print("❌ 前置检查失败，无法继续测试")
            return False
        
        print("\n🚀 开始最小化功能测试...")
        
        # 导入主要组件
        sys.path.append(str(Path.cwd().parent.parent.parent / "src"))
        from corrected_data_generator import Emotion2VecExtractor
        
        # 创建临时目录
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            print(f"📁 临时测试目录: {temp_path}")
            
            # 创建输出目录结构
            output_dirs = {
                'original_mels': temp_path / "original_mels",
                'neutral_mels': temp_path / "neutral_mels", 
                'emotion_features': temp_path / "emotion_features",
                'metadata': temp_path / "metadata"
            }
            
            for dir_path in output_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            
            print("✅ 输出目录创建成功")
            
            # 测试Emotion2Vec
            print("\n🧪 测试Emotion2Vec...")
            try:
                extractor = Emotion2VecExtractor(device='cpu')
                
                # 创建测试音频
                test_audio = np.random.randn(24000 * 2).astype(np.float32)  # 2秒
                test_tensor = torch.from_numpy(test_audio).unsqueeze(0)
                
                features = extractor.extract_features(test_tensor)
                print(f"✅ Emotion2Vec测试成功")
                print(f"  utterance形状: {features['utterance'].shape}")
                print(f"  frame形状: {features['frame'].shape}")
                
            except Exception as e:
                print(f"❌ Emotion2Vec测试失败: {e}")
                return False
            
            # 测试从现有缓存加载样本
            print("\n🧪 测试缓存样本加载...")
            try:
                cache_dir = Path("../emilia_cache")
                
                # 加载英文样本
                en_cache = cache_dir / "EN"
                en_files = list(en_cache.glob("*.wav"))[:2]  # 只测试2个
                
                for wav_file in en_files:
                    try:
                        audio, sr = sf.read(wav_file)
                        if sr != 24000:
                            import librosa
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                        
                        print(f"  ✅ 英文样本: {wav_file.name}, {len(audio)/24000:.1f}s")
                        
                        # 测试emotion2vec提取
                        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                        features = extractor.extract_features(audio_tensor)
                        
                        # 模拟保存
                        sample_id = f"test_en_{wav_file.stem}"
                        
                        np.savez_compressed(
                            output_dirs['emotion_features'] / f"{sample_id}_emotion_features.npz",
                            utterance=features['utterance'].cpu().numpy(),
                            frame=features['frame'].cpu().numpy(),
                            language='en',
                            source_type='original_audio_emotion'
                        )
                        
                        print(f"    ✅ 情感特征保存成功")
                        
                    except Exception as e:
                        print(f"    ❌ 处理失败: {wav_file.name} - {e}")
                
                # 加载中文样本
                zh_cache = cache_dir / "ZH" 
                zh_files = list(zh_cache.glob("*.wav"))[:2]  # 只测试2个
                
                for wav_file in zh_files:
                    try:
                        audio, sr = sf.read(wav_file)
                        if sr != 24000:
                            import librosa
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                        
                        print(f"  ✅ 中文样本: {wav_file.name}, {len(audio)/24000:.1f}s")
                        
                        # 测试emotion2vec提取
                        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                        features = extractor.extract_features(audio_tensor)
                        
                        # 模拟保存
                        sample_id = f"test_zh_{wav_file.stem}"
                        
                        np.savez_compressed(
                            output_dirs['emotion_features'] / f"{sample_id}_emotion_features.npz",
                            utterance=features['utterance'].cpu().numpy(),
                            frame=features['frame'].cpu().numpy(),
                            language='zh',
                            source_type='original_audio_emotion'
                        )
                        
                        print(f"    ✅ 情感特征保存成功")
                        
                    except Exception as e:
                        print(f"    ❌ 处理失败: {wav_file.name} - {e}")
                
            except Exception as e:
                print(f"❌ 缓存样本测试失败: {e}")
                return False
            
            # 检查生成的文件
            emotion_files = list(output_dirs['emotion_features'].glob("*.npz"))
            print(f"\n📊 生成了 {len(emotion_files)} 个情感特征文件")
            
            if emotion_files:
                # 验证文件内容
                test_file = emotion_files[0]
                data = np.load(test_file)
                print(f"✅ 文件验证成功: {test_file.name}")
                print(f"  utterance: {data['utterance'].shape}")
                print(f"  frame: {data['frame'].shape}")
                print(f"  language: {data.get('language', 'unknown')}")
                
                return True
            else:
                print("❌ 没有生成任何文件")
                return False
                
    except Exception as e:
        print(f"❌ 最小化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔍 调试测试 - 查找主脚本失败原因")
    
    success = run_minimal_test()
    
    if success:
        print("\n🎉 调试测试成功！")
        print("✅ 基本组件工作正常")
        print("💡 现在可以尝试运行完整的主脚本测试")
        return 0
    else:
        print("\n❌ 调试测试失败！")
        print("💡 请根据上述错误信息修复问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
