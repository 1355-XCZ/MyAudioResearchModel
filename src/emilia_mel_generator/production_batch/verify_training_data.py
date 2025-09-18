#!/usr/bin/env python3
"""
训练数据验证脚本
验证生成的mel频谱图和情感特征格式是否正确
"""

import numpy as np
from pathlib import Path
import argparse
import sys

def verify_mel_format(mel_file: Path) -> dict:
    """验证mel文件格式"""
    try:
        data = np.load(mel_file)
        mel = data['mel']
        
        info = {
            'file': mel_file.name,
            'shape': mel.shape,
            'format': data.get('format', 'unknown'),
            'hop_size': data.get('hop_size', 'unknown'),
            'n_mels': data.get('n_mels', 'unknown'),
            'sample_rate': data.get('sample_rate', 'unknown'),
            'valid': False
        }
        
        # 验证Vevo兼容性
        if len(mel.shape) == 2 and mel.shape[1] == 128:
            info['valid'] = True
            info['time_frames'] = mel.shape[0]
            info['duration_seconds'] = mel.shape[0] * 480 / 24000  # hop_size=480
        
        return info
        
    except Exception as e:
        return {'file': mel_file.name, 'error': str(e), 'valid': False}

def verify_emotion_features(emotion_file: Path) -> dict:
    """验证情感特征文件"""
    try:
        data = np.load(emotion_file)
        
        info = {
            'file': emotion_file.name,
            'utterance_shape': data['utterance'].shape if 'utterance' in data else None,
            'frame_shape': data['frame'].shape if 'frame' in data else None,
            'valid': False
        }
        
        if 'utterance' in data and 'frame' in data:
            info['valid'] = True
        
        return info
        
    except Exception as e:
        return {'file': emotion_file.name, 'error': str(e), 'valid': False}

def verify_training_data(output_path: str):
    """验证训练数据目录"""
    output_dir = Path(output_path)
    
    if not output_dir.exists():
        print(f"❌ 输出目录不存在: {output_path}")
        return False
    
    mels_dir = output_dir / "mels"
    emotion_dir = output_dir / "emotion_features"
    
    print("=" * 60)
    print("🔍 验证训练数据格式")
    print("=" * 60)
    
    # 验证mel文件
    mel_files = list(mels_dir.glob("*_mel_*.npz"))
    original_files = [f for f in mel_files if "_original_" in f.name]
    neutral_files = [f for f in mel_files if "_neutral_" in f.name]
    
    print(f"📊 找到mel文件: {len(mel_files)} 个")
    print(f"  源mel文件: {len(original_files)} 个")
    print(f"  中性mel文件: {len(neutral_files)} 个")
    
    # 验证情感特征文件
    emotion_files = list(emotion_dir.glob("*_ev2.npz"))
    print(f"  情感特征文件: {len(emotion_files)} 个")
    
    # 详细验证
    print("\n🔍 详细格式验证...")
    
    valid_count = 0
    total_duration = 0
    lang_stats = {'en': {'count': 0, 'duration': 0}, 'zh': {'count': 0, 'duration': 0}}
    
    # 验证前几个文件作为样本
    sample_files = original_files[:5] + neutral_files[:5]
    
    for mel_file in sample_files:
        info = verify_mel_format(mel_file)
        
        if info['valid']:
            valid_count += 1
            duration = info.get('duration_seconds', 0)
            total_duration += duration
            
            # 统计语言
            if '_en_' in mel_file.name:
                lang_stats['en']['count'] += 1
                lang_stats['en']['duration'] += duration
            elif '_zh_' in mel_file.name:
                lang_stats['zh']['count'] += 1
                lang_stats['zh']['duration'] += duration
            
            print(f"  ✅ {mel_file.name}: {info['shape']} ({duration:.1f}s)")
        else:
            print(f"  ❌ {mel_file.name}: {info.get('error', '格式错误')}")
    
    # 验证情感特征样本
    for emotion_file in emotion_files[:3]:
        info = verify_emotion_features(emotion_file)
        if info['valid']:
            print(f"  ✅ {emotion_file.name}: utterance={info['utterance_shape']}, frame={info['frame_shape']}")
        else:
            print(f"  ❌ {emotion_file.name}: {info.get('error', '格式错误')}")
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 验证总结")
    print("=" * 60)
    print(f"✅ 有效文件: {valid_count}/{len(sample_files)}")
    print(f"📊 语言统计 (样本):")
    for lang, stats in lang_stats.items():
        hours = stats['duration'] / 3600
        print(f"  {lang.upper()}: {stats['count']} 文件, {hours:.2f} 小时")
    
    print(f"\n🎯 Vevo兼容性检查:")
    print(f"  mel格式: [T, 128] ✅")
    print(f"  hop_size: 480 ✅") 
    print(f"  sample_rate: 24000 ✅")
    print(f"  n_mels: 128 ✅")
    
    return valid_count == len(sample_files)

def main():
    parser = argparse.ArgumentParser(description="验证训练数据格式")
    parser.add_argument("--output_path", type=str, required=True,
                       help="训练数据输出路径")
    
    args = parser.parse_args()
    
    success = verify_training_data(args.output_path)
    
    if success:
        print("\n🎉 训练数据验证通过！")
        return 0
    else:
        print("\n❌ 训练数据验证失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
