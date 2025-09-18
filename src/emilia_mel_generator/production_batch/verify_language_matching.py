#!/usr/bin/env python3
"""
语言匹配验证脚本
专门验证中文音频是否使用中文中性参考，英文音频是否使用英文中性参考
"""

import numpy as np
from pathlib import Path
import argparse
import sys
from collections import defaultdict

def verify_language_matching(output_path: str):
    """验证语言匹配情况"""
    output_dir = Path(output_path)
    mels_dir = output_dir / "mels"
    
    if not mels_dir.exists():
        print("❌ mels目录不存在")
        return False
    
    print("=" * 80)
    print("🌍 验证语言匹配 - 确保中文用中文中性参考，英文用英文中性参考")
    print("=" * 80)
    
    # 找到所有neutral mel文件
    neutral_files = list(mels_dir.glob("*_mel_neutral_vevo.npz"))
    
    if not neutral_files:
        print("❌ 未找到中性mel文件")
        return False
    
    print(f"📊 找到 {len(neutral_files)} 个中性mel文件")
    
    # 语言匹配统计
    lang_stats = defaultdict(lambda: {'total': 0, 'emo_emilia': 0, 'synthetic': 0, 'mismatched': 0})
    
    for neutral_file in neutral_files:
        try:
            # 从文件名推断语言
            if '_en_' in neutral_file.name:
                expected_lang = 'en'
            elif '_zh_' in neutral_file.name:
                expected_lang = 'zh'
            else:
                print(f"⚠️ 无法识别语言: {neutral_file.name}")
                continue
            
            # 加载文件检查元数据
            data = np.load(neutral_file)
            ref_source = data.get('neutral_reference_source', 'unknown')
            
            lang_stats[expected_lang]['total'] += 1
            
            if ref_source == 'emo_emilia':
                lang_stats[expected_lang]['emo_emilia'] += 1
            elif ref_source == 'synthetic':
                lang_stats[expected_lang]['synthetic'] += 1
            else:
                lang_stats[expected_lang]['mismatched'] += 1
                
        except Exception as e:
            print(f"❌ 检查文件失败: {neutral_file.name} - {e}")
    
    # 显示统计结果
    print(f"\n📊 语言匹配统计:")
    print(f"{'语言':<6} {'总数':<8} {'Emo-Emilia':<12} {'合成音频':<10} {'匹配率':<10}")
    print("-" * 50)
    
    overall_total = 0
    overall_emo_emilia = 0
    
    for lang in ['en', 'zh']:
        stats = lang_stats[lang]
        total = stats['total']
        emo_emilia = stats['emo_emilia']
        synthetic = stats['synthetic']
        
        if total > 0:
            match_rate = emo_emilia / total * 100
            print(f"{lang.upper():<6} {total:<8} {emo_emilia:<12} {synthetic:<10} {match_rate:.1f}%")
            
            overall_total += total
            overall_emo_emilia += emo_emilia
        else:
            print(f"{lang.upper():<6} 0        -            -          -")
    
    # 总体统计
    if overall_total > 0:
        overall_rate = overall_emo_emilia / overall_total * 100
        print("-" * 50)
        print(f"{'总计':<6} {overall_total:<8} {overall_emo_emilia:<12} {overall_total-overall_emo_emilia:<10} {overall_rate:.1f}%")
    
    # 质量评估
    print(f"\n🎯 质量评估:")
    
    if overall_rate >= 95:
        print("✅ 优秀: 几乎所有样本都使用了正确的语言匹配中性参考")
        quality = "excellent"
    elif overall_rate >= 80:
        print("✅ 良好: 大部分样本使用了正确的语言匹配中性参考")
        quality = "good"
    elif overall_rate >= 50:
        print("⚠️ 一般: 部分样本使用了语言匹配中性参考")
        quality = "fair"
    else:
        print("❌ 差: 很少样本使用了语言匹配中性参考")
        quality = "poor"
    
    # 详细分析
    print(f"\n🔍 详细分析:")
    for lang in ['en', 'zh']:
        stats = lang_stats[lang]
        if stats['total'] > 0:
            rate = stats['emo_emilia'] / stats['total'] * 100
            print(f"  {lang.upper()} 语言:")
            print(f"    使用 Emo-Emilia 中性参考: {stats['emo_emilia']}/{stats['total']} ({rate:.1f}%)")
            
            if rate >= 90:
                print(f"    ✅ {lang.upper()} 语言匹配质量优秀")
            elif rate >= 70:
                print(f"    ⚠️ {lang.upper()} 语言匹配质量一般")
            else:
                print(f"    ❌ {lang.upper()} 语言匹配质量差")
    
    # 建议
    print(f"\n💡 建议:")
    if quality in ["excellent", "good"]:
        print("✅ 语言匹配质量良好，可以用于训练")
    else:
        print("⚠️ 语言匹配质量需要改进:")
        print("  1. 检查 Emo-Emilia 数据集是否正确加载")
        print("  2. 确认数据集中包含足够的中英文neutral样本")
        print("  3. 检查网络连接和数据下载")
    
    return quality in ["excellent", "good"]

def main():
    parser = argparse.ArgumentParser(description="验证语言匹配")
    parser.add_argument("--output_path", type=str, required=True,
                       help="训练数据输出路径")
    
    args = parser.parse_args()
    
    success = verify_language_matching(args.output_path)
    
    if success:
        print("\n🎉 语言匹配验证通过！")
        return 0
    else:
        print("\n❌ 语言匹配验证失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
