#!/usr/bin/env python3
"""
检查音频文件长度并估算所需的上下文编码长度
"""

import os
import librosa
import numpy as np
from pathlib import Path
import argparse


def estimate_tokens(duration_seconds):
    """
    估算音频长度对应的 token 数量
    基于 Vevo 配置：采样率16000，hop_size=320
    """
    # 帧率 = 采样率 / hop_size = 16000 / 320 = 50 帧/秒
    frames_per_second = 50
    tokens = duration_seconds * frames_per_second
    return int(tokens)


def analyze_audio_directory(audio_dir):
    """分析音频目录中所有 WAV 文件的长度分布"""
    audio_dir = Path(audio_dir)
    wav_files = list(audio_dir.rglob("*.wav"))
    
    if not wav_files:
        print(f"❌ 在 {audio_dir} 中未找到 WAV 文件")
        return
    
    print(f"📂 分析目录: {audio_dir}")
    print(f"📊 找到 {len(wav_files)} 个 WAV 文件")
    print()
    
    durations = []
    token_counts = []
    
    print("🔍 分析前10个文件的详细信息:")
    for i, wav_file in enumerate(wav_files[:10]):
        try:
            # 使用 librosa 获取音频长度（不加载音频数据，速度快）
            duration = librosa.get_duration(path=str(wav_file))
            estimated_tokens = estimate_tokens(duration)
            
            durations.append(duration)
            token_counts.append(estimated_tokens)
            
            rel_path = wav_file.relative_to(audio_dir)
            print(f"  {i+1:2d}. {rel_path}: {duration:.2f}s → ~{estimated_tokens} tokens")
            
        except Exception as e:
            print(f"  ❌ 无法分析 {wav_file}: {e}")
    
    if len(wav_files) > 10:
        print(f"\n⏱️  分析剩余 {len(wav_files)-10} 个文件...")
        for wav_file in wav_files[10:]:
            try:
                duration = librosa.get_duration(path=str(wav_file))
                estimated_tokens = estimate_tokens(duration)
                durations.append(duration)
                token_counts.append(estimated_tokens)
            except Exception as e:
                print(f"  ❌ 跳过 {wav_file}: {e}")
    
    if not durations:
        print("❌ 没有成功分析的音频文件")
        return
    
    # 统计分析
    durations = np.array(durations)
    token_counts = np.array(token_counts)
    
    print("\n" + "="*60)
    print("📈 统计分析:")
    print("="*60)
    
    print(f"📊 音频长度统计:")
    print(f"  • 最短: {durations.min():.2f}s")
    print(f"  • 最长: {durations.max():.2f}s") 
    print(f"  • 平均: {durations.mean():.2f}s")
    print(f"  • 中位数: {np.median(durations):.2f}s")
    print(f"  • 95%分位数: {np.percentile(durations, 95):.2f}s")
    
    print(f"\n🎯 Token 数量统计:")
    print(f"  • 最少: ~{token_counts.min()} tokens")
    print(f"  • 最多: ~{token_counts.max()} tokens")
    print(f"  • 平均: ~{token_counts.mean():.0f} tokens")
    print(f"  • 中位数: ~{np.median(token_counts):.0f} tokens")
    print(f"  • 95%分位数: ~{np.percentile(token_counts, 95):.0f} tokens")
    
    # 推荐上下文长度
    print(f"\n💡 推荐的上下文设置:")
    max_tokens = token_counts.max()
    p95_tokens = int(np.percentile(token_counts, 95))
    
    if max_tokens <= 1000:
        rec_context = 1600
        rec_pos_emb = 2048
    elif max_tokens <= 1600:
        rec_context = 2000
        rec_pos_emb = 4096
    elif max_tokens <= 2000:
        rec_context = 2400
        rec_pos_emb = 4096
    else:
        rec_context = max_tokens + 400
        rec_pos_emb = max(4096, max_tokens + 1000)
    
    print(f"  • 推荐 max_context: {rec_context}")
    print(f"  • 推荐 max_position_embeddings: {rec_pos_emb}")
    print(f"  • 理由: 最长音频需要 ~{max_tokens} tokens")
    
    # 长度分布
    print(f"\n📊 长度分布:")
    bins = [0, 3, 5, 10, 15, 20, 30, float('inf')]
    labels = ['<3s', '3-5s', '5-10s', '10-15s', '15-20s', '20-30s', '>30s']
    
    for i in range(len(bins)-1):
        count = np.sum((durations >= bins[i]) & (durations < bins[i+1]))
        percentage = count / len(durations) * 100
        print(f"  • {labels[i]:>6s}: {count:3d} 文件 ({percentage:4.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="分析音频文件长度并估算上下文需求")
    parser.add_argument("audio_dir", help="音频文件目录路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_dir):
        print(f"❌ 目录不存在: {args.audio_dir}")
        return
    
    analyze_audio_directory(args.audio_dir)


if __name__ == "__main__":
    main()
