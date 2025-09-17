#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 MCSD_v1.xlsx 时间戳精确分割音频文件

用法:
    python split_audio_by_timestamps.py --season s5 --input-dir F:\vevo_data --output-dir F:\vevo_data_split

功能:
- 读取 MCSD_v1.xlsx 中的时间戳标注
- 使用 ffmpeg 按时间戳精确分割音频
- 按季节和集数组织输出目录结构
- 文件重命名为 utterance_id
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
except ImportError:
    print("[错误] 需要安装 pandas: pip install pandas openpyxl")
    sys.exit(1)


def parse_time_to_seconds(time_str: str) -> float:
    """将 HH:MM:SS 格式转换为秒数"""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        else:
            return float(parts[0])
    except (ValueError, IndexError):
        raise ValueError(f"无法解析时间格式: {time_str}")


def find_video_file(input_dir: str, item: str) -> str:
    """查找对应的视频文件"""
    # 可能的文件扩展名
    extensions = ['.webm', '.mp4', '.wav', '.m4a']
    
    # 在所有子目录中搜索
    input_path = Path(input_dir)
    
    # 方法1: 直接匹配文件名
    for ext in extensions:
        matches = list(input_path.rglob(f"{item}{ext}"))
        if matches:
            print(f"[找到] {item}{ext} -> {matches[0]}")
            return str(matches[0])
    
    # 方法2: 模糊匹配（文件名包含item）
    print(f"[搜索] 在 {input_dir} 中查找包含 '{item}' 的文件...")
    all_files = list(input_path.rglob("*"))
    audio_files = [f for f in all_files if f.suffix in extensions]
    
    # 优先匹配：文件名以item开头的
    exact_matches = [f for f in audio_files if f.stem.startswith(item)]
    if exact_matches:
        print(f"[找到] 精确匹配: {exact_matches[0]}")
        return str(exact_matches[0])
    
    # 次优匹配：文件名包含item的
    partial_matches = [f for f in audio_files if item in f.stem]
    if partial_matches:
        print(f"[找到] 部分匹配: {partial_matches[0]}")
        return str(partial_matches[0])
    
    # 调试信息
    print(f"[警告] 未找到文件: {item}")
    print(f"[调试] 搜索目录: {input_dir}")Demucs 音源分离
    print(f"[调试] 找到的音频文件数: {len(audio_files)}")
    
    # 显示前几个音频文件作为参考
    if audio_files:
        print("[参考] 目录中的音频文件样例:")
        for f in audio_files[:5]:
            print(f"  {f.name}")
    
    return None


def split_audio_segment(input_file: str, start_time: str, end_time: str, output_file: str) -> bool:
    """使用 ffmpeg 分割音频片段"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # ffmpeg 命令
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', start_time,
            '-to', end_time,
            '-c:a', 'pcm_s16le',  # WAV格式，16位PCM
            '-ar', '24000',       # 24kHz采样率
            '-ac', '1',           # 单声道
            '-y',                 # 覆盖输出文件
            output_file
        ]
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            return True
        else:
            print(f"[错误] ffmpeg 失败: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("[错误] 未找到 ffmpeg，请确保已安装并添加到 PATH")
        return False
    except Exception as e:
        print(f"[错误] 分割音频失败: {e}")
        return False


def process_season(mcsd_file: str, season: str, input_dir: str, output_dir: str) -> None:
    """处理指定季节的音频分割"""
    
    # 读取 MCSD 数据
    print(f"[读取] {mcsd_file}")
    df = pd.read_excel(mcsd_file, engine='openpyxl')
    
    # 过滤指定季节的数据
    season_pattern = f"tkxdh_{season}_"
    season_data = df[df['item'].str.startswith(season_pattern, na=False)]
    
    if season_data.empty:
        print(f"[错误] 未找到季节 {season} 的数据")
        return
    
    print(f"[统计] 季节 {season} 共有 {len(season_data)} 条记录")
    print(f"[统计] 涉及 {season_data['item'].nunique()} 个不同的视频文件")
    
    # 按 item 分组处理
    success_count = 0
    fail_count = 0
    
    for item, group in season_data.groupby('item'):
        print(f"\n[处理] {item} ({len(group)} 个片段)")
        
        # 查找对应的视频文件
        video_file = find_video_file(input_dir, item)
        if not video_file:
            print(f"[跳过] 未找到视频文件: {item}")
            fail_count += len(group)
            continue
        
        # 创建输出目录
        season_dir = os.path.join(output_dir, f"tkxdh_{season}")
        item_dir = os.path.join(season_dir, item)
        os.makedirs(item_dir, exist_ok=True)
        
        # 分割每个片段
        for _, row in group.iterrows():
            utterance_id = row['utterance_id']
            start_time = row['start_time']
            end_time = row['end_time']
            
            # 转换时间格式（处理pandas时间对象）
            if hasattr(start_time, 'strftime'):
                start_time_str = start_time.strftime('%H:%M:%S')
            else:
                start_time_str = str(start_time)
                
            if hasattr(end_time, 'strftime'):
                end_time_str = end_time.strftime('%H:%M:%S')
            else:
                end_time_str = str(end_time)
            
            output_file = os.path.join(item_dir, f"{utterance_id}.wav")
            
            print(f"  [{utterance_id}] {start_time_str} -> {end_time_str}")
            
            if split_audio_segment(video_file, start_time_str, end_time_str, output_file):
                success_count += 1
                # 验证输出文件
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f"    ✓ 成功: {output_file}")
                else:
                    print(f"    ✗ 输出文件异常: {output_file}")
                    fail_count += 1
            else:
                fail_count += 1
    
    print(f"\n[完成] 成功: {success_count}, 失败: {fail_count}")


def main():
    parser = argparse.ArgumentParser(
        description="基于时间戳分割音频文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理第5季
  python split_audio_by_timestamps.py --season s5 --input-dir F:\\vevo_data --output-dir F:\\vevo_data_split
  
  # 使用默认路径
  python split_audio_by_timestamps.py --season s5
        """
    )
    
    parser.add_argument(
        '--season', 
        type=str, 
        required=True,
        help='季节标识 (如: s5)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default="F:\\vevo_data",
        help='输入视频目录 (默认: F:\\vevo_data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="F:\\vevo_data_split",
        help='输出音频目录 (默认: F:\\vevo_data_split)'
    )
    parser.add_argument(
        '--mcsd-file',
        type=str,
        default=None,
        help='MCSD Excel 文件路径 (默认: 脚本同目录下的 MCSD_v1.xlsx)'
    )
    
    args = parser.parse_args()
    
    # 确定 MCSD 文件路径
    if args.mcsd_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.mcsd_file = os.path.join(script_dir, "MCSD_v1.xlsx")
    
    # 检查文件和目录
    if not os.path.exists(args.mcsd_file):
        print(f"[错误] MCSD 文件不存在: {args.mcsd_file}")
        sys.exit(1)
    
    if not os.path.exists(args.input_dir):
        print(f"[错误] 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[配置] 季节: {args.season}")
    print(f"[配置] 输入目录: {args.input_dir}")
    print(f"[配置] 输出目录: {args.output_dir}")
    print(f"[配置] MCSD 文件: {args.mcsd_file}")
    
    # 开始处理
    try:
        process_season(args.mcsd_file, args.season, args.input_dir, args.output_dir)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消操作")
    except Exception as e:
        print(f"[错误] 处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
