#!/usr/bin/env python3
"""
文件名: setup_mustard_data_local.py
用途  : 在本地下载并处理 MUStARD 数据集音频，准备英文上下文长度实验
"""

import os
import sys
import zipfile
import subprocess
import urllib.request
from pathlib import Path
from tqdm import tqdm
import argparse

def download_file(url: str, output_path: str):
    """下载文件并显示进度"""
    print(f"[下载] 正在下载到: {output_path}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\r[下载] 进度: {percent}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="")
    
    urllib.request.urlretrieve(url, output_path, progress_hook)
    print(f"\n[下载] 完成: {output_path}")

def extract_zip(zip_path: str, extract_to: str):
    """解压 ZIP 文件"""
    print(f"[解压] 正在解压: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[解压] 完成: {extract_to}")

def convert_audio(input_file: str, output_file: str) -> bool:
    """使用 FFmpeg 转换音频格式"""
    try:
        # 16kHz, 单声道, 16-bit PCM (与 Vevo 配置一致)
        cmd = [
            'ffmpeg', '-i', input_file,
            '-ar', '16000',
            '-ac', '1', 
            '-c:a', 'pcm_s16le',
            output_file, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"[错误] 转换失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="MUStARD 数据集本地处理")
    parser.add_argument("--output-dir", default="F:/MUStARD_data", help="输出目录")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载步骤")
    parser.add_argument("--skip-extract", action="store_true", help="跳过解压步骤")
    args = parser.parse_args()
    
    # 路径设置
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    audio_output_dir = output_dir / "audio_processed"
    zip_path = data_dir / "mmsd_raw_data.zip"
    
    print(f"[开始] MUStARD 数据集本地处理")
    print(f"[输出] 目录: {output_dir}")
    
    # 创建目录
    data_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载数据
    if not args.skip_download and not zip_path.exists():
        url = "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"
        download_file(url, str(zip_path))
    elif zip_path.exists():
        print(f"[跳过] 数据已存在: {zip_path}")
    
    # 解压数据
    context_dir = data_dir / "context_final"
    utterances_dir = data_dir / "utterances_final"
    
    if not args.skip_extract and (not context_dir.exists() or not utterances_dir.exists()):
        extract_zip(str(zip_path), str(data_dir))
    else:
        print(f"[跳过] 数据已解压")
    
    # 统计原始音频文件
    audio_files = []
    for subdir in ["context_final", "utterances_final"]:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            for ext in ["*.mp4", "*.wav", "*.mp3"]:
                audio_files.extend(list(subdir_path.glob(ext)))
    
    print(f"[统计] 找到 {len(audio_files)} 个音频/视频文件")
    
    if len(audio_files) == 0:
        print("[错误] 未找到音频文件")
        return
    
    # 音频格式转换
    print(f"[处理] 开始音频格式标准化...")
    
    success_count = 0
    fail_count = 0
    
    for audio_file in tqdm(audio_files, desc="转换音频"):
        # 确定子目录
        if "context_final" in str(audio_file):
            subdir = "context_final"
        elif "utterances_final" in str(audio_file):
            subdir = "utterances_final"
        else:
            continue
        
        # 输出路径
        output_subdir = audio_output_dir / subdir
        output_subdir.mkdir(exist_ok=True)
        
        name_without_ext = audio_file.stem
        output_file = output_subdir / f"{name_without_ext}.wav"
        
        # 如果已存在，跳过
        if output_file.exists():
            continue
        
        # 转换音频
        if convert_audio(str(audio_file), str(output_file)):
            success_count += 1
        else:
            fail_count += 1
            print(f"[转换] ✗ 失败: {subdir}/{audio_file.name}")
    
    # 最终统计
    processed_files = list(audio_output_dir.rglob("*.wav"))
    print(f"\n[完成] 处理完成")
    print(f"[统计] 成功: {success_count}, 失败: {fail_count}")
    print(f"[结果] 生成 {len(processed_files)} 个标准化音频文件")
    print(f"[路径] 处理后的音频位于: {audio_output_dir}")
    
    # 创建文件列表
    file_list_path = output_dir / "audio_file_list.txt"
    with open(file_list_path, 'w') as f:
        for wav_file in sorted(processed_files):
            f.write(str(wav_file) + '\n')
    
    print(f"[列表] 音频文件列表保存到: {file_list_path}")
    
    # 显示前10个文件作为示例
    print(f"[示例] 前10个音频文件:")
    for i, wav_file in enumerate(sorted(processed_files)[:10], 1):
        print(f"  {i:2d}. {wav_file.name}")

if __name__ == "__main__":
    main()
