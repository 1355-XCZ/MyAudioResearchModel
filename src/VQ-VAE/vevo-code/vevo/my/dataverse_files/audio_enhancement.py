#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频增强脚本：降噪 + 音源分离 + VAD

用法:
    python audio_enhancement.py --input-dir F:\vevo_data_split\tkxdh_s5 --output-dir F:\vevo_data_enhanced

功能:
- 使用 Demucs 分离人声
- 使用 noisereduce 降噪
- 可选：VAD 静音检测
- 批量处理所有 WAV 文件
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import soundfile as sf
    import numpy as np
    import noisereduce as nr
    from scipy.signal import wiener
except ImportError as e:
    print(f"[错误] 缺少依赖包: {e}")
    print("请安装: pip install librosa soundfile noisereduce scipy")
    sys.exit(1)

# 可选依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def check_demucs_available() -> bool:
    """检查 Demucs 是否可用"""
    try:
        result = subprocess.run(['python', '-c', 'import demucs'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def separate_vocals_demucs(audio_file: str, output_dir: str) -> Optional[str]:
    """使用 Demucs 分离人声"""
    if not check_demucs_available():
        print(f"[警告] Demucs 不可用，跳过音源分离: {os.path.basename(audio_file)}")
        return None
    
    try:
        # 使用 Demucs 分离
        cmd = [
            'python', '-m', 'demucs.separate', 
            '--two-stems=vocals',  # 只分离人声和伴奏
            '--out', output_dir,
            audio_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[错误] Demucs 分离失败: {result.stderr}")
            return None
            
        # 查找分离后的人声文件
        base_name = Path(audio_file).stem
        vocals_file = os.path.join(output_dir, 'htdemucs', base_name, 'vocals.wav')
        
        if os.path.exists(vocals_file):
            return vocals_file
        else:
            print(f"[警告] 未找到分离后的人声文件: {vocals_file}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"[错误] Demucs 处理超时: {os.path.basename(audio_file)}")
        return None
    except Exception as e:
        print(f"[错误] Demucs 处理异常: {e}")
        return None


def denoise_audio(audio_file: str, output_file: str, method: str = 'noisereduce') -> bool:
    """音频降噪"""
    try:
        # 加载音频
        audio, sr = librosa.load(audio_file, sr=None)
        
        if method == 'noisereduce':
            # 使用 noisereduce
            # 使用音频前0.5秒作为噪声样本
            noise_sample_length = min(int(0.5 * sr), len(audio) // 4)
            noise_sample = audio[:noise_sample_length]
            
            # 降噪处理
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=sr,
                y_noise=noise_sample,
                prop_decrease=0.8,  # 噪声抑制程度
                stationary=False    # 处理非稳态噪声
            )
            
        elif method == 'wiener':
            # 使用 Wiener 滤波
            reduced_noise = wiener(audio, noise=0.1)
            
        else:
            print(f"[警告] 未知降噪方法: {method}，跳过降噪")
            reduced_noise = audio
        
        # 音量归一化
        if np.max(np.abs(reduced_noise)) > 0:
            reduced_noise = reduced_noise / np.max(np.abs(reduced_noise)) * 0.95
        
        # 保存处理后的音频
        sf.write(output_file, reduced_noise, sr)
        return True
        
    except Exception as e:
        print(f"[错误] 降噪处理失败: {e}")
        return False


def apply_vad_silence(audio_file: str, output_file: str, threshold: float = 0.02) -> bool:
    """应用VAD，静音检测并减弱非语音段"""
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        
        # 计算短时能量
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms
        
        # 计算RMS能量
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 检测语音活动
        voice_activity = rms > threshold
        
        # 扩展到音频长度
        voice_mask = np.repeat(voice_activity, hop_length)[:len(audio)]
        
        # 对非语音段进行衰减而不是完全静音
        enhanced_audio = audio.copy()
        enhanced_audio[~voice_mask] *= 0.1  # 非语音段衰减到10%
        
        # 保存结果
        sf.write(output_file, enhanced_audio, sr)
        return True
        
    except Exception as e:
        print(f"[错误] VAD处理失败: {e}")
        return False


def process_audio_file(input_file: str, output_file: str, 
                      use_demucs: bool = True, 
                      use_denoise: bool = True,
                      use_vad: bool = False,
                      temp_dir: str = None) -> bool:
    """处理单个音频文件"""
    
    print(f"[处理] {os.path.basename(input_file)}")
    
    current_file = input_file
    
    # 步骤1: Demucs 音源分离
    if use_demucs and TORCH_AVAILABLE:
        print(f"  -> 音源分离...")
        vocals_file = separate_vocals_demucs(current_file, temp_dir)
        if vocals_file:
            current_file = vocals_file
            print(f"  ✓ 人声分离完成")
        else:
            print(f"  ! 音源分离失败，使用原始音频")
    
    # 步骤2: 降噪处理
    if use_denoise:
        print(f"  -> 降噪处理...")
        temp_denoised = os.path.join(temp_dir, f"denoised_{os.path.basename(input_file)}")
        if denoise_audio(current_file, temp_denoised, method='noisereduce'):
            current_file = temp_denoised
            print(f"  ✓ 降噪完成")
        else:
            print(f"  ! 降噪失败")
    
    # 步骤3: VAD处理
    if use_vad:
        print(f"  -> VAD处理...")
        temp_vad = os.path.join(temp_dir, f"vad_{os.path.basename(input_file)}")
        if apply_vad_silence(current_file, temp_vad):
            current_file = temp_vad
            print(f"  ✓ VAD完成")
        else:
            print(f"  ! VAD失败")
    
    # 最终复制到输出位置
    try:
        if current_file != output_file:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if current_file == input_file:
                # 没有处理，直接复制
                import shutil
                shutil.copy2(current_file, output_file)
            else:
                # 已处理，移动文件
                import shutil
                shutil.move(current_file, output_file)
        
        print(f"  ✓ 完成: {os.path.basename(output_file)}")
        return True
        
    except Exception as e:
        print(f"  ✗ 保存失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="音频增强：降噪+音源分离+VAD")
    parser.add_argument("--input-dir", required=True, help="输入目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--no-demucs", action="store_true", help="跳过 Demucs 音源分离")
    parser.add_argument("--no-denoise", action="store_true", help="跳过降噪处理")
    parser.add_argument("--use-vad", action="store_true", help="启用 VAD 处理")
    parser.add_argument("--temp-dir", help="临时文件目录（默认系统临时目录）")
    parser.add_argument("--limit", type=int, help="限制处理文件数（测试用）")
    
    args = parser.parse_args()
    
    # 设置临时目录
    if args.temp_dir:
        temp_dir = args.temp_dir
    else:
        import tempfile
        temp_dir = tempfile.mkdtemp()
    
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"[配置] 输入: {args.input_dir}")
    print(f"[配置] 输出: {args.output_dir}")
    print(f"[配置] 临时: {temp_dir}")
    print(f"[配置] Demucs: {not args.no_demucs and TORCH_AVAILABLE}")
    print(f"[配置] 降噪: {not args.no_denoise}")
    print(f"[配置] VAD: {args.use_vad}")
    
    # 查找所有 WAV 文件
    input_path = Path(args.input_dir)
    wav_files = list(input_path.rglob("*.wav"))
    
    if args.limit:
        wav_files = wav_files[:args.limit]
    
    print(f"[发现] {len(wav_files)} 个 WAV 文件")
    
    # 处理统计
    success_count = 0
    fail_count = 0
    
    for wav_file in wav_files:
        # 计算相对路径和输出路径
        rel_path = wav_file.relative_to(input_path)
        output_file = Path(args.output_dir) / rel_path
        
        # 处理文件
        if process_audio_file(
            str(wav_file), 
            str(output_file),
            use_demucs=not args.no_demucs,
            use_denoise=not args.no_denoise,
            use_vad=args.use_vad,
            temp_dir=temp_dir
        ):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n[完成] 成功: {success_count}, 失败: {fail_count}")
    
    # 清理临时文件
    if not args.temp_dir:  # 只清理自动创建的临时目录
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"[清理] 临时目录已删除: {temp_dir}")
        except:
            pass


if __name__ == "__main__":
    main()
