#!/usr/bin/env python3
"""
文件名: batch_infer_vevostyle_resume.py
用途  : Vevo 批量推理脚本 - 支持断点续传，跳过已存在的文件
基于  : batch_infer_vevostyle.py
改进  : 添加文件存在检查，支持断点续传
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import torch
import torchaudio

from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio


def build_pipeline(device: torch.device, ar_cfg_path: str = None) -> VevoInferencePipeline:
    """构建 Vevo 推理管道"""
    print("[init] building Vevo pipeline (first-time will download from HF cache)...")
    
    local_dir = "./ckpt/Vevo"
    
    content_tokenizer_ckpt_path = os.path.join(local_dir, "content_tokenizer")
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "content_style_tokenizer")
    
    if ar_cfg_path is None:
        ar_cfg_path = "./models/vc/vevo/config/Vq32ToVq8192.json"  # 默认配置
    
    ar_ckpt_path = os.path.join(local_dir, "acoustic_modeling/AR")
    
    fmt_cfg_path = "./models/vc/vevo/config/FlowMatchingTransformer.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/FlowMatching")
    
    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    pipeline = VevoInferencePipeline(
        content_tokenizer_ckpt_path=content_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )
    return pipeline


def collect_style_wavs(style_dir: Path) -> List[Path]:
    return sorted(list(style_dir.rglob("*.wav")))


def ffmpeg_trim_wav(input_path: str, output_path: str, max_seconds: float) -> bool:
    """用 ffmpeg 裁剪音频到前 max_seconds 秒，输出为 24kHz/单声道 WAV。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-t",
        str(max_seconds),
        "-ar",
        "24000",
        "-ac",
        "1",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Vevo 批量推理 - 支持断点续传")
    parser.add_argument("--content-wav", required=True, help="内容音频文件路径")
    parser.add_argument("--style-dir", required=True, help="风格参考音频目录")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--limit", type=int, help="限制处理文件数量（用于测试）")
    parser.add_argument("--style-max-seconds", type=float, default=10.0, 
                       help="风格音频最大秒数（避免上下文长度错误）")
    parser.add_argument("--content-max-seconds", type=float, default=8.0,
                       help="内容音频最大秒数")
    parser.add_argument("--config", type=str, default=None, 
                       help="AR 模型配置文件路径（用于上下文长度研究）")
    parser.add_argument("--force-overwrite", action="store_true",
                       help="强制覆盖已存在的文件")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    pipeline = build_pipeline(device, ar_cfg_path=args.config)

    style_root = Path(args.style_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    style_wavs = collect_style_wavs(style_root)
    if args.limit:
        style_wavs = style_wavs[:args.limit]

    print(f"[data] found {len(style_wavs)} style wavs under {style_root}")

    # 统计已存在和需要处理的文件
    existing_count = 0
    to_process = []
    
    for style_wav in style_wavs:
        rel = style_wav.relative_to(style_root)
        out_path = out_root / rel.parent / (style_wav.stem + "_vevostyle.wav")
        
        if out_path.exists() and not args.force_overwrite:
            existing_count += 1
            if existing_count <= 5:  # 显示前5个跳过的文件
                print(f"[skip] {out_path.name} already exists")
        else:
            to_process.append((style_wav, out_path))
    
    print(f"[resume] {existing_count} files already exist (skipping)")
    print(f"[resume] {len(to_process)} files to process")
    
    if len(to_process) == 0:
        print("[done] all files already processed!")
        return

    # 处理内容音频裁剪
    content_used = args.content_wav
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        
        if args.content_max_seconds and args.content_max_seconds > 0:
            trimmed_content = tmp_root / "content_trim.wav"
            if ffmpeg_trim_wav(args.content_wav, str(trimmed_content), args.content_max_seconds):
                content_used = str(trimmed_content)
                print(f"[trim] content trimmed to {args.content_max_seconds}s: {trimmed_content}")

        # 处理每个需要处理的文件
        for idx, (style_wav, out_path) in enumerate(to_process, 1):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 显示总体进度（包括跳过的文件）
            total_idx = existing_count + idx
            total_count = len(style_wavs)
            print(f"[{total_idx}/{total_count}] style={style_wav} -> {out_path}")

            # 裁剪 style，避免过长导致 AR 生成长度校验失败
            rel = style_wav.relative_to(style_root)
            trimmed_style = tmp_root / (rel.parent.as_posix().replace('/', '_') + "_" + style_wav.stem + "_trim.wav")
            trimmed_style.parent.mkdir(parents=True, exist_ok=True)
            style_used = str(style_wav)
            if args.style_max_seconds and args.style_max_seconds > 0:
                if ffmpeg_trim_wav(str(style_wav), str(trimmed_style), args.style_max_seconds):
                    style_used = str(trimmed_style)

            try:
                gen_audio = pipeline.inference_ar_and_fm(
                    src_wav_path=content_used,
                    src_text=None,
                    style_ref_wav_path=style_used,
                    timbre_ref_wav_path=content_used,
                )
                save_audio(gen_audio, output_path=str(out_path))
                print(f"[success] generated: {out_path.name}")
            except Exception as e:
                print(f"[error] failed on {style_wav}: {e}")
                # 跳过该条，继续下一个
                continue

    print(f"[done] processed {len(to_process)} new files, skipped {existing_count} existing files.")


if __name__ == "__main__":
    main()
