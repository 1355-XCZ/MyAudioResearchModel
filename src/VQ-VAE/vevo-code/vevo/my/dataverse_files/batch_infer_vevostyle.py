#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行 Vevo-Style：
- 使用“示例源音频”作为 content
- 使用“分割得到的音频”作为 style reference

用法（本地或集群均可）：
  python models/vc/vevo/my/dataverse_files/batch_infer_vevostyle.py \
    --style-dir /path/to/vevo_data_split/tkxdh_s5 \
    --out-dir   /path/to/vevo_style_out/tkxdh_s5 \
    --content-wav models/vc/vevo/wav/source.wav

说明：仅执行风格迁移（Vevo-Style），不做降噪/去笑声。
"""

import argparse
import os
from pathlib import Path
import subprocess
from typing import List

import torch
from huggingface_hub import snapshot_download

# 直接复用 vevo 推理所需工具与管线
from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio


def build_pipeline(device: torch.device, ar_cfg_path: str = None) -> VevoInferencePipeline:
    """按照 infer_vevostyle.py 的方式构建推理管线，一次加载复用。"""
    # ===== Content Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq32/*"],
    )
    content_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
    )

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # ===== Autoregressive Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["contentstyle_modeling/Vq32ToVq8192/*"],
    )
    # 使用传入的配置文件路径，或默认路径
    if ar_cfg_path is None:
        ar_cfg_path = "./models/vc/vevo/config/Vq32ToVq8192.json"
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/Vq32ToVq8192")

    # ===== Flow Matching Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )
    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
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
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch Vevo-Style inference")
    parser.add_argument("--style-dir", required=True, help="风格参考音频根目录（分割产物）")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument(
        "--content-wav",
        default="models/vc/vevo/wav/source.wav",
        help="示例源音频（content）",
    )
    parser.add_argument("--style-max-seconds", type=float, default=10.0, help="style 裁剪秒数（防止过长导致生成溢出）")
    parser.add_argument("--content-max-seconds", type=float, default=8.0, help="content 裁剪秒数（可选，防止过长）")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少个文件（调试用）")
    parser.add_argument("--config", type=str, default=None, help="AR 模型配置文件路径（用于上下文长度研究）")
    args = parser.parse_args()

    style_root = Path(args.style_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print("[init] building Vevo pipeline (first-time will download from HF cache)...")
    if args.config:
        print(f"[config] using custom AR config: {args.config}")
    pipeline = build_pipeline(device, ar_cfg_path=args.config)
    print("[init] pipeline ready")

    style_wavs = collect_style_wavs(style_root)
    if args.limit:
        style_wavs = style_wavs[: args.limit]
    print(f"[data] found {len(style_wavs)} style wavs under {style_root}")

    # 为裁剪文件准备临时目录
    tmp_root = out_root / "_tmp_trim"
    tmp_root.mkdir(parents=True, exist_ok=True)

    # 预裁剪 content（如需要）
    content_path = Path(args.content_wav)
    trimmed_content = tmp_root / ("content_trim.wav")
    content_used = str(content_path)
    if args.content_max_seconds and args.content_max_seconds > 0:
        if ffmpeg_trim_wav(str(content_path), str(trimmed_content), args.content_max_seconds):
            content_used = str(trimmed_content)

    # 逐个执行风格迁移
    for idx, style_wav in enumerate(style_wavs, 1):
        rel = style_wav.relative_to(style_root)
        out_path = out_root / rel.parent / (style_wav.stem + "_vevostyle.wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(style_wavs)}] style={style_wav} -> {out_path}")

        # 裁剪 style，避免过长导致 AR 生成长度校验失败
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
        except Exception as e:
            print(f"[warn] failed on {style_wav}: {e}")
            # 跳过该条，继续下一个
            continue

    print("[done] all files processed.")


if __name__ == "__main__":
    main()


