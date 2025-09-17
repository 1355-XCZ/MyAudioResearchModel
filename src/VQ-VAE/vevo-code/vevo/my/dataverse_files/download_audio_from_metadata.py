import argparse
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print("[错误] 需要安装 pandas: pip install pandas openpyxl", file=sys.stderr)
    raise

try:
    from yt_dlp import YoutubeDL
except Exception as exc:  # pragma: no cover
    print("[错误] 需要安装 yt-dlp: pip install yt-dlp", file=sys.stderr)
    raise


URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def detect_urls_from_excel(excel_path: str, sheet: Optional[str] = None) -> Tuple[List[str], List[Dict[str, str]]]:
    """Scan an Excel sheet and extract URLs from all columns.

    Returns
    -------
    urls : List[str]
        All URLs found (may contain duplicates before de-duplication)
    rows_info : List[Dict[str, str]]
        Row-wise info dicts with 'row_index', 'url', and an optional 'code' if any cell matches naming pattern
    """
    # Handle both single sheet and multi-sheet cases
    excel_data = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
    
    # If sheet_name=None, pandas returns a dict of DataFrames; otherwise a single DataFrame
    if isinstance(excel_data, dict):
        # Multiple sheets case: use first sheet
        sheet_names = list(excel_data.keys())
        if not sheet_names:
            return [], []
        df = excel_data[sheet_names[0]]
        print(f"[信息] 使用工作表: {sheet_names[0]}")
    else:
        # Single sheet case
        df = excel_data
    
    urls: List[str] = []
    rows_info: List[Dict[str, str]] = []

    # Try to find a canonical code like tkxdh_s5_ep1_1 from any column per row
    code_pattern = re.compile(r"\b[a-z0-9]+_s\d+_ep\d+_[a-z0-9]+\b", re.IGNORECASE)

    for i, row in df.iterrows():
        row_code: Optional[str] = None
        # First pass: detect code from any cell
        for col in df.columns:
            val = str(row[col]) if not pd.isna(row[col]) else ""
            m = code_pattern.search(val)
            if m:
                row_code = m.group(0)
                break

        # Second pass: collect URLs and bind row_code
        for col in df.columns:
            val = str(row[col]) if not pd.isna(row[col]) else ""
            if URL_PATTERN.match(val):
                urls.append(val)
                rows_info.append({"row_index": str(i), "url": val, "code": row_code or ""})

    return urls, rows_info


def dedupe_urls(urls: List[str]) -> List[str]:
    """Basic de-duplication by normalized string form."""
    seen: Set[str] = set()
    unique: List[str] = []
    for u in urls:
        nu = normalize_url(u)
        if nu not in seen:
            seen.add(nu)
            unique.append(u)
    return unique


def normalize_url(url: str) -> str:
    """Normalize a URL string for coarse de-duplication.

    - Lowercase scheme and host
    - Strip trailing slashes
    - Remove common YouTube tracking query params
    """
    try:
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
        p = urlparse(url)
        query = parse_qsl(p.query, keep_blank_values=True)
        # Drop unimportant params
        drop_keys = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "si", "pp"}
        query = [(k, v) for k, v in query if k not in drop_keys]
        p2 = p._replace(
            scheme=p.scheme.lower(),
            netloc=p.netloc.lower(),
            path=p.path.rstrip("/"),
            query=urlencode(query)
        )
        return urlunparse(p2)
    except Exception:
        return url.strip().rstrip("/")


def build_url_to_code(rows_info: List[Dict[str, str]]) -> Dict[str, str]:
    """Map URL -> detected canonical code (if any) from the same row."""
    mapping: Dict[str, str] = {}
    for item in rows_info:
        url = item.get("url", "")
        code = item.get("code", "")
        if url and code and url not in mapping:
            mapping[url] = code
    return mapping


def build_url_to_codes(rows_info: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Map URL -> list of detected codes (unique, preserve order)."""
    url2codes: Dict[str, List[str]] = {}
    for item in rows_info:
        url = item.get("url", "")
        code = item.get("code", "")
        if not url or not code:
            continue
        bucket = url2codes.setdefault(url, [])
        if code not in bucket:
            bucket.append(code)
    return url2codes


def sanitize_code(name: str) -> str:
    """Make code safe for filesystem: lowercase and replace invalid chars with '_'"""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name.strip())
    return safe.lower()


def input_if_absent(value: Optional[str], prompt: str) -> str:
    if value:
        return value
    try:
        return input(prompt).strip()
    except KeyboardInterrupt:
        print("\n[中止] 用户取消。")
        sys.exit(1)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_outtmpl(output_dir: str, rich_naming: bool, per_link_dir: bool) -> str:
    """Construct yt-dlp output template.

    rich_naming: include playlist index/id and title for traceability.
    """
    if per_link_dir:
        # 每个链接一个独立目录（以视频 id 命名目录）
        if rich_naming:
            # 在目录内包含原始顺序与标题，便于人读与追溯
            return os.path.join(
                output_dir,
                "%(id)s",
                "%(playlist_index|000)03d__%(title).80s.%(ext)s",
            )
        else:
            # 简洁命名：仅保存容器扩展名
            return os.path.join(output_dir, "%(id)s", "%(id)s.%(ext)s")
    else:
        if rich_naming:
            # 每个播放列表一个目录，文件名保留 index、id、title
            return os.path.join(
                output_dir,
                "%(playlist_id|NA)s",
                "%(playlist_index|000)03d__%(id)s__%(title).80s.%(ext)s",
            )
        else:
            return os.path.join(output_dir, "%(id)s.%(ext)s")


def run_download(urls: List[str], outtmpl: str, audio_format: str, split_chapters: bool, skip_existing: bool) -> None:
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": audio_format,          # e.g., bestaudio/best (只下载音频流，不转码)
        "noplaylist": False,            # allow playlist expansion
        "ignoreerrors": True,
        # 不设置音频转码的 postprocessor，保留原始音频容器（webm/m4a/opus/aac 等）
        "writethumbnail": False,
        "writesubtitles": False,
        "concurrent_fragment_downloads": 3,
        "no_warnings": True,
        "quiet": False,
        "nooverwrites": skip_existing,
        "postprocessor_args": [],
    }
    if split_chapters:
        ydl_opts["split_chapters"] = True

    with YoutubeDL(ydl_opts) as ydl:
        for u in urls:
            try:
                print(f"[下载] {u}")
                ydl.download([u])
            except Exception as exc:  # pragma: no cover
                print(f"[跳过] {u} 失败: {exc}")
            time.sleep(0.1)


def run_download_per_link(
    urls: List[str],
    compute_dir,
    file_template: str,
    audio_format: str,
    split_chapters: bool,
    skip_existing: bool,
) -> None:
    """Download each URL into its own concrete directory computed by compute_dir(url)."""
    for u in urls:
        try:
            link_dir = compute_dir(u)
            os.makedirs(link_dir, exist_ok=True)
            outtmpl = os.path.join(link_dir, file_template)
            ydl_opts = {
                "outtmpl": outtmpl,
                "format": audio_format,
                "noplaylist": False,
                "ignoreerrors": True,
                "writethumbnail": False,
                "writesubtitles": False,
                "concurrent_fragment_downloads": 3,
                "no_warnings": True,
                "quiet": False,
                "nooverwrites": skip_existing,
                "postprocessor_args": [],
            }
            if split_chapters:
                ydl_opts["split_chapters"] = True
            print(f"[下载] {u} -> {link_dir}")
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([u])
        except Exception as exc:  # pragma: no cover
            print(f"[跳过] {u} 失败: {exc}")
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "从 metadata_v1.xlsx 提取去重后的 URL，并以最高音频质量下载到本地。\n"
            "- 支持选择输出目录、命名策略；\n"
            "- 如为播放列表，将展开并在子目录中按序号命名；\n"
            "- 默认仅下载压缩音频(最佳质量)，不转 WAV。"
        )
    )
    parser.add_argument(
        "--excel",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "metadata_v1.xlsx"),
        help="Excel 路径（默认: metadata_v1.xlsx）",
    )
    parser.add_argument("--sheet", type=str, default=None, help="Excel 工作表名（默认: 第一个工作表）")
    parser.add_argument("--output-dir", type=str, default=None, help="下载保存目录（未提供则运行时交互输入）")
    parser.add_argument(
        "--rich-naming",
        action="store_true",
        help="使用富命名（含 playlist_index、id、title，并按 playlist_id 分子目录）",
    )
    parser.add_argument(
        "--per-link-dir",
        dest="per_link_dir",
        action="store_true",
        help="每个链接下载到独立目录（目录名为视频 id）",
    )
    parser.add_argument(
        "--flat",
        dest="per_link_dir",
        action="store_false",
        help="不按链接分目录（与 --per-link-dir 互斥，默认启用按链接分目录）",
    )
    parser.add_argument(
        "--split-chapters",
        action="store_true",
        help="若视频有章节信息，则按章节自动切分保存（可选）",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="bestaudio/best",
        help="yt-dlp format 选择（默认: bestaudio/best）",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅下载前 N 条（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要下载的 URL，不实际下载")
    parser.add_argument("--skip-existing", action="store_true", help="若目标文件存在则跳过下载（默认不覆盖）")

    args = parser.parse_args()

    # 默认启用“每个链接一个目录”
    if not hasattr(args, "per_link_dir") or args.per_link_dir is None:
        args.per_link_dir = True

    output_dir = input_if_absent(args.output_dir, "请输入下载保存目录(绝对/相对路径): ")
    ensure_dir(output_dir)

    print(f"[读取] Excel: {args.excel}")
    urls_raw, rows_info = detect_urls_from_excel(args.excel, args.sheet)
    if not urls_raw:
        print("[提示] 未在 Excel 中发现 URL。")
        return

    print(f"[统计] Excel 中共发现 {len(urls_raw)} 条 URL（含可能重复）")
    urls = dedupe_urls(urls_raw)
    if args.limit is not None:
        urls = urls[: args.limit]
    print(f"[去重] 计划下载 {len(urls)} 条")

    # 名称策略：若 Excel 行内解析到命名（如 tkxdh_s5_ep1_1），优先将其作为“链接目录名”；否则退回视频 id
    url2codes = build_url_to_codes(rows_info)
    print(f"[命名] 解析到具备命名标识的 URL 数: {len(url2codes)}")

    def compute_link_dir(url: str) -> str:
        # 若包含多个 code，使用第一个；多命名可在后续重命名时处理
        codes = url2codes.get(url)
        if codes and len(codes) > 0:
            code_dir = sanitize_code(codes[0])
            return os.path.join(output_dir, code_dir)
        # 退回到 id 目录：先用一个轻量 ydl extract 获取 id
        # 为避免频繁提取，这里用占位目录名，下载时 outtmpl 仍会用 %(id)s
        return os.path.join(output_dir, "%(id)s")

    if args.dry_run:
        print("[Dry-Run] 将要下载的 URL 清单: ")
        for u in urls:
            print(" ", u)
        return

    if args.per_link_dir:
        file_template = "%(playlist_index|000)03d__%(title).80s.%(ext)s" if args.rich_naming else "%(id)s.%(ext)s"
        run_download_per_link(
            urls=urls,
            compute_dir=compute_link_dir,
            file_template=file_template,
            audio_format=args.format,
            split_chapters=args.split_chapters,
            skip_existing=args.skip_existing,
        )
    else:
        outtmpl = make_outtmpl(output_dir, rich_naming=args.rich_naming, per_link_dir=args.per_link_dir)
        print(f"[命名] 输出模板: {outtmpl}")
        run_download(
            urls=urls,
            outtmpl=outtmpl,
            audio_format=args.format,
            split_chapters=args.split_chapters,
            skip_existing=args.skip_existing,
        )

    print("[完成] 下载流程结束。")


if __name__ == "__main__":
    main()


