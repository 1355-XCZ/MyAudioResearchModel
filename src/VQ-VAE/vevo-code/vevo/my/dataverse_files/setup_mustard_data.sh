#!/bin/bash
###############################################################################
#  文件名: setup_mustard_data.sh
#  用途  : 下载并处理 MUStARD 数据集音频，准备英文上下文长度实验
###############################################################################

set -euo pipefail

#############################
# ===== 路径设置 =====
#############################
PROJECT_ROOT=/data/gpfs/projects/punim2341/haoguangzhou
MUSTARD_DIR="${PROJECT_ROOT}/data/MUStARD"
AUDIO_OUTPUT_DIR="${PROJECT_ROOT}/data/MUStARD/audio_processed"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === MUStARD 数据集设置开始 ==="

#############################
# ===== 创建目录 =====
#############################
mkdir -p "${MUSTARD_DIR}/data"
mkdir -p "${AUDIO_OUTPUT_DIR}"

#############################
# ===== 下载音频数据 =====
#############################
cd "${MUSTARD_DIR}"

if [[ ! -f "data/mmsd_raw_data.zip" ]]; then
    echo "[下载] 正在下载 MUStARD 音频数据..."
    wget -O data/mmsd_raw_data.zip "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"
    echo "[下载] 音频数据下载完成"
else
    echo "[跳过] 音频数据已存在"
fi

#############################
# ===== 解压音频数据 =====
#############################
if [[ ! -d "${MUSTARD_DIR}/data/context_final" ]] || [[ ! -d "${MUSTARD_DIR}/data/utterances_final" ]]; then
    echo "[解压] 正在解压音频数据..."
    cd "${MUSTARD_DIR}/data"
    unzip -o mmsd_raw_data.zip
    echo "[解压] 音频数据解压完成"
    cd "${PROJECT_ROOT}/voice/Amphion-VevoDev"
else
    echo "[跳过] 音频数据已解压"
fi

#############################
# ===== 检查音频文件 =====
#############################
echo "[检查] 统计音频文件..."
AUDIO_COUNT=$(find "${MUSTARD_DIR}/data/context_final" "${MUSTARD_DIR}/data/utterances_final" -name "*.wav" -o -name "*.mp4" -o -name "*.mp3" 2>/dev/null | wc -l)
echo "[统计] 找到 ${AUDIO_COUNT} 个音频/视频文件"

if [[ ${AUDIO_COUNT} -eq 0 ]]; then
    echo "[错误] 未找到音频文件，请检查下载和解压过程"
    exit 1
fi

#############################
# ===== 音频格式转换 =====
#############################
echo "[处理] 开始音频格式标准化..."

# 查找所有音频/视频文件并转换为标准 WAV 格式
find "${MUSTARD_DIR}/data/context_final" "${MUSTARD_DIR}/data/utterances_final" -type f \( -name "*.mp4" -o -name "*.mp3" -o -name "*.wav" \) | while read -r file; do
    # 获取相对路径和文件名（简化路径处理）
    if [[ "$file" == *"/context_final/"* ]]; then
        subdir="context_final"
    elif [[ "$file" == *"/utterances_final/"* ]]; then
        subdir="utterances_final"
    else
        subdir="unknown"
    fi
    
    filename=$(basename "$file")
    name_without_ext="${filename%.*}"
    output_dir="${AUDIO_OUTPUT_DIR}/${subdir}"
    output_file="${output_dir}/${name_without_ext}.wav"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 如果输出文件已存在，跳过
    if [[ -f "$output_file" ]]; then
        continue
    fi
    
    # 使用 ffmpeg 转换为标准格式
    # 16kHz, 单声道, 16-bit PCM (与 Vevo 配置一致)
    if ffmpeg -i "$file" -ar 16000 -ac 1 -c:a pcm_s16le "$output_file" -y >/dev/null 2>&1; then
        echo "[转换] ✓ ${subdir}/$(basename "$file") -> $(basename "$output_file")"
    else
        echo "[转换] ✗ 转换失败: ${subdir}/$(basename "$file")"
    fi
done

#############################
# ===== 最终统计 =====
#############################
PROCESSED_COUNT=$(find "${AUDIO_OUTPUT_DIR}" -name "*.wav" | wc -l)
echo "[完成] 处理完成，生成 ${PROCESSED_COUNT} 个标准化音频文件"
echo "[路径] 处理后的音频位于: ${AUDIO_OUTPUT_DIR}"

#############################
# ===== 创建文件列表 =====
#############################
echo "[列表] 创建音频文件列表..."
find "${AUDIO_OUTPUT_DIR}" -name "*.wav" | sort > "${MUSTARD_DIR}/audio_file_list.txt"
echo "[列表] 音频文件列表保存到: ${MUSTARD_DIR}/audio_file_list.txt"

# 显示前10个文件作为示例
echo "[示例] 前10个音频文件:"
head -10 "${MUSTARD_DIR}/audio_file_list.txt" | nl

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === MUStARD 数据集设置完成 ==="
