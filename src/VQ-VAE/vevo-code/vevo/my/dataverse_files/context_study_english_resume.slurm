#!/bin/bash
###############################################################################
#  文件名: context_study_english_resume.slurm
#  用途  : 英文上下文长度对比实验 - 支持断点续传
#  基于  : context_study_english.slurm
#  改进  : 使用新的 batch_infer_vevostyle_resume.py，自动跳过已存在的文件
###############################################################################

#SBATCH --job-name=vevo-ctx-en-resume
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/vevo-ctx-en-resume-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/vevo-ctx-en-resume-%j.err

set -euo pipefail

#############################
# ===== 环境设置 =====
#############################
PROJECT_ROOT=/data/gpfs/projects/punim2341/haoguangzhou
VEVO_ROOT=${PROJECT_ROOT}/voice/Amphion-VevoDev
CONDA_ENV=vevo_env

# 研究参数 - 英文版
STYLE_DIR=${STYLE_DIR:-"${PROJECT_ROOT}/data/MUStARD/audio_processed"}
CONTENT_WAV=${CONTENT_WAV:-"models/vc/vevo/wav/source.wav"}  # 使用 Vevo 官方英文音频
LIMIT=${LIMIT:-}  # 处理全部英文音频文件

# 使用现有的批处理目录（断点续传）
EXISTING_BATCH_DIR="${PROJECT_ROOT}/data/MUStARD/context_study_english/batch_20250822_175551"
BASE_OUT_DIR=${BASE_OUT_DIR:-"${EXISTING_BATCH_DIR}"}

# 上下文长度设置
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-"1600 2000 2400"}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 英文上下文窗口长度研究开始（断点续传） ==="
echo "[批处理] 使用现有批次: $(basename ${BASE_OUT_DIR})"
echo "[输出目录] ${BASE_OUT_DIR}"

#############################
# ===== 环境激活 =====
#############################
cd "${VEVO_ROOT}"

# 加载系统模块
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate "${CONDA_ENV}"
echo "[conda] 已激活环境: $CONDA_ENV"

# 验证 Python 环境
PYTHON_BIN=$(which python)
echo "[python] 使用: $PYTHON_BIN"

# 加载系统模块
echo "[sys] loading system modules"
module load espeak-ng/1.52
echo "[sys] espeak-ng/1.52 module loaded successfully"

#############################
# ===== 依赖检查 =====
#############################
echo "[deps] 检查依赖..."
python -c "import torch; print('[deps] ✓ torch 已安装')"
python -c "import torchaudio; print('[deps] ✓ torchaudio 已安装')"
python -c "import transformers; print('[deps] ✓ transformers 已安装')"
python -c "import librosa; print('[deps] ✓ librosa 已安装')"
python -c "import soundfile; print('[deps] ✓ soundfile 已安装')"

#############################
# ===== 数据验证 =====
#############################
if [[ ! -d "${STYLE_DIR}" ]]; then
    echo "[错误] 风格音频目录不存在: ${STYLE_DIR}"
    exit 1
fi

if [[ ! -f "${CONTENT_WAV}" ]]; then
    echo "[错误] 内容音频文件不存在: ${CONTENT_WAV}"
    exit 1
fi

# 统计可用的 WAV 文件
AVAILABLE_WAVS=$(find "${STYLE_DIR}" -name "*.wav" | wc -l)
if [[ -n "${LIMIT}" ]]; then
    echo "[info] 发现 ${AVAILABLE_WAVS} 个 WAV 文件，将处理前 ${LIMIT} 个"
else
    echo "[info] 发现 ${AVAILABLE_WAVS} 个 WAV 文件，将处理全部"
fi

if [[ ${AVAILABLE_WAVS} -eq 0 ]]; then
    echo "[错误] 在 ${STYLE_DIR} 中未找到 WAV 文件"
    exit 1
fi

#############################
# ===== 上下文长度实验 =====
#############################
for ctx_len in ${CONTEXT_LENGTHS}; do
    echo ""
    echo "=== 开始上下文长度 ${ctx_len} ==="
    
    # 配置文件路径
    CONFIG_FILE="./models/vc/vevo/config/Vq32ToVq8192_ctx${ctx_len}.json"
    
    # 检查配置文件是否存在
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "[错误] 配置文件不存在: ${CONFIG_FILE}"
        exit 1
    fi
    
    # 输出目录
    OUTPUT_DIR="${BASE_OUT_DIR}/ctx_${ctx_len}"
    mkdir -p "${OUTPUT_DIR}"
    
    # 构建推理命令参数
    INFER_ARGS=(
        --content-wav "${CONTENT_WAV}"
        --style-dir "${STYLE_DIR}"
        --out-dir "${OUTPUT_DIR}"
        --config "${CONFIG_FILE}"
        --style-max-seconds 10
        --content-max-seconds 8
    )
    
    # 添加限制参数（如果指定）
    if [[ -n "${LIMIT}" ]]; then
        INFER_ARGS+=(--limit "${LIMIT}")
    fi
    
    echo "[执行] 上下文长度 ${ctx_len}"
    echo "[配置] ${CONFIG_FILE}"
    echo "[输出] ${OUTPUT_DIR}"
    
    # 运行推理（使用断点续传版本）
    python models/vc/vevo/my/dataverse_files/batch_infer_vevostyle_resume.py "${INFER_ARGS[@]}"
    
    # 统计生成的文件
    GENERATED_COUNT=$(find "${OUTPUT_DIR}" -name "*_vevostyle.wav" | wc -l)
    echo "[完成] 上下文长度 ${ctx_len}，生成 ${GENERATED_COUNT} 个文件"
done

#############################
# ===== 最终统计 =====
#############################
echo ""
echo "=== 英文上下文窗口长度研究完成 ==="

# 统计各上下文长度的文件数量
for ctx_len in ${CONTEXT_LENGTHS}; do
    OUTPUT_DIR="${BASE_OUT_DIR}/ctx_${ctx_len}"
    if [[ -d "${OUTPUT_DIR}" ]]; then
        COUNT=$(find "${OUTPUT_DIR}" -name "*_vevostyle.wav" | wc -l)
        echo "[统计] ctx_${ctx_len}: ${COUNT} 个文件"
    fi
done

# 总计
TOTAL_COUNT=$(find "${BASE_OUT_DIR}" -name "*_vevostyle.wav" | wc -l)
echo "[总计] 生成 ${TOTAL_COUNT} 个 Vevo 风格转换音频文件"
echo "[路径] 结果保存在: ${BASE_OUT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 英文上下文窗口长度研究完成 ==="
