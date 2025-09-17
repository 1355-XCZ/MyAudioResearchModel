#!/bin/bash
###############################################################################
#  文件名: audio_denoise.slurm
#  用途  : Spartan 上批量处理音频降噪去笑声 (tkxdh_s5 数据集)
###############################################################################

#############################
# ===== Slurm 资源申请 =====
#############################
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J audio-denoise
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.err
#SBATCH --mail-user=haoguang.zhou@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A punim2341

set -euo pipefail

#############################
# ===== 路径与环境变量  =====
#############################
PROJECT_ROOT=/data/gpfs/projects/punim2341/haoguangzhou
CODE_DIR="${PROJECT_ROOT}/voice/Amphion-VevoDev"
RUNS_ROOT="${PROJECT_ROOT}/runs"
CONDA_ENV=vevo_env

# 音频处理相关路径（按照你的实际数据位置，包含 MCSD）
INPUT_DIR="${PROJECT_ROOT}/data/MCSD/vevo_data_split/tkxdh_s5"
OUTPUT_DIR="${PROJECT_ROOT}/data/MCSD/vevo_data_enhanced/tkxdh_s5"
TEMP_DIR="${PROJECT_ROOT}/tmp/audio_denoise"

# 统一创建项目根 logs（给 Slurm -o/-e 用）
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TEMP_DIR}"

# 模型/权重缓存放项目目录
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

export TMPDIR=${TMPDIR:-/tmp}

# 线程亲和性与并发控制，降低 ORT/BLAS 亲和性报错与资源争用
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-4}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export ORT_NUM_THREADS=${ORT_NUM_THREADS:-4}
export ORT_DISABLE_CPU_AFFINITY=1

#############################
# ===== 生成本次运行目录 =====
#############################
timestamp=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/${timestamp}_audio_denoise"
LOG_DIR="${RUN_DIR}/logs"
STAT_DIR="${RUN_DIR}/stats"
mkdir -p "${LOG_DIR}" "${STAT_DIR}"

#############################
# ===== 加载与激活环境  =====
#############################
module purge
module load Anaconda3/2024.02-1
module load foss/2022a || { echo "[sys] ERROR: failed to load toolchain foss/2022a" | tee -a "${LOG_DIR}/run-info.txt"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# CUDA/cuDNN for GPU acceleration
module load CUDA/12.4.1
module load cuDNN/9.6.0.74-CUDA-12.4.1

# 确保 Python 能导入项目内模块
: "${PYTHONPATH:=}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

# 选用当前环境的 python 可执行文件
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}" | tee -a "${LOG_DIR}/run-info.txt"
PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(which python)"
echo "PYTHON_BIN: $PYTHON_BIN" | tee -a "${LOG_DIR}/run-info.txt"

#############################
# ===== 依赖检查与安装  =====
#############################
echo "[deps] Installing audio processing dependencies" | tee -a "${LOG_DIR}/run-info.txt"

# 安装音频处理必要依赖
pip install --quiet demucs librosa soundfile noisereduce scipy torch torchaudio || {
    echo "[deps] ERROR: failed to install audio processing dependencies" | tee -a "${LOG_DIR}/run-info.txt"
    exit 1
}

echo "[deps] Testing imports" | tee -a "${LOG_DIR}/run-info.txt"
"$PYTHON_BIN" -c "
import torch, librosa, soundfile, noisereduce
print('torch=', torch.__version__)
print('librosa=', librosa.__version__)
print('GPU available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)
" | tee -a "${LOG_DIR}/run-info.txt" || {
    echo "[deps] ERROR: import test failed" | tee -a "${LOG_DIR}/run-info.txt"
    exit 1
}

#############################
# ===== 运行前检查  =====
#############################
echo "==== PRE-RUN CHECK ====" | tee -a "${LOG_DIR}/run-info.txt"
echo "INPUT_DIR     : ${INPUT_DIR}"  | tee -a "${LOG_DIR}/run-info.txt"
echo "OUTPUT_DIR    : ${OUTPUT_DIR}" | tee -a "${LOG_DIR}/run-info.txt"
echo "TEMP_DIR      : ${TEMP_DIR}"   | tee -a "${LOG_DIR}/run-info.txt"

# 检查输入目录是否存在
if [ ! -d "${INPUT_DIR}" ]; then
    echo "[data] ERROR: INPUT_DIR does not exist: ${INPUT_DIR}" | tee -a "${LOG_DIR}/run-info.txt"
    echo "[data] Please upload your split audio data to the cluster first" | tee -a "${LOG_DIR}/run-info.txt"
    exit 1
fi

# 统计输入文件数量
WAV_COUNT=$(find "${INPUT_DIR}" -name "*.wav" | wc -l)
echo "[data] Found ${WAV_COUNT} WAV files to process" | tee -a "${LOG_DIR}/run-info.txt"

if [ "${WAV_COUNT}" -eq 0 ]; then
    echo "[data] ERROR: No WAV files found in ${INPUT_DIR}" | tee -a "${LOG_DIR}/run-info.txt"
    exit 1
fi

#############################
# ===== 运行前信息快照  =====
#############################
echo "==== RUN INFO ====" | tee -a "${LOG_DIR}/run-info.txt"
echo "JOB_ID        : ${SLURM_JOB_ID}"        | tee -a "${LOG_DIR}/run-info.txt"
echo "PARTITION     : ${SLURM_JOB_PARTITION}" | tee -a "${LOG_DIR}/run-info.txt"
echo "RUN_DIR       : ${RUN_DIR}"             | tee -a "${LOG_DIR}/run-info.txt"
echo "CODE_DIR      : ${CODE_DIR}"            | tee -a "${LOG_DIR}/run-info.txt"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | tee -a "${LOG_DIR}/run-info.txt"

#############################
# ===== 优雅退出与统计  =====
#############################
record_stats() {
  my-job-stats -a -n -s -j "${SLURM_JOB_ID}" > "${STAT_DIR}/${SLURM_JOB_ID}.stats" 2>&1 || true
  echo "==== my-job-stats summary ====" | tee -a "${LOG_DIR}/run-info.txt"
  tail -n +1 "${STAT_DIR}/${SLURM_JOB_ID}.stats" | tee -a "${LOG_DIR}/run-info.txt" || true
}
trap record_stats EXIT
trap "echo 'Got SIGTERM, wrapping up…' | tee -a '${LOG_DIR}/run-info.txt'; record_stats; exit 1" TERM

#############################
# ===== 切换到代码目录  =====
#############################
cd "${CODE_DIR}"

#############################
# ===== 执行音频处理  =====
#############################
echo "==== STARTING AUDIO DENOISING ====" | tee -a "${LOG_DIR}/run-info.txt"

# 处理策略选择（默认高级：demucs + noisereduce + VAD）
PROCESSING_MODE=${PROCESSING_MODE:-"advanced"}  # light, standard, advanced

case "$PROCESSING_MODE" in
    light)
        echo "[mode] Using light processing (noisereduce only)" | tee -a "${LOG_DIR}/run-info.txt"
        DENOISE_ARGS="--no-demucs"
        ;;
    standard)
        echo "[mode] Using standard processing (demucs + noisereduce)" | tee -a "${LOG_DIR}/run-info.txt"
        DENOISE_ARGS=""
        ;;
    advanced)
        echo "[mode] Using advanced processing (demucs + noisereduce + vad)" | tee -a "${LOG_DIR}/run-info.txt"
        DENOISE_ARGS="--use-vad"
        ;;
    *)
        echo "[mode] Unknown processing mode: $PROCESSING_MODE, using standard" | tee -a "${LOG_DIR}/run-info.txt"
        DENOISE_ARGS=""
        ;;
esac

# 执行音频增强处理
"$PYTHON_BIN" models/vc/vevo/my/dataverse_files/audio_enhancement.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --temp-dir "${TEMP_DIR}" \
    ${DENOISE_ARGS} \
    2>&1 | tee -a "${LOG_DIR}/denoise.log"

DENOISE_EXIT_CODE=${PIPESTATUS[0]}

if [ ${DENOISE_EXIT_CODE} -eq 0 ]; then
    echo "==== AUDIO DENOISING COMPLETED SUCCESSFULLY ====" | tee -a "${LOG_DIR}/run-info.txt"
    
    # 统计处理结果
    OUTPUT_WAV_COUNT=$(find "${OUTPUT_DIR}" -name "*.wav" | wc -l)
    echo "[result] Processed ${OUTPUT_WAV_COUNT} / ${WAV_COUNT} files" | tee -a "${LOG_DIR}/run-info.txt"
    
    # 计算目录大小
    INPUT_SIZE=$(du -sh "${INPUT_DIR}" 2>/dev/null | cut -f1 || echo "unknown")
    OUTPUT_SIZE=$(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1 || echo "unknown")
    echo "[result] Input size: ${INPUT_SIZE}, Output size: ${OUTPUT_SIZE}" | tee -a "${LOG_DIR}/run-info.txt"
    
    # 清理临时文件
    echo "[cleanup] Removing temporary files" | tee -a "${LOG_DIR}/run-info.txt"
    rm -rf "${TEMP_DIR}" || true
    
else
    echo "==== AUDIO DENOISING FAILED (exit code: ${DENOISE_EXIT_CODE}) ====" | tee -a "${LOG_DIR}/run-info.txt"
    echo "[error] Check ${LOG_DIR}/denoise.log for details" | tee -a "${LOG_DIR}/run-info.txt"
    exit ${DENOISE_EXIT_CODE}
fi

echo "==== DONE. Enhanced audio in: ${OUTPUT_DIR} ====" | tee -a "${LOG_DIR}/run-info.txt"


