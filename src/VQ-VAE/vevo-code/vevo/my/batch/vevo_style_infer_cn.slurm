#!/bin/bash
###############################################################################
#  文件名: vevo_style_infer_cn.slurm
#  用途  : Spartan 上批量运行 Vevo-Style 推理（中文示例 content）
#          - content: 中文示例 models/vc/vevo/wav/mandarin_female.wav
#          - style  : 你分割好的音频（整目录）
###############################################################################

#############################
# ===== Slurm 资源申请 =====
#############################
# 可按需调整时间与资源
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -J vevo-style-cn
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

# 覆盖变量（可 sbatch 前导出或在命令行传入 STYLE_DIR=... OUT_DIR=...）
STYLE_DIR=${STYLE_DIR:-"${PROJECT_ROOT}/data/MCSD/vevo_data_split/tkxdh_s5"}
OUT_DIR=${OUT_DIR:-"${PROJECT_ROOT}/data/MCSD/vevo_style_out/tkxdh_s5"}
CONTENT_WAV=${CONTENT_WAV:-"models/vc/vevo/wav/mandarin_female.wav"}
LIMIT=${LIMIT:-}

# 统一创建项目根 logs（给 Slurm -o/-e 用）
mkdir -p "${PROJECT_ROOT}/logs"

# 模型/权重缓存放项目目录
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

export TMPDIR=${TMPDIR:-/tmp}

# 线程亲和性与并发控制（尽量降低资源争用）
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export ORT_DISABLE_CPU_AFFINITY=1

#############################
# ===== 生成本次运行目录 =====
#############################
timestamp=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/${timestamp}_vevo_style"
LOG_DIR="${RUN_DIR}/logs"
STAT_DIR="${RUN_DIR}/stats"
mkdir -p "${LOG_DIR}" "${STAT_DIR}" "${OUT_DIR}"

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

# （可选）espeak-ng/phonemizer 依赖检查，与 vevo_run3 一致
module load espeak-ng/1.52 && {
  echo "[sys] espeak-ng/1.52 loaded" | tee -a "${LOG_DIR}/run-info.txt"
} || true

#############################
# ===== 运行前信息快照  =====
#############################
echo "==== RUN INFO ====" | tee -a "${LOG_DIR}/run-info.txt"
echo "JOB_ID        : ${SLURM_JOB_ID}"        | tee -a "${LOG_DIR}/run-info.txt"
echo "PARTITION     : ${SLURM_JOB_PARTITION}" | tee -a "${LOG_DIR}/run-info.txt"
echo "RUN_DIR       : ${RUN_DIR}"             | tee -a "${LOG_DIR}/run-info.txt"
echo "CODE_DIR      : ${CODE_DIR}"            | tee -a "${LOG_DIR}/run-info.txt"
echo "STYLE_DIR     : ${STYLE_DIR}"           | tee -a "${LOG_DIR}/run-info.txt"
echo "OUT_DIR       : ${OUT_DIR}"             | tee -a "${LOG_DIR}/run-info.txt"
echo "CONTENT_WAV   : ${CONTENT_WAV}"         | tee -a "${LOG_DIR}/run-info.txt"
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
# ===== 执行 Vevo-Style  =====
#############################
echo "==== STARTING VEVO-STYLE INFERENCE ====" | tee -a "${LOG_DIR}/run-info.txt"

EXTRA_LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then
  EXTRA_LIMIT_ARG="--limit ${LIMIT}"
fi

"$PYTHON_BIN" models/vc/vevo/my/dataverse_files/batch_infer_vevostyle.py \
  --style-dir "${STYLE_DIR}" \
  --out-dir   "${OUT_DIR}" \
  --content-wav "${CONTENT_WAV}" \
  ${EXTRA_LIMIT_ARG} \
  2>&1 | tee -a "${LOG_DIR}/vevo_style.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "==== VEVO-STYLE COMPLETED SUCCESSFULLY ====" | tee -a "${LOG_DIR}/run-info.txt"
  # 简要统计
  OUT_COUNT=$(find "${OUT_DIR}" -name "*_vevostyle.wav" | wc -l)
  echo "[result] generated files: ${OUT_COUNT}" | tee -a "${LOG_DIR}/run-info.txt"
else
  echo "==== VEVO-STYLE FAILED (exit code: ${EXIT_CODE}) ====" | tee -a "${LOG_DIR}/run-info.txt"
  echo "[error] Check ${LOG_DIR}/vevo_style.log for details" | tee -a "${LOG_DIR}/run-info.txt"
  exit ${EXIT_CODE}
fi

echo "==== DONE. Outputs in: ${OUT_DIR} ====" | tee -a "${LOG_DIR}/run-info.txt"


