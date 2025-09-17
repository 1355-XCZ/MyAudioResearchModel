#!/bin/bash
###############################################################################
#  文件名: vevo_run.slurm
#  用途  : Spartan 上一次跑完 VEVO 的 timbre/style/voice/tts 四种推理
###############################################################################

#############################
# ===== Slurm 资源申请 =====
#############################
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J vevo-infer
# ——把 Slurm 日志固定到项目根 logs/——
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.err
#SBATCH --signal=TERM@60
#SBATCH --mail-user=haoguangz@student.unimelb.edu.au
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

# 统一创建项目根 logs（给 Slurm -o/-e 用）
mkdir -p "${PROJECT_ROOT}/logs"

# 模型/权重缓存放项目目录
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

export TMPDIR=${TMPDIR:-/tmp}

#############################
# ===== 生成本次运行目录 =====
#############################
timestamp=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/${timestamp}_vevo_infer"
LOG_DIR="${RUN_DIR}/logs"       # 运行期快照日志（与 Slurm 日志不同）
OUT_DIR="${RUN_DIR}/outputs"
STAT_DIR="${RUN_DIR}/stats"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${STAT_DIR}"

#############################
# ===== 加载与激活环境  =====
#############################
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# 确保 Python 能导入项目内模块（更稳健）
# 避免 set -u 下 PYTHONPATH 未定义导致报错
: "${PYTHONPATH:=}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

# （已移除）自动加载系统模块以避免噪声日志；如需可手动开启
# module -t avail ffmpeg   >/dev/null 2>&1 && module load ffmpeg   >/dev/null 2>&1 || true
# module -t avail espeak   >/dev/null 2>&1 && module load espeak   >/dev/null 2>&1 || true
# module -t avail espeak-ng>/dev/null 2>&1 && module load espeak-ng>/dev/null 2>&1 || true

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
# ======= 执行 VEVO  =======
#############################
WAV_DIR="models/vc/vevo/wav"
mkdir -p "$WAV_DIR"

for mode in timbre style voice tts; do
  echo "==== Running VEVO_MODE=$mode ====" | tee -a "${LOG_DIR}/run-info.txt"

  # 清空旧 wav，避免不同模式的输出混淆
  rm -f "${WAV_DIR}"/*.wav 2>/dev/null || true

  case "$mode" in
    timbre) python -m models.vc.vevo.infer_vevotimbre ;;
    style)  python -m models.vc.vevo.infer_vevostyle  ;;
    voice)  python -m models.vc.vevo.infer_vevovoice  ;;
    tts)    python -m models.vc.vevo.infer_vevotts    ;;  # 若缺系统依赖可能不会产出
  esac

  # 收集产物到各自目录
  if ls "${WAV_DIR}"/*.wav >/dev/null 2>&1; then
    rsync -ah --ignore-existing "${WAV_DIR}/" "${OUT_DIR}/${mode}/"
  else
    echo "No WAV produced for mode=${mode} (check Slurm & run-info logs)" | tee -a "${LOG_DIR}/run-info.txt"
  fi
done

echo "==== DONE. Outputs in: ${OUT_DIR} ====" | tee -a "${LOG_DIR}/run-info.txt"
