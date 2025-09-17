#!/bin/bash
###############################################################################
#  文件名: vevo_run.slurm
#  用途  : Spartan 上一次跑完 VEVO 的 timbre/style/voice/tts 四种推理
###############################################################################

#############################
# ===== Slurm 资源申请 =====
#############################
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -J vevo-sanity
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

# 统一创建项目根 logs（给 Slurm -o/-e 用）
mkdir -p "${PROJECT_ROOT}/logs"

# 模型/权重缓存放项目目录（仅设置 HF_HOME，避免 transformers 弃用警告）
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

export TMPDIR=${TMPDIR:-/tmp}

# 线程亲和性与并发控制，降低 ORT/BLAS 亲和性报错与资源争用
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export ORT_NUM_THREADS=${ORT_NUM_THREADS:-1}
# Fix onnxruntime pthread_setaffinity_np spam by disabling CPU affinity inside ORT
export ORT_DISABLE_CPU_AFFINITY=1

#############################
# ===== 生成本次运行目录 =====
#############################
timestamp=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/${timestamp}_vevo_infer"
LOG_DIR="${RUN_DIR}/logs"       # 运行期快照日志（与 Slurm 日志不同）
OUT_DIR="${RUN_DIR}/outputs"
STAT_DIR="${RUN_DIR}/stats"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${STAT_DIR}"

# 供运行期兼容模块的临时 shim 目录
SHIM_ROOT="${RUN_DIR}/shims"
mkdir -p "${SHIM_ROOT}"

#############################
# ===== 加载与激活环境  =====
#############################
module purge
module load Anaconda3/2024.02-1
## Per Spartan modules docs, load a toolchain before app modules
## espeak-ng/1.52 lives under GCC/11.3.0 + OpenMPI/4.1.4 (foss/2022a includes these)
module load foss/2022a || { echo "[sys] ERROR: failed to load toolchain foss/2022a" | tee -a "${LOG_DIR}/run-info.txt"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# CUDA/cuDNN for ONNX Runtime GPU provider
module load CUDA/12.4.1
module load cuDNN/9.6.0.74-CUDA-12.4.1

# 确保 Python 能导入项目内模块（更稳健）
# 避免 set -u 下 PYTHONPATH 未定义导致报错
: "${PYTHONPATH:=}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

# 选用当前环境的 python 可执行文件，避免落到系统 python
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}" | tee -a "${LOG_DIR}/run-info.txt"
PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(which python)"
echo "PYTHON_BIN: $PYTHON_BIN" | tee -a "${LOG_DIR}/run-info.txt"

# 系统模块加载：espeak-ng 是 phonemizer 的必需依赖
echo "[sys] loading system modules" | tee -a "${LOG_DIR}/run-info.txt"
module load espeak-ng/1.52 && {
  echo "[sys] espeak-ng/1.52 module loaded successfully" | tee -a "${LOG_DIR}/run-info.txt"
  espeak-ng --version 2>/dev/null | head -1 | tee -a "${LOG_DIR}/run-info.txt" || true
  # Verify espeak-ng is in PATH
  which espeak-ng | tee -a "${LOG_DIR}/run-info.txt" || echo "[sys] WARNING: espeak-ng not found in PATH" | tee -a "${LOG_DIR}/run-info.txt"
} || {
  echo "[sys] ERROR: espeak-ng/1.52 module load failed, phonemizer will not work" | tee -a "${LOG_DIR}/run-info.txt"
  exit 1
}

# Minimal import test (no source changes): ensure phonemizer can init EspeakBackend
echo "[sys] testing phonemizer EspeakBackend import" | tee -a "${LOG_DIR}/run-info.txt"
if ! "$PYTHON_BIN" -c "from phonemizer.backend import EspeakBackend; EspeakBackend('en-us'); print('OK')" >/dev/null 2>&1; then
  echo "[sys] ERROR: phonemizer EspeakBackend import failed; check espeak-ng module/toolchain" | tee -a "${LOG_DIR}/run-info.txt"
  exit 1
else
  echo "[sys] phonemizer EspeakBackend test: SUCCESS" | tee -a "${LOG_DIR}/run-info.txt"
fi

#############################
# ===== 轻量运行前检查 =====
#############################
echo "==== LIGHT CHECK ====" | tee -a "${LOG_DIR}/run-info.txt"
"$PYTHON_BIN" - <<'PY' 2>/dev/null | sed 's/^/[py]/' | tee -a "${LOG_DIR}/run-info.txt" || true
try:
    import torch, phonemizer
    print('torch=', getattr(torch, '__version__', 'unknown'))
    print('phonemizer ok')
except Exception as e:
    print('python deps check:', e)
PY



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

# 不再在脚本中下载或生成占位音频；若缺示例则直接提示并退出
if [ ! -f "${WAV_DIR}/source.wav" ] || [ ! -f "${WAV_DIR}/arabic_male.wav" ] || [ ! -f "${WAV_DIR}/mandarin_female.wav" ]; then
  echo "[data] required input wavs not found in ${WAV_DIR}" | tee -a "${LOG_DIR}/run-info.txt"
  echo "[data] please place these files before submitting:" | tee -a "${LOG_DIR}/run-info.txt"
  echo "       - source.wav (content)" | tee -a "${LOG_DIR}/run-info.txt"
  echo "       - arabic_male.wav (style ref)" | tee -a "${LOG_DIR}/run-info.txt"
  echo "       - mandarin_female.wav (timbre ref)" | tee -a "${LOG_DIR}/run-info.txt"
  echo "[data] you can use official samples from Amphion repo: models/vc/vevo/wav" | tee -a "${LOG_DIR}/run-info.txt"
  exit 1
fi

for mode in timbre style voice tts; do
  echo "==== Running VEVO_MODE=$mode ====" | tee -a "${LOG_DIR}/run-info.txt"

  # 仅清理旧输出，保留示例与输入，避免删除必要输入导致 FileNotFoundError
  rm -f "${WAV_DIR}"/output_*.wav 2>/dev/null || true

  case "$mode" in
    timbre) "$PYTHON_BIN" -m models.vc.vevo.infer_vevotimbre ;;
    style)  "$PYTHON_BIN" -m models.vc.vevo.infer_vevostyle  ;;
    voice)  "$PYTHON_BIN" -m models.vc.vevo.infer_vevovoice  ;;
    tts)    "$PYTHON_BIN" -m models.vc.vevo.infer_vevotts    ;;  # 若缺系统依赖可能不会产出
  esac

  # 收集产物到各自目录
  if ls "${WAV_DIR}"/*.wav >/dev/null 2>&1; then
    rsync -ah --ignore-existing "${WAV_DIR}/" "${OUT_DIR}/${mode}/"
  else
    echo "No WAV produced for mode=${mode} (check Slurm & run-info logs)" | tee -a "${LOG_DIR}/run-info.txt"
  fi
done

echo "==== DONE. Outputs in: ${OUT_DIR} ====" | tee -a "${LOG_DIR}/run-info.txt"
