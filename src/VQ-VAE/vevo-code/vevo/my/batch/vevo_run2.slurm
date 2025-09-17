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
# ===== 依赖版本预检  =====
#############################
echo "==== DEP CHECK ====" | tee -a "${LOG_DIR}/run-info.txt"
CURR_TRF=$("$PYTHON_BIN" - <<'PY'
try:
    import transformers as t
    print(getattr(t, '__version__', '0'))
except Exception:
    print('0')
PY
)
echo "transformers version: ${CURR_TRF}" | tee -a "${LOG_DIR}/run-info.txt"
case "$CURR_TRF" in
  4.41.*) echo "[dep] transformers is compatible" | tee -a "${LOG_DIR}/run-info.txt" ;;
  *)
    echo "[dep] fixing transformers to 4.41.2 (and tokenizers<0.20)" | tee -a "${LOG_DIR}/run-info.txt"
    "$PYTHON_BIN" -m pip install --upgrade "transformers==4.41.2" "tokenizers<0.20" | tee -a "${LOG_DIR}/run-info.txt"
    NEW_TRF=$("$PYTHON_BIN" -c "import transformers,sys; print(transformers.__version__)" 2>/dev/null || echo "unknown")
    echo "[dep] transformers now: ${NEW_TRF}" | tee -a "${LOG_DIR}/run-info.txt"
  ;;
esac

# 修复 LangSegment 版本不兼容：已手动安装兼容版本，跳过重装
echo "[dep] LangSegment compatibility: using existing version" | tee -a "${LOG_DIR}/run-info.txt"
# 注释掉自动重装逻辑，因为会破坏已安装的工作版本
# "$PYTHON_BIN" -m pip uninstall -y LangSegment 2>/dev/null || true
# "$PYTHON_BIN" -m pip install "LangSegment==2.0.0" | tee -a "${LOG_DIR}/run-info.txt" || {
#   echo "[dep] LangSegment 2.0.0 failed, trying 1.x" | tee -a "${LOG_DIR}/run-info.txt"
#   "$PYTHON_BIN" -m pip install "LangSegment<2.0" | tee -a "${LOG_DIR}/run-info.txt"
# }

# g2p 依赖确保安装
for PKG in pypinyin jieba cn2an g2p_en phonemizer pyopenjtalk pykakasi; do
  if ! "$PYTHON_BIN" -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('${PKG}') else 1)" >/dev/null 2>&1; then
    echo "[dep] installing ${PKG}" | tee -a "${LOG_DIR}/run-info.txt"
    "$PYTHON_BIN" -m pip install "${PKG}" | tee -a "${LOG_DIR}/run-info.txt" || true
  fi
done

# 依赖快速自检与修复：核心三方与扩展（缺则安装）
REQ_PKGS=(
  json5 ruamel.yaml easydict huggingface_hub tqdm
  librosa soundfile ffmpeg-python onnxruntime
  encodec vocos
)
for PKG in "${REQ_PKGS[@]}"; do
  MOD="$PKG"
  [ "$PKG" = "ruamel.yaml" ] && MOD="ruamel"
  [ "$PKG" = "ffmpeg-python" ] && MOD="ffmpeg"
  if ! "$PYTHON_BIN" -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('${MOD}') else 1)" >/dev/null 2>&1; then
    echo "[dep] installing ${PKG}" | tee -a "${LOG_DIR}/run-info.txt"
    "$PYTHON_BIN" -m pip install "${PKG}" | tee -a "${LOG_DIR}/run-info.txt" || true
  fi
done

# 对齐 torchvision 版本以匹配 torch（避免经典不兼容）
"$PYTHON_BIN" - <<'PY'
import re, subprocess, sys
import importlib
def vtuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])
try:
    import torch
    tv = getattr(torch, '__version__', '0.0.0')
    tmaj, tmin = vtuple(tv)[:2]
    import importlib.util
    need = None
    if tmaj==2 and tmin>=5:
        # torch>=2.5 建议 torchvision 0.20.x
        need = '0.20.1'
    if need:
        ok = False
        if importlib.util.find_spec('torchvision'):
            import torchvision
            ok = vtuple(torchvision.__version__)[0:2] == (0,20)
        if not ok:
            print('[dep] fixing torchvision to', need)
            subprocess.run([sys.executable,'-m','pip','install',f'torchvision=={need}'], check=False)
            import importlib as _il; _il.invalidate_caches()
except Exception as e:
    print('[dep] torchvision check skipped:', e)
PY

# 验证 LangSegment 修复是否成功
"$PYTHON_BIN" - <<'PY'
try:
    import LangSegment
    # 测试关键函数是否存在
    funcs = ['setfilters', 'getTexts']
    missing = [f for f in funcs if not hasattr(LangSegment, f)]
    if missing:
        print(f'[dep] LangSegment still missing: {missing}')
    else:
        print('[dep] LangSegment verification passed')
except Exception as e:
    print(f'[dep] LangSegment verification failed: {e}')
PY

# 若 monotonic_align 扩展不可用则自动编译安装（Cython）
if ! "$PYTHON_BIN" -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('monotonic_align.core') else 1)" >/dev/null 2>&1; then
  echo "[dep] building monotonic_align" | tee -a "${LOG_DIR}/run-info.txt"
  (
    cd "${CODE_DIR}/modules/monotonic_align" && \
    mkdir -p monotonic_align && [ -f monotonic_align/__init__.py ] || echo "" > monotonic_align/__init__.py && \
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel cython && \
    "$PYTHON_BIN" setup.py build_ext --inplace && \
    "$PYTHON_BIN" -m pip install -e .
  ) || echo "[dep] monotonic_align build failed (will continue)" | tee -a "${LOG_DIR}/run-info.txt"
fi

#############################
# (已移除) 预下载模型权重
# 说明：Vevo 的推理脚本会在运行时用 huggingface_hub 自动下载所需权重，
#       因此这里不再提前下载，避免脚本里出现网络依赖。
#############################

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
