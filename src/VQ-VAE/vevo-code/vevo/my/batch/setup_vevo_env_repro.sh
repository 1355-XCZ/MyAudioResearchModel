#!/bin/bash
###############################################################################
#  文件名: setup_vevo_env_repro.sh
#  用途  : 在 Spartan 集群上可重复地重建/修复 Vevo 推理环境
#  说明  :
#   - 仅使用站点提供的模块与 pip/conda；不改系统级文件
#   - 多次执行是幂等的（已安装/已加载则跳过）
#   - 与 vevo_run3.slurm 的运行时环境保持一致
###############################################################################

set -euo pipefail

#############################
# ===== 站点模块与路径  =====
#############################
PROJECT_ROOT=${PROJECT_ROOT:-/data/gpfs/projects/punim2341/haoguangzhou}
CODE_DIR=${CODE_DIR:-"${PROJECT_ROOT}/voice/Amphion-VevoDev"}
CONDA_ENV=${CONDA_ENV:-vevo_env}

module purge
module load Anaconda3/2024.02-1

# 激活/创建 conda 环境（优先复用）
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "[conda] using existing env: ${CONDA_ENV}"
else
  echo "[conda] creating env: ${CONDA_ENV} (python=3.11)"
  conda create -y -n "${CONDA_ENV}" python=3.11
fi
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
echo "[conda] CONDA_PREFIX=${CONDA_PREFIX}"

#############################
# ===== CUDA / cuDNN / ORT =====
#############################
# 与 onnxruntime-gpu 1.22.x 兼容的站点模块（CUDA 12.x + cuDNN 9.x）
module load CUDA/12.4.1
module load cuDNN/9.6.0.74-CUDA-12.4.1 || true

# 兜底导出库路径（部分站点的 module 已设置，此处保证可见性）
export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)) 2>/dev/null)}
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

#############################
# ===== 缓存与通用环境变量 =====
#############################
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

# 线程与亲和性（避免 ORT/BLAS 报警与资源争用）
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export ORT_NUM_THREADS=${ORT_NUM_THREADS:-1}
export ORT_DISABLE_CPU_AFFINITY=1

#############################
# ===== Python 依赖（精简必要） =====
#############################
${PYTHON_BIN} - <<'PY'
import sys, subprocess
pkgs = [
  # 基础
  "setuptools", "wheel", "pip",
  # 运行核心
  "transformers==4.41.2", "accelerate==0.24.1",
  # IO / DSP
  "librosa", "soundfile", "ffmpeg-python",
  # 模型相关
  "encodec", "vocos",
  # G2P / 多语文本
  "phonemizer==3.2.1", "g2p_en", "pypinyin", "jieba", "cn2an", "LangSegment", "pyopenjtalk", "pykakasi",
  # ORT GPU（与 CUDA12/cuDNN9 匹配）
  "onnxruntime-gpu==1.22.0",
]
for p in pkgs:
    try:
        __import__(p.split("==")[0].split("[")[0].replace("-","_"))
    except Exception:
        print("[pip] installing", p)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-input", p], check=False)
print("[pip] done")
PY

#############################
# ===== 语音音素化依赖（系统级） =====
#############################
# 需加载 espeak-ng 模块供 phonemizer 使用
module load espeak-ng/1.52

#############################
# ===== 轻量验证 =====
#############################
${PYTHON_BIN} - <<'PY'
import onnxruntime as ort
from phonemizer.backend import EspeakBackend
import torch
print('[verify] torch:', getattr(torch, '__version__', 'unknown'))
print('[verify] ORT providers:', ort.get_available_providers())
EspeakBackend('en-us')
print('[verify] phonemizer EspeakBackend: OK')
PY

echo "[done] Vevo inference environment is ready."
echo "- CODE_DIR=${CODE_DIR}"
echo "- CONDA_ENV=${CONDA_ENV}"
echo "- CUDA_HOME=${CUDA_HOME:-unset}"

# 可选：完全离线（若缓存齐全）
# export HF_HUB_OFFLINE=1

exit 0


