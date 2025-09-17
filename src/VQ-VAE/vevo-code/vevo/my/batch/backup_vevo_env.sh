#!/bin/bash
###############################################################################
# 文件名: backup_vevo_env.sh
# 用途  : 备份/快照当前 Conda 与模块环境，便于后续可重复恢复
# 位置  : 放在与 vevo_run3.slurm 同一目录下执行
###############################################################################

set -euo pipefail

# 可通过环境变量或参数指定环境名：CONDA_ENV 或第1参数
CONDA_ENV=${CONDA_ENV:-${1:-vevo_env}}

# 选择合适的 Anaconda 模块（与运行脚本一致）
module load Anaconda3/2024.02-1 2>/dev/null || true

# 激活 Conda 环境
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
	echo "[ERR] conda env '${CONDA_ENV}' not found. Aborting." >&2
	exit 1
fi
conda activate "${CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

# 快照输出目录（同目录下 env_snapshots/<timestamp>）
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 允许外部指定快照根目录（默认放在脚本同目录下的 env_snapshots）
SNAP_BASE=${SNAP_BASE:-"${SCRIPT_DIR}/env_snapshots"}
TS=$(date +%Y%m%d_%H%M%S)
SNAP_DIR="${SNAP_BASE}/${TS}"
mkdir -p "${SNAP_DIR}"

echo "[info] snapshot dir: ${SNAP_DIR}"
echo "[info] CONDA_PREFIX : ${CONDA_PREFIX}"

# 1) 最小可复现（无 build 号，含 pip 依赖同步）
conda env export -n "${CONDA_ENV}" --no-builds > "${SNAP_DIR}/env.lock.yml"

# 2) 完整快照（含 build/渠道）
conda env export -n "${CONDA_ENV}" > "${SNAP_DIR}/env.full.yml"

# 3) conda 显式规格（精准恢复）
conda list -n "${CONDA_ENV}" --explicit > "${SNAP_DIR}/env.conda-spec.txt"

# 4) pip 精确版本
pip freeze > "${SNAP_DIR}/env.pip.txt"

# 5) 记录已加载模块（CUDA/cuDNN等）
module list 2>&1 | sed 's/\x1B\[[0-9;]*[mK]//g' > "${SNAP_DIR}/modules.txt" || true

# 6) 运行时验证（ORT providers、torch 版本、espeak 版本）
"${PYTHON_BIN}" - <<'PY' > "${SNAP_DIR}/verify.txt" 2>&1 || true
import onnxruntime as ort, torch
print("torch:", getattr(torch,'__version__','?'))
print("ORT providers:", ort.get_available_providers())
PY
espeak-ng --version 2>/dev/null >> "${SNAP_DIR}/verify.txt" || true

# 7) 记录关键环境变量（便于恢复缓存/线程策略）
{
  echo "HF_HOME=${HF_HOME:-}"
  echo "TORCH_HOME=${TORCH_HOME:-}"
  echo "CUDA_HOME=${CUDA_HOME:-}"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
  echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-}"
  echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-}"
  echo "KMP_AFFINITY=${KMP_AFFINITY:-}"
  echo "ORT_NUM_THREADS=${ORT_NUM_THREADS:-}"
  echo "ORT_DISABLE_CPU_AFFINITY=${ORT_DISABLE_CPU_AFFINITY:-}"
} > "${SNAP_DIR}/env.vars"

# 8) 打包归档（可直接搬运保存）
tar -C "${SNAP_BASE}" -czf "${SNAP_DIR}.tgz" "${TS}"
echo "[done] snapshot archived: ${SNAP_DIR}.tgz"

echo "恢复建议："
cat <<'TXT'
- 方式A（推荐）：
  conda env create -n vevo_env -f env.lock.yml
  pip install -r env.pip.txt

- 方式B（显式规格）：
  conda create -n vevo_env --file env.conda-spec.txt
  pip install -r env.pip.txt

- 运行前加载模块（示例，与 vevo_run3.slurm 一致）：
  module load CUDA/12.4.1
  module load cuDNN/9.6.0.74-CUDA-12.4.1
  module load Anaconda3/2024.02-1
  module load espeak-ng/1.52
TXT

exit 0


