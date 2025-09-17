#!/bin/bash

# 一键稳定 VEVO 推理环境（仅影响 conda 环境 vevo_env）
# - 卸载 frechet_audio_distance（避免强制回退 numpy/scipy/transformers）
# - 固定 numpy 到 2.0.x（与 numba 0.60 兼容）
# - 将 torchaudio 对齐到 torch==2.5.1
# - 同步 torchvision==0.20.1（与 torch==2.5.1 兼容）
# - 编译并以开发模式安装 modules/monotonic_align
# - 最终校验关键依赖

set -e

ENV_NAME="vevo_env"

echo "=== 初始化 conda 并激活环境 ==="
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME" || {
    echo "❌ 无法激活 conda 环境: $ENV_NAME"; exit 1;
  }
else
  echo "❌ 未检测到 conda，请先手动激活：conda activate $ENV_NAME"; exit 1;
fi
echo "✅ 当前环境: $CONDA_DEFAULT_ENV"

echo "=== 卸载 frechet_audio_distance（避免依赖冲突） ==="
pip uninstall -y frechet_audio_distance || true

echo "=== 固定 numpy 到 2.0.x（满足 numba） ==="
pip install "numpy==2.0.*"

echo "=== 将 torchaudio 对齐到 torch==2.5.1 ==="
pip install "torchaudio==2.5.1"

echo "=== 同步 torchvision==0.20.1（与 torch==2.5.1 兼容） ==="
pip install -U "torchvision==0.20.1"

echo "=== 编译并安装 monotonic_align ==="
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../../../.." && pwd)"
MONO_DIR="$PROJECT_ROOT/modules/monotonic_align"
echo "项目根目录: $PROJECT_ROOT"
echo "Monotonic align 目录: $MONO_DIR"
if [ -d "$MONO_DIR" ]; then
  cd "$MONO_DIR"
  mkdir -p monotonic_align
  [ -f monotonic_align/__init__.py ] || touch monotonic_align/__init__.py
  python setup.py build_ext --inplace || {
    echo "⚠️  Cython 编译有警告或失败，继续尝试可导入检查";
  }
  pip install -e .
  cd - >/dev/null
else
  echo "⚠️  未找到 $MONO_DIR，跳过编译"
fi

echo "=== 最终校验（版本与关键模块导入） ==="
python - <<'PY'
import sys
print('Python:', sys.version)
import torch, torchvision, numpy
print('torch:', torch.__version__)
try:
    import torchaudio
    print('torchaudio:', torchaudio.__version__)
except Exception as e:
    print('torchaudio ERROR:', e)
print('torchvision:', torchvision.__version__)
print('numpy:', numpy.__version__)
try:
    import numba
    print('numba:', numba.__version__)
except Exception as e:
    print('numba ERROR:', e)
try:
    import monotonic_align.core as mac
    print('monotonic_align: OK')
except Exception as e:
    print('monotonic_align ERROR:', e)
PY

echo "=== 完成：环境已稳定（仅影响 vevo_env）。如需回滚，重装对应版本即可。 ==="


