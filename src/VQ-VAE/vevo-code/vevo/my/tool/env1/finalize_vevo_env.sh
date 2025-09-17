#!/bin/bash

# 将 numpy 固定到 2.0.x（与 numba 0.60 兼容）并将 torchaudio 与 torch(2.5.1) 对齐
# 然后进行一次完整校验

set -e

ENV_NAME="vevo_env"

echo "=== 初始化 conda 并激活环境 ==="
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME" || {
    echo "❌ 无法激活 conda 环境: $ENV_NAME"; exit 1;
  }
else
  echo "❌ 未检测到 conda，请手动激活环境后再运行：conda activate $ENV_NAME"; exit 1;
fi
echo "✅ 当前环境: $CONDA_DEFAULT_ENV"

echo "=== 步骤1：将 numpy 固定到 2.0.x（满足 numba 要求） ==="
pip install "numpy==2.0.*"

echo "=== 步骤2：将 torchaudio 与 torch(2.5.1) 对齐 ==="
pip install "torchaudio==2.5.1"

echo "=== 校验版本一致性 ==="
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

echo "=== 完成。若需恢复，仅需重新 pip 安装想要的版本；本脚本不改动系统环境，仅影响当前 conda 环境。 ==="


