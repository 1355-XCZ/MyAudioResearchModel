#!/bin/bash

# VEVO 环境校验脚本（仅检查，不做修改）

set -e

ENV_NAME="vevo_env"

echo "=== 初始化 conda（若可用） ==="
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    conda activate "$ENV_NAME" || echo "⚠️  未能自动激活 $ENV_NAME，继续使用当前环境: $CONDA_DEFAULT_ENV"
  fi
fi
echo "当前环境: ${CONDA_DEFAULT_ENV:-system}"

echo "=== 检查系统可执行依赖（可选） ==="
command -v ffmpeg >/dev/null 2>&1 && echo "ffmpeg: OK" || echo "ffmpeg: 未找到（仅部分功能需要）"
command -v espeak-ng >/dev/null 2>&1 && echo "espeak-ng: OK" || echo "espeak-ng: 未找到（phonemizer 某些语言需要）"

echo "=== Python 依赖与功能校验 ==="
python - <<'PY'
import sys
print('Python:', sys.version)

def check(name, mod=None, extra=None):
    mod = mod or name
    try:
        __import__(mod)
        v = getattr(sys.modules[mod], '__version__', 'N/A')
        print(f'{name}: OK  ver={v}')
        return True
    except Exception as e:
        print(f'{name}: ERROR -> {e}')
        if extra:
            print('  hint:', extra)
        return False

ok = True
ok &= check('torch')
ok &= check('torchvision')
ok &= check('torchaudio')
ok &= check('numpy')
ok &= check('numba')

try:
    import torch
    print('cuda_available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('cuda_device:', torch.cuda.get_device_name(0))
except Exception as e:
    print('CUDA check error:', e)

ok &= check('monotonic_align.core', 'monotonic_align')
ok &= check('huggingface_hub')
ok &= check('librosa')
ok &= check('phonemizer', extra='缺 espeak-ng 时某些语言不可用')
ok &= check('g2p_en')
ok &= check('pypinyin')
ok &= check('jieba')
ok &= check('encodec')
ok &= check('vocos')

# 仅验证 Vevo 推理入口可导入
try:
    import models.vc.vevo.infer_vevotimbre as _
    print('vevo.infer_vevotimbre: OK')
except Exception as e:
    print('vevo.infer_vevotimbre: ERROR ->', e)
    ok = False

print('\nSUMMARY:', 'OK' if ok else 'ERROR')
sys.exit(0 if ok else 1)
PY

echo "=== 校验完成 ==="



