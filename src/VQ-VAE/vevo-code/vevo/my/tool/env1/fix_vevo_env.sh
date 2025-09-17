#!/bin/bash

# VEVO 环境一键修复脚本（Conda 环境 vevo_env）
# 作用：
# 1) 初始化 conda 并激活 vevo_env
# 2) 修复 monotonic_align Cython 编译失败
# 3) 同步 torchvision 版本以匹配已安装的 torch（当前 torch=2.8.0 ）
# 4) 解决 frechet_audio_distance 带来的依赖冲突（默认卸载）
# 5) 补全 fairseq 运行期依赖
# 6) 最终一致性校验

set -e

ENV_NAME="vevo_env"

echo "=== [1/6] 初始化 conda 并激活环境 ==="
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME" || {
    echo "❌ 无法激活 conda 环境: $ENV_NAME"; exit 1;
  }
else
  echo "❌ 未检测到 conda，请在交互终端先执行 'module load Anaconda' 或联系管理员"; exit 1;
fi
echo "✅ 当前环境: $CONDA_DEFAULT_ENV"

echo "=== [2/6] 修复 monotonic_align 编译 ==="
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../../../.." && pwd)"
MONO_DIR="$PROJECT_ROOT/modules/monotonic_align"
echo "项目根目录: $PROJECT_ROOT"
echo "Monotonic align目录: $MONO_DIR"
if [ -d "$MONO_DIR" ]; then
  cd "$MONO_DIR"
  mkdir -p monotonic_align
  [ -f monotonic_align/__init__.py ] || touch monotonic_align/__init__.py
  python setup.py build_ext --inplace || {
    echo "⚠️  monotonic_align 编译失败，请稍后手工重试";
  }
  cd - >/dev/null
else
  echo "⚠️  未找到 $MONO_DIR，跳过编译"
fi

echo "=== [3/6] 同步 Torch / TorchVision 版本 ==="
# 已安装 torch=2.8.0（从安装日志可见），同步安装兼容的 torchvision 版本
pip install -U torchvision==0.20.1 || true

echo "=== [4/6] 处理 frechet_audio_distance 冲突 ==="
# 推理环境不强依赖该包，默认移除，避免 numpy/scipy/transformers 被回退
pip uninstall -y frechet_audio_distance || true

echo "=== [5/6] 补全 fairseq 依赖（保持 omegaconf=2.3.0） ==="
pip install -U hydra-core==1.3.2 sacrebleu bitarray || true

echo "=== [6/6] 最终一致性校验 ==="
python - <<'PY'
import sys
print('Python:', sys.version)
try:
    import torch
    print('torch:', torch.__version__)
    import torchvision
    print('torchvision:', torchvision.__version__)
    import transformers, librosa
    print('transformers:', transformers.__version__, 'librosa:', librosa.__version__)
    # monotonic_align
    try:
        import monotonic_align.core as mac
        print('monotonic_align: OK')
    except Exception as e:
        print('monotonic_align: ERROR ->', e)
    # fairseq 关键依赖
    missing = []
    for m in ['hydra', 'hydra.core', 'sacrebleu', 'bitarray']:
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    print('fairseq 依赖缺失:', missing if missing else 'None')
except Exception as e:
    print('校验失败:', e)
    sys.exit(1)
PY

echo "=== 修复完成 ==="
echo "若仍有冲突，建议：\n- frechet_audio_distance 放到单独评测环境安装\n- 集群上通过 module 加载 ffmpeg / espeak-ng（或联系管理员安装）"


