#!/bin/bash
###############################################################################
#  文件名: vevo_sanity.slurm
#  用途  : Spartan 上快速校验 VEVO 运行环境（不做耗时推理）
#           - 激活 conda 环境
#           - 校验 PYTHONPATH
#           - 打印 GPU/环境信息
#           - 测试关键模块导入
###############################################################################

#############################
# ===== Slurm 资源申请 =====
#############################
# 默认可被 sbatch 参数覆盖：sbatch -p <PART> -t <TIME> vevo_sanity.slurm
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -J vevo-sanity
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.err
#SBATCH -A punim2341

set -euo pipefail

#############################
# ===== 路径与环境变量  =====
#############################
PROJECT_ROOT=/data/gpfs/projects/punim2341/haoguangzhou
CODE_DIR="${PROJECT_ROOT}/voice/Amphion-VevoDev"
CONDA_ENV=vevo_env

mkdir -p "${PROJECT_ROOT}/logs"

# HF/torch 缓存到项目目录，避免 HOME 爆仓（仅设置 HF_HOME，避免 transformers 的弃用警告）
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

#############################
# ===== 加载与激活环境  =====
#############################
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# 确保 Python 能导入项目内模块
: "${PYTHONPATH:=}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

# 选用当前环境的 python 可执行文件，避免落到系统 python
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}"
PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(which python)"
echo "PYTHON_BIN: $PYTHON_BIN"

#############################
# ===== 打印环境/GPU信息 =====
#############################
echo "==== SANITY START ===="
echo "HOSTNAME: $(hostname)"
echo "WORKDIR : ${CODE_DIR}"
echo "ENV     : ${CONDA_DEFAULT_ENV}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

#############################
# ===== 关键模块导入测试 =====
#############################
cd "${CODE_DIR}"
"$PYTHON_BIN" - <<'PY'
import sys
print('Python:', sys.version)

def check(name, mod=None):
    mod = mod or name
    try:
        __import__(mod)
        v = getattr(sys.modules[mod], '__version__', 'N/A')
        print(f'[OK] {name:24s} ver={v}')
    except Exception as e:
        print(f'[ERR] {name:24s}', e)
        raise

check('torch')
check('torchvision')
check('torchaudio')
check('numpy')
check('numba')
check('huggingface_hub')
check('librosa')
check('json5')
check('ruamel.yaml','ruamel')
check('easydict')
check('encodec')
check('vocos')
check('monotonic_align.core','monotonic_align')

# 仅校验 Vevo 推理入口能导入
__import__('models.vc.vevo.infer_vevotimbre')
print('[OK] models.vc.vevo.infer_vevotimbre import')
print('SUMMARY: OK')
PY

# 生成最小测试音频并尝试执行四个推理脚本（失败不终止，用于冒烟）
echo "==== INFER SMOKE TEST ===="
WAV_DIR="${CODE_DIR}/models/vc/vevo/wav"
mkdir -p "$WAV_DIR"
export WAV_DIR  # 传递到 Python 子进程
"$PYTHON_BIN" - <<'PY'
import numpy as np, soundfile as sf, os
sr=24000
t=np.linspace(0,0.6,int(sr*0.6),endpoint=False)
sig=np.sin(2*np.pi*440*t).astype(np.float32)
base=os.environ.get('WAV_DIR')
if not base:
    base=os.path.join('.', 'models','vc','vevo','wav')
    os.makedirs(base, exist_ok=True)
sf.write(os.path.join(base,'source.wav'),sig,sr)
sf.write(os.path.join(base,'arabic_male.wav'),sig*0.8,sr)
sf.write(os.path.join(base,'mandarin_female.wav'),sig*0.6,sr)
print('test wavs prepared in',base)
PY

set +e
echo "-- timbre --";   "$PYTHON_BIN" -m models.vc.vevo.infer_vevotimbre   || echo "timbre failed"
echo "-- style  --";   "$PYTHON_BIN" -m models.vc.vevo.infer_vevostyle    || echo "style failed"
echo "-- voice  --";   "$PYTHON_BIN" -m models.vc.vevo.infer_vevovoice    || echo "voice failed"
echo "-- tts    --";   "$PYTHON_BIN" -m models.vc.vevo.infer_vevotts      || echo "tts failed"
set -e

echo "==== SANITY DONE ===="

