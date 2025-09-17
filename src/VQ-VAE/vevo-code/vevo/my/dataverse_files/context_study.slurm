#!/bin/bash
###############################################################################
#  文件名: context_study.slurm
#  用途  : 上下文窗口长度对比研究 (1600/2000/2400)
#          每个长度设置生成一组输出，便于效果对比
###############################################################################

#############################
# ===== Slurm 资源申请 =====
#############################
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -J vevo-ctx-study
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

# 研究参数
STYLE_DIR=${STYLE_DIR:-"${PROJECT_ROOT}/data/MCSD/vevo_data_split/tkxdh_s5"}
CONTENT_WAV=${CONTENT_WAV:-"models/vc/vevo/wav/mandarin_female.wav"}
LIMIT=${LIMIT:-}  # 处理全部文件

# 生成批处理编号（基于时间戳）
BATCH_ID=${BATCH_ID:-$(date +"%Y%m%d_%H%M%S")}
BASE_OUT_DIR=${BASE_OUT_DIR:-"${PROJECT_ROOT}/data/MCSD/context_study/batch_${BATCH_ID}"}

# 上下文长度设置
CONTEXT_LENGTHS=(1600 2000 2400)

# 统一创建项目根 logs（给 Slurm -o/-e 用）
mkdir -p "${PROJECT_ROOT}/logs"

# 模型/权重缓存放项目目录
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

export TMPDIR=${TMPDIR:-/tmp}

#############################
# ===== 环境激活与路径  =====
#############################
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 上下文窗口长度研究开始 ===" | tee "${PROJECT_ROOT}/logs/context_study.log"
echo "[批处理] 批次编号: ${BATCH_ID}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
echo "[输出目录] ${BASE_OUT_DIR}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"

# 加载模块系统（参考 vevo_run3.slurm）
module purge
module load Anaconda3/2024.02-1
module load foss/2022a || { echo "[sys] ERROR: failed to load toolchain foss/2022a" | tee -a "${PROJECT_ROOT}/logs/context_study.log"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# CUDA/cuDNN for GPU support
module load CUDA/12.4.1
module load cuDNN/9.6.0.74-CUDA-12.4.1

# 线程控制（参考 vevo_run3.slurm）
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export ORT_NUM_THREADS=${ORT_NUM_THREADS:-1}
export ORT_DISABLE_CPU_AFFINITY=1

# 设置 PYTHONPATH
: "${PYTHONPATH:=}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

# 切换到代码目录
cd "${CODE_DIR}"

# 选用当前环境的 python
PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(which python)"
echo "[conda] 已激活环境: ${CONDA_ENV}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
echo "[python] 使用: ${PYTHON_BIN}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"

# 加载必要的系统模块
echo "[sys] loading system modules" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
module load espeak-ng/1.52 && {
  echo "[sys] espeak-ng/1.52 module loaded successfully" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
} || {
  echo "[sys] ERROR: espeak-ng/1.52 module load failed" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
  exit 1
}

#############################
# ===== 依赖检查与安装  =====
#############################
echo "[deps] 检查依赖..." | tee -a "${PROJECT_ROOT}/logs/context_study.log"

REQUIRED_PACKAGES=("torch" "torchaudio" "transformers" "librosa" "soundfile")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import ${pkg}" 2>/dev/null; then
        echo "[deps] 安装 ${pkg}..." | tee -a "${PROJECT_ROOT}/logs/context_study.log"
        pip install "${pkg}"
    else
        echo "[deps] ✓ ${pkg} 已安装" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    fi
done

#############################
# ===== 预检查  =====
#############################
if [[ ! -d "${STYLE_DIR}" ]]; then
    echo "[错误] 风格音频目录不存在: ${STYLE_DIR}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    exit 1
fi

if [[ ! -f "${CONTENT_WAV}" ]]; then
    echo "[错误] 内容音频文件不存在: ${CONTENT_WAV}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    exit 1
fi

WAV_COUNT=$(find "${STYLE_DIR}" -name "*.wav" | wc -l)
if [[ -n "${LIMIT}" ]]; then
    echo "[info] 发现 ${WAV_COUNT} 个 WAV 文件，将处理前 ${LIMIT} 个" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
else
    echo "[info] 发现 ${WAV_COUNT} 个 WAV 文件，将处理全部" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
fi

if [[ ${WAV_COUNT} -eq 0 ]]; then
    echo "[错误] 在 ${STYLE_DIR} 中未找到 WAV 文件" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    exit 1
fi

#############################
# ===== 主要研究循环  =====
#############################
echo "[研究] 开始上下文长度对比实验" | tee -a "${PROJECT_ROOT}/logs/context_study.log"

for ctx_len in "${CONTEXT_LENGTHS[@]}"; do
    echo "" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    echo "========================================" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    echo "[实验] 测试上下文长度: ${ctx_len}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    echo "========================================" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    
    # 设置输出目录
    OUT_DIR="${BASE_OUT_DIR}/ctx_${ctx_len}"
    mkdir -p "${OUT_DIR}"
    
    # 选择对应的配置文件
    if [[ ${ctx_len} -eq 1600 ]]; then
        CONFIG_FILE="models/vc/vevo/config/Vq32ToVq8192_ctx1600.json"
    elif [[ ${ctx_len} -eq 2000 ]]; then
        CONFIG_FILE="models/vc/vevo/config/Vq32ToVq8192.json"  # 默认配置
    elif [[ ${ctx_len} -eq 2400 ]]; then
        CONFIG_FILE="models/vc/vevo/config/Vq32ToVq8192_ctx2400.json"
    else
        echo "[错误] 未知的上下文长度: ${ctx_len}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
        continue
    fi
    
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "[错误] 配置文件不存在: ${CONFIG_FILE}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
        continue
    fi
    
    echo "[配置] 使用配置文件: ${CONFIG_FILE}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    echo "[输出] 保存到: ${OUT_DIR}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    
    # 构建 limit 参数
    EXTRA_LIMIT_ARG=""
    if [[ -n "${LIMIT}" ]]; then
        EXTRA_LIMIT_ARG="--limit ${LIMIT}"
    fi
    
    # 执行推理
    echo "[执行] 开始推理..." | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    
    START_TIME=$(date +%s)
    
    # 临时修改配置文件路径的环境变量（如果推理脚本支持的话）
    # 否则需要修改 batch_infer_vevostyle.py 来接受 config 参数
    
    "$PYTHON_BIN" models/vc/vevo/my/dataverse_files/batch_infer_vevostyle.py \
      --style-dir "${STYLE_DIR}" \
      --out-dir   "${OUT_DIR}" \
      --content-wav "${CONTENT_WAV}" \
      --config "${CONFIG_FILE}" \
      ${EXTRA_LIMIT_ARG} \
      2>&1 | tee -a "${PROJECT_ROOT}/logs/context_study_${ctx_len}.log"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "[完成] 上下文长度 ${ctx_len} 推理完成，用时 ${DURATION} 秒" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    
    # 统计输出文件数量
    OUTPUT_COUNT=$(find "${OUT_DIR}" -name "*.wav" | wc -l)
    echo "[统计] 生成了 ${OUTPUT_COUNT} 个音频文件" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    
done

#############################
# ===== 研究总结  =====
#############################
echo "" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
echo "========================================" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
echo "[总结] 上下文长度对比研究完成" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
echo "========================================" | tee -a "${PROJECT_ROOT}/logs/context_study.log"

for ctx_len in "${CONTEXT_LENGTHS[@]}"; do
    OUT_DIR="${BASE_OUT_DIR}/ctx_${ctx_len}"
    if [[ -d "${OUT_DIR}" ]]; then
        OUTPUT_COUNT=$(find "${OUT_DIR}" -name "*.wav" | wc -l)
        TOTAL_SIZE=$(du -sh "${OUT_DIR}" | cut -f1)
        echo "[结果] ctx_${ctx_len}: ${OUTPUT_COUNT} 文件, 总大小 ${TOTAL_SIZE}" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
    fi
done

echo "" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
echo "[提示] 结果保存在: ${BASE_OUT_DIR}/" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
echo "[提示] 可以通过听觉测试对比不同上下文长度的效果" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 研究结束 ===" | tee -a "${PROJECT_ROOT}/logs/context_study.log"
