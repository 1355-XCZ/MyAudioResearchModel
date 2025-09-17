#!/bin/bash
###############################################################################
#  文件名: context_study_english_resume_compat.slurm
#  用途  : 英文上下文长度对比实验（兼容旧推理脚本），通过软链接仅处理缺失文件
#  原理  : 为每个 ctx 长度创建一个临时 style 目录，仅链接未生成的输入文件，
#          然后调用原来的 batch_infer_vevostyle.py，这样会自然跳过已完成的样本。
###############################################################################

#SBATCH --job-name=vevo-ctx-en-resume2
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/vevo-ctx-en-resume2-%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/vevo-ctx-en-resume2-%j.err

set -euo pipefail

PROJECT_ROOT=/data/gpfs/projects/punim2341/haoguangzhou
VEVO_ROOT=${PROJECT_ROOT}/voice/Amphion-VevoDev
STYLE_DIR=${STYLE_DIR:-"${PROJECT_ROOT}/data/MUStARD/audio_processed"}
BASE_OUT_DIR=${BASE_OUT_DIR:-"${PROJECT_ROOT}/data/MUStARD/context_study_english/batch_20250822_175551"}
CONTENT_WAV=${CONTENT_WAV:-"models/vc/vevo/wav/source.wav"}
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-"1600 2000 2400"}
LIMIT=${LIMIT:-}

cd "${VEVO_ROOT}"
module load Anaconda3/2024.02-1
module load espeak-ng/1.52 || true
eval "$(conda shell.bash hook)"
conda activate vevo_env

# 确保 Python 能导入项目根目录（解决 ModuleNotFoundError: models）
export PYTHONPATH="${VEVO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN=$(which python)
echo "[python] ${PYTHON_BIN}"

if [[ ! -d "${STYLE_DIR}" ]]; then echo "[错误] STYLE_DIR 不存在: ${STYLE_DIR}"; exit 1; fi
if [[ ! -f "${CONTENT_WAV}" ]]; then echo "[错误] CONTENT_WAV 不存在: ${CONTENT_WAV}"; exit 1; fi

# 统计总输入
TOTAL_IN=$(find "${STYLE_DIR}" -name "*.wav" | wc -l)
echo "[info] 输入共 ${TOTAL_IN} 条"

process_ctx() {
  local ctx_len="$1"
  local cfg="./models/vc/vevo/config/Vq32ToVq8192_ctx${ctx_len}.json"
  local out_dir="${BASE_OUT_DIR}/ctx_${ctx_len}"
  local tmp_root="${PROJECT_ROOT}/tmp/vevo_resume_ctx${ctx_len}_$SLURM_JOB_ID"

  if [[ ! -f "${cfg}" ]]; then echo "[错误] 缺少配置: ${cfg}"; exit 1; fi
  mkdir -p "${out_dir}" "${tmp_root}"

  echo "[ctx ${ctx_len}] 构建缺失样本列表..."
  missing=0
  linked=0

  # 遍历所有 style 源，若对应输出不存在，则在 tmp_root 建立同结构的软链接
  while IFS= read -r src; do
    rel="${src#${STYLE_DIR}/}"
    stem="${rel%.wav}"
    out_path="${out_dir}/${stem}_vevostyle.wav"

    if [[ -f "${out_path}" ]]; then
      continue
    fi

    dest_dir="${tmp_root}/$(dirname "${rel}")"
    mkdir -p "${dest_dir}"
    ln -s "${src}" "${dest_dir}/$(basename "${rel}")" || true
    missing=$((missing+1))
  done < <(find "${STYLE_DIR}" -type f -name "*.wav" | sort)

  if [[ -n "${LIMIT}" ]]; then
    echo "[ctx ${ctx_len}] LIMIT=${LIMIT} 开启，将只链接前 ${LIMIT} 条"
    # 重新构建受限的 tmp 目录
    rm -rf "${tmp_root}"
    mkdir -p "${tmp_root}"
    count=0
    while IFS= read -r src; do
      rel="${src#${STYLE_DIR}/}"
      stem="${rel%.wav}"
      out_path="${out_dir}/${stem}_vevostyle.wav"
      if [[ -f "${out_path}" ]]; then continue; fi
      dest_dir="${tmp_root}/$(dirname "${rel}")"
      mkdir -p "${dest_dir}"
      ln -s "${src}" "${dest_dir}/$(basename "${rel}")" || true
      count=$((count+1))
      if [[ ${count} -ge ${LIMIT} ]]; then break; fi
    done < <(find "${STYLE_DIR}" -type f -name "*.wav" | sort)
    missing=${count}
  fi

  echo "[ctx ${ctx_len}] 需要处理: ${missing} 条（其余已存在将被跳过）"
  if [[ ${missing} -eq 0 ]]; then
    echo "[ctx ${ctx_len}] 已全部完成，跳过推理"
    rm -rf "${tmp_root}"
    return 0
  fi

  echo "[ctx ${ctx_len}] 开始推理（调用原脚本）"
  set +e
  python models/vc/vevo/my/dataverse_files/batch_infer_vevostyle.py \
    --content-wav "${CONTENT_WAV}" \
    --style-dir "${tmp_root}" \
    --out-dir "${out_dir}" \
    --config "${cfg}" \
    --style-max-seconds 10 \
    --content-max-seconds 8
  ret=$?
  set -e

  echo "[ctx ${ctx_len}] 推理返回码: ${ret}"
  echo "[ctx ${ctx_len}] 当前已生成: $(find "${out_dir}" -name "*_vevostyle.wav" | wc -l) 条"

  rm -rf "${tmp_root}"
}

for ctx in ${CONTEXT_LENGTHS}; do
  echo "=== 开始上下文长度 ${ctx} ==="
  process_ctx "${ctx}"
  echo "=== 完成上下文长度 ${ctx} ==="
done

TOTAL=$(find "${BASE_OUT_DIR}" -name "*_vevostyle.wav" | wc -l)
echo "[总计] 已生成 ${TOTAL} 条"
