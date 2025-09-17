#!/bin/bash
# Emotion2Vec VQ-VAE 训练脚本

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/../vevo-code"

# 默认参数
DATA_ROOT="./data"
EXP_DIR="./experiments/emotion2vec_vqvae"
CONFIG="config/emotion2vec_vqvae_config.json"
BATCH_SIZE=8
LEARNING_RATE=1e-4
MAX_STEPS=50000
USE_DUMMY="--use_dummy"  # 默认使用 dummy 特征进行测试

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --exp_dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --no_dummy)
            USE_DUMMY=""
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --data_root PATH        Path to training data (default: ./data)"
            echo "  --exp_dir PATH          Experiment directory (default: ./experiments/emotion2vec_vqvae)"
            echo "  --config PATH           Config file path (default: config/emotion2vec_vqvae_config.json)"
            echo "  --batch_size INT        Batch size (default: 8)"
            echo "  --learning_rate FLOAT   Learning rate (default: 1e-4)"
            echo "  --max_steps INT         Max training steps (default: 50000)"
            echo "  --no_dummy              Use real emotion2vec features instead of dummy"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 创建必要的目录
mkdir -p "$EXP_DIR"
mkdir -p "$(dirname "$DATA_ROOT")"

echo "Starting Emotion2Vec VQ-VAE Training..."
echo "Data root: $DATA_ROOT"
echo "Experiment dir: $EXP_DIR"
echo "Config: $CONFIG"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Max steps: $MAX_STEPS"
echo "Use dummy features: $([ -n "$USE_DUMMY" ] && echo "Yes" || echo "No")"

# 启动训练
python train_emotion2vec_vqvae.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --exp_dir "$EXP_DIR" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_steps "$MAX_STEPS" \
    $USE_DUMMY

echo "Training completed!"
