# Emotion Recognition under RVQ Information Bottleneck

Research on emotion recognition robustness under information bottleneck with grouped RVQ and entropy coding.

## Overview

This project studies how emotion recognition performance degrades under different compression rates using:
- **Grouped RVQ**: 12 groups × 16 layers with SKIP mechanism
- **Entropy Coding**: ECVQ decision making with precise bitrate control  
- **Entropy Model**: Unconditional autoregressive Transformer for q(z)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration System

Use YAML configs for different experiments:

```bash
# Training with default settings
python train_rvq.py --config configs/default.yaml

# Evaluation on IEMOCAP
python run_evaluation_iemocap.py --dataset configs/datasets/iemocap.yaml

# Quick test
python run_evaluation_iemocap.py \
  --dataset configs/datasets/iemocap.yaml \
  --experiment configs/experiments/quick_test.yaml

# Override parameters
python run_evaluation_iemocap.py \
  --dataset configs/datasets/iemocap.yaml \
  --samples-per-emotion 50 \
  --rates-bpf "10,30,50,100"
```

See `configs/README.md` for details.

### Training

```bash
# Train RVQ model
sbatch scripts/run_train_rvq.slurm

# Train entropy model
sbatch scripts/run_train_entropy.slurm
```

### Evaluation

```bash
# Extract features
sbatch scripts/run_extract_features.slurm

# Run evaluation (parallel)
sbatch scripts/run_eval_iemocap.slurm
sbatch scripts/run_eval_ravdess.slurm
sbatch scripts/run_eval_esd.slurm
```

## Project Structure

```
├── config.py              # Configuration
├── grouped_rvq.py         # Grouped RVQ model
├── entropy_model.py       # Entropy model
├── rate_controller.py     # Rate control
├── datasets/              # Dataset loaders
├── evaluation/            # Evaluation methods
├── train_rvq.py          # RVQ training
├── train_entropy.py      # Entropy training
├── run_evaluation_*.py   # Evaluation scripts
└── scripts/              # Slurm scripts
```

## Core Methods

### ECVQ Decision
```
J(k) = D(k) + λ · (-log₂ q(k|ctx))
decision = argmin(J) over {k ∈ codebook, SKIP}
```

### Rate Control
Binary search for λ to achieve target bitrate (tolerance < 1 BPF).

### Entropy Model
```
q(z) = ∏_{t,g,m} q(k_{t,g,m} | history_{<t,g,m})
```

## Datasets

- **IEMOCAP**: 400 samples (4 emotions × 100)
- **RAVDESS**: 700 samples (7 emotions × 100)
- **ESD**: 500 samples (5 emotions × 100, EN+ZH)

## Results

Results saved in `evaluation_results/`:
- `rate_sweep/` - Rate vs accuracy  
- `layer_sweep/` - Layer vs accuracy
- `analysis/` - Plots and analysis

## License

MIT License
