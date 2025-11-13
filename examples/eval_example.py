#!/usr/bin/env python3
"""
Example: Evaluation with new configuration system

Usage:
    # Default evaluation
    python examples/eval_example.py --dataset configs/datasets/iemocap.yaml
    
    # Quick test
    python examples/eval_example.py \
      --dataset configs/datasets/iemocap.yaml \
      --experiment configs/experiments/quick_test.yaml
    
    # Custom rate points
    python examples/eval_example.py \
      --dataset configs/datasets/iemocap.yaml \
      --rates-bpf "10,30,50,100"
    
    # Override samples
    python examples/eval_example.py \
      --dataset configs/datasets/iemocap.yaml \
      --samples-per-emotion 20
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import setup_config


def main():
    # Load configuration
    config, args = setup_config()
    
    print("="*60)
    print("Evaluation Configuration")
    print("="*60)
    
    # Dataset info
    print("\n[Dataset]")
    dataset_name = config.get('dataset.name', 'Not specified')
    print(f"  Name: {dataset_name}")
    print(f"  Emotions: {config.get('dataset.emotions', [])}")
    print(f"  Samples per emotion: {config.get('dataset.samples_per_emotion')}")
    
    # Evaluation methods
    print("\n[Evaluation Methods]")
    
    # Rate sweep
    if config.get('evaluation.rate_sweep.enabled'):
        print("  ✓ Rate Sweep:")
        rates = config.get('evaluation.rate_sweep.rates_bpf')
        print(f"    Rate points: {rates}")
        print(f"    Number of points: {len(rates)}")
    
    # Layer sweep
    if config.get('evaluation.layer_sweep.enabled'):
        print("  ✓ Layer Sweep:")
        layers = config.get('evaluation.layer_sweep.layers')
        print(f"    Layer points: {layers}")
        print(f"    Number of points: {len(layers)}")
    
    # Classifier
    print("\n[Classifier]")
    print(f"  Model: {config.get('evaluation.emotion2vec_model')}")
    print(f"  Hub: {config.get('evaluation.emotion2vec_hub')}")
    
    # Paths
    print("\n[Paths]")
    print(f"  Checkpoints: {config.get('paths.checkpoints', 'checkpoints')}")
    print(f"  Results: {config.get('paths.results', 'evaluation_results')}")
    
    print("\n" + "="*60)
    print("✅ Ready to run evaluation!")
    print("="*60)
    
    # Show command that would be run
    print("\n[Equivalent Command]")
    print(f"  Dataset: {dataset_name}")
    print(f"  Rates: {config.get('evaluation.rate_sweep.rates_bpf')}")
    print(f"  Samples: {config.get('dataset.samples_per_emotion')}")
    
    return config


if __name__ == "__main__":
    config = main()

