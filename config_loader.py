"""
Modern Configuration System for Emotion RVQ Bottleneck

Supports:
- Hierarchical YAML configurations
- Command-line argument overrides
- Multiple config file merging
- Environment variable substitution
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import os


class ConfigLoader:
    """Load and merge YAML configuration files"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_yaml(self, path: Path) -> Dict:
        """Load a single YAML file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            elif value is not None:  # Only override if not None
                result[key] = value
        
        return result
    
    def load(self, 
             config_files: List[str],
             overrides: Optional[Dict] = None) -> Dict:
        """
        Load and merge multiple config files
        
        Args:
            config_files: List of config file paths (loaded in order)
            overrides: Dictionary of override values
            
        Returns:
            Merged configuration dictionary
        """
        config = {}
        
        # Load each config file and merge
        for config_file in config_files:
            path = Path(config_file)
            if not path.is_absolute():
                path = self.config_dir / path
            
            if path.exists():
                file_config = self.load_yaml(path)
                config = self.merge_configs(config, file_config)
            else:
                print(f"Warning: Config file not found: {path}")
        
        # Apply overrides
        if overrides:
            config = self.merge_configs(config, overrides)
        
        return config
    
    def substitute_env_vars(self, config: Dict) -> Dict:
        """Substitute environment variables in config values"""
        result = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                result[key] = self.substitute_env_vars(value)
            elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                result[key] = os.environ.get(env_var, value)
            else:
                result[key] = value
        
        return result


class ArgumentParser:
    """Parse command-line arguments for configuration"""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description='Emotion RVQ Bottleneck Training/Evaluation',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Configuration files
        parser.add_argument('--config', type=str, default='default.yaml',
                          help='Base configuration file')
        parser.add_argument('--dataset', type=str, default=None,
                          help='Dataset config (e.g., datasets/iemocap.yaml)')
        parser.add_argument('--experiment', type=str, default=None,
                          help='Experiment config (e.g., experiments/quick_test.yaml)')
        
        # Common overrides
        parser.add_argument('--batch-size', type=int, default=None,
                          help='Override batch size')
        parser.add_argument('--num-epochs', type=int, default=None,
                          help='Override number of epochs')
        parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                          help='Override learning rate')
        
        # Paths
        parser.add_argument('--data-root', type=str, default=None,
                          help='Override data root directory')
        parser.add_argument('--checkpoint-dir', type=str, default=None,
                          help='Override checkpoint directory')
        parser.add_argument('--output-dir', type=str, default=None,
                          help='Override output directory')
        
        # Evaluation specific
        parser.add_argument('--samples-per-emotion', type=int, default=None,
                          help='Samples per emotion for evaluation')
        parser.add_argument('--rates-bpf', type=str, default=None,
                          help='Comma-separated rate points (e.g., "10,30,50")')
        
        # Mode
        parser.add_argument('--mode', type=str, choices=['train', 'eval'], default=None,
                          help='Run mode')
        
        # Device
        parser.add_argument('--device', type=str, default='cuda',
                          help='Device (cuda/cpu)')
        parser.add_argument('--seed', type=int, default=None,
                          help='Random seed')
        
        return parser
    
    @staticmethod
    def args_to_overrides(args: argparse.Namespace) -> Dict:
        """Convert parsed arguments to configuration overrides"""
        overrides = {}
        
        # Training overrides
        if args.batch_size is not None:
            overrides.setdefault('training', {})
            overrides['training']['batch_size'] = args.batch_size
        
        if args.num_epochs is not None:
            overrides.setdefault('training', {})
            overrides['training']['num_epochs'] = args.num_epochs
        
        if args.lr is not None:
            overrides.setdefault('training', {})
            overrides['training']['learning_rate'] = args.lr
        
        # Path overrides
        if args.data_root is not None:
            overrides.setdefault('paths', {})['data_root'] = args.data_root
        
        if args.checkpoint_dir is not None:
            overrides.setdefault('paths', {})['checkpoints'] = args.checkpoint_dir
        
        if args.output_dir is not None:
            overrides.setdefault('paths', {})['results'] = args.output_dir
        
        # Evaluation overrides
        if args.samples_per_emotion is not None:
            overrides.setdefault('dataset', {})['samples_per_emotion'] = args.samples_per_emotion
        
        if args.rates_bpf is not None:
            rates = [float(x) if x != 'inf' else float('inf') 
                    for x in args.rates_bpf.split(',')]
            overrides.setdefault('evaluation', {}).setdefault('rate_sweep', {})['rates_bpf'] = rates
        
        # Other
        if args.seed is not None:
            overrides['seed'] = args.seed
        
        return overrides


class Config:
    """Main configuration class"""
    
    def __init__(self, config_dict: Dict):
        self._config = config_dict
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path
        
        Example: config.get('training.rvq.batch_size')
        """
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, path: str, value: Any):
        """Set configuration value by dot-separated path"""
        keys = path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return self._config
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self._config[key]
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(args: Optional[argparse.Namespace] = None,
                config_files: Optional[List[str]] = None,
                overrides: Optional[Dict] = None) -> Config:
    """
    Main entry point for loading configuration
    
    Args:
        args: Parsed command-line arguments
        config_files: List of config files to load
        overrides: Additional overrides
        
    Returns:
        Config object
    """
    loader = ConfigLoader()
    
    # Determine config files to load
    if args is not None:
        files = [args.config]
        if args.dataset:
            files.append(args.dataset)
        if args.experiment:
            files.append(args.experiment)
        
        # Convert args to overrides
        arg_overrides = ArgumentParser.args_to_overrides(args)
        if overrides:
            overrides = loader.merge_configs(arg_overrides, overrides)
        else:
            overrides = arg_overrides
    elif config_files is None:
        files = ['default.yaml']
    else:
        files = config_files
    
    # Load configuration
    config_dict = loader.load(files, overrides)
    
    # Substitute environment variables
    config_dict = loader.substitute_env_vars(config_dict)
    
    return Config(config_dict)


# Convenience function for scripts
def setup_config():
    """Setup configuration from command-line arguments"""
    parser = ArgumentParser.create_parser()
    args = parser.parse_args()
    
    config = load_config(args)
    
    return config, args


if __name__ == "__main__":
    # Example usage
    config, args = setup_config()
    
    print("Configuration loaded:")
    print(f"  Batch size: {config.get('training.rvq.batch_size')}")
    print(f"  Learning rate: {config.get('training.rvq.learning_rate')}")
    print(f"  Dataset: {config.get('dataset.name', 'Not specified')}")

