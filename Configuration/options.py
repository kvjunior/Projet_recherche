import argparse
from pathlib import Path
from typing import Dict, Any
import yaml
import torch
from config import AdaptiveHierFLConfig

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AdaptiveHierFL')
    
    # Basic settings
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Name of experiment')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default=None,
                      help='Dataset name (mnist/cifar10)')
    parser.add_argument('--data-path', type=str, default=None,
                      help='Path to dataset')
    parser.add_argument('--num-workers', type=int, default=None,
                      help='Number of data loading workers')
    
    # System settings
    parser.add_argument('--num-clients', type=int, default=None,
                      help='Number of clients')
    parser.add_argument('--num-edges', type=int, default=None,
                      help='Number of edge servers')
    parser.add_argument('--client-sampling-rate', type=float, default=None,
                      help='Fraction of clients to sample per round')
    parser.add_argument('--device', type=str, default=None,
                      help='Device (cuda/cpu)')
    parser.add_argument('--gpu', type=int, default=None,
                      help='GPU ID')
    
    # Training settings
    parser.add_argument('--num-rounds', type=int, default=None,
                      help='Number of communication rounds')
    parser.add_argument('--local-epochs', type=int, default=None,
                      help='Number of local epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                      help='Learning rate')
    parser.add_argument('--momentum', type=float, default=None,
                      help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=None,
                      help='Weight decay')
    
    # Tier settings
    parser.add_argument('--min-tiers', type=int, default=None,
                      help='Minimum number of tiers')
    parser.add_argument('--max-tiers', type=int, default=None,
                      help='Maximum number of tiers')
    parser.add_argument('--adaptation-frequency', type=int, default=None,
                      help='Tier adaptation frequency')
    
    # Privacy settings
    parser.add_argument('--epsilon', type=float, default=None,
                      help='Privacy parameter epsilon')
    parser.add_argument('--delta', type=float, default=None,
                      help='Privacy parameter delta')
    parser.add_argument('--noise-multiplier', type=float, default=None,
                      help='Noise multiplier for DP')
    
    # Knowledge distillation settings
    parser.add_argument('--temperature', type=float, default=None,
                      help='Temperature for knowledge distillation')
    parser.add_argument('--distillation-weight', type=float, default=None,
                      help='Weight for distillation loss')
    
    # Meta-learning settings
    parser.add_argument('--meta-lr', type=float, default=None,
                      help='Meta learning rate')
    parser.add_argument('--meta-batch-size', type=int, default=None,
                      help='Meta batch size')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory')
    parser.add_argument('--checkpoint-freq', type=int, default=None,
                      help='Checkpoint frequency')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed')
    
    args = parser.parse_args()
    return args

def load_config(args: argparse.Namespace) -> AdaptiveHierFLConfig:
    """Load and update configuration based on command line arguments"""
    # Load base configuration
    config = AdaptiveHierFLConfig.from_yaml(args.config)
    
    # Update configuration with command line arguments
    arg_dict = vars(args)
    updates = {}
    
    for key, value in arg_dict.items():
        if value is not None:
            # Convert command line argument names to config names
            config_key = key.replace('-', '_')
            updates[config_key] = value
    
    # Special handling for GPU device
    if args.gpu is not None:
        updates['device'] = f'cuda:{args.gpu}'
    
    # Update configuration
    config.update(**updates)
    
    return config

def save_experiment_config(config: AdaptiveHierFLConfig,
                         experiment_dir: Path):
    """Save experiment configuration"""
    # Create experiment directory
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = experiment_dir / 'config.yaml'
    config.save_yaml(str(config_path))
    
    # Save command line arguments
    args_path = experiment_dir / 'args.yaml'
    with open(args_path, 'w') as f:
        yaml.dump(vars(parse_args()), f, default_flow_style=False)

def get_device(config: AdaptiveHierFLConfig) -> torch.device:
    """Get PyTorch device based on configuration"""
    if config.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print('Warning: CUDA is not available. Using CPU instead.')
            return torch.device('cpu')
        return torch.device(config.device)
    return torch.device('cpu')

def setup_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Setup experiment based on arguments"""
    # Load and update configuration
    config = load_config(args)
    
    # Set random seed
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
    
    # Create experiment directory
    experiment_dir = config.get_experiment_dir()
    save_experiment_config(config, experiment_dir)
    
    # Get device
    device = get_device(config)
    
    return {
        'config': config,
        'device': device,
        'experiment_dir': experiment_dir
    }

def main():
    """Main entry point for parsing arguments"""
    args = parse_args()
    experiment = setup_experiment(args)
    return experiment

if __name__ == '__main__':
    main()