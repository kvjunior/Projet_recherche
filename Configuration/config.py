from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
import os
from pathlib import Path

@dataclass
class PrivacyConfig:
    """Privacy-related configuration"""
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    clip_norm: float = 1.0
    min_clients_per_round: int = 10
    secure_aggregation: bool = True
    encryption_key_size: int = 2048
    differential_privacy: bool = True

@dataclass
class CommunicationConfig:
    """Communication-related configuration"""
    bandwidth_threshold: float = 1e6  # 1 MB/s
    latency_threshold: float = 0.1    # 100ms
    compression_threshold: float = 0.1
    monitor_interval: float = 1.0
    transfer_timeout: float = 300
    batch_size: int = 32
    max_message_size: int = 1024 * 1024  # 1MB
    retry_limit: int = 3
    aggregation_timeout: float = 30.0

@dataclass
class TierConfig:
    """Tier-related configuration"""
    min_tiers: int = 2
    max_tiers: int = 5
    adaptation_frequency: int = 10
    min_nodes_per_tier: int = 3
    tier_formation_method: str = 'dynamic'  # 'dynamic' or 'static'
    tier_stability_threshold: float = 0.8
    rebalancing_threshold: float = 0.2
    performance_window_size: int = 10

@dataclass
class ModelConfig:
    """Model-related configuration"""
    architecture: str = 'resnet18'
    num_classes: int = 10
    input_channels: int = 3
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    batch_norm: bool = True
    dropout_rate: float = 0.5
    activation: str = 'relu'
    pretrained: bool = False

@dataclass
class TrainingConfig:
    """Training-related configuration"""
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    optimizer: str = 'adam'
    loss_function: str = 'cross_entropy'
    learning_rate_schedule: str = 'cosine'
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    validation_frequency: int = 1

@dataclass
class KnowledgeDistillationConfig:
    """Knowledge distillation configuration"""
    temperature: float = 2.0
    alpha: float = 0.5
    distillation_loss_weight: float = 0.1
    teacher_model_update_frequency: int = 5
    soft_target_loss_weight: float = 0.5
    attention_transfer_weight: float = 0.5
    feature_transfer_layers: List[str] = field(default_factory=lambda: ['layer2', 'layer3'])

@dataclass
class MetaLearningConfig:
    """Meta-learning configuration"""
    meta_learning_rate: float = 0.001
    meta_batch_size: int = 4
    num_inner_steps: int = 5
    inner_learning_rate: float = 0.01
    task_adaptation_steps: int = 3
    meta_momentum: float = 0.9
    meta_weight_decay: float = 0.0001
    personalization_epochs: int = 5

@dataclass
class AdaptiveHierFLConfig:
    """Main configuration for AdaptiveHierFL"""
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    tier: TierConfig = field(default_factory=TierConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    knowledge_distillation: KnowledgeDistillationConfig = field(default_factory=KnowledgeDistillationConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    
    # General settings
    experiment_name: str = "adaptive_hierfl"
    random_seed: int = 42
    debug_mode: bool = False
    logging_level: str = "INFO"
    
    # Dataset settings
    dataset: str = "cifar10"
    data_path: str = "data/"
    num_workers: int = 4
    
    # System settings
    num_clients: int = 100
    client_sampling_rate: float = 0.1
    device: str = "cuda"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AdaptiveHierFLConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
            
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get all model-related parameters"""
        return {
            'architecture': self.model.architecture,
            'num_classes': self.model.num_classes,
            'input_channels': self.model.input_channels,
            'batch_norm': self.model.batch_norm,
            'dropout_rate': self.model.dropout_rate,
            'activation': self.model.activation
        }
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """Get all optimizer-related parameters"""
        return {
            'optimizer': self.training.optimizer,
            'learning_rate': self.model.learning_rate,
            'momentum': self.model.momentum,
            'weight_decay': self.model.weight_decay
        }
    
    def get_privacy_params(self) -> Dict[str, Any]:
        """Get all privacy-related parameters"""
        return {
            'epsilon': self.privacy.epsilon,
            'delta': self.privacy.delta,
            'noise_multiplier': self.privacy.noise_multiplier,
            'clip_norm': self.privacy.clip_norm
        }
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.tier.min_tiers <= self.tier.max_tiers, \
            "min_tiers must be <= max_tiers"
        
        assert 0 <= self.client_sampling_rate <= 1, \
            "client_sampling_rate must be between 0 and 1"
        
        assert self.training.batch_size > 0, \
            "batch_size must be positive"
        
        assert self.privacy.epsilon > 0, \
            "epsilon must be positive"
        
        assert 0 < self.privacy.delta < 1, \
            "delta must be between 0 and 1"
        
        assert self.knowledge_distillation.temperature > 0, \
            "temperature must be positive"
            
    def get_experiment_dir(self) -> Path:
        """Get directory for experiment outputs"""
        base_dir = Path("experiments")
        return base_dir / self.experiment_name