import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class AdaptiveModelBase(nn.Module):
    """Base class for models with knowledge distillation and meta-learning support"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = None
        self.classifier = None
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features for knowledge transfer"""
        if self.feature_extractor is None:
            raise NotImplementedError("Feature extractor not implemented")
        return self.feature_extractor(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps for knowledge distillation"""
        raise NotImplementedError
        
    def forward_with_attention(self, x: torch.Tensor) -> tuple:
        """Forward pass returning both predictions and attention maps"""
        attention_maps = self.get_attention_maps(x)
        output = self.forward(x)
        return output, attention_maps
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for meta-learning"""
        features = self.extract_features(x)
        return features.view(features.size(0), -1)

class AdaptiveCNNMNIST(AdaptiveModelBase):
    """Adaptive CNN for MNIST with knowledge transfer capabilities"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Self-attention module
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        attention = self.attention(features)
        features = features * attention
        features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.attention(features)

class AdaptiveResNetCIFAR(AdaptiveModelBase):
    """Adaptive ResNet for CIFAR with knowledge transfer capabilities"""
    def __init__(self, num_classes: int = 10, base_width: int = 64):
        super().__init__()
        
        self.in_channels = 64
        self.base_width = base_width
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(64, 2),
            self._make_layer(128, 2, stride=2),
            self._make_layer(256, 2, stride=2),
            self._make_layer(512, 2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Self-attention module
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(self._make_block(out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels))
        return nn.Sequential(*layers)
    
    def _make_block(self, out_channels: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        block = ResBlock(self.in_channels, out_channels, stride, downsample)
        self.in_channels = out_channels
        return block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        attention = self.attention(features)
        features = features * attention
        features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.attention(features)

class ResBlock(nn.Module):
    """Residual block with attention"""
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                              stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return F.relu(out)

def create_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """Factory function to create models"""
    if model_name == 'mnist_cnn':
        return AdaptiveCNNMNIST(num_classes=config.get('num_classes', 10))
    elif model_name == 'cifar_resnet':
        return AdaptiveResNetCIFAR(num_classes=config.get('num_classes', 10),
                                  base_width=config.get('base_width', 64))
    else:
        raise ValueError(f"Unknown model: {model_name}")