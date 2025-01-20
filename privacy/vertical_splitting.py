import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class SplitLayerConfig:
    """Configuration for split layer"""
    layer_name: str
    is_sensitive: bool
    encryption_required: bool
    aggregation_method: str

class VerticalSplitManager:
    """Manage vertical splitting of neural networks"""
    def __init__(self, model: nn.Module, split_config: List[SplitLayerConfig]):
        self.model = model
        self.split_config = split_config
        self.sensitive_layers = self._identify_sensitive_layers()
        self.local_parts = {}
        self.remote_parts = {}
        
        # Split the model
        self._split_model()
        
    def _identify_sensitive_layers(self) -> Dict[str, bool]:
        """Identify sensitive layers based on configuration"""
        sensitive_layers = {}
        for config in self.split_config:
            sensitive_layers[config.layer_name] = config.is_sensitive
        return sensitive_layers
    
    def _split_model(self):
        """Split model into local and remote parts"""
        modules = dict(self.model.named_modules())
        
        for config in self.split_config:
            if config.is_sensitive:
                self.local_parts[config.layer_name] = modules[config.layer_name]
            else:
                self.remote_parts[config.layer_name] = modules[config.layer_name]

class VerticalModelSegment:
    """Segment of vertically split model"""
    def __init__(self, layers: nn.ModuleDict, 
                 is_local: bool,
                 encryption_service: Optional['EncryptionService'] = None):
        self.layers = layers
        self.is_local = is_local
        self.encryption_service = encryption_service
        self.intermediate_outputs = {}
        
    def forward(self, x: torch.Tensor, 
                segment_id: int) -> torch.Tensor:
        """Forward pass through segment"""
        output = x
        for name, layer in self.layers.items():
            output = layer(output)
            
            # Encrypt output if needed
            if self.encryption_service and not self.is_local:
                output = self.encryption_service.encrypt_tensor(output)
                
        self.intermediate_outputs[segment_id] = output
        return output
    
    def get_encrypted_gradients(self, 
                              segment_id: int) -> Dict[str, torch.Tensor]:
        """Get encrypted gradients for remote segment"""
        if not self.encryption_service:
            raise ValueError("Encryption service not configured")
            
        gradients = {}
        for name, param in self.layers.named_parameters():
            if param.grad is not None:
                encrypted_grad = self.encryption_service.encrypt_tensor(param.grad)
                gradients[name] = encrypted_grad
                
        return gradients

class VerticalFederatedTrainer:
    """Trainer for vertical federated learning"""
    def __init__(self, local_segment: VerticalModelSegment,
                 remote_segment: VerticalModelSegment,
                 encryption_service: 'EncryptionService'):
        self.local_segment = local_segment
        self.remote_segment = remote_segment
        self.encryption_service = encryption_service
        
        self.training_stats = defaultdict(list)
        
    def train_step(self, local_data: torch.Tensor,
                  remote_data: torch.Tensor,
                  labels: torch.Tensor) -> Dict[str, float]:
        """Perform one step of vertical federated training"""
        # Forward pass on local segment
        local_output = self.local_segment.forward(local_data, 0)
        
        # Encrypt and send to remote
        encrypted_local = self.encryption_service.encrypt_tensor(local_output)
        
        # Forward pass on remote segment
        remote_input = torch.cat([encrypted_local, remote_data], dim=1)
        remote_output = self.remote_segment.forward(remote_input, 0)
        
        # Compute loss
        loss = self._compute_encrypted_loss(remote_output, labels)
        
        # Backward pass
        loss.backward()
        
        # Exchange gradients
        local_grads = self.local_segment.get_encrypted_gradients(0)
        remote_grads = self.remote_segment.get_encrypted_gradients(0)
        
        # Update statistics
        stats = {
            'loss': loss.item(),
            'local_grad_norm': self._compute_grad_norm(local_grads),
            'remote_grad_norm': self._compute_grad_norm(remote_grads)
        }
        
        self._update_stats(stats)
        return stats
    
    def _compute_encrypted_loss(self, output: torch.Tensor,
                              labels: torch.Tensor) -> torch.Tensor:
        """Compute loss while maintaining encryption"""
        if self.encryption_service:
            decrypted_output = self.encryption_service.decrypt_tensor(output)
            loss = nn.functional.cross_entropy(decrypted_output, labels)
            return self.encryption_service.encrypt_tensor(loss)
        return nn.functional.cross_entropy(output, labels)
    
    def _compute_grad_norm(self, grads: Dict[str, torch.Tensor]) -> float:
        """Compute norm of gradients"""
        total_norm = 0.0
        for grad in grads.values():
            if self.encryption_service:
                grad = self.encryption_service.decrypt_tensor(grad)
            total_norm += grad.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def _update_stats(self, stats: Dict[str, float]):
        """Update training statistics"""
        for key, value in stats.items():
            self.training_stats[key].append(value)

class SecureAggregator:
    """Secure aggregation for vertical federated learning"""
    def __init__(self, num_parties: int,
                 encryption_service: 'EncryptionService'):
        self.num_parties = num_parties
        self.encryption_service = encryption_service
        self.party_weights = torch.ones(num_parties) / num_parties
        
    def aggregate(self, 
                 encrypted_gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Securely aggregate encrypted gradients"""
        aggregated = {}
        
        for name in encrypted_gradients[0].keys():
            # Weight and sum gradients while encrypted
            weighted_sum = None
            for i, grads in enumerate(encrypted_gradients):
                weighted_grad = self.encryption_service.multiply(
                    grads[name],
                    self.party_weights[i]
                )
                
                if weighted_sum is None:
                    weighted_sum = weighted_grad
                else:
                    weighted_sum = self.encryption_service.add(
                        weighted_sum,
                        weighted_grad
                    )
                    
            aggregated[name] = weighted_sum
            
        return aggregated
    
    def set_party_weights(self, weights: torch.Tensor):
        """Set weights for different parties"""
        if len(weights) != self.num_parties:
            raise ValueError("Invalid number of weights")
        self.party_weights = weights / weights.sum()

class PrivacyMetrics:
    """Track privacy metrics for vertical federated learning"""
    def __init__(self):
        self.information_leakage = []
        self.reconstruction_error = []
        
    def update_metrics(self, 
                      leakage: float,
                      error: float):
        """Update privacy metrics"""
        self.information_leakage.append(leakage)
        self.reconstruction_error.append(error)
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current privacy metrics"""
        if not self.information_leakage:
            return {'leakage': 0.0, 'error': 0.0}
            
        return {
            'avg_leakage': np.mean(self.information_leakage[-10:]),
            'avg_error': np.mean(self.reconstruction_error[-10:]),
            'max_leakage': max(self.information_leakage[-10:]),
            'min_error': min(self.reconstruction_error[-10:])
        }
    
    def check_privacy_breach(self, threshold: float = 0.1) -> bool:
        """Check if privacy breach threshold is exceeded"""
        if not self.information_leakage:
            return False
            
        recent_leakage = np.mean(self.information_leakage[-5:])
        return recent_leakage > threshold

class EncryptionService:
    """Service for handling encryption in vertical federated learning"""
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.public_keys = {}
        self.private_keys = {}
        self._initialize_keys()
        
    def _initialize_keys(self):
        """Initialize encryption keys"""
        # This is a placeholder - in practice, use proper cryptographic library
        # such as phe (Paillier Homomorphic Encryption) or TenSEAL
        pass
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Encrypt tensor"""
        # Placeholder for actual encryption
        return tensor
    
    def decrypt_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decrypt tensor"""
        # Placeholder for actual decryption
        return tensor
    
    def add(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Add encrypted tensors"""
        # Placeholder for homomorphic addition
        return tensor1 + tensor2
    
    def multiply(self, tensor: torch.Tensor, scalar: float) -> torch.Tensor:
        """Multiply encrypted tensor by plaintext scalar"""
        # Placeholder for homomorphic multiplication
        return tensor * scalar

class VerticalPartyManager:
    """Manage communication and synchronization between vertical parties"""
    def __init__(self, party_id: int, 
                 num_parties: int,
                 encryption_service: EncryptionService):
        self.party_id = party_id
        self.num_parties = num_parties
        self.encryption_service = encryption_service
        self.party_status = {i: 'inactive' for i in range(num_parties)}
        self.sync_buffer = defaultdict(dict)
        
    def send_intermediate(self, 
                         data: torch.Tensor,
                         target_party: int,
                         round_id: int):
        """Send intermediate results to another party"""
        encrypted_data = self.encryption_service.encrypt_tensor(data)
        self.sync_buffer[round_id][f'from_{self.party_id}_to_{target_party}'] = encrypted_data
        
    def receive_intermediate(self, 
                           source_party: int,
                           round_id: int) -> Optional[torch.Tensor]:
        """Receive intermediate results from another party"""
        key = f'from_{source_party}_to_{self.party_id}'
        if key in self.sync_buffer[round_id]:
            data = self.sync_buffer[round_id][key]
            return self.encryption_service.decrypt_tensor(data)
        return None
    
    def synchronize(self, round_id: int):
        """Synchronize all parties for current round"""
        self.party_status[self.party_id] = 'ready'
        
        # Wait for all parties
        while not all(status == 'ready' for status in self.party_status.values()):
            continue
            
        # Clear previous round data
        if round_id > 0:
            del self.sync_buffer[round_id - 1]
        
        # Reset status
        self.party_status = {i: 'inactive' for i in range(self.num_parties)}
    
    def cleanup(self):
        """Clean up resources and sync buffers"""
        self.sync_buffer.clear()
        self.party_status = {i: 'inactive' for i in range(self.num_parties)}

class VerticalFederatedLearningClient:
    """Client for vertical federated learning"""
    def __init__(self, 
                 party_id: int,
                 local_features: List[str],
                 model_segment: VerticalModelSegment,
                 party_manager: VerticalPartyManager):
        self.party_id = party_id
        self.local_features = local_features
        self.model_segment = model_segment
        self.party_manager = party_manager
        self.privacy_metrics = PrivacyMetrics()
        
    def train(self, 
              local_data: torch.Tensor,
              num_rounds: int) -> Dict[str, float]:
        """Train using vertical federated learning"""
        metrics = defaultdict(list)
        
        for round_id in range(num_rounds):
            # Local computation
            local_output = self.model_segment.forward(local_data, round_id)
            
            # Send to next party
            next_party = (self.party_id + 1) % self.party_manager.num_parties
            self.party_manager.send_intermediate(local_output, next_party, round_id)
            
            # Wait for other parties
            self.party_manager.synchronize(round_id)
            
            # Receive and process results
            if self.party_id > 0:
                prev_party = (self.party_id - 1) % self.party_manager.num_parties
                received = self.party_manager.receive_intermediate(prev_party, round_id)
                if received is not None:
                    # Update metrics
                    self._update_metrics(received, round_id)
                    metrics['received_data_size'].append(received.numel())
            
            metrics['local_compute_time'].append(time.time())
            
        return dict(metrics)
    
    def _update_metrics(self, 
                       received_data: torch.Tensor,
                       round_id: int):
        """Update privacy metrics"""
        # Simulate information leakage and reconstruction error
        leakage = np.random.random() * 0.1  # Placeholder
        error = np.random.random() * 0.2    # Placeholder
        self.privacy_metrics.update_metrics(leakage, error)
        
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get privacy status and metrics"""
        metrics = self.privacy_metrics.get_metrics()
        privacy_breach = self.privacy_metrics.check_privacy_breach()
        
        return {
            'metrics': metrics,
            'privacy_breach_detected': privacy_breach,
            'feature_importance': self._get_feature_importance()
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get importance scores for local features"""
        # Placeholder for actual feature importance calculation
        return {feature: np.random.random() for feature in self.local_features}