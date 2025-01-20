import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch.nn as nn

class DPMechanism:
    """Base class for differential privacy mechanisms"""
    def __init__(self, epsilon: float, delta: float):
        self.epsilon = epsilon
        self.delta = delta
        
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class GaussianMechanism(DPMechanism):
    """Gaussian noise mechanism for differential privacy"""
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        super().__init__(epsilon, delta)
        self.sensitivity = sensitivity
        # Calculate sigma based on epsilon, delta, and sensitivity
        self.sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tensor"""
        noise = torch.normal(
            mean=0,
            std=self.sigma,
            size=tensor.shape,
            device=tensor.device
        )
        return tensor + noise

class LaplaceMechanism(DPMechanism):
    """Laplace noise mechanism for differential privacy"""
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        super().__init__(epsilon, delta)
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon
        
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Laplace noise to tensor"""
        noise = torch.tensor(
            np.random.laplace(0, self.scale, size=tensor.shape),
            device=tensor.device,
            dtype=tensor.dtype
        )
        return tensor + noise

class DPModelUpdates:
    """Apply DP to model updates in federated learning"""
    def __init__(self, dp_mechanism: DPMechanism, 
                 clip_norm: float,
                 noise_multiplier: float):
        self.dp_mechanism = dp_mechanism
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.privacy_budget = 0.0
        
    def process_update(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process model update with DP"""
        # Clip gradients
        clipped_update = self._clip_update(model_update)
        
        # Add noise
        noisy_update = {}
        for name, param in clipped_update.items():
            noisy_update[name] = self.dp_mechanism.add_noise(param)
            
        # Track privacy budget
        self._update_privacy_budget()
        
        return noisy_update
    
    def _clip_update(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clip model update by norm"""
        total_norm = 0.0
        for param in update.values():
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = min(self.clip_norm / (total_norm + 1e-6), 1.0)
        
        clipped_update = {}
        for name, param in update.items():
            clipped_update[name] = param.mul(clip_coef)
            
        return clipped_update
    
    def _update_privacy_budget(self):
        """Update privacy budget tracking"""
        # Simple privacy budget accounting
        self.privacy_budget += self.noise_multiplier ** 2

class DPAggregator:
    """Differentially private aggregation for federated learning"""
    def __init__(self, num_clients: int,
                 epsilon: float,
                 delta: float,
                 noise_scale: float = 1.0):
        self.num_clients = num_clients
        self.epsilon = epsilon
        self.delta = delta
        self.noise_scale = noise_scale
        self.mechanism = GaussianMechanism(epsilon, delta, sensitivity=1.0)
        
    def aggregate(self, updates: List[Dict[str, torch.Tensor]],
                 weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Perform DP aggregation of model updates"""
        if weights is None:
            weights = [1.0 / self.num_clients] * self.num_clients
            
        # Normalize weights
        weights = torch.tensor(weights) / sum(weights)
        
        # Initialize aggregated update
        aggregated = {}
        for name in updates[0].keys():
            # Weighted sum
            weighted_sum = sum(
                update[name] * weight 
                for update, weight in zip(updates, weights)
            )
            
            # Add noise scaled by number of clients
            noise_scale = self.noise_scale / self.num_clients
            noisy_sum = self.mechanism.add_noise(weighted_sum * noise_scale)
            
            aggregated[name] = noisy_sum
            
        return aggregated
    
    def get_privacy_spent(self, num_rounds: int) -> Tuple[float, float]:
        """Calculate spent privacy budget"""
        # Implement advanced composition theorem
        eps_spent = self.epsilon * np.sqrt(2 * num_rounds * np.log(1/self.delta))
        return eps_spent, self.delta

class AdaptiveDPMechanism(DPMechanism):
    """Adaptive DP mechanism that adjusts privacy parameters"""
    def __init__(self, initial_epsilon: float,
                 initial_delta: float,
                 sensitivity: float,
                 target_accuracy: float):
        super().__init__(initial_epsilon, initial_delta)
        self.sensitivity = sensitivity
        self.target_accuracy = target_accuracy
        self.accuracy_history = []
        
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add adaptive noise based on accuracy history"""
        if self.accuracy_history:
            # Adjust epsilon based on accuracy
            avg_accuracy = np.mean(self.accuracy_history[-5:])
            accuracy_diff = self.target_accuracy - avg_accuracy
            
            # Increase epsilon (less noise) if accuracy is too low
            if accuracy_diff > 0:
                self.epsilon *= 1.1
            # Decrease epsilon (more noise) if accuracy is above target
            else:
                self.epsilon *= 0.9
                
        # Use Gaussian mechanism with adapted parameters
        mechanism = GaussianMechanism(self.epsilon, self.delta, self.sensitivity)
        return mechanism.add_noise(tensor)
    
    def update_accuracy(self, accuracy: float):
        """Update accuracy history"""
        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) > 100:
            self.accuracy_history.pop(0)