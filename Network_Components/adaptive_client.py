import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import numpy as np
import time
from copy import deepcopy

class AdaptiveClient:
    def __init__(self, client_id: int, 
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 model: nn.Module,
                 config: Any,
                 device: torch.device):
        """Initialize adaptive client with personalization capabilities"""
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.personalized_model = None
        self.config = config
        self.device = device
        
        # Resource monitoring
        self.compute_capacity = self._measure_compute_capacity()
        self.network_reliability = 1.0
        self.battery_level = 100.0
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'personalization_gain': []
        }
        
        # Communication stats
        self.communication_stats = {
            'success_rate': 1.0,
            'latency': [],
            'bandwidth': []
        }
    
    def local_train(self, num_epochs: int) -> Tuple[Dict, float]:
        """Perform local training with adaptive computation"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        epoch_losses = []
        computation_time = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            start_time = time.time()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Adaptive batch processing
                if self._should_process_batch(batch_idx):
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = self._compute_adaptive_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            
            computation_time += time.time() - start_time
            epoch_losses.append(epoch_loss / len(self.train_loader))
            
            # Adjust learning rate
            self._adjust_learning_rate(optimizer, epoch)
        
        # Update training history
        self.training_history['loss'].extend(epoch_losses)
        
        return {
            'model_state': self.model.state_dict(),
            'train_loss': np.mean(epoch_losses)
        }, computation_time
    
    def personalize(self, meta_learner) -> Dict[str, float]:
        """Personalize model using meta-learning"""
        if not self.personalized_model:
            self.personalized_model = deepcopy(self.model)
        
        # Generate personalization tasks
        tasks = meta_learner.generate_tasks(self)
        
        # Personalize model
        self.personalized_model = meta_learner.personalize(
            model=self.personalized_model,
            client_data=self.train_loader
        )
        
        # Evaluate personalization gain
        base_performance = self.evaluate(use_personalized=False)
        personalized_performance = self.evaluate(use_personalized=True)
        
        gain = personalized_performance['accuracy'] - base_performance['accuracy']
        self.training_history['personalization_gain'].append(gain)
        
        return {
            'base_accuracy': base_performance['accuracy'],
            'personalized_accuracy': personalized_performance['accuracy'],
            'personalization_gain': gain
        }
    
    def evaluate(self, use_personalized: bool = False) -> Dict[str, float]:
        """Evaluate model performance"""
        model = self.personalized_model if use_personalized else self.model
        model.eval()
        
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += nn.functional.cross_entropy(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader)
        accuracy = correct / len(self.test_loader.dataset)
        
        return {
            'loss': test_loss,
            'accuracy': accuracy
        }
    
    def update_model(self, new_state_dict: Dict[str, torch.Tensor]):
        """Update model with new parameters"""
        self.model.load_state_dict(new_state_dict)
        # Reset personalized model to force re-personalization
        self.personalized_model = None
    
    def get_resource_state(self) -> Dict[str, float]:
        """Get current resource state"""
        return {
            'compute_capacity': self.compute_capacity,
            'network_reliability': self.network_reliability,
            'battery_level': self.battery_level
        }
    
    def _compute_adaptive_loss(self, output: torch.Tensor, 
                             target: torch.Tensor) -> torch.Tensor:
        """Compute loss with adaptive regularization"""
        base_loss = nn.functional.cross_entropy(output, target)
        
        if self.personalized_model:
            # Add knowledge distillation from personalized model
            temperature = self.config.distillation_temperature
            with torch.no_grad():
                teacher_output = self.personalized_model(output)
            
            distillation_loss = self._distillation_loss(
                output, teacher_output, temperature
            )
            
            # Adaptive weighting based on personalization gain
            alpha = self._compute_distillation_weight()
            return (1 - alpha) * base_loss + alpha * distillation_loss
        
        return base_loss
    
    def _distillation_loss(self, student_output: torch.Tensor,
                          teacher_output: torch.Tensor,
                          temperature: float) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        return nn.KLDivLoss()(
            torch.log_softmax(student_output / temperature, dim=1),
            torch.softmax(teacher_output / temperature, dim=1)
        ) * (temperature ** 2)
    
    def _compute_distillation_weight(self) -> float:
        """Compute adaptive weight for distillation loss"""
        if not self.training_history['personalization_gain']:
            return 0.5
        
        recent_gains = self.training_history['personalization_gain'][-5:]
        avg_gain = np.mean(recent_gains)
        return np.clip(avg_gain, 0.1, 0.9)
    
    def _should_process_batch(self, batch_idx: int) -> bool:
        """Decide whether to process a batch based on resource state"""
        if self.battery_level < self.config.min_battery_level:
            return batch_idx % 2 == 0  # Process every other batch
        return True
    
    def _adjust_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        """Adjust learning rate based on progress and resource state"""
        if epoch < self.config.warmup_epochs:
            # Linear warmup
            lr_scale = min(1., float(epoch + 1) / self.config.warmup_epochs)
        else:
            # Cosine decay
            progress = float(epoch - self.config.warmup_epochs) / (
                self.config.total_epochs - self.config.warmup_epochs
            )
            lr_scale = 0.5 * (1. + np.cos(np.pi * progress))
        
        # Adjust for resource state
        lr_scale *= self.compute_capacity
        
        for param_group in optimizer.parameters():
            param_group['lr'] = self.config.learning_rate * lr_scale
    
    def _measure_compute_capacity(self) -> float:
        """Measure computational capacity"""
        try:
            test_tensor = torch.randn(100, 100).to(self.device)
            start_time = time.time()
            for _ in range(100):
                _ = torch.mm(test_tensor, test_tensor)
            compute_time = time.time() - start_time
            
            # Normalize between 0 and 1
            return 1.0 / (1.0 + compute_time)
        except:
            return 0.5  # Default capacity
    
    def update_network_stats(self, success: bool, latency: float, bandwidth: float):
        """Update network communication statistics"""
        self.communication_stats['success_rate'] = (
            0.9 * self.communication_stats['success_rate'] +
            0.1 * float(success)
        )
        self.communication_stats['latency'].append(latency)
        self.communication_stats['bandwidth'].append(bandwidth)
        
        # Update network reliability
        recent_latencies = self.communication_stats['latency'][-10:]
        self.network_reliability = 1.0 / (1.0 + np.mean(recent_latencies))
    
    def simulate_battery_consumption(self, computation_time: float):
        """Simulate battery consumption based on computation"""
        energy_per_second = 0.1  # 0.1% battery per second of computation
        self.battery_level = max(0, self.battery_level - computation_time * energy_per_second)