import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import numpy as np

class CrossTierDistillation:
    def __init__(self, config):
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.distillation_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def distill_cloud_to_edge(self, teacher_model: nn.Module, 
                             student_model: nn.Module) -> nn.Module:
        """Perform knowledge distillation from cloud to edge model"""
        student_model.train()
        teacher_model.eval()
        
        optimizer = torch.optim.Adam(student_model.parameters(), 
                                   lr=self.config.distill_lr)
        
        for epoch in range(self.config.distill_epochs):
            for batch_idx, (data, _) in enumerate(self.config.distill_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                
                # Get soft targets from teacher
                with torch.no_grad():
                    teacher_logits = teacher_model(data)
                    soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
                
                # Get student predictions
                student_logits = student_model(data)
                student_soft = F.softmax(student_logits / self.temperature, dim=1)
                
                # Compute distillation loss
                loss = self._compute_distillation_loss(student_soft, 
                                                     soft_targets,
                                                     self.temperature)
                
                loss.backward()
                optimizer.step()
                
        return student_model
    
    def distill_edge_to_client(self, teacher_model: nn.Module,
                              student_model: nn.Module,
                              client_data: torch.utils.data.DataLoader) -> nn.Module:
        """Perform knowledge distillation from edge to client model"""
        student_model.train()
        teacher_model.eval()
        
        optimizer = torch.optim.Adam(student_model.parameters(),
                                   lr=self.config.distill_lr)
        
        for epoch in range(self.config.distill_epochs):
            for batch_idx, (data, targets) in enumerate(client_data):
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                # Get soft targets from teacher
                with torch.no_grad():
                    teacher_logits = teacher_model(data)
                    soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
                
                # Get student predictions
                student_logits = student_model(data)
                student_soft = F.softmax(student_logits / self.temperature, dim=1)
                
                # Compute combined loss (distillation + task-specific)
                distill_loss = self._compute_distillation_loss(student_soft,
                                                             soft_targets,
                                                             self.temperature)
                
                task_loss = F.cross_entropy(student_logits, targets)
                
                # Combined loss
                loss = (self.alpha * distill_loss + 
                       (1 - self.alpha) * task_loss)
                
                loss.backward()
                optimizer.step()
                
        return student_model
    
    def distill_at_edge(self, student_model: nn.Module,
                        teacher_model: nn.Module) -> nn.Module:
        """Perform knowledge distillation at edge server"""
        # Similar to cloud_to_edge but with edge-specific adaptations
        student_model.train()
        teacher_model.eval()
        
        optimizer = torch.optim.Adam(student_model.parameters(),
                                   lr=self.config.distill_lr)
        
        for epoch in range(self.config.distill_epochs):
            for batch_idx, (data, _) in enumerate(self.config.distill_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                
                # Get teacher predictions with attention
                with torch.no_grad():
                    teacher_logits, teacher_attention = self._forward_with_attention(
                        teacher_model, data
                    )
                    soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
                
                # Get student predictions with attention
                student_logits, student_attention = self._forward_with_attention(
                    student_model, data
                )
                student_soft = F.softmax(student_logits / self.temperature, dim=1)
                
                # Compute losses
                distill_loss = self._compute_distillation_loss(
                    student_soft, soft_targets, self.temperature
                )
                
                attention_loss = self._compute_attention_loss(
                    student_attention, teacher_attention
                )
                
                # Combined loss
                loss = distill_loss + self.config.attention_weight * attention_loss
                
                loss.backward()
                optimizer.step()
                
        return student_model
    
    def _compute_distillation_loss(self, student_soft: torch.Tensor,
                                 teacher_soft: torch.Tensor,
                                 temperature: float) -> torch.Tensor:
        """Compute the knowledge distillation loss"""
        distillation_loss = F.kl_div(
            F.log_softmax(student_soft / temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return distillation_loss
    
    def _compute_attention_loss(self, student_attention: torch.Tensor,
                              teacher_attention: torch.Tensor) -> torch.Tensor:
        """Compute the attention transfer loss"""
        attention_loss = F.mse_loss(
            self._normalize_attention(student_attention),
            self._normalize_attention(teacher_attention)
        )
        return attention_loss
    
    def _normalize_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """Normalize attention maps"""
        B, H, W, C = attention.size()
        attention = attention.view(B, H * W, C)
        attention = F.normalize(attention, p=2, dim=1)
        return attention
    
    def _forward_with_attention(self, model: nn.Module,
                              x: torch.Tensor) -> tuple:
        """Forward pass with attention maps"""
        if hasattr(model, 'get_attention_maps'):
            return model.forward_with_attention(x)
        else:
            return model(x), None
            
    def update_temperature(self, current_epoch: int):
        """Dynamically adjust temperature based on training progress"""
        if current_epoch > self.config.temperature_schedule_start:
            self.temperature = self.config.temperature * (
                1 + self.config.temperature_decay_rate * (
                    current_epoch - self.config.temperature_schedule_start
                )
            )
    
    def get_current_temperature(self) -> float:
        """Get current temperature value"""
        return self.temperature
        
    def adaptive_distillation(self, student_model: nn.Module, 
                            teacher_model: nn.Module,
                            validation_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Perform adaptive knowledge distillation with dynamic adjustments"""
        student_model.train()
        teacher_model.eval()
        
        # Initialize tracking metrics
        best_performance = float('-inf')
        patience = self.config.adaptive_patience
        patience_counter = 0
        
        optimizer = torch.optim.Adam(student_model.parameters(),
                                   lr=self.config.distill_lr)
        
        for epoch in range(self.config.adaptive_epochs):
            # Training phase
            epoch_loss = 0.0
            for batch_idx, (data, targets) in enumerate(self.config.distill_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = teacher_model(data)
                    soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
                
                # Get student predictions
                student_logits = student_model(data)
                student_soft = F.softmax(student_logits / self.temperature, dim=1)
                
                # Compute adaptive loss
                loss = self._compute_adaptive_loss(
                    student_logits=student_logits,
                    student_soft=student_soft,
                    teacher_soft=soft_targets,
                    targets=targets,
                    epoch=epoch
                )
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation phase
            current_performance = self._evaluate_distillation(
                student_model, validation_loader
            )
            
            # Update best model if improved
            if current_performance > best_performance:
                best_performance = current_performance
                best_state_dict = student_model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Adaptive adjustments
            self._adjust_distillation_params(
                epoch_loss=epoch_loss,
                current_performance=current_performance,
                best_performance=best_performance
            )
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Restore best model
        student_model.load_state_dict(best_state_dict)
        return student_model
    
    def _compute_adaptive_loss(self, student_logits: torch.Tensor,
                             student_soft: torch.Tensor,
                             teacher_soft: torch.Tensor,
                             targets: torch.Tensor,
                             epoch: int) -> torch.Tensor:
        """Compute adaptive loss with dynamic weighting"""
        # Distillation loss
        distill_loss = self._compute_distillation_loss(
            student_soft, teacher_soft, self.temperature
        )
        
        # Task-specific loss
        task_loss = F.cross_entropy(student_logits, targets)
        
        # Dynamic alpha based on training progress
        current_alpha = self._compute_dynamic_alpha(epoch)
        
        # Combined loss
        loss = (current_alpha * distill_loss + 
                (1 - current_alpha) * task_loss)
        
        return loss
    
    def _compute_dynamic_alpha(self, epoch: int) -> float:
        """Compute dynamic weighting factor alpha"""
        if epoch < self.config.warmup_epochs:
            # Linear warmup
            return self.alpha * (epoch / self.config.warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - self.config.warmup_epochs) / (
                self.config.adaptive_epochs - self.config.warmup_epochs
            )
            return self.alpha * (1 + np.cos(np.pi * progress)) / 2
    
    def _evaluate_distillation(self, model: nn.Module,
                             validation_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate current distillation performance"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in validation_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def _adjust_distillation_params(self, epoch_loss: float,
                                  current_performance: float,
                                  best_performance: float):
        """Adjust distillation parameters based on performance"""
        # Adjust temperature
        if current_performance < best_performance * 0.95:
            self.temperature *= self.config.temperature_decay_rate
        
        # Adjust learning rate if needed
        if epoch_loss > self.config.loss_threshold:
            self.config.distill_lr *= self.config.lr_decay_rate
            
    def save_distillation_state(self, path: str):
        """Save distillation state for future use"""
        state = {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'config': self.config.__dict__
        }
        torch.save(state, path)
    
    def load_distillation_state(self, path: str):
        """Load saved distillation state"""
        state = torch.load(path)
        self.temperature = state['temperature']
        self.alpha = state['alpha']
        self.config.__dict__.update(state['config'])