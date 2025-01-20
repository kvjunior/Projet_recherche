import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple
import numpy as np
import copy

class TaskBatch:
    """Helper class to manage meta-learning tasks"""
    def __init__(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                 query_data: torch.Tensor, query_labels: torch.Tensor):
        self.support_data = support_data
        self.support_labels = support_labels
        self.query_data = query_data
        self.query_labels = query_labels
        
class MetaLearner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_lr = config.meta_lr
        self.inner_lr = config.inner_lr
        self.meta_batch_size = config.meta_batch_size
        self.num_inner_steps = config.num_inner_steps
        
    def generate_tasks(self, client) -> List[TaskBatch]:
        """Generate meta-learning tasks from client's data"""
        tasks = []
        
        # Get client's dataset
        dataset = client.get_dataset()
        
        # Generate multiple tasks for the client
        for _ in range(self.meta_batch_size):
            # Sample support and query sets
            support_data, support_labels, query_data, query_labels = \
                self._sample_task(dataset)
            
            task = TaskBatch(
                support_data.to(self.device),
                support_labels.to(self.device),
                query_data.to(self.device),
                query_labels.to(self.device)
            )
            tasks.append(task)
            
        return tasks
        
    def meta_train(self, model: nn.Module, tasks: List[TaskBatch]) -> nn.Module:
        """Perform meta-training using MAML"""
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=self.meta_lr)
        
        for epoch in range(self.config.meta_epochs):
            meta_loss = 0.0
            
            for task in tasks:
                # Clone model for inner loop optimization
                local_model = self._clone_model(model)
                
                # Inner loop optimization on support set
                for _ in range(self.num_inner_steps):
                    support_loss = self._compute_loss(
                        local_model,
                        task.support_data,
                        task.support_labels
                    )
                    
                    # Compute gradients for inner loop
                    grads = torch.autograd.grad(
                        support_loss,
                        local_model.parameters(),
                        create_graph=True,
                        allow_unused=True
                    )
                    
                    # Update local model parameters
                    self._update_model_params(
                        local_model,
                        grads,
                        self.inner_lr
                    )
                
                # Compute meta-loss on query set
                query_loss = self._compute_loss(
                    local_model,
                    task.query_data,
                    task.query_labels
                )
                
                meta_loss += query_loss
            
            # Meta-optimization
            meta_optimizer.zero_grad()
            meta_loss = meta_loss / len(tasks)
            meta_loss.backward()
            meta_optimizer.step()
            
            # Adaptive adjustment of meta-learning rate
            self._adjust_meta_lr(epoch, meta_loss.item())
            
        return model
    
    def personalize(self, model: nn.Module, client_data: DataLoader) -> nn.Module:
        """Personalize model for a specific client"""
        personalized_model = self._clone_model(model)
        optimizer = torch.optim.Adam(personalized_model.parameters(), lr=self.inner_lr)
        
        for epoch in range(self.config.personalization_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, labels) in enumerate(client_data):
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = personalized_model(data)
                loss = F.cross_entropy(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Adaptive adjustments during personalization
            self._adjust_personalization(
                model=personalized_model,
                epoch=epoch,
                loss=epoch_loss/len(client_data)
            )
            
        return personalized_model
    
    def _sample_task(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        """Sample support and query sets for a task"""
        # Get number of classes and samples per class
        n_classes = self.config.n_way
        n_support = self.config.k_shot
        n_query = self.config.n_query
        
        # Sample classes
        classes = np.random.choice(
            self.config.n_classes,
            n_classes,
            replace=False
        )
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx, class_label in enumerate(classes):
            # Get all samples for this class
            class_samples = self._get_class_samples(dataset, class_label)
            
            # Sample support and query sets
            perm = np.random.permutation(len(class_samples))
            support_idx = perm[:n_support]
            query_idx = perm[n_support:n_support + n_query]
            
            for idx in support_idx:
                sample = class_samples[idx]
                support_data.append(sample[0])
                support_labels.append(class_idx)
                
            for idx in query_idx:
                sample = class_samples[idx]
                query_data.append(sample[0])
                query_labels.append(class_idx)
        
        # Convert to tensors
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels)
        
        return support_data, support_labels, query_data, query_labels
    
    def _compute_loss(self, model: nn.Module,
                     data: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for meta-learning"""
        outputs = model(data)
        return F.cross_entropy(outputs, labels)
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a copy of the model with same parameters"""
        clone = copy.deepcopy(model)
        clone = clone.to(self.device)
        return clone
    
    def _update_model_params(self, model: nn.Module,
                           grads: List[torch.Tensor],
                           learning_rate: float):
        """Update model parameters using computed gradients"""
        for param, grad in zip(model.parameters(), grads):
            if grad is not None:
                param.data = param.data - learning_rate * grad
    
    def _adjust_meta_lr(self, epoch: int, loss: float):
        """Adjust meta learning rate based on progress"""
        if epoch > self.config.meta_lr_schedule_start:
            self.meta_lr *= self.config.meta_lr_decay_rate
    
    def _adjust_personalization(self, model: nn.Module,
                              epoch: int,
                              loss: float):
        """Adjust personalization parameters"""
        # Implement adaptive personalization strategies
        pass
    
    def _get_class_samples(self, dataset: Dataset,
                          class_label: int) -> List[Tuple[torch.Tensor, int]]:
        """Get all samples for a specific class"""
        class_samples = []
        for i in range(len(dataset)):
            sample, label = dataset[i]
            if label == class_label:
                class_samples.append((sample, label))
        return class_samples
    
    def save_meta_state(self, path: str):
        """Save meta-learning state"""
        state = {
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr,
            'config': self.config.__dict__
        }
        torch.save(state, path)
    
    def load_meta_state(self, path: str):
        """Load meta-learning state"""
        state = torch.load(path)
        self.meta_lr = state['meta_lr']
        self.inner_lr = state['inner_lr']
        self.config.__dict__.update(state['config'])
        
    def evaluate_meta_learning(self, model: nn.Module, 
                             validation_tasks: List[TaskBatch]) -> Dict[str, float]:
        """Evaluate meta-learning performance"""
        model.eval()
        total_accuracy = 0.0
        total_loss = 0.0
        
        with torch.no_grad():
            for task in validation_tasks:
                # Clone and adapt model for this task
                adapted_model = self._clone_model(model)
                
                # Adapt using support set
                for _ in range(self.num_inner_steps):
                    support_loss = self._compute_loss(
                        adapted_model,
                        task.support_data,
                        task.support_labels
                    )
                    
                    grads = torch.autograd.grad(
                        support_loss,
                        adapted_model.parameters()
                    )
                    
                    self._update_model_params(
                        adapted_model,
                        grads,
                        self.inner_lr
                    )
                
                # Evaluate on query set
                outputs = adapted_model(task.query_data)
                loss = F.cross_entropy(outputs, task.query_labels)
                
                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == task.query_labels).float().mean().item()
                
                total_accuracy += accuracy
                total_loss += loss.item()
        
        # Compute averages
        avg_accuracy = total_accuracy / len(validation_tasks)
        avg_loss = total_loss / len(validation_tasks)
        
        return {
            'meta_test_accuracy': avg_accuracy,
            'meta_test_loss': avg_loss
        }
    
    def _compute_prototypes(self, support_data: torch.Tensor,
                          support_labels: torch.Tensor,
                          model: nn.Module) -> torch.Tensor:
        """Compute class prototypes from support set embeddings"""
        model.eval()
        embeddings = model.get_embeddings(support_data)
        
        prototypes = []
        for class_idx in range(self.config.n_way):
            # Get embeddings for this class
            mask = support_labels == class_idx
            class_embeddings = embeddings[mask]
            
            # Compute prototype (mean embedding)
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
            
        return torch.stack(prototypes)
    
    def _compute_prototype_loss(self, query_embeddings: torch.Tensor,
                              query_labels: torch.Tensor,
                              prototypes: torch.Tensor) -> torch.Tensor:
        """Compute prototype-based classification loss"""
        # Compute distances to prototypes
        dists = torch.cdist(query_embeddings, prototypes)
        
        # Convert distances to probabilities
        logits = -dists
        log_probs = F.log_softmax(logits, dim=1)
        
        # Compute cross entropy loss
        loss = F.nll_loss(log_probs, query_labels)
        return loss
    
    def _update_prototype_memory(self, prototypes: torch.Tensor):
        """Update running memory of class prototypes"""
        if not hasattr(self, 'prototype_memory'):
            self.prototype_memory = prototypes
        else:
            # Exponential moving average update
            momentum = self.config.prototype_momentum
            self.prototype_memory = (momentum * self.prototype_memory + 
                                   (1 - momentum) * prototypes)
    
    def _compute_confidence_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute confidence weights for meta-learning"""
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        weights = torch.sigmoid((max_probs - self.config.confidence_threshold) * 
                              self.config.confidence_scaling)
        return weights
    
    def _initialize_task_memory(self):
        """Initialize episodic memory for meta-learning tasks"""
        self.task_memory = {
            'support_data': [],
            'support_labels': [],
            'query_data': [],
            'query_labels': [],
            'performance': []
        }
        
    def _update_task_memory(self, task: TaskBatch, performance: float):
        """Update episodic memory with new task"""
        if len(self.task_memory['support_data']) >= self.config.memory_size:
            # Remove oldest task
            for key in self.task_memory.keys():
                self.task_memory[key].pop(0)
        
        # Add new task
        self.task_memory['support_data'].append(task.support_data)
        self.task_memory['support_labels'].append(task.support_labels)
        self.task_memory['query_data'].append(task.query_data)
        self.task_memory['query_labels'].append(task.query_labels)
        self.task_memory['performance'].append(performance)
    
    def _sample_memory_task(self) -> TaskBatch:
        """Sample a task from episodic memory"""
        if not self.task_memory['support_data']:
            raise ValueError("Task memory is empty")
            
        # Sample based on performance (prioritize difficult tasks)
        probs = F.softmax(torch.tensor(self.task_memory['performance']), dim=0)
        idx = torch.multinomial(probs, 1).item()
        
        task = TaskBatch(
            self.task_memory['support_data'][idx],
            self.task_memory['support_labels'][idx],
            self.task_memory['query_data'][idx],
            self.task_memory['query_labels'][idx]
        )
        
        return task