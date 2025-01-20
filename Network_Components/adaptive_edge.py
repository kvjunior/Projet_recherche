import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict
import time

from knowledge_distillation import CrossTierDistillation
from meta_learner import MetaLearner

class AdaptiveEdge:
    def __init__(self, edge_id: int, 
                 model: nn.Module,
                 config: Any,
                 device: torch.device):
        """Initialize adaptive edge server"""
        self.edge_id = edge_id
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize components
        self.distillation = CrossTierDistillation(config)
        self.meta_learner = MetaLearner(config)
        
        # Client management
        self.active_clients = set()
        self.client_performances = defaultdict(list)
        self.client_resources = {}
        
        # Edge state
        self.aggregation_weights = None
        self.knowledge_buffer = []
        self.performance_history = []
        
        # Communication tracking
        self.communication_stats = defaultdict(list)
        
    def register_client(self, client_id: int, initial_resources: Dict[str, float]):
        """Register a new client with the edge server"""
        self.active_clients.add(client_id)
        self.client_resources[client_id] = initial_resources
        
    def unregister_client(self, client_id: int):
        """Unregister a client"""
        self.active_clients.remove(client_id)
        if client_id in self.client_resources:
            del self.client_resources[client_id]
    
    def aggregate_models(self, client_updates: Dict[int, Dict],
                        round_number: int) -> Dict[str, Any]:
        """Aggregate client models with adaptive weighting"""
        if not client_updates:
            return None
            
        # Update client performances and resources
        self._update_client_states(client_updates)
        
        # Compute aggregation weights
        self.aggregation_weights = self._compute_aggregation_weights(client_updates)
        
        # Perform weighted aggregation
        aggregated_state = {}
        for key in client_updates[list(client_updates.keys())[0]]['model_state'].keys():
            weighted_sum = torch.stack([
                update['model_state'][key] * self.aggregation_weights[client_id]
                for client_id, update in client_updates.items()
            ]).sum(dim=0)
            aggregated_state[key] = weighted_sum
            
        # Apply knowledge distillation if available
        if self.knowledge_buffer:
            aggregated_state = self._apply_knowledge_distillation(aggregated_state)
        
        # Update model and evaluate
        self.model.load_state_dict(aggregated_state)
        performance = self._evaluate_aggregation()
        self.performance_history.append(performance)
        
        return {
            'model_state': aggregated_state,
            'performance': performance,
            'weights': self.aggregation_weights
        }
    
    def receive_from_cloud(self, cloud_model_state: Dict[str, torch.Tensor],
                          global_knowledge: Dict[str, Any]):
        """Receive and process updates from cloud"""
        # Store previous state for knowledge distillation
        prev_state = self.model.state_dict()
        
        # Update model with cloud state
        self.model.load_state_dict(cloud_model_state)
        
        # Store knowledge for future distillation
        self.knowledge_buffer.append({
            'teacher_state': prev_state,
            'global_knowledge': global_knowledge
        })
        
        # Maintain limited buffer size
        if len(self.knowledge_buffer) > self.config.knowledge_buffer_size:
            self.knowledge_buffer.pop(0)
    
    def get_client_meta_tasks(self, client_id: int) -> List[Dict[str, torch.Tensor]]:
        """Generate meta-learning tasks for client personalization"""
        if client_id not in self.active_clients:
            raise ValueError(f"Client {client_id} not registered with this edge")
            
        return self.meta_learner.generate_tasks(
            model=self.model,
            client_history=self.client_performances[client_id],
            num_tasks=self.config.meta_tasks_per_client
        )
    
    def _update_client_states(self, client_updates: Dict[int, Dict]):
        """Update client performance and resource states"""
        for client_id, update in client_updates.items():
            # Update performance history
            self.client_performances[client_id].append(update['train_loss'])
            if len(self.client_performances[client_id]) > self.config.performance_history_size:
                self.client_performances[client_id].pop(0)
            
            # Update resource state if provided
            if 'resources' in update:
                self.client_resources[client_id] = update['resources']
                
            # Update communication stats
            if 'communication' in update:
                for key, value in update['communication'].items():
                    self.communication_stats[f"client_{client_id}_{key}"].append(value)
    
    def _compute_aggregation_weights(self, client_updates: Dict[int, Dict]) -> Dict[int, float]:
        """Compute adaptive weights for model aggregation"""
        weights = {}
        total_weight = 0.0
        
        for client_id in client_updates.keys():
            # Base weight from data size
            base_weight = len(client_updates[client_id].get('data_size', 1))
            
            # Performance factor
            performance_factor = self._compute_performance_factor(client_id)
            
            # Resource factor
            resource_factor = self._compute_resource_factor(client_id)
            
            # Reliability factor
            reliability_factor = self._compute_reliability_factor(client_id)
            
            # Combined weight
            weight = base_weight * performance_factor * resource_factor * reliability_factor
            weights[client_id] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for client_id in weights:
                weights[client_id] /= total_weight
                
        return weights
    
    def _compute_performance_factor(self, client_id: int) -> float:
        """Compute performance-based factor for aggregation"""
        if not self.client_performances[client_id]:
            return 1.0
            
        recent_losses = self.client_performances[client_id][-5:]
        avg_loss = np.mean(recent_losses)
        
        # Convert loss to factor (higher loss = lower weight)
        return np.exp(-avg_loss)
    
    def _compute_resource_factor(self, client_id: int) -> float:
        """Compute resource-based factor for aggregation"""
        if client_id not in self.client_resources:
            return 1.0
            
        resources = self.client_resources[client_id]
        
        # Combine different resource metrics
        compute_factor = resources.get('compute_capacity', 0.5)
        battery_factor = resources.get('battery_level', 100) / 100.0
        
        return compute_factor * (battery_factor ** 0.5)
    
    def _compute_reliability_factor(self, client_id: int) -> float:
        """Compute reliability-based factor for aggregation"""
        if client_id not in self.client_resources:
            return 1.0
            
        reliability = self.client_resources[client_id].get('network_reliability', 0.8)
        return reliability
    
    def _apply_knowledge_distillation(self, aggregated_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply knowledge distillation using buffered knowledge"""
        if not self.knowledge_buffer:
            return aggregated_state
            
        # Load aggregated state into temporary model
        temp_model = type(self.model)().to(self.device)
        temp_model.load_state_dict(aggregated_state)
        
        # Get most recent knowledge
        recent_knowledge = self.knowledge_buffer[-1]
        
        # Create teacher model with previous state
        teacher_model = type(self.model)().to(self.device)
        teacher_model.load_state_dict(recent_knowledge['teacher_state'])
        
        # Apply distillation
        distilled_model = self.distillation.distill_at_edge(
            student_model=temp_model,
            teacher_model=teacher_model,
            global_knowledge=recent_knowledge['global_knowledge']
        )
        
        return distilled_model.state_dict()
    
    def _evaluate_aggregation(self) -> Dict[str, float]:
        """Evaluate the aggregated model performance"""
        # This should be implemented based on validation data availability
        return {
            'loss': sum(self.client_performances[cid][-1] 
                       for cid in self.active_clients) / len(self.active_clients)
        }
    
    def get_edge_metrics(self) -> Dict[str, Any]:
        """Get current edge server metrics"""
        return {
            'num_clients': len(self.active_clients),
            'avg_client_performance': np.mean([
                np.mean(perfs) for perfs in self.client_performances.values()
                if perfs
            ]),
            'communication_stats': dict(self.communication_stats),
            'performance_history': self.performance_history
        }
    
    def prune_inactive_clients(self, inactive_threshold: int = 3):
        """Remove inactive clients based on update history"""
        current_time = time.time()
        
        inactive_clients = set()
        for client_id in self.active_clients:
            if len(self.client_performances[client_id]) < inactive_threshold:
                inactive_clients.add(client_id)
        
        for client_id in inactive_clients:
            self.unregister_client(client_id)
    
    def save_edge_state(self, path: str):
        """Save edge server state"""
        state = {
            'model_state': self.model.state_dict(),
            'client_performances': dict(self.client_performances),
            'client_resources': self.client_resources,
            'knowledge_buffer': self.knowledge_buffer,
            'performance_history': self.performance_history,
            'communication_stats': dict(self.communication_stats)
        }
        torch.save(state, path)
    
    def load_edge_state(self, path: str):
        """Load edge server state"""
        state = torch.load(path)
        self.model.load_state_dict(state['model_state'])
        self.client_performances = defaultdict(list, state['client_performances'])
        self.client_resources = state['client_resources']
        self.knowledge_buffer = state['knowledge_buffer']
        self.performance_history = state['performance_history']
        self.communication_stats = defaultdict(list, state['communication_stats'])