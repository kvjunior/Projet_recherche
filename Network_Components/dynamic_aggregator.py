import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import ot  # optimal transport

class DynamicAggregator:
    def __init__(self, config: Any):
        """Initialize dynamic aggregator with different strategies"""
        self.config = config
        self.strategies = {
            'client': self._client_tier_strategy,
            'edge': self._edge_tier_strategy,
            'cloud': self._cloud_tier_strategy
        }
        
        # Tracking aggregation performance
        self.aggregation_stats = defaultdict(list)
        
    def aggregate(self, tier: str,
                 updates: Dict[int, Dict[str, torch.Tensor]],
                 metadata: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """Aggregate updates using tier-specific strategy"""
        if tier not in self.strategies:
            raise ValueError(f"Unknown tier: {tier}")
            
        # Select and apply appropriate strategy
        strategy = self.strategies[tier]
        aggregated_state = strategy(updates, metadata)
        
        # Track performance
        self._update_aggregation_stats(tier, updates, aggregated_state)
        
        return aggregated_state
    
    def _client_tier_strategy(self, updates: Dict[int, Dict[str, torch.Tensor]],
                            metadata: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """Client tier strategy: Contrastive learning based aggregation"""
        if not updates:
            return {}
            
        # Extract features for contrastive learning
        features = self._extract_update_features(updates)
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(features)
        
        # Compute weights using contrastive learning
        weights = self._contrastive_weights(similarity_matrix)
        
        # Weighted aggregation
        aggregated_state = {}
        for key in updates[list(updates.keys())[0]].keys():
            weighted_sum = torch.stack([
                update[key] * weights[i]
                for i, update in enumerate(updates.values())
            ]).sum(dim=0)
            aggregated_state[key] = weighted_sum
            
        return aggregated_state
    
    def _edge_tier_strategy(self, updates: Dict[int, Dict[str, torch.Tensor]],
                          metadata: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """Edge tier strategy: Attention-based aggregation"""
        if not updates:
            return {}
            
        # Extract features for attention
        features = self._extract_update_features(updates)
        
        # Compute attention scores
        attention_scores = self._compute_attention_weights(features)
        
        # Get performance weights if available
        performance_weights = self._get_performance_weights(metadata)
        
        # Combine attention and performance weights
        combined_weights = self._combine_weights(attention_scores, performance_weights)
        
        # Attention-weighted aggregation
        aggregated_state = {}
        for key in updates[list(updates.keys())[0]].keys():
            params = [update[key] for update in updates.values()]
            aggregated_state[key] = self._attention_aggregation(params, combined_weights)
            
        return aggregated_state
    
    def _cloud_tier_strategy(self, updates: Dict[int, Dict[str, torch.Tensor]],
                           metadata: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """Cloud tier strategy: Optimal transport based aggregation"""
        if not updates:
            return {}
            
        # Extract features for optimal transport
        features = self._extract_update_features(updates)
        
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(features)
        
        # Solve optimal transport problem
        transport_weights = self._solve_optimal_transport(cost_matrix)
        
        # Apply transport plan to updates
        aggregated_state = {}
        for key in updates[list(updates.keys())[0]].keys():
            params = [update[key] for update in updates.values()]
            aggregated_state[key] = self._transport_aggregation(params, transport_weights)
            
        return aggregated_state
    
    def _extract_update_features(self, updates: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Extract features from model updates for similarity computation"""
        features = []
        
        for update in updates.values():
            # Compute statistical features for each layer
            layer_features = []
            for param in update.values():
                flat_param = param.view(-1).cpu().numpy()
                stats = [
                    np.mean(flat_param),
                    np.std(flat_param),
                    np.percentile(flat_param, 25),
                    np.percentile(flat_param, 75)
                ]
                layer_features.extend(stats)
            features.append(layer_features)
            
        return torch.tensor(features)
    
    def _compute_similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix between updates"""
        # Normalize features
        normalized_features = nn.functional.normalize(features, dim=1)
        
        # Compute cosine similarity
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # Apply temperature scaling
        similarity_matrix /= self.config.temperature
        
        return similarity_matrix
    
    def _contrastive_weights(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """Compute weights using contrastive learning"""
        # Apply softmax to get probabilities
        weights = nn.functional.softmax(similarity_matrix, dim=1)
        
        # Average across all pairs
        weights = weights.mean(dim=1)
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def _compute_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Compute attention weights for edge-tier aggregation"""
        # Project features to query, key, value spaces
        query = features
        key = features
        value = features
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.t())
        attention_scores = attention_scores / np.sqrt(features.shape[1])
        
        # Apply softmax
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        
        return attention_weights
    
    def _get_performance_weights(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """Extract performance-based weights from metadata"""
        if not metadata or 'performance' not in metadata:
            return None
            
        performances = torch.tensor([
            metadata['performance'].get(i, 0.0)
            for i in range(len(metadata['performance']))
        ])
        
        # Normalize performances to weights
        weights = nn.functional.softmax(performances, dim=0)
        
        return weights
    
    def _combine_weights(self, attention_weights: torch.Tensor,
                        performance_weights: torch.Tensor = None) -> torch.Tensor:
        """Combine attention and performance weights"""
        if performance_weights is None:
            return attention_weights
            
        # Combine using adaptive weighting
        alpha = self.config.performance_weight
        combined = (1 - alpha) * attention_weights + alpha * performance_weights
        
        return combined
    
    def _compute_cost_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute cost matrix for optimal transport"""
        # Use Euclidean distance as cost
        n = len(features)
        cost_matrix = torch.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                cost_matrix[i,j] = torch.norm(features[i] - features[j])
                
        return cost_matrix
    
    def _solve_optimal_transport(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Solve optimal transport problem"""
        n = len(cost_matrix)
        
        # Source and target distributions (uniform)
        a = torch.ones(n) / n
        b = torch.ones(n) / n
        
        # Solve optimal transport
        transport_matrix = ot.sinkhorn(
            a.numpy(),
            b.numpy(),
            cost_matrix.numpy(),
            reg=self.config.ot_regularization
        )
        
        return torch.from_numpy(transport_matrix)
    
    def _transport_aggregation(self, params: List[torch.Tensor],
                             transport_weights: torch.Tensor) -> torch.Tensor:
        """Aggregate parameters using optimal transport plan"""
        n = len(params)
        aggregated_param = torch.zeros_like(params[0])
        
        for i in range(n):
            for j in range(n):
                aggregated_param += transport_weights[i,j] * params[j]
                
        return aggregated_param
    
    def _attention_aggregation(self, params: List[torch.Tensor],
                             attention_weights: torch.Tensor) -> torch.Tensor:
        """Aggregate parameters using attention weights"""
        aggregated_param = torch.zeros_like(params[0])
        
        for i, param in enumerate(params):
            aggregated_param += attention_weights[i] * param
            
        return aggregated_param
    
    def _update_aggregation_stats(self, tier: str,
                                updates: Dict[int, Dict[str, torch.Tensor]],
                                aggregated_state: Dict[str, torch.Tensor]):
        """Update aggregation performance statistics"""
        # Compute diversity metric
        diversity = self._compute_update_diversity(updates)
        
        # Compute compression ratio
        compression = self._compute_compression_ratio(updates, aggregated_state)
        
        # Store stats
        self.aggregation_stats[f"{tier}_diversity"].append(diversity)
        self.aggregation_stats[f"{tier}_compression"].append(compression)
    
    def _compute_update_diversity(self, updates: Dict[int, Dict[str, torch.Tensor]]) -> float:
        """Compute diversity metric for updates"""
        if len(updates) < 2:
            return 0.0
            
        # Compute pairwise distances between updates
        distances = []
        update_list = list(updates.values())
        
        for i in range(len(update_list)):
            for j in range(i + 1, len(update_list)):
                distance = 0
                for key in update_list[i].keys():
                    param_dist = torch.norm(
                        update_list[i][key] - update_list[j][key]
                    ).item()
                    distance += param_dist
                distances.append(distance)
                
        return np.mean(distances) if distances else 0.0
    
    def _compute_compression_ratio(self, updates: Dict[int, Dict[str, torch.Tensor]],
                                 aggregated_state: Dict[str, torch.Tensor]) -> float:
        """Compute compression ratio achieved by aggregation"""
        input_size = sum(
            param.numel() * param.element_size()
            for update in updates.values()
            for param in update.values()
        )
        
        output_size = sum(
            param.numel() * param.element_size()
            for param in aggregated_state.values()
        )
        
        return input_size / output_size if output_size > 0 else 0.0
    
    def get_aggregation_stats(self) -> Dict[str, List[float]]:
        """Get aggregation performance statistics"""
        return dict(self.aggregation_stats)
    
    def save_aggregator_state(self, path: str):
        """Save aggregator state"""
        state = {
            'config': self.config.__dict__,
            'aggregation_stats': dict(self.aggregation_stats)
        }
        torch.save(state, path)
    
    def load_aggregator_state(self, path: str):
        """Load aggregator state"""
        state = torch.load(path)
        self.config.__dict__.update(state['config'])
        self.aggregation_stats = defaultdict(list, state['aggregation_stats'])