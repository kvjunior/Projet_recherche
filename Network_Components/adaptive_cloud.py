import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict
import time
from sklearn.cluster import KMeans

class AdaptiveCloud:
    def __init__(self, model: nn.Module, config: Any, device: torch.device):
        """Initialize cloud server with global pattern extraction capabilities"""
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Edge server management
        self.active_edges = set()
        self.edge_performances = defaultdict(list)
        self.edge_resources = {}
        
        # Global state
        self.global_patterns = {}
        self.pattern_evolution = []
        self.performance_history = []
        
        # Knowledge management
        self.knowledge_bank = {
            'patterns': [],
            'feature_stats': defaultdict(list),
            'model_stats': defaultdict(list)
        }
        
        # Initialize pattern extraction
        self.feature_extractor = self._build_feature_extractor()
        
    def register_edge(self, edge_id: int, initial_state: Dict[str, Any]):
        """Register a new edge server"""
        self.active_edges.add(edge_id)
        self.edge_resources[edge_id] = initial_state
        
    def aggregate_edge_models(self, edge_updates: Dict[int, Dict],
                            round_number: int) -> Dict[str, Any]:
        """Aggregate edge models with global pattern extraction"""
        if not edge_updates:
            return None
            
        # Extract and update global patterns
        self._update_global_patterns(edge_updates)
        
        # Compute adaptive weights
        aggregation_weights = self._compute_edge_weights(edge_updates)
        
        # Pattern-aware aggregation
        aggregated_state = {}
        for key in edge_updates[list(edge_updates.keys())[0]]['model_state'].keys():
            # Get all parameter tensors for this key
            params = [update['model_state'][key] for update in edge_updates.values()]
            
            # Apply pattern-based aggregation
            aggregated_param = self._pattern_based_aggregation(
                params, key, aggregation_weights
            )
            aggregated_state[key] = aggregated_param
        
        # Update global model
        self.model.load_state_dict(aggregated_state)
        
        # Extract global knowledge
        global_knowledge = self._extract_global_knowledge()
        
        # Evaluate and store performance
        performance = self._evaluate_global_model()
        self.performance_history.append(performance)
        
        return {
            'model_state': aggregated_state,
            'global_knowledge': global_knowledge,
            'performance': performance
        }
    
    def _update_global_patterns(self, edge_updates: Dict[int, Dict]):
        """Extract and update global patterns from edge updates"""
        # Extract features from edge models
        edge_features = {}
        for edge_id, update in edge_updates.items():
            features = self._extract_model_features(update['model_state'])
            edge_features[edge_id] = features
        
        # Cluster features to identify patterns
        patterns = self._cluster_model_features(edge_features)
        
        # Update pattern evolution
        self.pattern_evolution.append(patterns)
        if len(self.pattern_evolution) > self.config.pattern_history_size:
            self.pattern_evolution.pop(0)
        
        # Update global patterns
        self.global_patterns = self._merge_patterns(
            self.global_patterns,
            patterns
        )
    
    def _extract_model_features(self, model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from model parameters"""
        features = []
        
        for name, param in model_state.items():
            # Compute statistical features
            flat_param = param.view(-1).cpu().numpy()
            stats = {
                'mean': np.mean(flat_param),
                'std': np.std(flat_param),
                'sparsity': np.sum(np.abs(flat_param) < 1e-6) / len(flat_param)
            }
            
            features.extend([stats['mean'], stats['std'], stats['sparsity']])
            
            # Store in knowledge bank
            self.knowledge_bank['feature_stats'][name].append(stats)
        
        return torch.tensor(features)
    
    def _cluster_model_features(self, edge_features: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """Cluster model features to identify patterns"""
        features = torch.stack(list(edge_features.values()))
        
        # Determine optimal number of clusters
        max_clusters = min(len(features), self.config.max_patterns)
        best_n_clusters = self._find_optimal_clusters(features, max_clusters)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
        clusters = kmeans.fit_transform(features.numpy())
        
        return {
            'centroids': kmeans.cluster_centers_,
            'assignments': kmeans.labels_,
            'edge_clusters': dict(zip(edge_features.keys(), kmeans.labels_))
        }
    
    def _find_optimal_clusters(self, features: torch.Tensor, max_clusters: int) -> int:
        """Find optimal number of clusters using elbow method"""
        if len(features) < 3:
            return len(features)
            
        inertias = []
        for k in range(1, min(max_clusters + 1, len(features) + 1)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features.numpy())
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        diffs = np.diff(inertias)
        rel_diffs = diffs[1:] / diffs[:-1]
        elbow = np.argmin(rel_diffs) + 2
        
        return int(elbow)
    
    def _merge_patterns(self, old_patterns: Dict[str, Any],
                       new_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Merge old and new patterns"""
        if not old_patterns:
            return new_patterns
            
        # Compute similarity between old and new centroids
        old_centroids = old_patterns['centroids']
        new_centroids = new_patterns['centroids']
        
        similarity = np.zeros((len(old_centroids), len(new_centroids)))
        for i, old_c in enumerate(old_centroids):
            for j, new_c in enumerate(new_centroids):
                similarity[i, j] = np.linalg.norm(old_c - new_c)
        
        # Match similar patterns
        merged_centroids = []
        used_old = set()
        used_new = set()
        
        while len(used_old) < len(old_centroids) and len(used_new) < len(new_centroids):
            i, j = np.unravel_index(similarity.argmin(), similarity.shape)
            if i not in used_old and j not in used_new:
                # Merge centroids
                merged_centroids.append(
                    0.7 * old_centroids[i] + 0.3 * new_centroids[j]
                )
                used_old.add(i)
                used_new.add(j)
            similarity[i, j] = float('inf')
        
        # Add remaining centroids
        for i in range(len(old_centroids)):
            if i not in used_old:
                merged_centroids.append(old_centroids[i])
        
        for j in range(len(new_centroids)):
            if j not in used_new:
                merged_centroids.append(new_centroids[j])
        
        return {
            'centroids': np.array(merged_centroids),
            'assignments': new_patterns['assignments'],
            'edge_clusters': new_patterns['edge_clusters']
        }
    
    def _compute_edge_weights(self, edge_updates: Dict[int, Dict]) -> Dict[int, float]:
        """Compute adaptive weights for edge aggregation"""
        weights = {}
        total_weight = 0.0
        
        for edge_id in edge_updates.keys():
            # Base weight from data size
            base_weight = edge_updates[edge_id].get('total_samples', 1)
            
            # Performance weight
            performance_weight = self._compute_performance_weight(edge_id)
            
            # Pattern similarity weight
            pattern_weight = self._compute_pattern_weight(edge_id)
            
            # Resource weight
            resource_weight = self._compute_resource_weight(edge_id)
            
            # Combined weight
            weight = base_weight * performance_weight * pattern_weight * resource_weight
            weights[edge_id] = weight
            total_weight += weight
        
        # Normalize weights
        return {eid: w/total_weight for eid, w in weights.items()}
        
    def _pattern_based_aggregation(self, params: List[torch.Tensor], 
                                 key: str,
                                 weights: Dict[int, float]) -> torch.Tensor:
        """Perform pattern-aware parameter aggregation"""
        if key not in self.knowledge_bank['model_stats']:
            # Fall back to weighted average if no pattern info
            return sum(p * w for p, w in zip(params, weights.values()))
            
        # Get pattern info for this layer
        pattern_info = self.knowledge_bank['model_stats'][key][-1]
        
        if 'pattern_mask' in pattern_info:
            # Apply pattern-based masking
            mask = pattern_info['pattern_mask']
            masked_params = [p * mask for p in params]
            return sum(p * w for p, w in zip(masked_params, weights.values()))
        
        return sum(p * w for p, w in zip(params, weights.values()))
    
    def _compute_performance_weight(self, edge_id: int) -> float:
        """Compute performance-based weight factor"""
        if not self.edge_performances[edge_id]:
            return 1.0
            
        recent_perfs = self.edge_performances[edge_id][-5:]
        avg_perf = np.mean(recent_perfs)
        
        # Convert to weight (higher performance = higher weight)
        return 1.0 + np.tanh(avg_perf)
    
    def _compute_pattern_weight(self, edge_id: int) -> float:
        """Compute pattern-similarity based weight"""
        if not self.global_patterns or 'edge_clusters' not in self.global_patterns:
            return 1.0
            
        # Get cluster assignment for this edge
        cluster_id = self.global_patterns['edge_clusters'].get(edge_id, -1)
        if cluster_id == -1:
            return 1.0
            
        # Count edges in same cluster
        cluster_size = sum(1 for cid in self.global_patterns['edge_clusters'].values() 
                         if cid == cluster_id)
                         
        # Reward edges in larger clusters
        return 1.0 + (cluster_size / len(self.active_edges))
    
    def _compute_resource_weight(self, edge_id: int) -> float:
        """Compute resource-based weight factor"""
        if edge_id not in self.edge_resources:
            return 1.0
            
        resources = self.edge_resources[edge_id]
        
        # Combine different resource metrics
        computation_cap = resources.get('computation_capacity', 0.5)
        bandwidth = resources.get('bandwidth', 0.5)
        reliability = resources.get('reliability', 0.8)
        
        return computation_cap * bandwidth * reliability
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction network"""
        try:
            # Try to use model's feature extraction capability if available
            if hasattr(self.model, 'extract_features'):
                return self.model.extract_features
            
            # Otherwise, use the model up to the last layer
            layers = list(self.model.children())[:-1]
            return nn.Sequential(*layers)
        except:
            return None
    
    def _extract_global_knowledge(self) -> Dict[str, Any]:
        """Extract global knowledge for knowledge distillation"""
        return {
            'patterns': self.global_patterns,
            'feature_distributions': self._compute_feature_distributions(),
            'layer_importance': self._compute_layer_importance()
        }
    
    def _compute_feature_distributions(self) -> Dict[str, Dict[str, float]]:
        """Compute statistical distributions of features"""
        distributions = {}
        
        for name, stats_list in self.knowledge_bank['feature_stats'].items():
            if not stats_list:
                continue
                
            recent_stats = stats_list[-self.config.feature_window_size:]
            distributions[name] = {
                'mean': np.mean([s['mean'] for s in recent_stats]),
                'std': np.mean([s['std'] for s in recent_stats]),
                'sparsity': np.mean([s['sparsity'] for s in recent_stats])
            }
            
        return distributions
    
    def _compute_layer_importance(self) -> Dict[str, float]:
        """Compute importance scores for each layer"""
        importance = {}
        
        for name, stats_list in self.knowledge_bank['feature_stats'].items():
            if not stats_list:
                continue
                
            # Compute variation in layer statistics
            recent_stats = stats_list[-self.config.feature_window_size:]
            variations = np.std([s['mean'] for s in recent_stats])
            
            # Higher variation = higher importance
            importance[name] = float(variations)
            
        # Normalize importance scores
        if importance:
            max_importance = max(importance.values())
            importance = {k: v/max_importance for k, v in importance.items()}
            
        return importance
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model performance"""
        # This should be implemented based on validation data availability
        if self.performance_history:
            return {'loss': self.performance_history[-1]['loss']}
        return {'loss': float('inf')}
    
    def save_cloud_state(self, path: str):
        """Save cloud server state"""
        state = {
            'model_state': self.model.state_dict(),
            'global_patterns': self.global_patterns,
            'pattern_evolution': self.pattern_evolution,
            'knowledge_bank': self.knowledge_bank,
            'performance_history': self.performance_history
        }
        torch.save(state, path)
    
    def load_cloud_state(self, path: str):
        """Load cloud server state"""
        state = torch.load(path)
        self.model.load_state_dict(state['model_state'])
        self.global_patterns = state['global_patterns']
        self.pattern_evolution = state['pattern_evolution']
        self.knowledge_bank = state['knowledge_bank']
        self.performance_history = state['performance_history']