import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Optional
import torch

class DistributionTracker:
    """Track and analyze data distribution patterns"""
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.class_indices = defaultdict(set)
        self.access_counts = defaultdict(int)
        self.access_history = []
        self.distribution_stats = {}
        
        # Drift detection
        self.drift_window_size = 1000
        self.drift_threshold = 0.1
        self.distribution_history = []
        
    def initialize_distribution(self, targets: List[int]):
        """Initialize distribution tracking with targets"""
        for idx, target in enumerate(targets):
            self.class_counts[target] += 1
            self.class_indices[target].add(idx)
            
        self._compute_distribution_stats()
        
    def update_access(self, index: int, target: int):
        """Update access patterns"""
        self.access_counts[index] += 1
        self.access_history.append((index, target))
        
        # Keep limited history for drift detection
        if len(self.access_history) > self.drift_window_size:
            self.access_history.pop(0)
            
        # Check for distribution drift
        if len(self.access_history) == self.drift_window_size:
            self._check_distribution_drift()
    
    def get_class_indices(self, class_idx: int) -> Set[int]:
        """Get indices for a specific class"""
        return self.class_indices[class_idx]
    
    def get_stats(self) -> Dict:
        """Get current distribution statistics"""
        return {
            'class_counts': dict(self.class_counts),
            'access_patterns': self._analyze_access_patterns(),
            'distribution_stats': self.distribution_stats,
            'drift_detected': self._get_drift_status()
        }
    
    def _compute_distribution_stats(self):
        """Compute distribution statistics"""
        total_samples = sum(self.class_counts.values())
        class_probs = {
            cls: count/total_samples 
            for cls, count in self.class_counts.items()
        }
        
        # Compute entropy
        entropy = -sum(p * np.log2(p) for p in class_probs.values())
        
        # Compute imbalance ratio
        majority_class = max(class_probs.values())
        minority_class = min(class_probs.values())
        imbalance_ratio = majority_class / minority_class if minority_class > 0 else float('inf')
        
        self.distribution_stats = {
            'entropy': entropy,
            'imbalance_ratio': imbalance_ratio,
            'class_proportions': class_probs
        }
    
    def _analyze_access_patterns(self) -> Dict:
        """Analyze data access patterns"""
        if not self.access_history:
            return {}
            
        recent_accesses = self.access_history[-self.drift_window_size:]
        
        # Compute class frequencies in recent window
        class_freq = defaultdict(int)
        for _, target in recent_accesses:
            class_freq[target] += 1
            
        # Compute access concentration
        access_concentration = len(set(idx for idx, _ in recent_accesses)) / len(recent_accesses)
        
        return {
            'recent_class_frequencies': dict(class_freq),
            'access_concentration': access_concentration
        }
    
    def _check_distribution_drift(self):
        """Check for distribution drift"""
        if not self.distribution_history:
            # First window, just store distribution
            current_dist = self._get_current_distribution()
            self.distribution_history.append(current_dist)
            return
        
        current_dist = self._get_current_distribution()
        reference_dist = self.distribution_history[0]
        
        # Compute distribution difference
        drift_magnitude = self._compute_distribution_difference(
            current_dist, reference_dist
        )
        
        # Update distribution history
        self.distribution_history.append({
            'distribution': current_dist,
            'drift_magnitude': drift_magnitude
        })
        
        # Keep limited history
        if len(self.distribution_history) > 10:
            self.distribution_history.pop(0)
    
    def _get_current_distribution(self) -> Dict[int, float]:
        """Get current class distribution"""
        recent_accesses = self.access_history[-self.drift_window_size:]
        class_counts = defaultdict(int)
        
        for _, target in recent_accesses:
            class_counts[target] += 1
            
        total = len(recent_accesses)
        return {cls: count/total for cls, count in class_counts.items()}
    
    def _compute_distribution_difference(self, dist1: Dict[int, float],
                                      dist2: Dict[int, float]) -> float:
        """Compute difference between two distributions using KL divergence"""
        # Get union of classes
        all_classes = set(dist1.keys()) | set(dist2.keys())
        
        # Add smoothing for zero probabilities
        eps = 1e-10
        kl_div = 0
        
        for cls in all_classes:
            p = dist1.get(cls, 0) + eps
            q = dist2.get(cls, 0) + eps
            kl_div += p * np.log(p/q)
            
        return kl_div
    
    def _get_drift_status(self) -> Dict:
        """Get current drift status"""
        if len(self.distribution_history) < 2:
            return {'detected': False, 'magnitude': 0.0}
            
        recent_drifts = [d['drift_magnitude'] for d in self.distribution_history[-5:]]
        avg_drift = np.mean(recent_drifts)
        
        return {
            'detected': avg_drift > self.drift_threshold,
            'magnitude': avg_drift
        }
    
    def get_imbalanced_classes(self, threshold: float = 0.5) -> List[int]:
        """Get classes with significant imbalance"""
        if not self.distribution_stats:
            return []
            
        avg_proportion = 1.0 / len(self.class_counts)
        proportions = self.distribution_stats['class_proportions']
        
        return [
            cls for cls, prop in proportions.items()
            if prop < avg_proportion * threshold
        ]
    
    def suggest_rebalancing(self) -> Optional[Dict]:
        """Suggest rebalancing strategy"""
        imbalanced_classes = self.get_imbalanced_classes()
        if not imbalanced_classes:
            return None
            
        avg_samples = np.mean(list(self.class_counts.values()))
        
        suggestions = {}
        for cls in imbalanced_classes:
            current_samples = self.class_counts[cls]
            suggestions[cls] = {
                'current_samples': current_samples,
                'target_samples': int(avg_samples),
                'samples_needed': int(avg_samples - current_samples)
            }
            
        return suggestions
    
    def save_state(self, path: str):
        """Save tracker state"""
        state = {
            'class_counts': dict(self.class_counts),
            'distribution_stats': self.distribution_stats,
            'distribution_history': self.distribution_history
        }
        torch.save(state, path)
    
    def load_state(self, path: str):
        """Load tracker state"""
        state = torch.load(path)
        self.class_counts = defaultdict(int, state['class_counts'])
        self.distribution_stats = state['distribution_stats']
        self.distribution_history = state['distribution_history']