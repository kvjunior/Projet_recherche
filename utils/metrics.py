import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time
from sklearn.metrics import confusion_matrix, roc_auc_score

class MetricsTracker:
    """Track and analyze various performance metrics"""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_round = 0
        self.start_time = time.time()
        
    def update(self, metrics_dict: Dict[str, float], round_num: Optional[int] = None):
        """Update metrics with new values"""
        if round_num is not None:
            self.current_round = round_num
            
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
            
    def get_current_metrics(self) -> Dict[str, float]:
        """Get most recent metrics"""
        return {key: values[-1] if values else 0.0 
                for key, values in self.metrics.items()}
        
    def get_average_metrics(self, window_size: int = 5) -> Dict[str, float]:
        """Get average metrics over recent window"""
        return {key: np.mean(values[-window_size:]) if values else 0.0
                for key, values in self.metrics.items()}
                
    def get_metric_history(self) -> Dict[str, List[float]]:
        """Get complete history of metrics"""
        return dict(self.metrics)

class AdaptiveMetrics:
    """Metrics for adaptive federated learning"""
    def __init__(self):
        self.tier_metrics = defaultdict(MetricsTracker)
        self.adaptation_history = []
        self.communication_costs = []
        self.computation_times = []
        
    def log_adaptation(self, event: Dict[str, Any]):
        """Log adaptation event"""
        event['timestamp'] = time.time()
        self.adaptation_history.append(event)
        
    def update_tier_metrics(self, tier: str,
                          metrics: Dict[str, float],
                          round_num: int):
        """Update metrics for specific tier"""
        self.tier_metrics[tier].update(metrics, round_num)
        
    def log_communication(self, source: str,
                         destination: str,
                         data_size: int):
        """Log communication cost"""
        self.communication_costs.append({
            'timestamp': time.time(),
            'source': source,
            'destination': destination,
            'size': data_size
        })
        
    def log_computation(self, node_id: str,
                       operation: str,
                       duration: float):
        """Log computation time"""
        self.computation_times.append({
            'timestamp': time.time(),
            'node': node_id,
            'operation': operation,
            'duration': duration
        })
        
    def get_tier_performance(self, tier: str) -> Dict[str, Any]:
        """Get performance metrics for tier"""
        metrics = self.tier_metrics[tier]
        current = metrics.get_current_metrics()
        average = metrics.get_average_metrics()
        
        return {
            'current': current,
            'average': average,
            'trend': self._compute_trend(metrics.get_metric_history())
        }
        
    def get_system_efficiency(self) -> Dict[str, float]:
        """Get overall system efficiency metrics"""
        if not self.communication_costs or not self.computation_times:
            return {}
            
        recent_comm = self.communication_costs[-100:]
        recent_comp = self.computation_times[-100:]
        
        total_data = sum(log['size'] for log in recent_comm)
        total_time = sum(log['duration'] for log in recent_comp)
        
        return {
            'communication_efficiency': total_data / len(recent_comm),
            'computation_efficiency': total_time / len(recent_comp),
            'adaptations_per_round': len(self.adaptation_history) / max(1, self.tier_metrics['cloud'].current_round)
        }
        
    def get_adaptation_effectiveness(self) -> Dict[str, float]:
        """Analyze effectiveness of adaptations"""
        if not self.adaptation_history:
            return {}
            
        effectiveness = defaultdict(list)
        for i in range(1, len(self.adaptation_history)):
            prev_event = self.adaptation_history[i-1]
            curr_event = self.adaptation_history[i]
            
            # Compare metrics before and after adaptation
            for metric in ['accuracy', 'loss', 'communication_cost']:
                if metric in prev_event and metric in curr_event:
                    improvement = curr_event[metric] - prev_event[metric]
                    effectiveness[f'{metric}_improvement'].append(improvement)
                    
        return {key: np.mean(values) for key, values in effectiveness.items()}
        
    def _compute_trend(self, metric_history: Dict[str, List[float]]) -> Dict[str, str]:
        """Compute trend direction for metrics"""
        trends = {}
        for metric, values in metric_history.items():
            if len(values) < 2:
                trends[metric] = 'stable'
                continue
                
            recent = values[-5:]
            slope = np.polyfit(range(len(recent)), recent, 1)[0]
            
            if abs(slope) < 0.01:
                trends[metric] = 'stable'
            elif slope > 0:
                trends[metric] = 'improving'
            else:
                trends[metric] = 'degrading'
                
        return trends

class PrivacyMetrics:
    """Track privacy-related metrics"""
    def __init__(self, epsilon: float, delta: float):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_costs = defaultdict(list)
        
    def log_privacy_cost(self, tier: str,
                        epsilon_cost: float,
                        delta_cost: float):
        """Log privacy cost for operation"""
        self.privacy_costs[tier].append({
            'timestamp': time.time(),
            'epsilon': epsilon_cost,
            'delta': delta_cost
        })
        
    def get_privacy_budget_consumed(self) -> Dict[str, Dict[str, float]]:
        """Get consumed privacy budget per tier"""
        consumption = {}
        for tier, costs in self.privacy_costs.items():
            total_epsilon = sum(cost['epsilon'] for cost in costs)
            total_delta = sum(cost['delta'] for cost in costs)
            
            consumption[tier] = {
                'epsilon_consumed': total_epsilon,
                'epsilon_remaining': max(0, self.epsilon - total_epsilon),
                'delta_consumed': total_delta,
                'delta_remaining': max(0, self.delta - total_delta)
            }
            
        return consumption

class ModelPerformanceTracker:
    """Track model performance metrics"""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.predictions = []
        self.targets = []
        self.losses = []
        
    def update(self, predictions: torch.Tensor,
               targets: torch.Tensor,
               loss: float):
        """Update with new predictions"""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
        
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive performance metrics"""
        if not self.predictions or not self.targets:
            return {}
            
        # Convert to numpy arrays
        y_pred = np.array(self.predictions)
        y_true = np.array(self.targets)
        
        metrics = {}
        
        # Basic metrics
        metrics['average_loss'] = np.mean(self.losses)
        metrics['accuracy'] = np.mean(y_pred == y_true)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        # Per-class metrics
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            # Precision, recall, and F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f'class_{i}_precision'] = precision
            metrics[f'class_{i}_recall'] = recall
            metrics[f'class_{i}_f1'] = f1
        
        # Macro-averaged metrics
        metrics['macro_precision'] = np.mean([metrics[f'class_{i}_precision'] for i in range(self.num_classes)])
        metrics['macro_recall'] = np.mean([metrics[f'class_{i}_recall'] for i in range(self.num_classes)])
        metrics['macro_f1'] = np.mean([metrics[f'class_{i}_f1'] for i in range(self.num_classes)])
        
        # ROC AUC score (if applicable)
        if self.num_classes == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            except:
                metrics['roc_auc'] = 0.0
                
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if not self.predictions or not self.targets:
            return np.zeros((self.num_classes, self.num_classes))
            
        return confusion_matrix(self.targets, self.predictions,
                              labels=range(self.num_classes))
    
    def reset(self):
        """Reset tracker state"""
        self.predictions = []
        self.targets = []
        self.losses = []

class ResourceMetrics:
    """Track resource utilization metrics"""
    def __init__(self):
        self.resource_usage = defaultdict(list)
        self.bottlenecks = defaultdict(int)
        
    def log_resource_usage(self, node_id: str,
                          metrics: Dict[str, float]):
        """Log resource usage for node"""
        metrics['timestamp'] = time.time()
        self.resource_usage[node_id].append(metrics)
        
        # Check for bottlenecks
        self._check_bottlenecks(node_id, metrics)
        
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage"""
        summary = defaultdict(dict)
        
        for node_id, usage_history in self.resource_usage.items():
            if not usage_history:
                continue
                
            recent = usage_history[-10:]
            for metric in recent[0].keys():
                if metric != 'timestamp':
                    values = [u[metric] for u in recent]
                    summary[node_id][metric] = {
                        'current': values[-1],
                        'average': np.mean(values),
                        'peak': max(values)
                    }
                    
        return dict(summary)
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Get analysis of system bottlenecks"""
        total_incidents = sum(self.bottlenecks.values())
        if total_incidents == 0:
            return {}
            
        analysis = {
            'total_bottlenecks': total_incidents,
            'bottleneck_distribution': {
                node: count / total_incidents
                for node, count in self.bottlenecks.items()
            }
        }
        
        # Add resource correlation analysis
        analysis['resource_correlations'] = self._compute_resource_correlations()
        
        return analysis
    
    def _check_bottlenecks(self, node_id: str,
                          metrics: Dict[str, float]):
        """Check for resource bottlenecks"""
        # CPU bottleneck
        if metrics.get('cpu_utilization', 0) > 90:
            self.bottlenecks[f'{node_id}_cpu'] += 1
            
        # Memory bottleneck
        if metrics.get('memory_utilization', 0) > 90:
            self.bottlenecks[f'{node_id}_memory'] += 1
            
        # Network bottleneck
        if metrics.get('network_latency', 0) > 1.0:  # 1 second threshold
            self.bottlenecks[f'{node_id}_network'] += 1
    
    def _compute_resource_correlations(self) -> Dict[str, float]:
        """Compute correlations between different resources"""
        correlations = {}
        
        for node_id, usage_history in self.resource_usage.items():
            if len(usage_history) < 2:
                continue
                
            metrics = usage_history[0].keys()
            for m1 in metrics:
                for m2 in metrics:
                    if m1 >= m2 or m1 == 'timestamp' or m2 == 'timestamp':
                        continue
                        
                    values1 = [u[m1] for u in usage_history]
                    values2 = [u[m2] for u in usage_history]
                    
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    correlations[f'{node_id}_{m1}_vs_{m2}'] = correlation
                    
        return correlations
    
    def save_metrics(self, path: str):
        """Save metrics to file"""
        state = {
            'resource_usage': dict(self.resource_usage),
            'bottlenecks': dict(self.bottlenecks)
        }
        torch.save(state, path)
    
    def load_metrics(self, path: str):
        """Load metrics from file"""
        state = torch.load(path)
        self.resource_usage = defaultdict(list, state['resource_usage'])
        self.bottlenecks = defaultdict(int, state['bottlenecks'])