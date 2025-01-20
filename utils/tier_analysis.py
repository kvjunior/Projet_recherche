import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import networkx as nx

class TierAnalyzer:
    """Analyze and optimize tier structure"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tier_history = []
        self.performance_metrics = defaultdict(list)
        self.network_metrics = defaultdict(list)
        
    def analyze_tier_structure(self, 
                             node_metrics: Dict[str, Dict[str, float]],
                             network_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze current tier structure and suggest optimizations"""
        # Record current state
        self.tier_history.append({
            'node_metrics': node_metrics,
            'network_stats': network_stats,
            'timestamp': time.time()
        })
        
        # Compute tier-level metrics
        tier_metrics = self._compute_tier_metrics(node_metrics)
        
        # Analyze network topology
        topology_analysis = self._analyze_network_topology(network_stats)
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(node_metrics, network_stats)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(
            tier_metrics, topology_analysis, bottlenecks
        )
        
        return {
            'tier_metrics': tier_metrics,
            'topology_analysis': topology_analysis,
            'bottlenecks': bottlenecks,
            'suggestions': suggestions
        }
    
    def suggest_tier_adaptation(self, 
                              current_structure: Dict[str, Any],
                              performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Suggest adaptations to tier structure"""
        # Update performance history
        for metric, value in performance_metrics.items():
            self.performance_metrics[metric].append(value)
        
        # Analyze performance trends
        trends = self._analyze_performance_trends()
        
        # Check adaptation criteria
        should_adapt = self._check_adaptation_criteria(trends)
        
        if not should_adapt:
            return {
                'adapt': False,
                'reason': 'Current structure is performing well'
            }
        
        # Determine optimal number of tiers
        optimal_tiers = self._determine_optimal_tiers(current_structure)
        
        # Generate new structure
        new_structure = self._generate_new_structure(
            current_structure,
            optimal_tiers
        )
        
        return {
            'adapt': True,
            'new_structure': new_structure,
            'reason': self._get_adaptation_reason(trends)
        }
    
    def _compute_tier_metrics(self, 
                            node_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each tier"""
        tier_metrics = defaultdict(dict)
        
        for node_id, metrics in node_metrics.items():
            tier = self._get_node_tier(node_id)
            
            # Compute average metrics per tier
            for metric, value in metrics.items():
                if metric not in tier_metrics[tier]:
                    tier_metrics[tier][metric] = []
                tier_metrics[tier][metric].append(value)
        
        # Calculate statistics
        for tier in tier_metrics:
            for metric in tier_metrics[tier]:
                values = tier_metrics[tier][metric]
                tier_metrics[tier][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return dict(tier_metrics)
    
    def _analyze_network_topology(self, 
                                network_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze network topology between tiers"""
        # Create graph
        G = nx.Graph()
        
        # Add edges with weights based on network metrics
        for connection, stats in network_stats.items():
            source, dest = connection.split('_')
            weight = stats.get('latency', 0) + 1/max(stats.get('bandwidth', 1e-6), 1e-6)
            G.add_edge(source, dest, weight=weight)
        
        # Compute topology metrics
        analysis = {
            'diameter': nx.diameter(G),
            'average_path_length': nx.average_shortest_path_length(G),
            'clustering_coefficient': nx.average_clustering(G),
            'centrality': {
                node: score for node, score in 
                nx.betweenness_centrality(G).items()
            }
        }
        
        # Identify critical paths
        critical_paths = self._identify_critical_paths(G)
        analysis['critical_paths'] = critical_paths
        
        return analysis
    
    def _detect_bottlenecks(self,
                           node_metrics: Dict[str, Dict[str, float]],
                           network_stats: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Detect system bottlenecks"""
        bottlenecks = []
        
        # Check computational bottlenecks
        for node_id, metrics in node_metrics.items():
            if metrics.get('cpu_utilization', 0) > 90 or \
               metrics.get('memory_utilization', 0) > 90:
                bottlenecks.append({
                    'type': 'computational',
                    'node': node_id,
                    'metrics': metrics
                })
        
        # Check network bottlenecks
        for connection, stats in network_stats.items():
            if stats.get('latency', 0) > self.config['latency_threshold'] or \
               stats.get('bandwidth', float('inf')) < self.config['bandwidth_threshold']:
                bottlenecks.append({
                    'type': 'network',
                    'connection': connection,
                    'stats': stats
                })
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self,
                                        tier_metrics: Dict[str, Dict[str, float]],
                                        topology_analysis: Dict[str, Any],
                                        bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Check tier balance
        balance_suggestion = self._check_tier_balance(tier_metrics)
        if balance_suggestion:
            suggestions.append(balance_suggestion)
        
        # Check topology optimization
        topology_suggestion = self._check_topology_optimization(topology_analysis)
        if topology_suggestion:
            suggestions.append(topology_suggestion)
        
        # Handle bottlenecks
        for bottleneck in bottlenecks:
            suggestion = self._generate_bottleneck_suggestion(bottleneck)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _analyze_performance_trends(self) -> Dict[str, str]:
        """Analyze performance metric trends"""
        trends = {}
        window_size = min(10, len(next(iter(self.performance_metrics.values()))))
        
        for metric, values in self.performance_metrics.items():
            recent = values[-window_size:]
            slope = np.polyfit(range(window_size), recent, 1)[0]
            
            if abs(slope) < 0.01:
                trends[metric] = 'stable'
            elif slope > 0:
                trends[metric] = 'improving'
            else:
                trends[metric] = 'degrading'
        
        return trends
    
    def _check_adaptation_criteria(self, trends: Dict[str, str]) -> bool:
        """Check if tier adaptation is needed"""
        # Check for degrading performance
        degrading_metrics = sum(1 for trend in trends.values() 
                              if trend == 'degrading')
        
        return degrading_metrics >= len(trends) // 2
    
    def _determine_optimal_tiers(self, 
                               current_structure: Dict[str, Any]) -> int:
        """Determine optimal number of tiers"""
        features = self._extract_node_features(current_structure)
        
        # Try different numbers of tiers
        scores = []
        for n_tiers in range(self.config['min_tiers'], self.config['max_tiers'] + 1):
            if len(features) < n_tiers:
                continue
                
            kmeans = KMeans(n_clusters=n_tiers, random_state=42)
            labels = kmeans.fit_predict(features)
            
            # Compute clustering quality scores
            silhouette = silhouette_score(features, labels)
            calinski = calinski_harabasz_score(features, labels)
            
            scores.append({
                'n_tiers': n_tiers,
                'silhouette': silhouette,
                'calinski': calinski
            })
        
        if not scores:
            return self.config['min_tiers']
            
        # Select optimal number of tiers
        return max(scores, key=lambda x: x['silhouette'])['n_tiers']
    
    def _extract_node_features(self, 
                             structure: Dict[str, Any]) -> np.ndarray:
        """Extract features for node clustering"""
        features = []
        
        for node_id, node_data in structure['nodes'].items():
            node_features = [
                node_data.get('compute_capacity', 0),
                node_data.get('network_reliability', 0),
                node_data.get('data_size', 0),
                node_data.get('importance_score', 0)
            ]
            features.append(node_features)
            
        return np.array(features)
    
    def _generate_new_structure(self,
                              current_structure: Dict[str, Any],
                              optimal_tiers: int) -> Dict[str, Any]:
        """Generate new tier structure"""
        features = self._extract_node_features(current_structure)
        
        # Cluster nodes
        kmeans = KMeans(n_clusters=optimal_tiers, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # Create new structure
        new_structure = {
            'n_tiers': optimal_tiers,
            'nodes': {},
            'connections': []
        }
        
        # Assign nodes to tiers
        for node_idx, (node_id, node_data) in enumerate(current_structure['nodes'].items()):
            tier = int(labels[node_idx])
            node_data['tier'] = tier
            new_structure['nodes'][node_id] = node_data
        
        # Establish connections
        new_structure['connections'] = self._establish_tier_connections(new_structure)
        
        return new_structure
    
    def _establish_tier_connections(self,
                                  structure: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Establish connections between tiers"""
        connections = []
        nodes_by_tier = defaultdict(list)
        
        # Group nodes by tier
        for node_id, node_data in structure['nodes'].items():
            nodes_by_tier[node_data['tier']].append(node_id)
        
        # Connect adjacent tiers
        tiers = sorted(nodes_by_tier.keys())
        for i in range(len(tiers)-1):
            current_tier = tiers[i]
            next_tier = tiers[i+1]
            
            # Connect each node in current tier to nearest node in next tier
            for node1 in nodes_by_tier[current_tier]:
                nearest = self._find_nearest_node(
                    node1,
                    nodes_by_tier[next_tier],
                    structure
                )
                connections.append((node1, nearest))
        
        return connections
    
    def _find_nearest_node(self,
                          source: str,
                          candidates: List[str],
                          structure: Dict[str, Any]) -> str:
        """Find nearest node from candidates"""
        min_distance = float('inf')
        nearest = None
        
        source_data = structure['nodes'][source]
        for candidate in candidates:
            candidate_data = structure['nodes'][candidate]
            distance = self._compute_node_distance(source_data, candidate_data)
            
            if distance < min_distance:
                min_distance = distance
                nearest = candidate
                
        return nearest
    
    def _compute_node_distance(self,
                             node1_data: Dict[str, Any],
                             node2_data: Dict[str, Any]) -> float:
        """Compute distance between nodes based on features"""
        features1 = np.array([
            node1_data.get('compute_capacity', 0),
            node1_data.get('network_reliability', 0),
            node1_data.get('data_size', 0)
        ])
        
        features2 = np.array([
            node2_data.get('compute_capacity', 0),
            node2_data.get('network_reliability', 0),
            node2_data.get('data_size', 0)
        ])
        
        return np.linalg.norm(features1 - features2)