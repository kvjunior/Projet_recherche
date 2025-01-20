import torch
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from collections import defaultdict

class DynamicTierManager:
    def __init__(self, config):
        self.config = config
        self.min_tiers = config.min_tiers
        self.max_tiers = config.max_tiers
        
        # Metrics for tier adaptation
        self.communication_costs = []
        self.computation_loads = []
        self.data_distributions = []
        
    def form_initial_tiers(self, clients, edges, cloud) -> Dict[str, Any]:
        """Form initial tier structure based on initial conditions"""
        tier_structure = {
            'num_tiers': self.min_tiers,
            'tier_assignments': {},
            'tier_connections': defaultdict(list)
        }
        
        # Analyze initial conditions
        client_metrics = self._analyze_clients(clients)
        network_conditions = self._analyze_network_conditions(clients, edges)
        
        # Form initial clusters based on analyzed metrics
        tier_structure['tier_assignments'] = self._cluster_nodes(
            clients=clients,
            edges=edges,
            client_metrics=client_metrics,
            network_conditions=network_conditions
        )
        
        # Establish initial connections between tiers
        tier_structure['tier_connections'] = self._establish_connections(
            tier_structure['tier_assignments']
        )
        
        return tier_structure
    
    def adapt_tiers(self, current_structure: Dict, metrics: Dict) -> Dict[str, Any]:
        """Adapt tier structure based on runtime metrics"""
        new_structure = current_structure.copy()
        
        # 1. Analyze adaptation needs
        should_adapt, adaptation_type = self._analyze_adaptation_needs(metrics)
        
        if not should_adapt:
            return current_structure
            
        # 2. Perform adaptation based on type
        if adaptation_type == 'split':
            new_structure = self._split_tier(current_structure, metrics)
        elif adaptation_type == 'merge':
            new_structure = self._merge_tiers(current_structure, metrics)
        elif adaptation_type == 'reassign':
            new_structure = self._reassign_nodes(current_structure, metrics)
            
        # 3. Update connections
        new_structure['tier_connections'] = self._establish_connections(
            new_structure['tier_assignments']
        )
        
        return new_structure
    
    def _analyze_clients(self, clients) -> Dict[int, Dict]:
        """Analyze client characteristics"""
        client_metrics = {}
        
        for client in clients:
            metrics = {
                'compute_capacity': self._measure_compute_capacity(client),
                'data_distribution': self._analyze_data_distribution(client),
                'network_reliability': self._measure_network_reliability(client)
            }
            client_metrics[client.id] = metrics
            
        return client_metrics
    
    def _analyze_network_conditions(self, clients, edges) -> Dict:
        """Analyze network conditions between nodes"""
        network_conditions = {
            'latency_matrix': self._measure_latency_matrix(clients, edges),
            'bandwidth_matrix': self._measure_bandwidth_matrix(clients, edges)
        }
        return network_conditions
    
    def _cluster_nodes(self, clients, edges, client_metrics, network_conditions) -> Dict:
        """Cluster nodes into tiers based on characteristics"""
        # Prepare feature matrix for clustering
        features = self._prepare_clustering_features(
            clients, edges, client_metrics, network_conditions
        )
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=self.min_tiers-1,  # -1 because cloud is always top tier
            random_state=42
        )
        cluster_assignments = kmeans.fit_predict(features)
        
        # Convert clustering results to tier assignments
        tier_assignments = self._convert_clusters_to_tiers(
            clients, edges, cluster_assignments
        )
        
        return tier_assignments
    
    def _establish_connections(self, tier_assignments: Dict) -> Dict[int, List]:
        """Establish connections between tiers"""
        connections = defaultdict(list)
        
        # Sort tiers by level
        tier_levels = sorted(set(tier_assignments.values()))
        
        # Connect each tier to the tier above it
        for i in range(len(tier_levels)-1):
            current_tier = tier_levels[i]
            next_tier = tier_levels[i+1]
            
            # Get nodes in current and next tier
            current_nodes = [n for n, t in tier_assignments.items() if t == current_tier]
            next_nodes = [n for n, t in tier_assignments.items() if t == next_tier]
            
            # Establish connections based on optimal routing
            connections.update(
                self._optimize_tier_connections(current_nodes, next_nodes)
            )
            
        return connections
    
    def _analyze_adaptation_needs(self, metrics: Dict) -> tuple:
        """Analyze if and how the tier structure should be adapted"""
        # Extract relevant metrics
        communication_cost = metrics['communication_cost'][-1]
        accuracy = metrics['test_accuracy'][-1]
        tier_efficiency = self._compute_tier_efficiency(metrics)
        
        # Define adaptation thresholds
        COMM_THRESHOLD = self.config.communication_threshold
        ACC_THRESHOLD = self.config.accuracy_threshold
        EFF_THRESHOLD = self.config.efficiency_threshold
        
        # Determine adaptation need and type
        if communication_cost > COMM_THRESHOLD and tier_efficiency < EFF_THRESHOLD:
            return True, 'split'
        elif communication_cost < COMM_THRESHOLD/2 and accuracy > ACC_THRESHOLD:
            return True, 'merge'
        elif tier_efficiency < EFF_THRESHOLD:
            return True, 'reassign'
            
        return False, None
    
    def _split_tier(self, current_structure: Dict, metrics: Dict) -> Dict:
        """Split a tier into two tiers"""
        new_structure = current_structure.copy()
        
        # Identify tier to split based on metrics
        tier_to_split = self._identify_tier_to_split(metrics)
        
        # Get nodes in the tier
        nodes_in_tier = [n for n, t in current_structure['tier_assignments'].items() 
                        if t == tier_to_split]
        
        # Split nodes into two groups
        split_assignments = self._perform_tier_split(nodes_in_tier, metrics)
        
        # Update tier assignments
        new_structure['tier_assignments'].update(split_assignments)
        new_structure['num_tiers'] += 1
        
        return new_structure
    
    def _merge_tiers(self, current_structure: Dict, metrics: Dict) -> Dict:
        """Merge two tiers into one"""
        new_structure = current_structure.copy()
        
        # Identify tiers to merge
        tier1, tier2 = self._identify_tiers_to_merge(metrics)
        
        # Merge tier assignments
        new_assignments = self._perform_tier_merge(
            current_structure['tier_assignments'],
            tier1,
            tier2
        )
        
        new_structure['tier_assignments'] = new_assignments
        new_structure['num_tiers'] -= 1
        
        return new_structure
    
    def _reassign_nodes(self, current_structure: Dict, metrics: Dict) -> Dict:
        """Reassign nodes to different tiers"""
        new_structure = current_structure.copy()
        
        # Identify nodes to reassign
        nodes_to_reassign = self._identify_nodes_to_reassign(metrics)
        
        # Compute new assignments
        new_assignments = self._compute_new_assignments(
            nodes_to_reassign,
            current_structure,
            metrics
        )
        
        new_structure['tier_assignments'].update(new_assignments)
        
        return new_structure
    
    def _compute_tier_efficiency(self, metrics: Dict) -> float:
        """Compute efficiency metric for current tier structure"""
        communication_cost = np.mean(metrics['communication_cost'][-5:])
        accuracy = np.mean(metrics['test_accuracy'][-5:])
        adaptability = self._compute_adaptability(metrics)
        
        # Normalize metrics
        norm_comm_cost = communication_cost / self.config.max_communication_cost
        norm_accuracy = accuracy / 1.0  # accuracy is already between 0 and 1
        norm_adaptability = adaptability / 1.0  # adaptability is normalized
        
        # Weighted combination
        efficiency = (
            self.config.weight_comm * (1 - norm_comm_cost) +
            self.config.weight_acc * norm_accuracy +
            self.config.weight_adapt * norm_adaptability
        )
        
        return efficiency
    
    def _compute_adaptability(self, metrics: Dict) -> float:
        """Compute adaptability score based on historical adaptations"""
        if len(metrics['tier_adaptations']) < 2:
            return 1.0
            
        # Analyze recent adaptations
        recent_adaptations = metrics['tier_adaptations'][-5:]
        
        # Check if adaptations improved performance
        improvements = []
        for i in range(len(recent_adaptations) - 1):
            before_acc = metrics['test_accuracy'][recent_adaptations[i]]
            after_acc = metrics['test_accuracy'][recent_adaptations[i + 1]]
            improvements.append(1 if after_acc > before_acc else 0)
            
        adaptability = sum(improvements) / len(improvements)
        return adaptability
    
    def _measure_compute_capacity(self, client) -> float:
        """Measure computational capacity of a client"""
        if hasattr(client, 'compute_capacity'):
            return client.compute_capacity
            
        # Default estimation based on batch processing time
        try:
            start_time = time.time()
            client.process_dummy_batch()
            processing_time = time.time() - start_time
            
            # Normalize between 0 and 1
            capacity = 1.0 / (1.0 + processing_time)
            return capacity
        except:
            return 0.5  # Default medium capacity
    
    def _analyze_data_distribution(self, client) -> np.ndarray:
        """Analyze data distribution characteristics of a client"""
        # Get data distribution metrics
        try:
            distribution = client.get_data_distribution()
            return distribution
        except:
            return np.ones(self.config.num_classes) / self.config.num_classes
    
    def _measure_network_reliability(self, client) -> float:
        """Measure network reliability of a client"""
        if hasattr(client, 'network_reliability'):
            return client.network_reliability
            
        # Use recent communication success rate
        try:
            success_rate = client.get_communication_success_rate()
            return success_rate
        except:
            return 0.8  # Default reliability
    
    def _measure_latency_matrix(self, clients, edges) -> np.ndarray:
        """Measure latency between all nodes"""
        num_nodes = len(clients) + len(edges)
        latency_matrix = np.zeros((num_nodes, num_nodes))
        
        # Measure or estimate latencies
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                latency = self._estimate_latency(i, j)
                latency_matrix[i, j] = latency
                latency_matrix[j, i] = latency
                
        return latency_matrix
    
    def _measure_bandwidth_matrix(self, clients, edges) -> np.ndarray:
        """Measure bandwidth between all nodes"""
        num_nodes = len(clients) + len(edges)
        bandwidth_matrix = np.zeros((num_nodes, num_nodes))
        
        # Measure or estimate bandwidth
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                bandwidth = self._estimate_bandwidth(i, j)
                bandwidth_matrix[i, j] = bandwidth
                bandwidth_matrix[j, i] = bandwidth
                
        return bandwidth_matrix
    
    def _estimate_latency(self, node1_idx: int, node2_idx: int) -> float:
        """Estimate latency between two nodes"""
        # This could be replaced with actual network measurements
        base_latency = 0.1  # base latency in seconds
        distance_factor = abs(node1_idx - node2_idx) * 0.01
        return base_latency + distance_factor
    
    def _estimate_bandwidth(self, node1_idx: int, node2_idx: int) -> float:
        """Estimate bandwidth between two nodes"""
        # This could be replaced with actual network measurements
        base_bandwidth = 100  # base bandwidth in Mbps
        distance_penalty = abs(node1_idx - node2_idx) * 5
        return max(base_bandwidth - distance_penalty, 10)  # minimum 10 Mbps
    
    def _prepare_clustering_features(self, clients, edges, client_metrics, network_conditions) -> np.ndarray:
        """Prepare feature matrix for clustering"""
        num_nodes = len(clients) + len(edges)
        num_features = 4  # compute, network, data_dist, location
        features = np.zeros((num_nodes, num_features))
        
        # Combine all relevant metrics into feature matrix
        for i, client in enumerate(clients):
            metrics = client_metrics[client.id]
            features[i] = [
                metrics['compute_capacity'],
                metrics['network_reliability'],
                np.mean(metrics['data_distribution']),
                i / num_nodes  # normalized location
            ]
            
        # Add edge features
        for i, edge in enumerate(edges, start=len(clients)):
            features[i] = [
                1.0,  # high compute capacity
                0.95,  # high reliability
                0.5,  # neutral data distribution
                i / num_nodes  # normalized location
            ]
            
        return features
    
    def _identify_tier_to_split(self, metrics: Dict) -> int:
        """Identify which tier should be split"""
        # Analyze tier loads and performance
        tier_loads = self._compute_tier_loads(metrics)
        tier_performances = self._compute_tier_performances(metrics)
        
        # Find most overloaded tier
        return max(tier_loads.items(), key=lambda x: x[1])[0]
    
    def _identify_tiers_to_merge(self, metrics: Dict) -> tuple:
        """Identify which tiers should be merged"""
        # Analyze tier efficiency
        tier_loads = self._compute_tier_loads(metrics)
        
        # Find two least loaded adjacent tiers
        sorted_tiers = sorted(tier_loads.items(), key=lambda x: x[1])
        return sorted_tiers[0][0], sorted_tiers[1][0]
    
    def _compute_tier_loads(self, metrics: Dict) -> Dict[int, float]:
        """Compute load for each tier"""
        tier_loads = defaultdict(float)
        
        # Analyze recent metrics to determine tier loads
        for metric in metrics['communication_cost'][-5:]:
            # Add load computation logic here
            pass
            
        return tier_loads
    
    def _compute_tier_performances(self, metrics: Dict) -> Dict[int, float]:
        """Compute performance metrics for each tier"""
        tier_performances = defaultdict(float)
        
        # Analyze recent metrics to determine tier performance
        for acc in metrics['test_accuracy'][-5:]:
            # Add performance computation logic here
            pass
            
        return tier_performances